"""Queue and process lifecycle tests that never import or start a CFD backend."""

from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import sys
import time
from typing import Any

import pytest

from pyurbanair.jobs.processes import identity, matching, owned, signal_owned
from pyurbanair.jobs.registry import TERMINAL, Registry
from pyurbanair.jobs.results import inspect_results
from pyurbanair.jobs.supervisor import Supervisor

REPO = pathlib.Path(__file__).resolve().parents[2]


def wait_until(service: Supervisor, predicate: Any, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        service.tick()
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError(f"Jobs did not settle: {service.registry.unfinished()}")


def fake_command(job: dict[str, Any]) -> list[str]:
    root = pathlib.Path(job["run_root"])
    mode = job["payload"].get("mode", "success")
    code = """
import json, pathlib, subprocess, sys, time, signal
root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
print('native output captured', flush=True)
if mode == 'detached':
    child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'], start_new_session=True)
    (root / 'child.pid').write_text(str(child.pid))
    time.sleep(60)
elif mode == 'sleep':
    time.sleep(60)
elif mode == 'failure':
    raise RuntimeError('intentional fake solver failure')
else:
    (root / 'completion.json').write_text(json.dumps({'state': 'succeeded'}))
"""
    return [sys.executable, "-c", code, str(root), mode]


def test_registry_atomic_idempotency_and_logs(tmp_path: pathlib.Path) -> None:
    registry = Registry(tmp_path)
    payload = {"kind": "forward", "plan_id": "one"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(
            pool.map(lambda _: Registry(tmp_path).submit(payload, "same"), range(16))
        )
    assert len({job["id"] for job in jobs}) == 1
    with pytest.raises(ValueError, match="different inputs"):
        registry.submit({**payload, "plan_id": "two"}, "same")
    job = jobs[0]
    root = pathlib.Path(job["run_root"])
    root.mkdir(parents=True)
    (root / "worker.log").write_text("abcdefgh")
    assert registry.logs(job["id"], limit=3)["text"] == "abc"
    assert registry.logs(job["id"], cursor=3, limit=3)["text"] == "def"
    with pytest.raises(ValueError):
        registry.logs(job["id"], cursor=-1)
    assert registry.cancel(job["id"])["state"] == "cancelled"
    assert registry.cancel(job["id"])["state"] == "cancelled"


def test_worker_uses_launcher_pixi_outside_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    python = repo / ".pixi/envs/dev/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    pixi = tmp_path / "desktop-pixi"
    pixi.write_text("#!/bin/sh\nexit 0\n")
    pixi.chmod(0o700)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("PYURBANAIR_PIXI", str(pixi))
    supervisor = Supervisor(repo, tmp_path / "store")
    command = supervisor.worker_command(
        {"payload": {}, "run_root": str(tmp_path / "run")}
    )
    assert command[0] == str(pixi)


def test_queue_failure_and_recovery(tmp_path: pathlib.Path) -> None:
    supervisor = Supervisor(
        REPO, tmp_path, command_factory=fake_command, grace_seconds=0.1
    )
    one = supervisor.registry.submit({"kind": "forward", "mode": "failure"}, "one")
    two = supervisor.registry.submit({"kind": "forward"}, "two")
    supervisor.tick()
    assert supervisor.registry.get(one["id"])["state"] == "preparing"
    assert supervisor.registry.get(two["id"])["state"] == "queued"
    wait_until(
        supervisor, lambda: supervisor.registry.get(two["id"])["state"] in TERMINAL
    )
    assert supervisor.registry.get(one["id"])["state"] == "failed"
    assert supervisor.registry.get(two["id"])["state"] == "succeeded"
    assert "native output" in supervisor.registry.logs(two["id"])["text"]


def test_cancellation_detached_child_and_reconnect(tmp_path: pathlib.Path) -> None:
    first = Supervisor(REPO, tmp_path, command_factory=fake_command, grace_seconds=0.1)
    job = first.registry.submit({"kind": "forward", "mode": "detached"}, "cancel")
    first.tick()
    child_file = pathlib.Path(job["run_root"]) / "child.pid"
    try:
        wait_until(first, child_file.exists)
        child_identity = identity(int(child_file.read_text()))
        # A new supervisor has no Popen handles; persistent identity still suffices.
        recovered = Supervisor(
            REPO, tmp_path, command_factory=fake_command, grace_seconds=0.1
        )
        recovered.registry.cancel(job["id"])
        wait_until(
            recovered, lambda: recovered.registry.get(job["id"])["state"] == "cancelled"
        )
        assert matching(child_identity) is None
        first.children[job["id"]].wait(timeout=3)
    finally:
        current = first.registry.get(job["id"])
        signal_owned(owned(current["details"]["token"]), force=True)


def test_preparation_cancellation_and_reused_pid(tmp_path: pathlib.Path) -> None:
    supervisor = Supervisor(
        REPO, tmp_path, command_factory=fake_command, grace_seconds=0.1
    )
    job = supervisor.registry.submit(
        {"kind": "forward", "mode": "sleep"}, "cancel-prepare"
    )
    supervisor.tick()
    assert supervisor.registry.get(job["id"])["state"] == "preparing"
    supervisor.registry.cancel(job["id"])
    wait_until(
        supervisor, lambda: supervisor.registry.get(job["id"])["state"] == "cancelled"
    )
    assert matching({"pid": os.getpid(), "created": 0}) is None
    stale = supervisor.registry.submit({"kind": "forward"}, "stale")
    supervisor.registry.update(
        stale["id"],
        "running",
        token="not-a-real-token",
        process={"pid": os.getpid(), "created": 0},
    )
    supervisor.tick()
    assert supervisor.registry.get(stale["id"])["state"] == "interrupted"


def test_result_selection_bounds_and_containment(tmp_path: pathlib.Path) -> None:
    import numpy as np
    import xarray as xr

    xr.Dataset({"u": (("x", "y"), np.ones((100, 100)))}).to_netcdf(
        tmp_path / "state.nc"
    )
    index: dict[str, Any] = {
        "status": "complete",
        "artifacts": [{"kind": "state", "path": "state.nc"}],
    }
    (tmp_path / "artifact_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="4096"):
        inspect_results(tmp_path, 0, "u")
    assert inspect_results(tmp_path, 0, "u", {"x": 0, "y": [0, 3]})["selection"][
        "values"
    ] == [1, 1, 1]
    index["artifacts"][0]["path"] = "../outside.nc"
    (tmp_path / "artifact_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="owned run"):
        inspect_results(tmp_path, 0)


def test_supervisor_clients_share_queue_and_reconnect(tmp_path: pathlib.Path) -> None:
    from pyurbanair.jobs.supervisor import SupervisorClient

    clients = [SupervisorClient(REPO, tmp_path) for _ in range(3)]
    processes = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            pings = list(pool.map(lambda client: client.request("ping"), clients))
        assert len({ping["pid"] for ping in pings}) == 1
        processes.append(identity(pings[0]["pid"]))
        # No numerical launch: an already-cancelled registry record survives restart.
        registry = Registry(tmp_path)
        job = registry.submit({"kind": "forward"}, "persisted")
        registry.cancel(job["id"])
        signal_owned(processes, force=True)
        deadline = time.monotonic() + 5
        while matching(processes[0]) is not None and time.monotonic() < deadline:
            time.sleep(0.03)
        assert clients[1].request("status", job_id=job["id"])["state"] == "cancelled"
        processes.append(identity(clients[1].request("ping")["pid"]))
        assert processes[0] != processes[1]
    finally:
        signal_owned(processes, force=True)


def test_post_render_failure_preserves_simulation(tmp_path: pathlib.Path) -> None:
    def command(job: dict[str, Any]) -> list[str]:
        if job["kind"] == "visualization":
            job = {**job, "payload": {**job["payload"], "mode": "failure"}}
        return fake_command(job)

    supervisor = Supervisor(REPO, tmp_path, command_factory=command)
    parent = supervisor.registry.submit(
        {"kind": "forward", "post_render": {"member": 0}}, "auto-render"
    )
    wait_until(
        supervisor,
        lambda: supervisor.registry.get(parent["id"])["state"] == "succeeded",
    )
    parent = supervisor.registry.get(parent["id"])
    render_id = parent["details"]["visualization_id"]
    assert render_id
    wait_until(
        supervisor, lambda: supervisor.registry.get(render_id)["state"] in TERMINAL
    )
    assert supervisor.registry.get(render_id)["state"] == "failed"
    assert supervisor.registry.get(parent["id"])["state"] == "succeeded"


def test_solver_input_inspection_is_bounded_text(tmp_path: pathlib.Path) -> None:
    (tmp_path / "infile.in").write_text("a" * 40000)
    (tmp_path / "artifact_index.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "artifacts": [{"kind": "solver_input", "path": "infile.in"}],
            }
        )
    )
    result = inspect_results(tmp_path, artifact_id=0)
    assert result["truncated"] and len(result["text"]) == 32768
    with pytest.raises(ValueError, match="text inspection"):
        inspect_results(tmp_path, artifact_id=0, variable="u")
