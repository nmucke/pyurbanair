"""Isolated execution entry point. All native output belongs in worker.log."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import signal
import sys
import threading
import time
import traceback
from typing import Any

from mcp_server.jobs.registry import atomic_json


class Cancelled(BaseException):
    pass


def execute(job: dict[str, Any]) -> None:
    root = pathlib.Path(job["run_root"])
    payload = job["payload"]
    phase = "preparing"
    stopping = threading.Event()
    build_lock = None

    def heartbeat() -> None:
        while not stopping.is_set():
            atomic_json(
                root / "worker_status.json",
                {
                    "phase": phase,
                    "heartbeat": time.time(),
                    "pid": os.getpid(),
                },
            )
            stopping.wait(1)

    def cancel(signum: int, frame: Any) -> None:
        raise Cancelled(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    outcome: dict[str, Any] = {"state": "failed"}
    try:
        if payload["kind"] == "forward":
            from mcp_server.jobs.paths import bind_job_paths
            from mcp_server.jobs.preparation import (
                PreparationService,
                load_script,
                verify_worker_toolchain_environment,
            )
            from omegaconf import OmegaConf

            service = PreparationService(job["repo_root"], payload["store_root"])
            plan = service.verify(payload["plan_id"], check_environment=False)
            if plan["digest"] != payload["plan_digest"]:
                raise ValueError("Prepared plan digest changed after enqueueing")
            verify_worker_toolchain_environment(plan, dict(os.environ))
            config, environment = bind_job_paths(plan["config"], root)
            for directory in environment.values():
                pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            os.environ.pop("PYLBM_LBM_PATH", None)
            os.environ["PYPALM_SKIP_AUTOINSTALL"] = "1"
            os.environ["PYLBM_AUTOSYNC_SUBMODULE"] = "0"
            os.environ.update(environment)
            working = root / "work"
            working.mkdir(exist_ok=True)
            os.chdir(working)
            atomic_json(
                root / "launch.json",
                {
                    "plan": plan,
                    "effective_config": config,
                    "environment": {
                        **environment,
                        "PYLBM_LBM_PATH": None,
                        "PYPALM_SKIP_AUTOINSTALL": "1",
                        "PYLBM_AUTOSYNC_SUBMODULE": "0",
                    },
                    "shared_build_lock": str(
                        pathlib.Path(job["repo_root"]) / ".temp/local-backend.lock"
                    ),
                    "python": sys.executable,
                    "started": time.time(),
                },
            )
            # Shared source/cache mutations are a declared exception to job paths.
            # One checkout-wide lock also covers clients using different stores.
            lock_path = pathlib.Path(job["repo_root"]) / ".temp" / "local-backend.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            build_lock = lock_path.open("a")
            fcntl.flock(build_lock, fcntl.LOCK_EX)
            run_forward = load_script(job["repo_root"], "scripts/run_forward.py")

            if (root / "cancel.requested").exists():
                raise Cancelled("Cancelled during preparation")
            phase = "running"
            run_forward.run(OmegaConf.create(config))
        elif payload["kind"] == "visualization":
            from visualization import render

            phase = "running"
            render(
                pathlib.Path(payload["source_root"]),
                root / "bundle",
                payload.get("options"),
            )
        else:
            raise ValueError(f"Unsupported worker kind {payload['kind']!r}")
        phase = "finalizing"
        outcome = {"state": "succeeded", "finished": time.time()}
    except Cancelled as exc:
        outcome = {"state": "cancelled", "error": str(exc), "phase": phase}
    except BaseException as exc:
        traceback.print_exc()
        outcome = {
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "phase": phase,
        }
    finally:
        if build_lock is not None:
            build_lock.close()
        stopping.set()
        thread.join(timeout=2)
        atomic_json(root / "completion.json", outcome)
    if outcome["state"] != "succeeded":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=pathlib.Path, required=True)
    args = parser.parse_args()
    execute(json.loads(args.job.read_text()))


if __name__ == "__main__":
    main()
