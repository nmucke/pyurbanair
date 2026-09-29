"""One detached local supervisor owns admission, workers and viewer serving."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from pyurbanair.jobs.paths import ensure_private_directory
from pyurbanair.jobs.processes import TOKEN_ENV, identity, matching, owned, signal_owned
from pyurbanair.jobs.registry import ACTIVE, Registry, atomic_json
from pyurbanair.jobs.rendering_environment import select_render_environment

MAX_MESSAGE = 4 * 1024 * 1024


def socket_path(root: pathlib.Path) -> pathlib.Path:
    digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]
    directory = pathlib.Path(tempfile.gettempdir()) / f"pyurbanair-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    stat = directory.lstat()
    if directory.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError(f"Supervisor socket directory must be private: {directory}")
    return directory / f"{digest}.sock"


class Supervisor:
    def __init__(
        self,
        repo_root: pathlib.Path,
        root: pathlib.Path,
        *,
        command_factory: Callable[[dict[str, Any]], list[str]] | None = None,
        grace_seconds: float = 5,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.registry = Registry(root)
        self.command_factory = command_factory or self.worker_command
        self.grace_seconds = grace_seconds
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self.asset_server: Any = None
        self.last_request = time.monotonic()
        self._lock = threading.RLock()

    def worker_command(self, job: dict[str, Any]) -> list[str]:
        environment = job["payload"].get("environment", "dev")
        if environment not in {"dev", "cuda", "rendering"}:
            raise ValueError(
                "Workers require the dev, cuda or rendering Pixi environment"
            )
        python = self.repo_root / ".pixi" / "envs" / environment / "bin" / "python"
        if not python.is_file():
            raise ValueError(
                f"Pixi environment {environment!r} is not installed; run pixi install -e {environment}"
            )
        pixi = shutil.which(os.environ.get("PYURBANAIR_PIXI", "pixi"))
        if not pixi:
            raise ValueError(
                "pixi is unavailable; set PYURBANAIR_PIXI to its absolute path"
            )
        return [
            pixi,
            "run",
            "--locked",
            "--manifest-path",
            str(self.repo_root / "pyproject.toml"),
            "-e",
            environment,
            "python",
            "-m",
            "pyurbanair.jobs.worker",
            "--job",
            str(pathlib.Path(job["run_root"]) / "job.json"),
        ]

    def dispatch(self, operation: str, args: dict[str, Any]) -> Any:
        self.last_request = time.monotonic()
        with self._lock:
            if operation == "ping":
                return {
                    "pid": os.getpid(),
                    "root": str(self.registry.root),
                    "repo_root": str(self.repo_root),
                }
            if operation == "existing":
                return self.registry.existing(args["payload"], args["key"])
            if operation == "submit":
                return self.registry.submit(args["payload"], args["key"])
            if operation == "status":
                return self.registry.get(args["job_id"])
            if operation == "list":
                return self.registry.list(**args)
            if operation == "logs":
                return self.registry.logs(**args)
            if operation == "cancel":
                return self.registry.cancel(args["job_id"])
            if operation == "viewer":
                job = self.registry.get(args["job_id"])
                if job["kind"] != "visualization" or job["state"] != "succeeded":
                    raise ValueError("Viewer requires a successful visualization job")
                if self.asset_server is None or self.asset_server.closed:
                    from pyurbanair.visualization.assets import BundleAssetServer

                    self.asset_server = BundleAssetServer()
                return {
                    "url": self.asset_server.register(
                        pathlib.Path(job["run_root"]) / "bundle"
                    )
                }
            raise ValueError(f"Unknown supervisor operation: {operation}")

    def _start(self, job: dict[str, Any]) -> None:
        root = pathlib.Path(job["run_root"])
        root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        job["repo_root"] = str(self.repo_root)
        atomic_json(root / "job.json", job)
        self.registry.update(job["id"], "preparing", token=token, heartbeat=time.time())
        try:
            command = self.command_factory(job)
            env = dict(os.environ)
            # Prevent an activated MCP environment from contaminating worker imports.
            for key in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "JAX_PLATFORMS"):
                env.pop(key, None)
            env.update(
                {TOKEN_ENV: token, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
            )
            env["PYTHONPATH"] = str(self.repo_root / "src")
            with (root / "worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.children[job["id"]] = process
            self.registry.update(
                job["id"], process=identity(process.pid), command=command
            )
        except Exception as exc:
            self.registry.update(
                job["id"], "failed", error=str(exc), phase="preparation"
            )

    def _poll(self, job: dict[str, Any]) -> None:
        details = job["details"]
        root = pathlib.Path(job["run_root"])
        token = details.get("token")
        if not token:
            self.registry.update(
                job["id"],
                "interrupted",
                error="No verified worker identity after supervisor restart",
            )
            return
        processes = owned(token, details.get("descendants", []))
        process = self.children.get(job["id"])
        exit_code = process.poll() if process is not None else None
        running = matching(details.get("process", {})) is not None
        completion_path = root / "completion.json"
        completion = (
            json.loads(completion_path.read_text())
            if completion_path.exists()
            else None
        )
        worker_status_path = root / "worker_status.json"
        worker_status = (
            json.loads(worker_status_path.read_text())
            if worker_status_path.exists()
            else {}
        )
        cancelling = job["state"] == "cancelling"
        # Also tear down descendants left behind by a worker that exited or crashed.
        teardown = cancelling or not running
        if teardown and processes:
            started = details.get("teardown_started", time.time())
            (root / "cancel.requested").touch()
            signal_owned(processes, force=time.time() - started >= self.grace_seconds)
            self.registry.update(
                job["id"],
                teardown_started=started,
                descendants=processes,
                heartbeat=time.time(),
            )
            return
        if not running and not processes:
            if cancelling:
                state = "cancelled"
            elif completion is not None:
                state = completion.get("state", "failed")
                if state == "succeeded" and exit_code not in (None, 0):
                    state = "failed"
            else:
                state = "failed" if process is not None else "interrupted"
            visualization_id = None
            render_options = job["payload"].get("post_render")
            if (
                state == "succeeded"
                and job["kind"] == "forward"
                and render_options is not None
            ):
                try:
                    render_job = self.registry.submit(
                        {
                            "kind": "visualization",
                            "parent_run_id": job["id"],
                            "source_root": job["run_root"],
                            "options": render_options,
                            "environment": select_render_environment(
                                self.repo_root, render_options
                            ),
                        },
                        f"post-render:{job['id']}",
                    )
                    visualization_id = render_job["id"]
                except Exception as exc:
                    self.registry.update(job["id"], post_render_error=str(exc))

            self.registry.update(
                job["id"],
                state,
                visualization_id=visualization_id,
                exit_code=exit_code,
                completion=completion,
                phase=worker_status.get("phase"),
                error=(completion or {}).get(
                    "error",
                    (
                        "Worker exited without a completion record"
                        if completion is None
                        else None
                    ),
                ),
                descendants=[],
                finished=time.time(),
            )
            self.children.pop(job["id"], None)
            return
        phase = worker_status.get("phase", "preparing")
        state = (
            phase if phase in {"preparing", "running", "finalizing"} else job["state"]
        )
        self.registry.update(
            job["id"],
            state,
            heartbeat=worker_status.get("heartbeat", time.time()),
            descendants=processes,
            phase=phase,
        )

    def tick(self) -> None:
        with self._lock:
            jobs = self.registry.unfinished()
            for job in jobs:
                if job["state"] in ACTIVE:
                    self._poll(job)
            jobs = self.registry.unfinished()
            if not any(job["state"] in ACTIVE for job in jobs):
                queued = next((job for job in jobs if job["state"] == "queued"), None)
                if queued is not None:
                    self._start(queued)

    def close(self) -> None:
        if self.asset_server is not None:
            self.asset_server.close()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(30)
        try:
            line = self.rfile.readline(MAX_MESSAGE + 1)
            if len(line) > MAX_MESSAGE:
                raise ValueError("Supervisor request exceeds byte limit")
            request = json.loads(line)
            result = self.server.service.dispatch(request["operation"], request.get("args", {}))  # type: ignore[attr-defined]
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: pathlib.Path, service: Supervisor) -> None:
        self.service = service
        super().__init__(str(path), _Handler)


def serve(
    repo_root: pathlib.Path, root: pathlib.Path, idle_seconds: float = 1800
) -> None:
    root = root.resolve()
    ensure_private_directory(root)
    with (root / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path = socket_path(root)
        path.unlink(missing_ok=True)
        service = Supervisor(repo_root, root)
        try:
            with _Server(path, service) as server:
                os.chmod(path, 0o600)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    while True:
                        service.tick()
                        if (
                            not service.registry.unfinished()
                            and (
                                service.asset_server is None
                                or service.asset_server.closed
                            )
                            and time.monotonic() - service.last_request > idle_seconds
                        ):
                            break
                        time.sleep(0.2)
                finally:
                    server.shutdown()
        finally:
            service.close()
            path.unlink(missing_ok=True)


class SupervisorClient:
    def __init__(self, repo_root: str | pathlib.Path, root: str | pathlib.Path) -> None:
        self.repo_root = pathlib.Path(repo_root).resolve()
        self.root = pathlib.Path(root).resolve()

    def _request(self, operation: str, args: dict[str, Any]) -> Any:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(30)
            sock.connect(str(socket_path(self.root)))
            sock.sendall(
                json.dumps({"operation": operation, "args": args}).encode() + b"\n"
            )
            with sock.makefile("rb") as stream:
                data = stream.readline(MAX_MESSAGE + 1)
            if len(data) > MAX_MESSAGE:
                raise ValueError("Supervisor response exceeds byte limit")
        response = json.loads(data)
        if not response["ok"]:
            raise ValueError(response["error"])
        return response["result"]

    def ensure(self) -> None:
        ensure_private_directory(self.root)
        with (self.root / "startup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                current = self._request("ping", {})
                if current["repo_root"] != str(self.repo_root):
                    raise ValueError("Job storage is already owned by another checkout")
                return
            except (ConnectionError, FileNotFoundError):
                pass
            env = dict(os.environ)
            env["PYTHONPATH"] = str(self.repo_root / "src")
            with (self.root / "supervisor.log").open("ab") as log:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "pyurbanair.jobs.supervisor",
                        "--repo-root",
                        str(self.repo_root),
                        "--root",
                        str(self.root),
                    ],
                    cwd=self.repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    self._request("ping", {})
                    return
                except (ConnectionError, FileNotFoundError):
                    time.sleep(0.05)
            raise RuntimeError(
                f"Supervisor did not start; see {self.root / 'supervisor.log'}"
            )

    def request(self, operation: str, **args: Any) -> Any:
        self.ensure()
        return self._request(operation, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    serve(args.repo_root, args.root)


if __name__ == "__main__":
    main()
