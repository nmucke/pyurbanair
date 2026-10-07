"""Verified process ownership, including detached MPI and forkserver children."""

from __future__ import annotations

import os
import signal
from typing import Any

import psutil

TOKEN_ENV = "PYURBANAIR_JOB_TOKEN"


def identity(pid: int) -> dict[str, Any]:
    process = psutil.Process(pid)
    return {"pid": pid, "created": process.create_time()}


def matching(record: dict[str, Any]) -> psutil.Process | None:
    try:
        process = psutil.Process(int(record["pid"]))
        if (
            process.create_time() == record["created"]
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            return process
    except (psutil.Error, KeyError):
        pass
    return None


def owned(
    token: str, known: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Match inherited job tokens, then descendants, never bare reused PIDs."""
    found: dict[int, dict[str, Any]] = {}
    for record in known or []:
        process = matching(record)
        if process is not None:
            found[process.pid] = record
    for process in psutil.process_iter(["pid", "uids", "status"]):
        try:
            if process.pid == os.getpid() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            if (
                process.uids().real == os.getuid()
                and process.environ().get(TOKEN_ENV) == token
            ):
                found[process.pid] = identity(process.pid)
        except (psutil.Error, OSError):
            continue
    for record in list(found.values()):
        process = matching(record)
        if process is None:
            continue
        try:
            for child in process.children(recursive=True):
                if child.status() != psutil.STATUS_ZOMBIE:
                    found[child.pid] = identity(child.pid)
        except psutil.Error:
            pass
    return list(found.values())


def signal_owned(records: list[dict[str, Any]], force: bool = False) -> None:
    for record in reversed(records):
        process = matching(record)
        if process is not None:
            try:
                process.send_signal(signal.SIGKILL if force else signal.SIGTERM)
            except psutil.Error:
                pass
