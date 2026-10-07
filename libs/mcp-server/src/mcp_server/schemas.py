"""Transport limits shared by tool schemas and response conversion."""

from typing import Any, TypedDict

MAX_PNG_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class LaunchResult(TypedDict):
    run_id: str
    job_id: str
    state: str
    kind: str
    run_root: str
    details: dict[str, Any]


def summary(job: dict[str, Any]) -> LaunchResult:
    return LaunchResult(
        run_id=job["id"],
        job_id=job["id"],
        state=job["state"],
        kind=job["kind"],
        run_root=job["run_root"],
        details=job["details"],
    )
