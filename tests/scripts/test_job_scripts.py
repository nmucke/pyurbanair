"""The SLURM job scripts in job_scripts/ stay in step with scripts/."""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from tests.conftest import REPO

CLUSTERS = ("snellius", "delftblue")
# Helpers and quick login-node tools are not jobs.
NOT_JOBS = ("scripts/utils/", "scripts/tools/")


def _runnable() -> set[str]:
    scripts = (str(p.relative_to(REPO)) for p in (REPO / "scripts").rglob("*.py"))
    return {s for s in scripts if not s.startswith(NOT_JOBS)}


def _called(path: pathlib.Path, follow: bool = True) -> set[str]:
    """The scripts a job or workflow runs, plus those of the workflows it calls."""
    text = path.read_text()
    # A workflow picks `scripts/run_${method}.py` from its usage check.
    choices = re.search(r"=~ \^\(([\w|]+)\)\$", text)
    called = set()
    for ref in re.findall(r"scripts/[\w/${}]+\.py", text):
        if "${method}" in ref and choices:
            called |= {ref.replace("${method}", m) for m in choices[1].split("|")}
        else:
            called.add(ref)
    for workflow in re.findall(r"workflows/\w+\.sh", text) if follow else []:
        called |= _called(REPO / workflow, follow=False)
    return called


@pytest.mark.parametrize("cluster", CLUSTERS)  # type: ignore[misc]
def test_every_script_has_a_job(cluster: str) -> None:
    jobs = sorted((REPO / "job_scripts" / cluster).glob("*.slurm"))
    called = set().union(*map(_called, jobs))
    assert _runnable() - called == set()
    assert called - _runnable() == set()


@pytest.mark.parametrize(  # type: ignore[misc]
    "job",
    sorted((REPO / "job_scripts").glob("*/*.s*")),
    ids=lambda p: str(p.relative_to(REPO / "job_scripts")),
)
def test_job_script_parses(job: pathlib.Path) -> None:
    subprocess.run(["bash", "-n", str(job)], check=True)
