"""Case inputs must be safe to snapshot into the local job store."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mcp_server.jobs import preparation
from mcp_server.jobs.preparation import PreparationService


def test_case_containing_store_rejected_before_fingerprint(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_fingerprint: Callable[[str | Path], dict[str, Any]] = preparation.fingerprint

    def checked_fingerprint(path: str | Path) -> dict[str, Any]:
        assert Path(path).resolve() != checkout
        return source_fingerprint(path)

    monkeypatch.setattr(preparation, "fingerprint", checked_fingerprint)
    service = PreparationService(checkout, checkout / ".temp" / "jobs")
    with pytest.raises(ValueError, match="overlaps managed job storage"):
        service.prepare(["model.forward_model.case_dir=."])
    assert not list(service.store_root.glob("plans/*/inputs/*/case"))


def test_symlink_alias_to_store_rejected_before_fingerprint(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = checkout / ".temp" / "jobs"
    alias = checkout / "case_alias"
    alias.symlink_to(checkout, target_is_directory=True)
    source_fingerprint: Callable[[str | Path], dict[str, Any]] = preparation.fingerprint

    def checked_fingerprint(path: str | Path) -> dict[str, Any]:
        assert Path(path).resolve() != checkout
        return source_fingerprint(path)

    monkeypatch.setattr(preparation, "fingerprint", checked_fingerprint)
    with pytest.raises(ValueError, match="overlaps managed job storage"):
        PreparationService(checkout, store).prepare(
            ["model.forward_model.case_dir=case_alias"]
        )


def test_case_directory_symlink_loop_is_rejected(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    (case / "loop").symlink_to(case, target_is_directory=True)
    service = PreparationService(checkout, checkout.parent / "store")
    with pytest.raises(ValueError, match="case_dir contains a symlink: loop"):
        service.prepare()
    assert not list(service.store_root.glob("plans/*/inputs/*/case"))


def test_repo_file_symlink_stages_as_regular_file(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    shared = checkout / "geometries/shared.stl"
    shared.write_text("solid shared\nendsolid shared\n")
    (case / "linked.stl").symlink_to(Path("..") / shared.name)
    service = PreparationService(checkout, checkout.parent / "store")
    plan = service.prepare()
    staged = Path(plan["config"]["model"]["forward_model"]["case_dir"])
    assert not (staged / "linked.stl").is_symlink()
    assert (staged / "linked.stl").read_bytes() == shared.read_bytes()
    assert service.verify(plan["plan_id"])["digest"] == plan["digest"]


def test_case_rejects_file_symlink_outside_checkout(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    outside = checkout.parent / "outside.txt"
    outside.write_text("private")
    (case / "outside.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe symlink: outside.txt"):
        PreparationService(checkout, checkout.parent / "store").prepare()


def test_case_rejects_fifo_before_copy(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    os.mkfifo(case / "pipe")
    service = PreparationService(checkout, checkout.parent / "store")
    with pytest.raises(ValueError, match="non-regular entry: pipe"):
        service.prepare()
    assert not list(service.store_root.glob("plans/*/inputs/*/case"))


def test_replaced_subdirectory_cannot_redirect_case_copy(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = checkout / "geometries/xie_and_castro"
    nested = case / "nested"
    nested.mkdir()
    (nested / "input.txt").write_text("inside")
    outside = checkout.parent / "outside"
    outside.mkdir()
    (outside / "input.txt").write_text("secret")
    original_entries: Callable[
        [Path, Path, Path, dict[str, int]], list[tuple[Path, bool, int, Path]]
    ] = preparation._case_entries

    def swap_after_scan(
        source: Path, repo_root: Path, store_root: Path, limits: dict[str, int]
    ) -> list[tuple[Path, bool, int, Path]]:
        entries = original_entries(source, repo_root, store_root, limits)
        nested.rename(checkout.parent / "moved")
        nested.symlink_to(outside, target_is_directory=True)
        return entries

    monkeypatch.setattr(preparation, "_case_entries", swap_after_scan)
    service = PreparationService(checkout, checkout.parent / "store")
    with pytest.raises(ValueError, match="entry changed or contains a symlink"):
        service.prepare()
    assert not list(service.store_root.glob("plans/*/inputs/*/case/nested/input.txt"))


@pytest.mark.parametrize(  # type: ignore[misc]
    "limit,amount",
    [("max_case_input_bytes", 1), ("max_case_input_entries", 1)],
)
def test_case_limits_reject_before_copy(
    checkout: Path, limit: str, amount: int
) -> None:
    (checkout / "geometries/xie_and_castro/second.txt").write_text("second")
    service = PreparationService(checkout, checkout.parent / "store", {limit: amount})
    with pytest.raises(ValueError, match=limit):
        service.prepare()
    assert not list(service.store_root.glob("plans/*/inputs/*/case"))


def test_regular_case_stages_and_verifies(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    (case / "empty").mkdir()
    original = case / "namoptions.300"
    original.chmod(0o750)
    service = PreparationService(checkout, checkout.parent / "store")
    plan = service.prepare()
    staged = Path(plan["config"]["model"]["forward_model"]["case_dir"])
    assert (staged / "namoptions.300").read_bytes() == original.read_bytes()
    assert stat.S_IMODE((staged / "namoptions.300").stat().st_mode) == 0o750
    assert (staged / "empty").is_dir()
    assert service.verify(plan["plan_id"])["digest"] == plan["digest"]


def test_verify_reapplies_case_traversal_limit(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    service = PreparationService(
        checkout,
        checkout.parent / "store",
        {"max_case_input_entries": len(list(case.iterdir()))},
    )
    plan = service.prepare()
    (case / "added.txt").write_text("changed")
    with pytest.raises(ValueError, match="max_case_input_entries"):
        service.verify(plan["plan_id"])


def test_verify_rejects_symlink_added_to_case(checkout: Path) -> None:
    case = checkout / "geometries/xie_and_castro"
    service = PreparationService(checkout, checkout.parent / "store")
    plan = service.prepare()
    (case / "loop").symlink_to(case, target_is_directory=True)
    with pytest.raises(ValueError, match="case_dir contains a symlink: loop"):
        service.verify(plan["plan_id"])
