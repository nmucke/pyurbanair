"""Every relative Markdown link in the maintained docs resolves to a file."""

from __future__ import annotations

import pathlib
import re

import pytest

from tests.conftest import REPO

# Finished plans and the archive are historical: their links may point at code
# that has since moved.
SKIP = ("docs/archive/", "docs/plans/implemented/", "docs/plans/rejected/")
DOCS = sorted(
    p
    for pattern in (
        "docs/**/*.md",
        "configs/**/*.md",
        "tests/**/*.md",
        "README.md",
        "AGENTS.md",
    )
    for p in REPO.glob(pattern)
    if not p.relative_to(REPO).as_posix().startswith(SKIP)
)
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO).as_posix())  # type: ignore[misc]
def test_relative_links_resolve(doc: pathlib.Path) -> None:
    targets = (m.split("#")[0] for m in LINK.findall(doc.read_text()))
    missing = [
        t
        for t in targets
        if t and "://" not in t and not t.startswith("mailto:")
        if not (doc.parent / t).exists()
    ]
    assert not missing, f"{doc.relative_to(REPO)} links to missing {missing}"
