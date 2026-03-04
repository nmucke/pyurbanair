import pathlib

import pytest
from pyudales.utils.namoptions_utils import rename_namoptions_file


def test_rename_namoptions_file_raises_for_ambiguous_sources(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "namoptions.999").write_text("&RUN\niexpnr = 999\n/\n")
    (tmp_path / "namoptions.old").write_text("&RUN\niexpnr = old\n/\n")
    (tmp_path / "namoptions.backup").write_text("&RUN\niexpnr = backup\n/\n")

    with pytest.raises(RuntimeError, match="Ambiguous namoptions sources"):
        rename_namoptions_file(tmp_path, "000")


def test_rename_namoptions_file_prefers_single_source_and_updates_iexpnr(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "namoptions.999"
    source.write_text("&RUN\niexpnr = 999\n/\n")

    rename_namoptions_file(tmp_path, "000")

    target = tmp_path / "namoptions.000"
    assert target.exists()
    assert not source.exists()
    assert "iexpnr       = 000" in target.read_text()
