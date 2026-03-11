import pathlib
import re

from pylbm.stl_to_lbm import process_stl_to_fortran

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "stl_to_lbm"


def _extract_blanking_ranges(
    fortran_text: str,
) -> set[tuple[int, int, int, int, int, int]]:
    pattern = re.compile(
        r"blanking\(\s*ioff\+\s*(\d+)\s*:\s*ioff\+\s*(\d+)\s*,\s*"
        r"joff\+\s*(\d+)\s*:\s*joff\+\s*(\d+)\s*,\s*"
        r"(\d+)\s*:\s*(\d+)\s*\)\s*=\s*\.true\.",
        flags=re.IGNORECASE,
    )
    return {tuple(map(int, match)) for match in pattern.findall(fortran_text)}  # type: ignore[misc]


def test_stl_to_lbm_matches_reference_on_subdomain(tmp_path: pathlib.Path) -> None:
    stl_path = FIXTURE_DIR / "xie_castro_2008_STL.stl"
    reference_fortran_path = FIXTURE_DIR / "m_city3.F90"
    generated_fortran_path = tmp_path / "generated_city3.F90"

    generated_text = process_stl_to_fortran(
        stl_path=stl_path,
        output_path=generated_fortran_path,
        nx=32,
        ny=32,
        nz=8,
        bounds={
            "xmin": 0.0,
            "xmax": 80.0,
            "ymin": 0.0,
            "ymax": 80.0,
            "zmin": 0.0,
            "zmax": 40.0,
        },
    )

    reference_text = reference_fortran_path.read_text()

    generated_ranges = _extract_blanking_ranges(generated_text)
    reference_ranges = _extract_blanking_ranges(reference_text)

    # On the [0, 80] x [0, 80] subdomain, both sources should identify the
    # same building footprint locations (start indices in x/y and count).
    generated_starts = {(is_, js) for is_, _, js, _, _, _ in generated_ranges}
    reference_starts = {(is_, js) for is_, _, js, _, _, _ in reference_ranges}

    assert len(generated_ranges) == len(reference_ranges)
    assert generated_starts == reference_starts

    # Regression check for the right-wall artifact: no building should be
    # clamped to the outer x boundary for this benchmark setup.
    assert max(r[1] for r in generated_ranges) < 32
