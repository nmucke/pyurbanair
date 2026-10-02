"""
Tests for the per-run LBM build tree (``pylbm.utils.build_tree_utils``).

The build tree is a private copy of the LBM submodule's ``src/`` that every run
compiles in, so the submodule itself stays byte-identical to its checked-out
commit. The subtle part is that the mirror runs *more than once per run* --
``get_lbm_directory_paths`` is called by ``ForwardModel.__init__`` and again by
``create_new_forward_model`` for each ensemble member, after the binary has been
built. Files the wrapper rewrote (the grid in ``mod_dimensions.F90``, the geometry
case in ``m_solid_objects_init.F90``) must survive that second pass: ensemble
members are deepcopies, so nothing regenerates them, and a mirror that refreshed
them would restore upstream's own grid under a binary compiled for the configured
one. That desync only surfaced later, as a state/grid shape mismatch from
``_prepare_warmstart``.

No compilation happens here, so these stay fast.
"""

import json
import pathlib

import pytest
from pylbm import LBM_PATH
from pylbm.forward_model import ForwardModel
from pylbm.utils.build_tree_utils import MIRROR_MANIFEST_NAME, materialize_build_tree
from pylbm.utils.forward_model_utils import create_new_forward_model
from pylbm.utils.mod_dimensions_utils import ModDimensions

STL_PATH = pathlib.Path("examples/xie_and_castro/xie_castro_2008_STL.stl")

# LBM_PATH is Optional at the module level (it stays None when .gitmodules cannot
# be read). Every test here needs the real submodule, so resolve it once.
assert LBM_PATH is not None, "LBM submodule path could not be resolved"
LBM_SOURCE: pathlib.Path = pathlib.Path(LBM_PATH)

# Deliberately unlike any case in upstream's mod_dimensions.F90 (whose active
# case is nx=200, ny=120, nz=2), so a reverted file cannot masquerade as correct.
NX, NY, NZ = 18, 14, 6


def _make_model(temp_dir: pathlib.Path) -> ForwardModel:
    """Construct a ForwardModel without compiling or running anything."""
    return ForwardModel(
        stl_path=STL_PATH,
        temp_dir=temp_dir,
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=((0.0, 20.0), (0.0, 20.0), (0.0, 10.0)),
        cuda=False,
        verbose=False,
    )


def _active_grid(build_root: pathlib.Path) -> dict:
    """Read the active experiment's dimensions out of the build tree."""
    mod = ModDimensions(build_root / "src" / "mod_dimensions.F90")
    active = mod.get_active_experiment_name()
    assert active is not None, "no active experiment in mod_dimensions.F90"
    params = mod.get_experiment_params(active)
    assert params is not None
    return params


@pytest.fixture  # type: ignore[misc]
def build_root(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "lbm_build"


class TestWrapperOwnedSourcesSurviveRemirroring:
    """The regression that broke test_run_esmda[state_and_param] in CI."""

    def test_grid_survives_ensemble_member_creation(
        self, tmp_path: pathlib.Path
    ) -> None:
        model = _make_model(tmp_path)
        build_root = model.dirs.lbm_src_path.parent

        assert _active_grid(build_root)["nx"] == NX

        # Every member re-enters get_lbm_directory_paths, re-running the mirror.
        for member in range(3):
            create_new_forward_model(
                forward_model=model,
                experiment_base_dir=tmp_path / "members",
                experiment_name=f"member_{member}",
            )

        grid = _active_grid(build_root)
        assert (grid["nx"], grid["ny"], grid["nz"]) == (NX, NY, NZ), (
            "the mirror reverted mod_dimensions.F90 to the submodule copy; a "
            "binary built for the configured grid would then be run against "
            "upstream's grid"
        )

    def test_geometry_case_survives_ensemble_member_creation(
        self, tmp_path: pathlib.Path
    ) -> None:
        model = _make_model(tmp_path)
        build_root = model.dirs.lbm_src_path.parent
        solid_objects = build_root / "src" / "m_solid_objects_init.F90"

        assert "case('runcase')" in solid_objects.read_text()

        create_new_forward_model(
            forward_model=model,
            experiment_base_dir=tmp_path / "members",
            experiment_name="member_0",
        )

        text = solid_objects.read_text()
        assert "case('runcase')" in text
        assert "use m_read_bathymetry" in text

    def test_repeated_materialize_does_not_revert(self, tmp_path: pathlib.Path) -> None:
        """The mirror alone must be idempotent w.r.t. wrapper-owned files."""
        model = _make_model(tmp_path)
        build_root = model.dirs.lbm_src_path.parent

        before = (build_root / "src" / "mod_dimensions.F90").read_text()
        for _ in range(3):
            materialize_build_tree(lbm_path=LBM_SOURCE, build_root=build_root)
        after = (build_root / "src" / "mod_dimensions.F90").read_text()

        assert after == before


class TestMirrorMechanics:
    def test_seeds_sources_and_helper_scripts(
        self, tmp_path: pathlib.Path, build_root: pathlib.Path
    ) -> None:
        materialize_build_tree(lbm_path=LBM_SOURCE, build_root=build_root)

        assert (build_root / "src" / "main.F90").exists()
        assert (build_root / "src" / "makefile").exists()
        # target.mk defines TARGET; without it `make install` degrades to
        # `cp ../build/ <bindir>` and fails with an unrelated-looking error.
        assert (build_root / "src" / "target.mk").exists()
        assert (build_root / "bin" / "mkdepend.pl").exists()
        assert (build_root / "build").is_dir()

    def test_prunes_only_what_it_mirrored(
        self, tmp_path: pathlib.Path, build_root: pathlib.Path
    ) -> None:
        materialize_build_tree(lbm_path=LBM_SOURCE, build_root=build_root)

        # A file the wrapper generated into the tree, never mirrored from source.
        generated = build_root / "src" / "m_generated_by_wrapper.F90"
        generated.write_text("module m_generated_by_wrapper\nend module\n")

        # A file that looks mirrored but is no longer in the source tree.
        vanished = build_root / "src" / "m_removed_upstream.F90"
        vanished.write_text("module m_removed_upstream\nend module\n")
        manifest = build_root / MIRROR_MANIFEST_NAME
        manifest.write_text(
            json.dumps(sorted(set(json.loads(manifest.read_text())) | {vanished.name}))
        )

        materialize_build_tree(lbm_path=LBM_SOURCE, build_root=build_root)

        assert generated.exists(), "pruned a file the mirror never placed"
        assert not vanished.exists(), "kept a source that vanished upstream"

    def test_missing_source_tree_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="LBM sources not found"):
            materialize_build_tree(
                lbm_path=tmp_path / "nope", build_root=tmp_path / "build"
            )
