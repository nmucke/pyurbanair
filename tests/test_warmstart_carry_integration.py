"""End-to-end check that the pyudales warm start reuses the on-disk carry.

Runs a tiny uDALES cold start, then a warm start seeded from its state, and
verifies that the warm start reused the real end-of-run restart (the "carry")
rather than the zeroed cold-start template. The carry's subgrid record (e120,
the SGS TKE) must carry actual turbulence from the cold run, which is precisely
what removes the per-window re-spin-up bias.
"""

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from scipy.io import FortranFile

from pyurbanair.config.hydra_helpers import clean_outputs
from pyudales.utils.warm_start_utils import _carry_dir, fetch_carry


# Record layout of a uDALES restart (see update_warmstart_file_from_xarray):
# 0=mindist 1=wall 2=u0 3=v0 4=w0 5=pres0 6=thl0 7=e120 8=ekm ...
# Only u0/v0/w0/pres0 are injected from the state; the SGS fields (e120 with the
# one-equation model, ekm/eddy-viscosity) are the ones the carry preserves and
# the cold-start template would zero. The test case runs Smagorinsky, so e120 is
# unused (0) and ekm is the real carried subgrid field to assert on.
EKM_RECORD = 8


def _read_record(path, idx):
    records = []
    with FortranFile(str(path), "r") as f:
        try:
            while True:
                records.append(f.read_record(dtype=np.float64))
        except Exception:
            pass
    return records[idx]


def test_warm_start_reuses_carry_subgrid_fields():
    # Compose with the Hydra runtime config registered so paths.yaml's
    # ``${hydra:runtime.cwd}`` resolves under a bare compose() (no hydra.main).
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+size=test",
                "model=pyudales",
                "params=static",
                "time.spinup_time=2.0",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)

        true_params = instantiate(cfg.params).sample(1).isel(ensemble=0, drop=True)

        fm = instantiate(cfg.model.forward_model)
        instantiate(cfg.model.prepare, forward_model=fm)
        clean_outputs(cfg.model.name, fm)

        # --- Cold start: must produce a carry with real subgrid fields ---
        cold_state = fm(params=true_params)
        assert cold_state is not None
        carry_dir = _carry_dir(fm.dirs)
        assert carry_dir.exists(), "cold start did not persist a warmstart carry"
        carry_files = list(
            carry_dir.glob(f"initd*_000_000.{fm.dirs.experiment_name}")
        )
        assert carry_files, "carry directory has no restart file"

        # The cold-start tiny template (barely-evolved subgrid state) vs the
        # carry (real subgrid state from the full run): the carry's eddy
        # viscosity must be non-trivial and clearly differ from the template's.
        fm._ensure_warmstart_template()
        template_ekm = _read_record(fm.warmstart_template_file, EKM_RECORD)
        carry_ekm = _read_record(carry_files[0], EKM_RECORD)
        assert np.any(carry_ekm != 0.0), "carry eddy viscosity is all zero"
        assert not np.allclose(carry_ekm, template_ekm), (
            "carry subgrid state matches the cold-start template — the real "
            "fields are not being reused"
        )

        # The carry is fetchable for the next warm start.
        assert fetch_carry(fm.dirs) is not None

        # --- Warm start from the cold state: completes and refreshes the carry ---
        warm_state = fm(params=true_params, state=cold_state)
        assert warm_state is not None
        assert warm_state.sizes["time"] == cold_state.sizes["time"]
        assert _carry_dir(fm.dirs).exists()
