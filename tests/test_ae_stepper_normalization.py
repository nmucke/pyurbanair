"""M2: the AE -> time-stepper normalization contract must fail loud.

The frozen encoder inside ``TadpoleTimeStepper`` was pre-trained on the AE's
standardised distribution, so it can only be fed inputs standardised with the
AE's ``state_mean``/``state_std``. Two silent-failure paths the repo convention
(fail-loud) must close:

* ``_load_ae_state_stats`` used to only ``warnings.warn`` when the AE export had
  no stats, then run the encoder on out-of-distribution inputs (a whole wasted
  training run). It must now RAISE, unless the caller explicitly opts out because
  ``recompute_normalization=true`` will install fresh stats anyway.
* the fine-tune script's ``_check_ae_stepper_match`` cross-checked size /
  geometry / SDF but not the AE's ``normalize`` flag, so a ``normalize: false``
  AE could silently pair with a ``normalize: true`` stepper.

These are unit tests: they call the two functions directly rather than driving a
full fine-tune (the e2e cross-check lives in ``test_ae_to_timestepper.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import TadpoleTimeStepper
from omegaconf import OmegaConf

_WORKTREE = Path(__file__).resolve().parents[1]
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "finetune_neural_surrogate.py"

CROP = 16  # encoder_crop_size must be a multiple of 16


def _load_finetune_module():
    spec = importlib.util.spec_from_file_location(
        "finetune_ns_norm_under_test", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stepper(**kw) -> TadpoleTimeStepper:
    """A fresh (random-init, no AE dir) stepper on CPU smoke shapes."""
    kw.setdefault("encoder_crop_size", CROP)
    kw.setdefault("sdf_clamp_cells", 8)
    return TadpoleTimeStepper(
        n_state_channels=3,
        n_params=2,
        size="S",
        pretrained_ae_dir=None,
        skip_pretrained_load=True,
        **kw,
    )


# --------------------------------------------------------------------------- #
# _load_ae_state_stats: fail loud on missing / absent stats.
# --------------------------------------------------------------------------- #


def test_missing_weights_file_raises_actionable(tmp_path):
    """No weights.pt in the AE dir -> raise (not warn) with an actionable msg."""
    model = _stepper(normalize=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        model._load_ae_state_stats(str(tmp_path), require=True)
    msg = str(excinfo.value)
    assert "weights.pt" in msg
    assert "recompute_normalization" in msg


def test_weights_without_state_stats_raises_actionable(tmp_path):
    """weights.pt present but carrying no state_mean/state_std -> raise."""
    torch.save({"some_weight": torch.zeros(3)}, tmp_path / "weights.pt")
    model = _stepper(normalize=True)
    with pytest.raises(ValueError) as excinfo:
        model._load_ae_state_stats(str(tmp_path), require=True)
    msg = str(excinfo.value)
    assert "state_mean" in msg
    assert "recompute_normalization" in msg


def test_missing_stats_opt_out_warns_not_raises(tmp_path):
    """The recompute_normalization opt-out (require=False) warns and continues --
    the stats are dead because fresh ones get installed afterwards."""
    model = _stepper(normalize=True)
    with pytest.warns(UserWarning):
        model._load_ae_state_stats(str(tmp_path), require=False)


def test_present_stats_are_copied(tmp_path):
    """A well-formed AE export copies state_mean/state_std into the buffers."""
    torch.save(
        {
            "state_mean": torch.tensor([1.0, 2.0, 3.0]),
            "state_std": torch.tensor([4.0, 5.0, 6.0]),
        },
        tmp_path / "weights.pt",
    )
    model = _stepper(normalize=True)
    model._load_ae_state_stats(str(tmp_path), require=True)
    assert torch.allclose(model.state_mean, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(model.state_std, torch.tensor([4.0, 5.0, 6.0]))


# --------------------------------------------------------------------------- #
# _check_ae_stepper_match: the normalize flag must agree.
# --------------------------------------------------------------------------- #


def test_normalize_flag_mismatch_raises():
    """A normalize:false AE paired with a normalize:true stepper must fail loud."""
    mod = _load_finetune_module()
    model = _stepper(normalize=True)  # stepper.normalize == True
    ae_arch = OmegaConf.create(
        {
            "size": "S",
            "encode_geometry": True,
            "sdf_features": "none",
            "normalize": False,
        }
    )
    with pytest.raises(ValueError) as excinfo:
        mod._check_ae_stepper_match(ae_arch, model)
    assert "normalize" in str(excinfo.value)


def test_normalize_flag_match_passes():
    """Matching normalize flags (plus matching size/geometry/SDF) pass the check."""
    mod = _load_finetune_module()
    model = _stepper(normalize=True)
    ae_arch = OmegaConf.create(
        {
            "size": "S",
            "encode_geometry": True,
            "sdf_features": "none",
            "normalize": True,
        }
    )
    mod._check_ae_stepper_match(ae_arch, model)  # no raise
