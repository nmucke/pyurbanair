"""LoRA fine-tuning (plan 01): unit + end-to-end smoke tests.

Two layers:

* **Unit** (``neural_surrogates.finetuning``): LoRA injection is a no-op at init
  (B=0), merging round-trips to a plain state dict byte-identical to the base at
  init, and only the adapter (+ ``modules_to_save``) parameters train. Run on P3D
  (the plan's focus); skipped if ``p3d_surrogate`` is absent.
* **End-to-end** (``finetune_neural_surrogate.run``): the actual Hydra
  composition + script, fabricating a pretrained ``model_dir`` and a tiny dataset,
  fine-tuning for 2 epochs, and asserting the exported dir loads into
  ``NeuralSurrogateForwardModel`` and rolls out. Uses ``UNetConvNeXt`` so it needs
  no heavy P3D dependency and stays CPU-fast.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from neural_surrogates import UNetConvNeXt
from neural_surrogates.finetuning import (
    all_adaptable_module_names,
    inject_lora,
    merge_to_state_dict,
    resolve_target_modules,
)

from pyurbanair.base_forward_model import BaseForwardModel

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "finetune_neural_surrogate.py"

STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")


def _load_finetune_run():
    spec = importlib.util.spec_from_file_location("finetune_ns_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


# --------------------------------------------------------------------------- #
# Unit tests on P3D (the plan's focus architecture).
# --------------------------------------------------------------------------- #


def _p3d(n_params=2):
    p3d = pytest.importorskip("neural_surrogates.architectures.p3d")
    pytest.importorskip("p3d_surrogate")
    return p3d.P3D(
        n_state_channels=3,
        n_params=n_params,
        size="S",
        normalize=True,
        predict_residual=True,
        param_conditioning="native",
    )


def _p3d_inputs(b=2, grid=(16, 16, 16)):
    state = torch.randn(b, 3, *grid)
    params = torch.randn(b, 2)
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, params, geom


def test_inject_is_identity_at_init():
    """LoRA B=0 at init => the wrapped model's output equals the base's."""
    model = _p3d().eval()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention"),
    ).eval()
    state, params, geom = _p3d_inputs()
    with torch.no_grad():
        base = model(state, params, geom)
        wrapped = peft_model(state, params, geom)
    assert torch.allclose(base, wrapped, atol=1e-6)


def test_merge_round_trips_to_plain_state_dict():
    """merge_to_state_dict yields a plain base state dict, identical at init."""
    model = _p3d().eval()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention+conv"),
    )
    merged = merge_to_state_dict(peft_model)
    # loads strictly into a freshly instantiated architecture (no lora_ keys)
    fresh = _p3d()
    fresh.load_state_dict(merged)  # raises on any missing/unexpected key
    # B=0 at init => merged weights are byte-identical to the base
    for k, v in base_state.items():
        assert torch.allclose(merged[k], v, atol=1e-6), k


def test_merge_parity_at_nonzero_lora_b():
    """merge_to_state_dict reproduces the wrapped forward at *nonzero* LoRA B.

    The init round-trip (B=0) only proves the pass-through path; it never
    exercises peft's actual delta math -- the alpha/r scaling and the Conv3d
    weight fold. This repo has already hit two peft-0.19 conv-merge bugs, so
    assert the merged base weights reproduce the adapted forward end-to-end:
    perturb the trainable ``lora_B`` matrices to nonzero, then compare the
    wrapped model's output against a fresh base model loaded with the merged
    state dict, on a nontrivial input. The tolerance is loosened to ~1e-5 for
    the conv-fold arithmetic.
    """
    model = _p3d().eval()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention+conv"),
    )
    # Drive the adapter off the zero fixed point: bump every trainable lora_B
    # (zero-initialised by peft) so the merged delta is genuinely nonzero.
    with torch.no_grad():
        n_bumped = 0
        for n, p in peft_model.named_parameters():
            if "lora_B" in n and p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
                n_bumped += 1
    assert n_bumped, "no trainable lora_B parameters were perturbed"

    peft_model.eval()
    state, params, geom = _p3d_inputs()
    merged = merge_to_state_dict(peft_model)
    fresh = _p3d().eval()
    fresh.load_state_dict(merged)  # strict: plain base state dict, no lora_ keys
    with torch.no_grad():
        base = model(state, params, geom)
        wrapped = peft_model(state, params, geom)
        reloaded = fresh(state, params, geom)

    # sanity: nonzero B actually moved the output away from the base model
    assert not torch.allclose(
        wrapped, base, atol=1e-4
    ), "perturbation did not change the output; delta math is untested"
    # the folded base weights must reproduce the adapted (nonzero-B) forward
    assert torch.allclose(wrapped, reloaded, atol=1e-5)


def test_only_adapter_and_modules_to_save_train():
    """Only lora_* and the explicit modules_to_save head require grad."""
    model = _p3d().eval()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention"),
        modules_to_save=["param_to_scalar"],
    )
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    for n in trainable:
        assert "lora_" in n or "param_to_scalar" in n, n
    # the frozen base weights (e.g. the qkv base_layer) must not train
    frozen = [
        n
        for n, p in peft_model.named_parameters()
        if not p.requires_grad and "attn.qkv.base_layer" in n
    ]
    assert frozen, "expected a frozen base qkv layer"


def test_merge_preserves_trained_modules_to_save_head():
    """merge_and_unload must fold a trained modules_to_save head back in.

    PEFT wraps a modules_to_save target in a ModulesToSaveWrapper; the merge has
    to unwrap it into a plain module carrying the *trained* weights (the riskiest
    merge path). Perturb the head's trainable copy, merge, load into a fresh
    architecture, and assert the head is the trained one — not the base.
    """
    model = _p3d().eval()  # n_params=2 => param_to_scalar (a Linear(2,1)) exists
    base_head = model.param_to_scalar.weight.detach().clone()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention"),
        modules_to_save=["param_to_scalar"],
    )
    # "Train" the head: bump its trainable (modules_to_save) copy by a constant.
    with torch.no_grad():
        n_bumped = 0
        for n, p in peft_model.named_parameters():
            if "param_to_scalar" in n and "weight" in n and p.requires_grad:
                p.add_(1.0)
                n_bumped += 1
    assert n_bumped == 1, "expected exactly one trainable param_to_scalar weight"

    merged = merge_to_state_dict(peft_model)
    fresh = _p3d()
    fresh.load_state_dict(merged)  # strict: no lora_/modules_to_save keys survive
    # the merged head must be the trained one (base + 1), not the base
    assert torch.allclose(fresh.param_to_scalar.weight, base_head + 1.0, atol=1e-6)
    assert not torch.allclose(fresh.param_to_scalar.weight, base_head, atol=1e-6)


# --------------------------------------------------------------------------- #
# Unit test on the generic enumerator (no P3D needed).
# --------------------------------------------------------------------------- #


def test_all_preset_skips_grouped_and_1x1_convs():
    """all_adaptable_module_names drops depthwise + 1x1x1 convs (unmergeable)."""
    model = UNetConvNeXt(
        n_state_channels=3,
        n_params=2,
        base_channels=4,
        channel_mults=(1, 2),
        depths=(1, 1),
        kernel_size=3,
        expansion=2,
    )
    names = set(all_adaptable_module_names(model))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and (
            module.groups != 1 or all(k == 1 for k in module.kernel_size)
        ):
            assert name not in names, f"{name} should be excluded"
    # injecting on the resolved 'all' targets must not raise (grouped/1x1 excluded)
    targets = resolve_target_modules(model, preset="all")
    inject_lora(copy.deepcopy(model), rank=4, alpha=8, target_modules=targets)


# --------------------------------------------------------------------------- #
# End-to-end: compose finetuning.yaml + run the script, load into the forward
# model and roll out. UNetConvNeXt keeps it dependency-light and CPU-fast.
# --------------------------------------------------------------------------- #

NZ, NY, NX = 4, 8, 8


class _StubSpinup(BaseForwardModel):
    """Synthetic spin-up backend returning a developed pylbm-style field.

    A trimmed copy of the stub in ``test_neural_surrogate_forward_model`` kept
    local so this test does not import that module (whose top-level
    ``importorskip('trimesh')`` would raise a Skip mid-test here).
    """

    def __init__(self, results_dir=None) -> None:
        super().__init__(results_dir=results_dir)
        self.spinup_time = 0.0
        self._rng = np.random.default_rng(0)

    def _apply_inflow_settings(self, params) -> None:
        pass

    def save_results(self, state, sim_name: str = "state") -> None:
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        pass

    def disable_spinup(self) -> None:
        self.spinup_time = 0.0

    def run_single(self, state=None, params=None, sim_name="state") -> xr.Dataset:
        coords = {
            "z": np.arange(NZ) + 0.5,
            "y": np.arange(NY) + 0.5,
            "x": np.arange(NX) + 0.5,
            "time": [0],
        }
        data = {
            v: (("time", "z", "y", "x"), self._rng.standard_normal((1, NZ, NY, NX)))
            for v in STATE_VARS
        }
        return xr.Dataset(data, coords=coords)


def _params() -> xr.Dataset:
    t = np.linspace(0.0, 3.0, 4)
    return xr.Dataset(
        {
            "inflow_angle": ("time", np.linspace(10.0, 20.0, t.size)),
            "velocity_magnitude": ("time", np.linspace(3.0, 4.0, t.size)),
        },
        coords={"time": t},
    )


def _write_dataset(root: Path, *, nz=NZ, ny=NY, nx=NX, t=4) -> None:
    rng = np.random.default_rng(0)
    for split, n in {"train": 2, "val": 1}.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        (root / "param" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            blank = np.zeros((nz, ny, nx), "f4")
            blank[0] = 1.0
            state = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((t, nz, ny, nx)).astype("f4"),
                )
                for v in STATE_VARS
            }
            state["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                state,
                coords=dict(
                    time=np.arange(t) * 1.0,
                    z=np.arange(nz),
                    y=np.arange(ny),
                    x=np.arange(nx),
                ),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")
            xr.Dataset(
                {
                    "inflow_angle": (("time",), rng.standard_normal(t).astype("f4")),
                    "velocity_magnitude": (
                        ("time",),
                        (7 + rng.standard_normal(t)).astype("f4"),
                    ),
                },
                coords=dict(time=np.arange(t) * 1.0),
            ).to_netcdf(root / "param" / split / f"sample_{i:04d}.nc")


def _unet_arch() -> dict:
    return {
        "_target_": "neural_surrogates.UNetConvNeXt",
        "base_channels": 4,
        "channel_mults": [1, 2],
        "depths": [1, 1],
        "kernel_size": 3,
        "expansion": 2,
    }


def _make_pretrained_model_dir(root: Path, data_dir: Path) -> Path:
    """Fabricate a trained model_dir (config.yaml + weights.pt) + data config."""
    from hydra.utils import instantiate

    arch = _unet_arch()
    model_dir = root / "model_weights" / "unet_base"
    model_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": arch,
                "dataset": {
                    "root_dir": str(data_dir),
                    "state_vars": list(STATE_VARS),
                    "param_vars": list(PARAM_VARS),
                    "sdf_features": "none",
                    "sdf_clamp_cells": 32,
                },
            }
        ),
        model_dir / "config.yaml",
    )
    model = instantiate(
        arch, n_state_channels=len(STATE_VARS), n_params=len(PARAM_VARS)
    )
    torch.save(model.state_dict(), model_dir / "weights.pt")
    # training-data config carrying the trained domain + step size for the loader
    OmegaConf.save(
        OmegaConf.create(
            {
                "domain": {
                    "nx": NX,
                    "ny": NY,
                    "nz": NZ,
                    "bounds": [[0.0, NX], [0.0, NY], [0.0, NZ]],
                },
                "time": {
                    "simulation_time": 4.0,
                    "output_frequency": 1.0,
                    "spinup_time": 0.0,
                },
            }
        ),
        data_dir / "config.yaml",
    )
    return model_dir


def _shrink_for_cpu(cfg) -> None:
    cfg.dataset.pushforward_steps = 1
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.trainer.num_epochs = 2
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.patience = None
    cfg.trainer.pushforward_epochs_per_step = None
    cfg.trainer.lr_warmup_epochs = None
    cfg.trainer.cudnn_benchmark = False
    cfg.trainer.tf32 = False
    cfg.trainer.resume = False


def test_finetune_end_to_end(tmp_path, monkeypatch):
    """compose finetuning.yaml -> run -> exported dir loads + rolls out."""
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)
    pretrained_dir = _make_pretrained_model_dir(tmp_path, data_dir)

    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="neural_surrogate/finetuning",
            overrides=[
                f"pretrained_model_dir={pretrained_dir}",
                "model_name=unet_ft_test",
                f"dataset.root_dir={data_dir}",
                "lora.target_preset=all",
                "lora.rank=4",
                "lora.alpha=8",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    _shrink_for_cpu(cfg)

    # The finetune_mode group bundles Trainer + MSELoss.
    assert cfg.trainer._target_.split(".")[-1] == "Trainer"
    assert cfg.loss._target_.split(".")[-1] == "MSELoss"

    monkeypatch.chdir(tmp_path)
    _load_finetune_run()(cfg)

    out = tmp_path / "model_weights" / "unet_ft_test"
    assert (out / "weights.pt").exists()
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "adapter" / "adapter_model.safetensors").exists()

    saved = OmegaConf.load(out / "config.yaml")
    assert saved.architecture._target_.split(".")[-1] == "UNetConvNeXt"
    assert list(saved.dataset.state_vars) == list(STATE_VARS)
    assert list(saved.dataset.param_vars) == list(PARAM_VARS)
    assert saved.dataset.root_dir == str(data_dir)
    assert saved.pretrained.model_dir == str(pretrained_dir)

    # merged weights.pt is a plain state dict loadable into a fresh architecture
    from hydra.utils import instantiate

    fresh = instantiate(
        _unet_arch(), n_state_channels=len(STATE_VARS), n_params=len(PARAM_VARS)
    )
    fresh.load_state_dict(torch.load(out / "weights.pt"))

    # and the exported dir drops into NeuralSurrogateForwardModel + rolls out
    from neural_surrogates import NeuralSurrogateForwardModel

    model = NeuralSurrogateForwardModel(
        spinup_forward_model=_StubSpinup(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
        simulation_time=4.0,
        output_frequency=1.0,
        model_dir=out,
    )
    result = model(params=_params())
    assert result.sizes["time"] == 4
    assert isinstance(model.model, UNetConvNeXt)
