"""Optional adversarial (GAN) extension of the AE pre-training loop (plan 02).

The extension must be a *strict* no-op when it is off, so the first test locks
that down (same loss value, same ``_aux_terms`` keys as the pure-VAE path) and
the rest exercise the on-path: the step-based warm-up, the 1:1 discriminator
update, the geometry channels reaching the critic, the validation path staying
adversary-free under ``no_grad``, checkpoint/resume of the critic half (including
a pre-extension checkpoint), and an end-to-end config-composition run of
``pretrain_autoencoder.run`` with the discriminator block enabled.

Same conventions as ``test_autoencoder_pretraining.py``: ``importorskip`` for the
vendored autoencoder's runtime deps, CPU, ``CROP = 16`` and tiny smoke shapes.
"""

from __future__ import annotations

import csv
import importlib.util
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import AutoencoderTrainer, TadpoleAE, TadpoleDiscriminator

if TYPE_CHECKING:
    # `torch` above is a *variable* (importorskip returns a module object), so
    # `torch.Tensor` is not a usable annotation. Import the names statically for
    # the type checker only; `from __future__ import annotations` keeps every
    # annotation below lazy, so nothing is evaluated at runtime.
    from torch import Tensor
    from torch.nn import Module

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "pretrain_autoencoder.py"

STATE_VARS = ("u", "v", "w")
CROP = 16  # encoder_crop_size must be a multiple of 16
GRID = (16, 16, 16)
C = len(STATE_VARS)


# --------------------------------------------------------------------------- #
# Fixtures / helpers.
# --------------------------------------------------------------------------- #


class _BatchLoader:
    """Minimal stand-in for a DataLoader over pre-collated snapshot batches.

    ``BaseTraining`` only needs ``.dataset`` (for the pushforward curriculum,
    which a snapshot AE does not have) plus iteration, so a real DataLoader would
    just add worker/collate machinery this suite does not exercise."""

    persistent_workers = False

    def __init__(self, batches: list[dict[str, Tensor]]) -> None:
        self.batches = batches
        self.dataset: list[Any] = []

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _ae(encode_geometry: bool = True, sdf_features: str = "none") -> TadpoleAE:
    """A tiny, *deterministic* (``latent_type="mode"``) autoencoder.

    Determinism matters for the regression test: a sampled latent would make two
    forwards of the same batch differ for reasons unrelated to the extension."""
    ae = TadpoleAE(
        n_state_channels=C,
        size="S",
        encoder_crop_size=CROP,
        latent_type="mode",
        encode_geometry=encode_geometry,
        sdf_features=sdf_features,
        sdf_clamp_cells=8,
    )
    ae.set_normalization([0.0] * C, [1.0] * C)
    return ae


def _disc(model: TadpoleAE, **overrides: Any) -> TadpoleDiscriminator:
    """Critic matching ``model``'s working-space channels (what the script does)."""
    kwargs: dict[str, Any] = dict(
        n_state_channels=model.n_state_channels,
        size="S",
        encode_geometry=model.encode_geometry,
        sdf_features=model.sdf_feature_mode,
    )
    kwargs.update(overrides)
    return TadpoleDiscriminator(**kwargs)


def _batch(b: int = 2, seed: int = 0) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    geom = (torch.rand(1, *GRID, generator=g) > 0.2).float()
    return {
        "state": torch.randn(b, C, *GRID, generator=g),
        "geometry": geom,  # shared geometry: (1, *grid)
    }


def _trainer(
    model: Module,
    batches: list[dict[str, Tensor]] | None = None,
    **kw: Any,
) -> AutoencoderTrainer:
    loader = _BatchLoader(batches if batches is not None else [_batch()])
    kwargs: dict[str, Any] = dict(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )
    kwargs.update(kw)
    return AutoencoderTrainer(**kwargs)


def _snapshot(module: Module) -> list[Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _any_changed(module: Module, before: list[Tensor]) -> bool:
    return any(
        not torch.equal(p.detach(), b) for p, b in zip(module.parameters(), before)
    )


def _aux(trainer: AutoencoderTrainer) -> dict[str, Tensor]:
    """The last batch's ``_aux_terms``, asserted present (``_loss`` always sets
    it, so ``None`` here would itself be a regression)."""
    terms: dict[str, Tensor] | None = trainer._aux_terms
    assert terms is not None
    return terms


# --------------------------------------------------------------------------- #
# Regression: the extension is a strict no-op when it is off.
# --------------------------------------------------------------------------- #


def test_no_discriminator_is_unchanged() -> None:
    """``discriminator=None`` keeps the pure-VAE loss and term keys exactly."""
    torch.manual_seed(0)
    model = _ae()
    trainer = _trainer(model)
    batch = _batch()

    loss = trainer._forward(batch)

    assert set(_aux(trainer)) == {"recon", "geom", "kl"}
    assert trainer.discriminator is None
    assert torch.isfinite(loss)
    # Nothing is stashed for a discriminator step, and the hook is inert.
    trainer._after_optimizer_step(batch)
    assert trainer._adv_real is None and trainer._global_step == 0


def test_warmup_loss_matches_pure_vae_path() -> None:
    """During the warm-up the total loss is bit-identical to the no-critic loss.

    Both trainers wrap the *same* model, so any difference would come from the
    adversarial term rather than from a different random init."""
    torch.manual_seed(0)
    model = _ae()
    batch = _batch()

    plain = _trainer(model)
    loss_plain = plain._forward(batch)

    disc = _disc(model)
    adversarial = _trainer(
        model,
        discriminator=disc,
        disc_optimizer=torch.optim.SGD(disc.parameters(), lr=0.1),
        adv_start_step=10,  # far beyond this batch: still in the warm-up
    )
    loss_adv = adversarial._forward(batch)

    assert torch.equal(loss_plain.detach(), loss_adv.detach())
    # the columns exist from step 0 (a mid-run new CSV column would break the
    # appended metrics.csv header), but carry zeros during the warm-up
    terms = _aux(adversarial)
    assert set(terms) == {"recon", "geom", "kl", "adv", "adv_w", "d"}
    assert float(terms["adv"]) == 0.0
    assert float(terms["adv_w"]) == 0.0


def test_discriminator_without_optimizer_raises() -> None:
    model = _ae()
    with pytest.raises(ValueError, match="disc_optimizer"):
        _trainer(model, discriminator=_disc(model))
    with pytest.raises(ValueError, match="discriminator"):
        _trainer(model, disc_optimizer=torch.optim.SGD(model.parameters(), lr=0.1))


# --------------------------------------------------------------------------- #
# The on-path: an epoch trains both networks.
# --------------------------------------------------------------------------- #


def _adversarial_trainer(
    model: Module,
    disc: Module,
    batches: list[dict[str, Tensor]],
    **kw: Any,
) -> AutoencoderTrainer:
    kwargs: dict[str, Any] = dict(
        discriminator=disc,
        disc_optimizer=torch.optim.SGD(disc.parameters(), lr=0.1),
        adv_start_step=0,
        adv_ramp_steps=0,
        adv_weight=1.0,  # exaggerated: a 1e-4 default would not move SGD params
        adaptive_adv_weight=False,
    )
    kwargs.update(kw)
    return _trainer(model, batches=batches, **kwargs)


def test_adversarial_epoch_updates_both_networks() -> None:
    torch.manual_seed(0)
    model = _ae()
    disc = _disc(model)
    batches = [_batch(seed=1), _batch(seed=2)]
    trainer = _adversarial_trainer(model, disc, batches)

    disc_before, model_before = _snapshot(disc), _snapshot(model)
    trainer._train_epoch()

    assert _any_changed(disc, disc_before), "discriminator was never updated"
    assert _any_changed(model, model_before), "autoencoder was never updated"
    assert trainer._global_step == len(batches)
    # the stash is cleared after every discriminator step
    assert trainer._adv_real is None and trainer._adv_fake is None
    for key in ("adv", "adv_w", "d"):
        assert key in trainer._train_terms
    assert trainer._train_terms["adv_w"] == pytest.approx(1.0)


def test_warmup_leaves_discriminator_untouched() -> None:
    """Before ``adv_start_step`` the critic neither trains nor enters the loss."""
    torch.manual_seed(0)
    model = _ae()
    disc = _disc(model)
    batches = [_batch(seed=1), _batch(seed=2)]
    trainer = _adversarial_trainer(model, disc, batches, adv_start_step=99)

    disc_before = _snapshot(disc)
    trainer._train_epoch()

    assert not _any_changed(disc, disc_before), "critic updated during the warm-up"
    assert trainer._train_terms["adv"] == 0.0
    assert trainer._train_terms["adv_w"] == 0.0
    assert trainer._train_terms["d"] == 0.0
    assert trainer._global_step == len(batches)  # the clock still ticks


def test_ramp_scales_the_effective_weight() -> None:
    """``adv_ramp_steps`` interpolates the coefficient linearly from 0 to full."""
    torch.manual_seed(0)
    model = _ae()
    disc = _disc(model)
    trainer = _adversarial_trainer(
        model, disc, [_batch()], adv_start_step=2, adv_ramp_steps=4, adv_weight=1.0
    )
    for step, expected in [(0, 0.0), (2, 0.0), (3, 0.25), (6, 1.0), (99, 1.0)]:
        trainer._global_step = step
        assert trainer._adv_ramp() == pytest.approx(expected)


def test_disc_recon_threshold_gates_the_update() -> None:
    """With an unreachable threshold the critic is never stepped."""
    torch.manual_seed(0)
    model = _ae()
    disc = _disc(model)
    trainer = _adversarial_trainer(
        model, disc, [_batch(seed=1)], disc_recon_threshold=0.0
    )

    disc_before = _snapshot(disc)
    trainer._train_epoch()
    assert not _any_changed(disc, disc_before)


# --------------------------------------------------------------------------- #
# Validation must stay adversary-free.
# --------------------------------------------------------------------------- #


def test_validation_has_no_adversarial_term() -> None:
    """``_validate`` runs under ``no_grad`` + ``eval``; the adaptive weight's
    ``autograd.grad`` would raise there, and an adversarial term would make the
    val curve jump at the warm-up boundary."""
    torch.manual_seed(0)
    model = _ae()
    disc = _disc(model)
    trainer = _adversarial_trainer(model, disc, [_batch(seed=1)])

    disc_before = _snapshot(disc)
    val = trainer._validate()  # must not raise under no_grad

    assert np.isfinite(val)
    assert set(trainer._val_terms) == {"recon", "geom", "kl"}
    assert not _any_changed(disc, disc_before)
    assert trainer._global_step == 0  # validation is not an optimizer step


# --------------------------------------------------------------------------- #
# The geometry channels reach the critic.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "encode_geometry, sdf_features, disc_kwargs",
    [
        (True, "none", {}),  # mask rides along:            3 + 1
        (True, "sdf", {}),  # mask + SDF:                   3 + 2
        (False, "none", {"encode_geometry": False}),  # state only: 3
        # geometry-blind AE, geometry-aware critic: the mask comes from the raw
        # `geometry` argument instead of the (absent) working-space block
        (False, "none", {"encode_geometry": True}),  # 3 + 1
    ],
)
def test_discriminator_sees_declared_channels(
    encode_geometry: bool,
    sdf_features: str,
    disc_kwargs: dict[str, Any],
) -> None:
    torch.manual_seed(0)
    model = _ae(encode_geometry=encode_geometry, sdf_features=sdf_features)
    disc = _disc(model, **disc_kwargs)
    trainer = _adversarial_trainer(model, disc, [_batch(seed=1)])

    seen: list[int] = []

    def _record(_module: Module, args: tuple[Any, ...]) -> None:
        seen.append(int(args[0].shape[1]))

    disc.register_forward_pre_hook(_record)
    trainer._train_epoch()

    assert seen, "the discriminator was never called"
    # one generator pass + two critic passes (real, fake), all at the declared width
    assert set(seen) == {disc.n_input_channels}
    assert len(seen) == 3


def test_channel_mismatch_fails_at_construction() -> None:
    """A critic built for a different geometry contract fails loudly up front."""
    model = _ae(encode_geometry=True, sdf_features="both")  # 3 + 5 channels
    disc = TadpoleDiscriminator(
        n_state_channels=C, size="S", encode_geometry=True, sdf_features="none"
    )  # 3 + 1
    with pytest.raises(ValueError, match="n_input_channels"):
        _trainer(
            model,
            discriminator=disc,
            disc_optimizer=torch.optim.SGD(disc.parameters(), lr=0.1),
        )


def test_discriminator_is_conditioned_on_true_geometry() -> None:
    """The critic's geometry channels come from the TARGET, never the recon.

    Otherwise the autoencoder could hide flow errors behind a distorted obstacle
    field. Compared against the block the AE actually consumed."""
    torch.manual_seed(0)
    model = _ae(encode_geometry=True, sdf_features="sdf")
    disc = _disc(model)
    trainer = _adversarial_trainer(model, disc, [_batch(seed=1)])
    batch = _batch(seed=1)

    trainer._forward(batch)
    state, geometry, features = trainer._prepare_ae_batch(batch)
    target = model._assemble_working_input(state, geometry, features)

    real, fake = trainer._adv_real, trainer._adv_fake
    assert real is not None and fake is not None
    for stash in (real, fake):
        torch.testing.assert_close(stash[:, C:], target[:, C:])
    # ... and the real pass is the target's own state block
    torch.testing.assert_close(real[:, :C], target[:, :C])


# --------------------------------------------------------------------------- #
# Checkpoint / resume of the critic half.
# --------------------------------------------------------------------------- #


def _fit_one_epoch(
    tmp_path: Path, model: Module, disc: Module, **kw: Any
) -> AutoencoderTrainer:
    trainer = _adversarial_trainer(
        model,
        disc,
        [_batch(seed=1), _batch(seed=2)],
        weights_path=tmp_path / "weights.pt",
        **kw,
    )
    trainer.fit()
    return trainer


def test_checkpoint_round_trip_restores_discriminator(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _ae()
    disc = _disc(model)
    first = _fit_one_epoch(tmp_path, model, disc)
    ckpt_path = tmp_path / "checkpoint.pt"
    assert ckpt_path.exists()
    saved = _snapshot(disc)
    assert first._global_step == 2

    # A fresh (differently initialised) critic + trainer resuming the same run.
    torch.manual_seed(1)
    model2 = _ae()
    disc2 = _disc(model2)
    assert _any_changed(disc2, saved), "fixture bug: the fresh critic already matches"
    second = _adversarial_trainer(
        model2,
        disc2,
        [_batch(seed=1)],
        weights_path=tmp_path / "weights.pt",
        resume=True,
    )
    second.fit()

    assert second._global_step == first._global_step
    for p, q in zip(disc2.parameters(), saved):
        torch.testing.assert_close(p.detach(), q)
    # the optimizer state travelled too (SGD without momentum keeps only counters,
    # so assert on the structure rather than on moment tensors)
    assert second.disc_optimizer is not None and first.disc_optimizer is not None
    assert second.disc_optimizer.state_dict()["param_groups"] == (
        first.disc_optimizer.state_dict()["param_groups"]
    )


def test_resume_from_pre_extension_checkpoint(tmp_path: Path) -> None:
    """A checkpoint written before this extension has no critic keys; resuming
    from it must work and simply start the warm-up clock at zero."""
    torch.manual_seed(0)
    model = _ae()
    plain = _trainer(
        model,
        batches=[_batch(seed=1)],
        weights_path=tmp_path / "weights.pt",
    )
    plain.fit()
    ckpt = torch.load(tmp_path / "checkpoint.pt")
    assert "discriminator" not in ckpt

    torch.manual_seed(1)
    model2 = _ae()
    disc2 = _disc(model2)
    resumed = _adversarial_trainer(
        model2,
        disc2,
        [_batch(seed=1)],
        weights_path=tmp_path / "weights.pt",
        resume=True,
    )
    resumed.fit()  # must not raise
    assert resumed._global_step == 0


# --------------------------------------------------------------------------- #
# End-to-end: compose pretrain_autoencoder.yaml with the discriminator enabled.
# --------------------------------------------------------------------------- #

NZ, NY, NX, T = 16, 16, 16, 4


def _load_pretrain_run() -> Callable[[DictConfig], None]:
    spec = importlib.util.spec_from_file_location("pretrain_ae_adv_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run: Callable[[DictConfig], None] = mod.run
    return run


def _write_dataset(root: Path) -> None:
    rng = np.random.default_rng(0)
    # The state variables are 4-D (time, z, y, x) while `blanking` is a static 3-D
    # mask, so the value type is annotated with a variable-length dim tuple rather
    # than inferred from the (state-only) comprehension.
    for split, n in {"train": 2, "val": 1}.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        blank = np.zeros((NZ, NY, NX), "f4")
        blank[0] = 1.0
        for i in range(n):
            data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((T, NZ, NY, NX)).astype("f4"),
                )
                for v in STATE_VARS
            }
            data["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                data,
                coords=dict(
                    time=np.arange(T) * 1.0,
                    z=np.arange(NZ),
                    y=np.arange(NY),
                    x=np.arange(NX),
                ),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")


def test_pretrain_end_to_end_with_discriminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="neural_surrogate/pretrain_autoencoder",
            overrides=[
                f"dataset.root_dir={data_dir}",
                "model_name=tadpole_ae_gan_test",
                # `++` because `discriminator` ships as an explicit null and
                # `disc_optimizer` ships only as a comment.
                "++discriminator={_target_: neural_surrogates.TadpoleDiscriminator}",
                "++disc_optimizer={_target_: torch.optim.AdamW, lr: 1.0e-4}",
                "loss.adv_start_step=0",
                "loss.adv_ramp_steps=0",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.architecture.encoder_crop_size = CROP
    cfg.architecture.size = "S"
    # The shipped config pairs architecture.sdf_features=both with
    # dataset.sdf_features=sdf, which the script rejects; align them so this test
    # exercises the wiring rather than that (pre-existing) config mismatch. It
    # also puts a 5-channel geometry block in front of the critic.
    cfg.dataset.sdf_features = cfg.architecture.sdf_features
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.trainer.num_epochs = 1
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.patience = None
    cfg.trainer.lr_warmup_epochs = None
    cfg.trainer.cudnn_benchmark = False
    cfg.trainer.tf32 = False
    cfg.trainer.resume = False

    monkeypatch.chdir(tmp_path)
    _load_pretrain_run()(cfg)

    out = tmp_path / "model_weights" / "tadpole_ae_gan_test"
    assert (out / "weights.pt").exists()
    with (out / "metrics.csv").open() as f:
        header = next(csv.reader(f))
    for col in ("train_recon", "train_adv", "train_adv_w", "train_d"):
        assert col in header, f"missing metrics column {col!r} in {header}"
    # validation stays adversary-free, so no val_adv column is ever written
    assert "val_adv" not in header
    # the critic's state rides along in the checkpoint
    ckpt = torch.load(out / "checkpoint.pt")
    assert {"discriminator", "disc_optimizer", "disc_scaler", "global_step"} <= set(
        ckpt
    )
