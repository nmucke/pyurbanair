"""Neural surrogates: training, evaluation and data generation."""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tests.conftest import (
    CONFIGS,
    TRAIN_ORDER,
    compose,
    inlet_turbulence_corpus,
    load_script,
)


@pytest.mark.parametrize("name", TRAIN_ORDER)  # type: ignore[misc]
def test_train_writes_a_rebuildable_model(name: str, trained: dict[str, Any]) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = trained[name]
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert (model_dir / "config.yaml").exists()
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) > 0


def test_autoencoder_exports_encoder_and_decoder(trained: dict[str, Any]) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = trained["train_autoencoder"]
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))  # weights.pt
    parts = {"encoder.pt": model.ae.encoder, "decoder.pt": model.ae.decoder}
    if model.geometry_branch is not None:
        parts["geometry_branch.pt"] = model.geometry_branch
    for file, module in parts.items():
        exported = torch.load(model_dir / file)
        expected = module.state_dict()
        assert exported.keys() == expected.keys(), file
        assert all(torch.equal(exported[k], expected[k]) for k in expected), file


def test_prechunk_prepare_only(
    tmp_path: pathlib.Path, training_data: pathlib.Path
) -> None:
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_autoencoder",
        "+test=train_autoencoder",
        f"paths.data_dir={training_data}",
        f"prechunk.output_root={prepared}",
        "prechunk.prepare_only=true",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    sources = sorted(
        p.relative_to(training_data) for p in training_data.glob("state/*/sample_*.nc")
    )
    assert sources
    assert sorted(p.relative_to(prepared) for p in prepared.glob("state/*/*.nc")) == (
        sources
    )
    manifest = json.loads((prepared / ".rechunk-manifest.json").read_text())
    assert manifest["complete"]
    assert sorted(manifest["files"]) == [str(p) for p in sources]
    assert not pathlib.Path(cfg.paths.weights_dir).exists()


def test_prechunked_training_keeps_the_source_root(
    tmp_path: pathlib.Path, training_data: pathlib.Path
) -> None:
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_autoencoder",
        "+test=train_autoencoder",
        f"paths.data_dir={training_data}",
        f"prechunk.output_root={prepared}",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    saved = OmegaConf.load(model_dir / "config.yaml")
    assert saved.dataset.root_dir == str(training_data)
    assert (prepared / "normalization_stats" / "train.npz").exists()
    assert (model_dir / "weights.pt").exists()


def test_prechunked_stepper_reads_source_params_and_stats(
    tmp_path: pathlib.Path,
    training_data: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_surrogates.training import data_utils

    compute = data_utils._compute_normalization_stats
    streamed: list[Any] = []

    def spy(train_ds: Any, source_root: Any = None) -> Any:
        streamed.append(source_root)
        return compute(train_ds, source_root=source_root)

    monkeypatch.setattr(data_utils, "_compute_normalization_stats", spy)
    eval_common = load_script("scripts/utils/eval_common.py")
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_stepper",
        "+test=train_stepper",
        f"paths.data_dir={training_data}",
        f"prechunk.output_root={prepared}",
        "prechunk.spatial_chunks=null",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    assert not (prepared / "param").exists()  # params come from the source
    # The state stats stream from the source and are cached under the copy.
    assert streamed == [str(training_data)]
    assert (prepared / "normalization_stats" / "train.npz").is_file()
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert OmegaConf.load(model_dir / "config.yaml").dataset.root_dir == str(
        training_data
    )
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) > 0


def test_prechunked_dft_reads_whole_frames_and_source_params(
    tmp_path: pathlib.Path, training_data: pathlib.Path, trained: dict[str, Any]
) -> None:
    ae = trained["train_autoencoder"]
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_dft",
        "+test=train_dft",
        f"paths.data_dir={training_data}",
        f"pretrained_dir={pathlib.Path(ae.paths.weights_dir) / ae.name}",
        f"prechunk.output_root={prepared}",
        "prechunk.spatial_chunks=null",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    manifest = json.loads((prepared / ".rechunk-manifest.json").read_text())
    assert manifest["complete"] and manifest["options"]["spatial_chunks"] is None
    assert not (prepared / "param").exists()  # params come from the source
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert OmegaConf.load(model_dir / "config.yaml").dataset.root_dir == str(
        training_data
    )
    assert (model_dir / "weights.pt").exists()


def _finetune_dft_cfg(
    root: pathlib.Path, training_data: pathlib.Path, trained: dict[str, Any]
) -> Any:
    dft = trained["train_dft"]
    return compose(
        "surrogate/finetune_dft",
        "+test=finetune_dft",
        f"paths.data_dir={training_data}",
        f"pretrained_dir={pathlib.Path(dft.paths.weights_dir) / dft.name}",
        root=root,
    )


def test_finetune_dft_trains_adapters_and_skips_only(
    tmp_path: pathlib.Path, training_data: pathlib.Path, trained: dict[str, Any]
) -> None:
    cfg = _finetune_dft_cfg(tmp_path, training_data, trained)
    OmegaConf.set_struct(cfg, False)
    setup = load_script("scripts/utils/tasks.py").TASKS[cfg.task](cfg, tmp_path)
    trainable = [n for n, p in setup.model.named_parameters() if p.requires_grad]
    lora = [n for n in trainable if ".lora_" in n]
    for part in (".dft.encoder.", ".dft.decoder.", ".dft.subnetwork.seqmodel."):
        assert any(part in n for n in lora), part
    assert all(
        ".dft.encoder." in n or ".dft.decoder." in n or ".seqmodel." in n for n in lora
    )
    full = [n for n in trainable if ".lora_" not in n]
    for module in cfg.trainable_modules:  # every listed module trains, fully
        params = [n for n, _ in setup.model.named_parameters() if f".{module}" in n]
        assert params and all(n in full for n in params), module
    assert all(any(f".{m}" in n for m in cfg.trainable_modules) for n in full)
    assert not any("geometry_branch" in n or "_film" in n for n in trainable)
    assert not any(".param_mlp." in n for n in trainable)


def test_finetune_dft_rollout_and_export(
    tmp_path: pathlib.Path,
    training_data: pathlib.Path,
    trained: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_surrogates import Trainer
    from neural_surrogates.finetuning import resolve_target_modules

    # Every training-mode model call carries gradients: backprop through the
    # whole rollout.
    calls: list[bool] = []
    model_forward = Trainer._model_forward

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.model.training:
            calls.append(torch.is_grad_enabled())
        return model_forward(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "_model_forward", spy)
    cfg = _finetune_dft_cfg(tmp_path, training_data, trained)
    load_script("scripts/surrogate/train.py").run(cfg)
    assert calls and all(calls)
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    with open(model_dir / "metrics.csv") as f:
        horizons = [int(row["pushforward_steps"]) for row in csv.DictReader(f)]
    assert horizons == [2, 3]  # the curriculum, up to dataset.pushforward_steps

    # A plain merged dict that rebuilds the architecture from config.yaml alone.
    dft = trained["train_dft"]
    before = torch.load(pathlib.Path(dft.paths.weights_dir) / dft.name / "weights.pt")
    after = torch.load(model_dir / "weights.pt")
    assert after.keys() == before.keys()
    eval_common = load_script("scripts/utils/eval_common.py")
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    adapted = resolve_target_modules(model, preset=cfg.lora.target_preset)
    trained_prefixes = (*adapted, *cfg.trainable_modules)
    for key in before:  # frozen: neither adapted nor listed
        if not any(f".{p}." in f".{key}." for p in trained_prefixes):
            assert torch.equal(after[key], before[key]), key
    for key in ("dft.subnetwork.seqmodel.input_proj.weight", "dft.decoder"):
        changed = [k for k in before if k.startswith(key)]
        assert not all(torch.equal(after[k], before[k]) for k in changed), key
    assert all(torch.equal(model.state_dict()[k], after[k]) for k in after)
    assert (model_dir / "adapter" / "adapter_config.json").exists()


def _latent_cache_cfg(
    tmp_path: pathlib.Path, trained: dict[str, Any], *overrides: str
) -> Any:
    ae = trained["train_autoencoder"]
    return compose(
        "surrogate/train_latent_generator",
        "+test=train_latent_generator",
        f"paths.data_dir={trained['train_latent_generator'].paths.data_dir}",
        f"autoencoder_dir={pathlib.Path(ae.paths.weights_dir) / ae.name}",
        f"latent_cache.output_root={tmp_path / 'latents'}",
        *overrides,
        root=tmp_path,
    )


def test_latent_cache_prepare_only(
    tmp_path: pathlib.Path, trained: dict[str, Any]
) -> None:
    cfg = _latent_cache_cfg(tmp_path, trained, "latent_cache.prepare_only=true")
    load_script("scripts/surrogate/train.py").run(cfg)
    manifest = json.loads((tmp_path / "latents" / "manifest.json").read_text())
    assert manifest["complete"]
    assert not pathlib.Path(cfg.paths.weights_dir).exists()


def test_latent_generator_trains_from_the_cache(
    tmp_path: pathlib.Path, trained: dict[str, Any]
) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = _latent_cache_cfg(tmp_path, trained)
    load_script("scripts/surrogate/train.py").run(cfg)
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    saved = OmegaConf.load(model_dir / "config.yaml")
    assert saved.dataset.root_dir == cfg.paths.data_dir  # evaluation reads states
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert bool(model.latent_stats_installed)


def test_adaln_latent_generator_rebuilds_from_its_config(
    tmp_path: pathlib.Path, trained: dict[str, Any]
) -> None:
    from neural_surrogates.architectures.latent_velocity_adaln import AdaLNVelocityNet

    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = _latent_cache_cfg(
        tmp_path, trained, "architecture.velocity_conditioning=adaln"
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    saved = OmegaConf.load(model_dir / "config.yaml")
    assert saved.architecture.velocity_conditioning == "adaln"
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert isinstance(model.velocity_net, AdaLNVelocityNet)


@pytest.mark.parametrize(  # type: ignore[misc]
    "option",
    sorted(p.stem for p in (CONFIGS / "latent_generator_architecture").glob("*.yaml")),
)
def test_latent_generator_architecture_builds(
    option: str, tmp_path: pathlib.Path
) -> None:
    """Each velocity-net option, made shallow and narrow, builds on a tiny
    autoencoder and gives a finite velocity of the latent's shape."""
    pytest.importorskip("diffusers")
    from tests.neural_surrogates._latent_generator_fixtures import (
        GRID,
        HP,
        NET,
        C,
        P,
        make_ae_export,
    )

    cfg = compose(
        "surrogate/train_latent_generator",
        f"latent_generator_architecture={option}",
        root=tmp_path,
    )
    arch = cfg.architecture
    tiny = {**NET, "param_history_steps": HP, "max_latent_tokens": None}
    if arch.hidden_size is not None:
        tiny["hidden_size"] = 16
    for key in ("head_hidden", "mlp_hidden"):
        if key in arch:
            tiny[key] = 32
    model = instantiate(
        OmegaConf.merge(arch, tiny),
        n_state_channels=C,
        n_params=P,
        pretrained_ae_dir=str(make_ae_export(tmp_path)),
    )
    assert model.velocity_conditioning == arch.velocity_conditioning
    assert model.latent_skip == arch.latent_skip
    assert model.gaussian_skip == arch.gaussian_skip
    n, d = model.working_latent_dim, model.state_latent_dim
    model.set_latent_normalization(
        torch.zeros(n), torch.ones(n), second_moment=torch.eye(d)
    )
    cond = model.geometry_condition(torch.ones(*GRID), batch_size=2)
    z = torch.randn(2, d, *model.latent_grid_for(GRID))
    with torch.no_grad():
        v = model.velocity(z, torch.rand(2), torch.randn(2, HP, P), cond)
    assert v.shape == z.shape and torch.isfinite(v).all()


@pytest.mark.parametrize(  # type: ignore[misc]
    "design",
    [
        # Design A: narrow adaLN trunk + per-token head, clean-sample prediction.
        [
            "architecture.velocity_conditioning=adaln",
            "+architecture.head_hidden=32",
            "+architecture.head_layers=1",
            "+architecture.prediction=data",
        ],
        # Design B: D-wide residual stream, narrow attention, shifted time grid.
        [
            "architecture.velocity_conditioning=adaln_bottleneck",
            "+architecture.mlp_hidden=32",
            "+architecture.time_shift=3.0",
        ],
    ],
    ids=["design_a", "design_b"],
)
def test_full_reach_latent_generator_trains_samples_and_evaluates(
    tmp_path: pathlib.Path, trained: dict[str, Any], design: list[str]
) -> None:
    """A sub-D net with the Gaussian skip trains from the cache (which stores
    the second moment), rebuilds strictly, and goes through the evaluation."""
    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = _latent_cache_cfg(
        tmp_path,
        trained,
        "architecture.hidden_size=16",
        # The Gaussian skip replaces the per-channel latent_skip.
        "architecture.latent_skip=false",
        "architecture.gaussian_skip=true",
        *design,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert model.gaussian_skip and "gaussian_skip=True" in model.extra_repr()
    eig = model.latent_pca_eig
    assert torch.isfinite(eig).all() and float(eig.min()) > 0
    assert not torch.equal(model.latent_pca_basis, torch.eye(len(eig)))

    ev = compose(
        "surrogate/eval",
        "+test=eval",
        f"latent_generator.model_dir={model_dir}",
        "latent_generator.max_snapshots=2",
        "latent_generator.num_steps_sweep=[2]",
        root=tmp_path,
    )
    load_script("scripts/surrogate/evaluate_latent_generator.py").run(ev)
    summary = json.loads(
        (pathlib.Path(ev.latent_generator.output_dir) / "summary.json").read_text()
    )
    assert np.isfinite(summary["scalars"]["generated"]["w1"])
    assert all(np.isfinite(r) for r in summary["latent_spectrum"]["var_ratio"].values())


def test_evaluate_stepper(session_root: pathlib.Path, trained: dict[str, Any]) -> None:
    cfg = compose("surrogate/eval", "+test=eval", root=session_root)
    load_script("scripts/surrogate/evaluate_stepper.py").run(cfg)
    out = pathlib.Path(cfg.stepper.output_dir)
    for name in (
        "metrics.csv",
        "rmse.png",
        "slices.png",
        "tke_slices.png",
        "params.png",
    ):
        assert (out / name).exists(), name


def test_evaluate_stepper_inlet_turbulence(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    trained: dict[str, Any],
) -> None:
    """The forcing changes the first predicted frame only upstream of the
    building (x < 7)."""
    script = load_script("scripts/surrogate/evaluate_stepper.py")
    rollouts: dict[bool, Any] = {}
    rollout = script._rollout
    for enabled in (False, True):

        def capture(*args: Any, enabled: bool = enabled) -> Any:
            rollouts[enabled] = rollout(*args)
            return rollouts[enabled]

        monkeypatch.setattr(script, "_rollout", capture)
        cfg = compose(
            "surrogate/eval",
            "+test=eval",
            f"data_dir={inlet_turbulence_corpus(training_data, tmp_path / str(enabled))}",
            f"stepper.output_dir={tmp_path / str(enabled) / 'eval'}",
            "stepper.max_steps=3",
            f"inlet_turbulence.enabled={str(enabled).lower()}",
            root=session_root,
        )
        script.run(cfg)
        assert (pathlib.Path(cfg.stepper.output_dir) / "metrics.csv").exists()
    model_cfg = OmegaConf.load(session_root / "weights" / "stepper" / "config.yaml")
    first = int(model_cfg.dataset.get("num_history_steps") or 1)  # first predicted
    forced, plain = rollouts[True], rollouts[False]
    assert np.isfinite(forced).all()
    np.testing.assert_array_equal(forced[:first], plain[:first])  # the truth
    for c in range(forced.shape[1]):
        assert not np.allclose(forced[first, c, ..., 0], plain[first, c, ..., 0]), c
    np.testing.assert_array_equal(forced[first, ..., 7:], plain[first, ..., 7:])


def test_evaluate_stepper_inlet_turbulence_input(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    trained: dict[str, Any],
) -> None:
    """`apply: input` stores the model's own output: the first predicted frame
    (from the truth) is unforced, the next one sees the forced input."""
    script = load_script("scripts/surrogate/evaluate_stepper.py")
    rollouts: dict[bool, Any] = {}
    rollout = script._rollout
    for enabled in (False, True):

        def capture(*args: Any, enabled: bool = enabled) -> Any:
            rollouts[enabled] = rollout(*args)
            return rollouts[enabled]

        monkeypatch.setattr(script, "_rollout", capture)
        cfg = compose(
            "surrogate/eval",
            "+test=eval",
            f"data_dir={inlet_turbulence_corpus(training_data, tmp_path / str(enabled))}",
            f"stepper.output_dir={tmp_path / str(enabled) / 'eval'}",
            "stepper.max_steps=3",
            f"inlet_turbulence.enabled={str(enabled).lower()}",
            "inlet_turbulence.apply=input",
            root=session_root,
        )
        script.run(cfg)
    model_cfg = OmegaConf.load(session_root / "weights" / "stepper" / "config.yaml")
    first = int(model_cfg.dataset.get("num_history_steps") or 1)  # first predicted
    forced, plain = rollouts[True], rollouts[False]
    assert np.isfinite(forced).all()
    np.testing.assert_array_equal(forced[: first + 1], plain[: first + 1])
    assert not np.allclose(forced[first + 1], plain[first + 1])


def test_evaluate_stepper_periodic_y(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    session_root: pathlib.Path,
    trained: dict[str, Any],
) -> None:
    """periodic_y changes the predictions at the y edges and adds a
    `periodic_y` column to metrics.csv; the default run has none."""
    script = load_script("scripts/surrogate/evaluate_stepper.py")
    rollouts: dict[bool, Any] = {}
    rollout = script._rollout
    for enabled in (False, True):

        def capture(*args: Any, enabled: bool = enabled) -> Any:
            rollouts[enabled] = rollout(*args)
            return rollouts[enabled]

        monkeypatch.setattr(script, "_rollout", capture)
        cfg = compose(
            "surrogate/eval",
            "+test=eval",
            f"stepper.output_dir={tmp_path / str(enabled)}",
            "stepper.max_steps=3",
            f"periodic_y={str(enabled).lower()}",
            root=session_root,
        )
        script.run(cfg)
        with open(pathlib.Path(cfg.stepper.output_dir) / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        if enabled:
            assert all(float(r["periodic_y"]) == 1 for r in rows)
        else:
            assert "periodic_y" not in rows[0]
    model_cfg = OmegaConf.load(session_root / "weights" / "stepper" / "config.yaml")
    first = int(model_cfg.dataset.get("num_history_steps") or 1)  # first predicted
    periodic, plain = rollouts[True], rollouts[False]
    assert np.isfinite(periodic).all()
    np.testing.assert_array_equal(periodic[:first], plain[:first])  # the truth
    for edge in (0, -1):
        assert not np.allclose(
            periodic[first, ..., edge, :], plain[first, ..., edge, :]
        )


def test_evaluate_autoencoder(
    session_root: pathlib.Path, trained: dict[str, Any]
) -> None:
    cfg = compose("surrogate/eval", "+test=eval", root=session_root)
    load_script("scripts/surrogate/evaluate_autoencoder.py").run(cfg)
    out = pathlib.Path(cfg.autoencoder.output_dir)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["per_variable"]) == {"u", "v", "w"}
    assert (out / "latent_stats.png").exists()
    assert list(out.glob("vertical_profiles_*.png"))


def test_evaluate_latent_generator(
    session_root: pathlib.Path, trained: dict[str, Any]
) -> None:
    cfg = compose(
        "surrogate/eval",
        "+test=eval",
        f"latent_generator.rollout_stepper_dir={session_root / 'weights' / 'stepper'}",
        root=session_root,
    )
    load_script("scripts/surrogate/evaluate_latent_generator.py").run(cfg)
    out = pathlib.Path(cfg.latent_generator.output_dir)
    summary = json.loads((out / "summary.json").read_text())
    assert "passed" in summary["acceptance"]
    assert [row["num_steps"] for row in summary["step_sweep"]] == [2, 5]
    assert set(summary["rollout_kinetic_energy"]) == {"real", "ae_recon", "generated"}
    spectrum = summary["latent_spectrum"]
    assert spectrum["var_ratio"] and all(
        np.isfinite(r) and r > 0 for r in spectrum["var_ratio"].values()
    )
    assert len(spectrum["real_eigenvalues"]) == len(spectrum["generated_variance"])
    with open(out / "metrics.csv") as f:
        generated = next(r for r in csv.DictReader(f) if r["source"] == "generated")
    assert all(f"latent_var_ratio_{b}" in generated for b in spectrum["var_ratio"])
    assert (out / "latent_spectrum.png").exists()


def test_evaluate_latent_generator_decode_modes_and_samplers(
    session_root: pathlib.Path, trained: dict[str, Any]
) -> None:
    """Decode-mode variants and a per-sampler step sweep add their own sources;
    the trained mode adds none, and the gate still judges `generated`."""
    cfg = compose(
        "surrogate/eval",
        "+test=eval",
        "latent_generator.output_dir=" + str(session_root / "lg_eval_variants"),
        "latent_generator.decode_modes=[local,halo,global]",
        "latent_generator.samplers=[euler,heun]",
        "latent_generator.num_steps=5",
        "latent_generator.max_snapshots=6",
        "latent_generator.max_internal_batchsize=2",
        root=session_root,
    )
    trained_mode = trained["train_autoencoder"].architecture.spatial_mode
    load_script("scripts/surrogate/evaluate_latent_generator.py").run(cfg)
    out = pathlib.Path(cfg.latent_generator.output_dir)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["num_steps"] == 5
    others = [m for m in ("local", "halo", "global") if m != trained_mode]
    assert summary["decode_modes"] == others
    expected = {f"{k}_decode_{m}" for m in others for k in ("generated", "ae_recon")}
    expected |= {"generated_steps2", "generated_steps5"}
    expected |= {"generated_heun_steps2", "generated_heun_steps5"}
    assert expected <= set(summary["scalars"])
    assert f"generated_decode_{trained_mode}" not in summary["scalars"]
    for m in others:
        assert "diversity" in summary["scalars"][f"generated_decode_{m}"]
    assert [
        (r["sampler"], r["num_steps"], r["nfe"]) for r in summary["step_sweep"]
    ] == [
        ("euler", 2, 2),
        ("euler", 5, 5),
        ("heun", 2, 4),
        ("heun", 5, 10),
    ]
    with open(out / "metrics.csv") as f:
        sources = {line.split(",")[0] for line in f.readlines()[1:]}
    assert expected <= sources
    for name in ("profiles.png", "histograms.png", "spectra.png", "step_sweep.png"):
        assert (out / name).exists(), name


@pytest.mark.integration  # type: ignore[misc]
def test_generate_data(tmp_path: pathlib.Path) -> None:
    cfg = compose("surrogate/generate_data", "+test=generate_data", root=tmp_path)
    load_script("scripts/surrogate/generate_data.py").run(cfg)
    out = pathlib.Path(cfg.paths.results_dir)
    counts = {
        s: len(list((out / "state" / s).glob("*.nc"))) for s in ("train", "val", "test")
    }
    assert counts == {"train": 2, "val": 1, "test": 1}


def _seed_ensemble_corpus(root: pathlib.Path) -> pathlib.Path:
    """A reference corpus as the archived generator wrote it: two test samples."""
    import xarray

    (root / "state" / "test").mkdir(parents=True)
    (root / "param" / "test").mkdir(parents=True)
    (root / "geometries").mkdir()
    fields = "split,sample,stl_file,nx,ny,nz,lx_domain_m,ly_domain_m,lz_domain_m"
    rows = [
        f"{fields},x0_domain_m,y0_domain_m",
        "train,0000,a.stl,16,16,8,64,64,32,-8,-8",
    ]
    for i, (stl, spinup) in enumerate([("b.stl", 20.0), ("c.stl", 30.0)]):
        rows.append(f"test,{i:04d},{stl},16,32,8,64,128,32,-8,-16")
        (root / "geometries" / stl).touch()
        state = xarray.Dataset(attrs={"spinup_time_s": spinup})
        state.to_netcdf(root / "state" / "test" / f"sample_{i:04d}.nc")
        times = np.arange(0.0, 20.0, 5.0, dtype=np.float32)
        xarray.Dataset(
            {"inflow_angle": ("time", np.full(4, 10.0 * i))},
            coords={"time": times},
        ).to_netcdf(root / "param" / "test" / f"sample_{i:04d}.nc")
    (root / "geometries.csv").write_text("\n".join(rows) + "\n")
    return root


def test_seed_ensemble_reads_the_reference_samples(tmp_path: pathlib.Path) -> None:
    script = load_script("scripts/surrogate/generate_data.py")
    ref = _seed_ensemble_corpus(tmp_path / "ref")
    cfg = compose(
        "surrogate/generate_seed_ensemble",
        "+test=generate_seed_ensemble",
        f"data.seed_ensemble.reference_dir={ref}",
        root=tmp_path,
    )
    refs = script._references(cfg.data.seed_ensemble)
    assert [(r.index, r.geometry.stl.name, r.spinup) for r in refs] == [
        (0, "b.stl", 20.0),
        (1, "c.stl", 30.0),
    ]
    assert refs[0].geometry.bounds == ((-8.0, 56.0), (-16.0, 112.0), (0.0, 32.0))
    assert (refs[0].geometry.nx, refs[0].geometry.ny) == (16, 32)
    cfg.data.seed_ensemble.samples = [1]
    assert [r.index for r in script._references(cfg.data.seed_ensemble)] == [1]
    cfg.data.seed_ensemble.samples = [5]
    with pytest.raises(ValueError, match="only"):
        script._references(cfg.data.seed_ensemble)


@pytest.mark.parametrize(  # type: ignore[misc]
    "vary", [["initial_condition"], ["initial_condition", "inlet_turbulence"]]
)
def test_seed_ensemble_varies_only_the_chosen_seeds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, vary: list[str]
) -> None:
    """Each member is a cold start on a fresh clock with its own seeds; member 0
    keeps the reference's, and rerunning skips what is on disk."""
    import xarray

    script = load_script("scripts/surrogate/generate_data.py")
    ref = _seed_ensemble_corpus(tmp_path / "ref")
    cfg = compose(
        "surrogate/generate_seed_ensemble",
        "+test=generate_seed_ensemble",
        f"data.seed_ensemble.reference_dir={ref}",
        f"data.seed_ensemble.vary=[{','.join(vary)}]",
        root=tmp_path,
    )
    runs: list[dict[str, Any]] = []

    class FakeModel:
        def __init__(self, experiment_dir: pathlib.Path) -> None:
            self.inlet_turbulence: dict[str, Any] = {"seed": None}
            self._elapsed_time = 0.0
            self.dirs = type(
                "Dirs", (), {"experiment_dir": experiment_dir, "experiment_name": "999"}
            )()
            (experiment_dir / "namoptions.999").write_text("&RUN\nirandom = 43\n/\n")

        def __call__(self, state: Any, params: xarray.Dataset) -> xarray.Dataset:
            from pyudales.utils.namoptions_utils import NamoptionsFile

            namoptions = NamoptionsFile(self.dirs.experiment_dir / "namoptions.999")
            runs.append(
                {
                    "state": state,
                    "clock": self._elapsed_time,
                    "irandom": int(namoptions.get_value("RUN", "irandom")),
                    "inlet_seed": self.inlet_turbulence["seed"],
                    "angle": float(params.inflow_angle[0]),
                }
            )
            self._elapsed_time += 25.0  # a cold start advances the clock
            times = np.array([5.0, 10.0])
            return xarray.Dataset({"u": ("time", times)}, coords={"time": times})

    def make_model(cfg: Any, geometry: Any, spinup: float) -> FakeModel:
        experiment_dir = tmp_path / "scratch" / geometry.stl.stem
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return FakeModel(experiment_dir)

    def save(cfg: Any, model: Any, state: Any, params: Any, sp: Any, pp: Any) -> None:
        for path in (sp, pp):
            path.parent.mkdir(parents=True, exist_ok=True)
        state.to_netcdf(sp)
        params.to_netcdf(pp)

    monkeypatch.setattr(script, "_make_model", make_model)
    monkeypatch.setattr(script, "_save", save)
    script.run(cfg)

    out = pathlib.Path(cfg.paths.results_dir)
    assert len(runs) == 6
    assert all(r["state"] is None and r["clock"] == 0.0 for r in runs)
    assert [r["angle"] for r in runs] == [0.0] * 3 + [10.0] * 3
    # runs[0:3] are sample 0's members, runs[3:6] sample 1's.
    for member0 in (runs[0], runs[3]):
        assert member0["irandom"] == 43 and member0["inlet_seed"] is None
    others = runs[1:3] + runs[4:6]
    # Every varied (sample, member) pair has its own seeds.
    assert len({r["irandom"] for r in others} | {43}) == 5
    inlet_seeds = [r["inlet_seed"] for r in others]
    if "inlet_turbulence" in vary:
        assert None not in inlet_seeds and len(set(inlet_seeds)) == 4
    else:
        assert inlet_seeds == [None] * 4
    state = xarray.open_dataset(out / "state" / "sample_0001" / "member_0002.nc")
    assert state.sizes["time"] == 1 and float(state.time[0]) == 10.0
    assert state.attrs["member"] == 2 and state.attrs["spinup_time_s"] == 30.0
    assert (out / "param" / "sample_0001" / "member_0002.nc").exists()

    runs.clear()
    script.run(cfg)
    assert runs == []
