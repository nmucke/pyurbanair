"""Neural surrogates: training, evaluation and data generation."""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import TRAIN_ORDER, compose, inlet_turbulence_corpus, load_script


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
        "+architecture.gaussian_skip=true",
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
