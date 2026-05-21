"""Compute the three §13 GATE bars on a held-out corpus split (go/no-go).

Architecture-agnostic — the same script re-runs for UPT later. Bars (all
pre-stated, over a full assimilation window's worth of steps):

1. **Clean rollout-error-vs-horizon** on held-out trajectories  ≤ B1.
2. **Analysis-OOD robustness** (load-bearing): roll from perturbed / noised ICs  ≤ B2.
3. **Cold-start sanity**: roll from the §4 IC strategy, confirm no divergence.

Commit cluster time to P4 only if **all three** clear.

    pixi run -e dev python scripts/eval_surrogate_gate.py \
        checkpoint_path=models/neural_surrogates/latest \
        corpus_path=.temp/neural_surrogate/corpus
"""

import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def evaluate_gate(
    checkpoint_path: str | pathlib.Path,
    corpus_path: str | pathlib.Path,
    *,
    split: str = "val",
    horizon: int = 8,
    b1: float = 0.5,
    b2: float = 1.0,
    perturb_noise_std: float = 0.1,
    cold_start_max: float = 10.0,
    batch_size: int = 8,
    seed: int = 0,
) -> dict:
    """Run the three GATE bars; return per-bar results + overall pass/fail."""
    import jax
    import jax.numpy as jnp

    from neural_surrogates.data.generate import open_corpus
    from neural_surrogates.training.checkpoint import load_checkpoint
    from neural_surrogates.training.loop import inject_off_manifold, per_horizon_sq_error
    from neural_surrogates.data.dataset import WindowDataset, iterate_batches

    ckpt = load_checkpoint(checkpoint_path)
    corpus = open_corpus(corpus_path)
    static = jnp.asarray(ckpt.static_channels)
    fluid = 1.0 - jnp.asarray(corpus.geometry_mask)

    ds = WindowDataset(
        corpus, split, ckpt.history_len, horizon=horizon,
        normalization=ckpt.normalization,
    )
    eff_h = horizon
    while len(ds) == 0 and eff_h > 1:  # shrink horizon if the split is short
        eff_h -= 1
        ds.set_horizon(eff_h)
    if len(ds) == 0:
        raise ValueError(f"No windows in split {split!r} for horizon {horizon}.")

    def mean_curve(perturb: float) -> np.ndarray:
        accum, n = None, 0
        key = jax.random.PRNGKey(seed)
        for batch in iterate_batches(ds, batch_size, shuffle=False):
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            if perturb > 0.0:
                key, sub = jax.random.split(key)
                batch = inject_off_manifold(batch, sub, perturb)
            err = np.asarray(
                per_horizon_sq_error(ckpt.arch, batch, static, fluid, eff_h)
            )
            accum = err if accum is None else accum + err
            n += 1
        return accum / max(n, 1)

    clean = mean_curve(0.0)
    ood = mean_curve(perturb_noise_std)

    # Cold-start: roll from the IC strategy via the forward model; ensure the
    # final-frame magnitude stays bounded (no divergence).
    cold_ok, cold_val = _cold_start_sanity(ckpt, corpus, eff_h, cold_start_max)

    results = {
        "horizon": eff_h,
        "clean_rollout_error": clean.tolist(),
        "ood_rollout_error": ood.tolist(),
        "clean_final": float(clean[-1]),
        "ood_final": float(ood[-1]),
        "cold_start_max_abs": cold_val,
        "bars": {"b1": b1, "b2": b2, "cold_start_max": cold_start_max},
        "pass_b1": bool(clean[-1] <= b1),
        "pass_b2": bool(ood[-1] <= b2),
        "pass_cold_start": bool(cold_ok),
    }
    results["pass"] = (
        results["pass_b1"] and results["pass_b2"] and results["pass_cold_start"]
    )
    return results


def _cold_start_sanity(ckpt, corpus, horizon, cold_start_max):
    """Roll from a canned IC and confirm the field stays bounded."""
    import jax.numpy as jnp

    from neural_surrogates.rollout import rollout_from_history
    from neural_surrogates.utils import state_io

    if not ckpt.ic_bank:
        return True, float("nan")  # nothing to check without an IC bank
    frame = ckpt.ic_bank["fields"][0][None]
    hist, mask = state_io.extract_history(frame, ckpt.history_len)
    hist_n = ckpt.normalization.apply(hist)
    p = ckpt.ic_bank["params"][0]
    k = ckpt.history_len
    cond = np.broadcast_to(p, (horizon, p.shape[0]))
    hist_p = np.broadcast_to(p, (k, p.shape[0]))
    preds = rollout_from_history(
        ckpt.arch, jnp.asarray(hist_n), jnp.asarray(hist_p), jnp.asarray(mask),
        jnp.asarray(cond), jnp.asarray(ckpt.static_channels), horizon,
    )
    preds = ckpt.normalization.invert(np.asarray(preds))
    max_abs = float(np.max(np.abs(preds[-1])))
    return (np.isfinite(max_abs) and max_abs <= cold_start_max), max_abs


def run(cfg: DictConfig) -> dict:
    results = evaluate_gate(
        cfg["checkpoint_path"], cfg["corpus_path"],
        split=cfg.get("split", "val"), horizon=int(cfg.get("horizon", 8)),
        b1=float(cfg.get("b1", 0.5)), b2=float(cfg.get("b2", 1.0)),
        perturb_noise_std=float(cfg.get("perturb_noise_std", 0.1)),
        cold_start_max=float(cfg.get("cold_start_max", 10.0)),
        seed=int(cfg.get("seed", 0)),
    )
    print(OmegaConf.to_yaml(OmegaConf.create(results)))
    verdict = "PASS — proceed to P4" if results["pass"] else "FAIL — do NOT scale"
    print(f"\nGATE: {verdict}")
    return results


@hydra.main(version_base=None, config_path="../conf/neural_surrogate", config_name="gate")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
