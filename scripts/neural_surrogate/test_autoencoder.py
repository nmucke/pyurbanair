"""Evaluate a trained Tadpole autoencoder (plan 02) on a data split.

Loads a ``model_dir`` written by ``pretrain_autoencoder.py`` (its ``config.yaml``
rebuilds the ``TadpoleAE``; ``weights.pt`` is the best-validation checkpoint),
reconstructs snapshots from the chosen split, and writes:

* ``reconstruction_traj{t}_snap{s}.png`` -- per (trajectory, snapshot) figure:
  rows ``truth / recon / |err|`` of velocity magnitude ``|u|``, one column per
  z-height. Obstacle cells are blanked.
* ``per_channel_metrics.png`` -- masked RMSE / MAE / relative-L2 per state channel.
* ``height_profile.png`` -- reconstruction RMSE vs normalised height ``z/nz``
  (aggregated across all metric snapshots and grids), i.e. where in the column
  the AE reconstructs well/poorly (near-ground vs top).
* ``error_hist.png`` -- distribution of per-cell reconstruction errors (fluid).
* ``pred_vs_true.png`` -- density of reconstructed vs true ``|u|`` over fluid
  cells (the identity line is a perfect AE).
* ``latent_stats.png`` -- per-snapshot KL and latent-activation summary.
* ``metrics.json`` / stdout -- the headline numbers.

All metrics are restricted to fluid cells (the ``mask_loss`` convention).
Reconstruction uses ``latent_type="mode"`` by default so the numbers are
reproducible (no VAE sampling noise).

    pixi run -e dev python scripts/neural_surrogate/test_autoencoder.py \
        model_dir=model_weights/tadpole_ae_s split=test
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_model(cfg: DictConfig, train_cfg: DictConfig, device: torch.device):
    dtype = getattr(torch, train_cfg.dataset.dtype)
    model = (
        instantiate(
            train_cfg.architecture,
            n_state_channels=len(train_cfg.dataset.state_vars),
        )
        .to(dtype=dtype)
        .to(device)
    )
    model.load_state_dict(
        torch.load(Path(cfg.model_dir) / "weights.pt", map_location=device)
    )
    model.eval()
    # Deterministic latent for reproducible reconstruction metrics (governs the
    # inner autoencoder's sampling).
    model.ae.latent_type = str(cfg.latent_type)
    return model, dtype


def _load_snapshots(dataset, traj: int, t_indices, dtype: torch.dtype):
    """Stack snapshots ``(len(t), C, *grid)`` of one trajectory + its geometry."""
    with xr.open_dataset(dataset._state_files[traj]) as ds:
        arr = np.stack(
            [np.asarray(ds[v].isel(time=t_indices).values) for v in dataset.state_vars],
            axis=1,
        )  # (len(t), C, *grid)
    state = torch.from_numpy(arr).to(dtype)
    geometry = dataset.geometry_for(traj)
    features = dataset.geom_features_for(traj)
    return state, geometry, features


@torch.no_grad()
def _reconstruct(model, state, geometry, features, device):
    """Physical-units reconstruction ``(B, C, *grid)`` for a batch on one grid."""
    b = state.shape[0]
    geom = geometry.unsqueeze(0).expand(b, *geometry.shape).to(device)
    feat = None
    if features is not None:
        feat = features.unsqueeze(0).expand(b, *features.shape).to(device)
    recon = model(state.to(device), geom, feat)
    return recon.cpu()


# --------------------------------------------------------------------------- #
# Metric accumulation (fluid cells only; grid-agnostic scalars)
# --------------------------------------------------------------------------- #


class _Accum:
    def __init__(self, n_channels: int, n_height_bins: int = 20) -> None:
        self.c = n_channels
        self.sse = np.zeros(n_channels)  # sum sq error per channel
        self.sae = np.zeros(n_channels)  # sum abs error
        self.s_true = np.zeros(n_channels)  # sum truth
        self.s_true2 = np.zeros(n_channels)  # sum truth^2
        self.s_tnorm2 = np.zeros(n_channels)  # sum truth^2 (for rel-L2, == s_true2)
        self.count = np.zeros(n_channels)
        self.nb = n_height_bins
        self.h_sse = np.zeros(n_height_bins)  # magnitude sse per normalised-height bin
        self.h_cnt = np.zeros(n_height_bins)
        # sampled scatter/hist reservoirs (fluid-cell magnitudes / errors)
        self.mag_true: list[np.ndarray] = []
        self.mag_pred: list[np.ndarray] = []
        self.err: list[np.ndarray] = []
        self.kl: list[float] = []

    def add(self, truth, recon, fluid):
        """truth/recon: (C, nz, ny, nx); fluid: (nz, ny, nx) bool."""
        t = truth.numpy()
        p = recon.numpy()
        f = fluid.numpy().astype(bool)
        for c in range(self.c):
            tc, pc = t[c][f], p[c][f]
            e = pc - tc
            self.sse[c] += np.sum(e**2)
            self.sae[c] += np.sum(np.abs(e))
            self.s_true[c] += np.sum(tc)
            self.s_true2[c] += np.sum(tc**2)
            self.count[c] += tc.size
        self.s_tnorm2 = self.s_true2
        # height profile on velocity magnitude
        mag_t = np.linalg.norm(t, axis=0)  # (nz, ny, nx)
        mag_p = np.linalg.norm(p, axis=0)
        nz = f.shape[0]
        for z in range(nz):
            fz = f[z]
            if not fz.any():
                continue
            b = min(int(z / max(nz - 1, 1) * self.nb), self.nb - 1)
            e = (mag_p[z][fz] - mag_t[z][fz]) ** 2
            self.h_sse[b] += e.sum()
            self.h_cnt[b] += fz.sum()
        # reservoir sample for scatter/hist (cap memory)
        mt, mp = mag_t[f], mag_p[f]
        idx = np.random.default_rng(0).choice(
            mt.size, size=min(mt.size, 4000), replace=False
        )
        self.mag_true.append(mt[idx])
        self.mag_pred.append(mp[idx])
        self.err.append((mp - mt)[idx])

    def summary(self) -> dict:
        rmse = np.sqrt(self.sse / np.maximum(self.count, 1))
        mae = self.sae / np.maximum(self.count, 1)
        rel_l2 = np.sqrt(self.sse / np.maximum(self.s_tnorm2, 1e-12))
        var = (
            self.s_true2 / np.maximum(self.count, 1)
            - (self.s_true / np.maximum(self.count, 1)) ** 2
        )
        ss_tot = var * self.count
        r2 = 1.0 - self.sse / np.maximum(ss_tot, 1e-12)
        overall_rmse = float(np.sqrt(self.sse.sum() / max(self.count.sum(), 1)))
        overall_rel_l2 = float(
            np.sqrt(self.sse.sum() / max(self.s_tnorm2.sum(), 1e-12))
        )
        mag_true = np.concatenate(self.mag_true) if self.mag_true else np.zeros(1)
        data_range = float(mag_true.max() - mag_true.min()) or 1.0
        psnr = float(20 * np.log10(data_range / max(overall_rmse, 1e-12)))
        return {
            "overall_rmse": overall_rmse,
            "overall_rel_l2": overall_rel_l2,
            "psnr_db": psnr,
            "per_channel_rmse": rmse.tolist(),
            "per_channel_mae": mae.tolist(),
            "per_channel_rel_l2": rel_l2.tolist(),
            "per_channel_r2": r2.tolist(),
            "mean_kl": float(np.mean(self.kl)) if self.kl else float("nan"),
        }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def _blank_obstacles(mag: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    out = mag.copy()
    out[~fluid] = np.nan
    return out


def _plot_reconstruction(truth, recon, fluid, z_levels, title, path) -> None:
    """rows truth/recon/|err| of |u|; one column per z-height."""
    mag_t = np.linalg.norm(truth.numpy(), axis=0)  # (nz, ny, nx)
    mag_p = np.linalg.norm(recon.numpy(), axis=0)
    f = fluid.numpy().astype(bool)
    vmax = float(np.nanmax(_blank_obstacles(mag_t, f))) or 1.0
    err = np.abs(mag_p - mag_t)
    emax = float(np.nanmax(_blank_obstacles(err, f))) or 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    ecmap = plt.get_cmap("magma").copy()
    ecmap.set_bad("lightgrey")

    n = len(z_levels)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)
    for j, z in enumerate(z_levels):
        fz = f[z]
        for r, (arr, cm, vm) in enumerate(
            [(mag_t[z], cmap, vmax), (mag_p[z], cmap, vmax), (err[z], ecmap, emax)]
        ):
            im = axes[r, j].imshow(
                _blank_obstacles(arr, fz), origin="lower", vmin=0, vmax=vm, cmap=cm
            )
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if r == 0:
                axes[r, j].set_title(f"z={z}")
            if j == n - 1:
                fig.colorbar(im, ax=axes[r, j], fraction=0.046)
    for r, lab in enumerate(["truth |u|", "recon |u|", "|err|"]):
        axes[r, 0].set_ylabel(lab, fontsize=12)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_per_channel(summary, channel_names, path) -> None:
    fig, ax = plt.subplots(figsize=(1.6 * len(channel_names) + 3, 4))
    x = np.arange(len(channel_names))
    w = 0.27
    ax.bar(x - w, summary["per_channel_rmse"], w, label="RMSE")
    ax.bar(x, summary["per_channel_mae"], w, label="MAE")
    ax.bar(x + w, summary["per_channel_rel_l2"], w, label="rel-L2")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names)
    ax.set_ylabel("error")
    ax.set_title("Per-channel reconstruction error (fluid cells)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_height_profile(acc: _Accum, path) -> None:
    rmse = np.sqrt(acc.h_sse / np.maximum(acc.h_cnt, 1))
    centres = (np.arange(acc.nb) + 0.5) / acc.nb
    valid = acc.h_cnt > 0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rmse[valid], centres[valid], "-o", ms=3)
    ax.set_xlabel("|u| reconstruction RMSE")
    ax.set_ylabel("normalised height  z / nz")
    ax.set_title("RMSE vs height")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_error_hist(acc: _Accum, path) -> None:
    err = np.concatenate(acc.err) if acc.err else np.zeros(1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err, bins=80, color="steelblue")
    ax.set_xlabel("recon − truth  (|u|, fluid cells)")
    ax.set_ylabel("count")
    ax.set_title(f"Error distribution  (mean={err.mean():.3g}, std={err.std():.3g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pred_vs_true(acc: _Accum, path) -> None:
    t = np.concatenate(acc.mag_true) if acc.mag_true else np.zeros(1)
    p = np.concatenate(acc.mag_pred) if acc.mag_pred else np.zeros(1)
    lim = float(max(t.max(), p.max())) or 1.0
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(
        t, p, gridsize=50, cmap="viridis", bins="log", extent=(0, lim, 0, lim)
    )
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="identity")
    ax.set_xlabel("true |u|")
    ax.set_ylabel("reconstructed |u|")
    ax.set_title("Reconstructed vs true (fluid cells)")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_latent_stats(kl_per_sample, latent_active, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(kl_per_sample, bins=30, color="darkorange")
    axes[0].set_xlabel("KL element (mean per snapshot)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Posterior KL")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(["active latent\nchannels (frac)"], [latent_active], color="teal")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Latent activation")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _pick(n_available: int, n_want: int) -> list[int]:
    n = min(n_want, n_available)
    if n <= 0:
        return []
    return np.linspace(0, n_available - 1, n, dtype=int).tolist()


def run(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    model_dir = Path(cfg.model_dir)
    train_cfg = OmegaConf.load(model_dir / "config.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = instantiate(
        train_cfg.dataset,
        split=cfg.split,
        dtype=getattr(torch, train_cfg.dataset.dtype),
    )
    model, dtype = _load_model(cfg, train_cfg, device)
    channel_names = list(train_cfg.dataset.state_vars)
    print(
        f"loaded {type(model).__name__} (size={getattr(model, 'size', '?')}, "
        f"latent_type={model.ae.latent_type}) on {device}; "
        f"split '{cfg.split}' has {len(dataset._state_files)} trajectories, "
        f"{len(dataset)} snapshots"
    )
    metrics_path = model_dir / "metrics.csv"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        print(f"training: {len(lines) - 1} epochs logged; last: {lines[-1]}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Reconstruction figures across trajectories / snapshots ---
    traj_ids = _pick(len(dataset._state_files), int(cfg.num_trajectories))
    for traj in traj_ids:
        n_t = dataset._traj_lengths[traj]
        t_ids = _pick(n_t, int(cfg.snapshots_per_traj))
        state, geometry, features = _load_snapshots(dataset, traj, t_ids, dtype)
        recon = _reconstruct(model, state, geometry, features, device)
        nz = geometry.shape[0]
        z_levels = _pick(nz, int(cfg.num_heights))
        for s, t in enumerate(t_ids):
            _plot_reconstruction(
                state[s],
                recon[s],
                geometry.bool(),
                z_levels,
                f"trajectory {traj}, snapshot t={t}  (|u|)",
                out_dir / f"reconstruction_traj{traj}_snap{t}.png",
            )
    print(f"wrote {len(traj_ids)} reconstruction figure group(s)")

    # --- Aggregate metrics over a sample of snapshots ---
    all_pairs = list(dataset.sample_index)  # (traj, t)
    rng = np.random.default_rng(int(cfg.seed))
    if cfg.get("num_metric_samples") is not None:
        k = min(int(cfg.num_metric_samples), len(all_pairs))
        sel = [all_pairs[i] for i in rng.choice(len(all_pairs), k, replace=False)]
    else:
        sel = all_pairs
    # group by trajectory so each grid is reconstructed in batches
    by_traj: dict[int, list[int]] = {}
    for traj, t in sel:
        by_traj.setdefault(traj, []).append(t)

    acc = _Accum(len(channel_names))
    latent_stds: list[np.ndarray] = []
    bs = int(cfg.batch_size)
    for traj, ts in by_traj.items():
        geometry = dataset.geometry_for(traj)
        features = dataset.geom_features_for(traj)
        fluid = geometry.bool()
        for start in range(0, len(ts), bs):
            batch_t = ts[start : start + bs]
            state, _, _ = _load_snapshots(dataset, traj, batch_t, dtype)
            recon = _reconstruct(model, state, geometry, features, device)
            # KL per snapshot (mean of the per-crop element)
            b = state.shape[0]
            geom = geometry.unsqueeze(0).expand(b, *geometry.shape).to(device)
            feat = (
                features.unsqueeze(0).expand(b, *features.shape).to(device)
                if features is not None
                else None
            )
            with torch.no_grad():
                _, kl = model(state.to(device), geom, feat, return_kl_element=True)
                lat = model.encode(state.to(device), geom, feat, latent_type="mode")
            per_snap_kl = kl.reshape(b, -1).mean(dim=1).cpu().numpy()
            acc.kl.extend(per_snap_kl.tolist())
            # latent activation: std per latent channel across crops+space
            latC = lat.shape[1]
            latent_stds.append(
                lat.reshape(-1, latC).std(dim=0).cpu().numpy()
                if lat.shape[0] > 1
                else np.abs(lat.reshape(-1, latC).mean(dim=0).cpu().numpy())
            )
            for s in range(b):
                acc.add(state[s], recon[s], fluid)

    summary = acc.summary()
    latent_active = float("nan")
    if latent_stds:
        mean_std = np.mean(np.stack(latent_stds, 0), axis=0)
        latent_active = float(np.mean(mean_std > 1e-2))
    summary["latent_active_fraction"] = latent_active
    summary["n_metric_snapshots"] = len(acc.kl)

    # --- Plots ---
    _plot_per_channel(summary, channel_names, out_dir / "per_channel_metrics.png")
    _plot_height_profile(acc, out_dir / "height_profile.png")
    _plot_error_hist(acc, out_dir / "error_hist.png")
    _plot_pred_vs_true(acc, out_dir / "pred_vs_true.png")
    _plot_latent_stats(acc.kl, latent_active, out_dir / "latent_stats.png")

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Headline metrics ---
    print("\n================ autoencoder reconstruction metrics ================")
    print(f"snapshots evaluated : {len(acc.kl)}")
    print(f"overall RMSE        : {summary['overall_rmse']:.6f}")
    print(f"overall rel-L2      : {summary['overall_rel_l2'] * 100:.3f} %")
    print(f"PSNR                : {summary['psnr_db']:.2f} dB")
    print(f"mean posterior KL   : {summary['mean_kl']:.4g}")
    print(f"latent active frac  : {latent_active:.3f}")
    print("per-channel:")
    for i, name in enumerate(channel_names):
        print(
            f"  {name:>4}: RMSE={summary['per_channel_rmse'][i]:.5f}  "
            f"MAE={summary['per_channel_mae'][i]:.5f}  "
            f"relL2={summary['per_channel_rel_l2'][i] * 100:.2f}%  "
            f"R2={summary['per_channel_r2'][i]:.4f}"
        )
    print("====================================================================")
    print(f"plots + metrics.json written to {out_dir}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/testing_autoencoder",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
