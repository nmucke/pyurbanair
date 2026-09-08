"""History-conditioned inference: forward model, ensemble and ESMDA seeding.

Covers what changes on the *inference* side when the trained network consumes
``num_history_steps = H > 1`` past frames instead of a single snapshot:

* the rollout carries an ``(B, H * C, ...)`` ring buffer — one frame is dropped
  and the prediction appended per step, and the emitted frame is the prediction
  (not the buffer),
* the buffer is seeded with the last ``H`` frames of the supplied state, or by
  repeating the only available frame (with a one-time warning) when the caller
  hands over a single snapshot, and
* :func:`write_initial_state_files` writes ``H`` frames per member so an ESMDA
  warm start seeds the buffer with real history.

``H == 1`` must stay exactly what it was; that is guarded by
``tests/test_neural_surrogate_forward_model.py`` running unchanged, plus the
byte-identity check below.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates import NeuralSurrogateEnsembleForwardModel
from neural_surrogates import forward_model as fm_mod
from neural_surrogates.training_spinup import (
    list_split_samples,
    write_initial_state_files,
)

from tests.test_neural_surrogate_forward_model import (
    NX,
    NY,
    NZ,
    PARAM_VARS,
    STATE_VARS,
    _ensemble_stub_factory,
    _make_model,
    _params,
    _patch_backend,
    _regular_snapshot,
    _StubSpinup,
)
from tests.test_training_spinup import _write_training_data

C = len(STATE_VARS)


class _RecordingModel(torch.nn.Module):
    """Stub network that records its inputs and adds 1 to the newest frame.

    Stands in for the real architectures so these tests exercise the forward
    model's buffer bookkeeping (not the networks'): it advertises the
    ``num_history_steps`` / ``n_state_channels`` contract, asserts it is fed
    ``H * C`` channels, and returns a deterministic ``C``-channel prediction.
    """

    def __init__(self, n_state_channels: int = C, num_history_steps: int = 1) -> None:
        super().__init__()
        self.n_state_channels = n_state_channels
        self.num_history_steps = num_history_steps
        self.n_input_state_channels = num_history_steps * n_state_channels
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, state, params, geometry, *extra):  # noqa: D102
        assert state.shape[1] == self.n_input_state_channels, (
            f"expected {self.n_input_state_channels} input channels, "
            f"got {state.shape[1]}"
        )
        self.inputs.append(state.detach().clone())
        pred = state[:, -self.n_state_channels :] + 1.0
        self.outputs.append(pred.detach().clone())
        return pred


def _history_model(num_history_steps: int, **overrides):
    """Forward model wrapping a fresh :class:`_RecordingModel`."""
    net = _RecordingModel(num_history_steps=num_history_steps)
    return _make_model(architecture=net, **overrides)


def _multi_frame_state(n_frames: int) -> xr.Dataset:
    """Regular-grid trajectory whose frame ``t`` is filled with ``t``-derived values.

    Channel ``c`` of frame ``t`` is the constant ``100 * t + c``, so a test can
    read straight off a stacked tensor which frames (and in which order) seeded
    the history buffer.
    """
    values = {
        v: np.stack(
            [np.full((NZ, NY, NX), 100.0 * t + c) for t in range(n_frames)], axis=0
        )
        for c, v in enumerate(STATE_VARS)
    }
    return xr.Dataset(
        {v: (("time", "z", "y", "x"), arr) for v, arr in values.items()},
        coords={
            "time": np.arange(n_frames, dtype=float),
            "z": np.arange(NZ) + 0.5,
            "y": np.arange(NY) + 0.5,
            "x": np.arange(NX) + 0.5,
        },
    )


# -- defaults: H = 1 is unchanged -------------------------------------------


def test_default_num_history_steps_is_one() -> None:
    """A model whose network advertises no history is a one-step surrogate."""
    model = _make_model()
    assert model.num_history_steps == 1
    assert model.n_state_channels == len(STATE_VARS)


def test_stack_history_at_h1_equals_stack_state() -> None:
    """H == 1 seeding is byte-identical to the historic single-snapshot stack."""
    model = _make_model()
    snapshot = _regular_snapshot(0)
    assert torch.equal(model._stack_history(snapshot), model._stack_state(snapshot))


# -- seeding policy ----------------------------------------------------------


def test_cold_start_repeats_single_frame_and_warns_once(monkeypatch) -> None:
    """A single spin-up frame fills the H-buffer by repetition, warning once."""
    monkeypatch.setattr(fm_mod, "_REPEAT_SEEDING_WARNED", False)
    model = _history_model(2, spinup_forward_model=_StubSpinup())

    with pytest.warns(RuntimeWarning, match="num_history_steps=2"):
        out = model(params=_params())

    # Same frame count as the H = 1 rollout: seeding does not change the
    # emit schedule (simulation_time / output_frequency = 3 predicted frames).
    assert out.sizes["time"] == 3
    # The buffer was seeded with the one spin-up frame, twice.
    first = model.model.inputs[0]
    assert first.shape[1] == 2 * C
    torch.testing.assert_close(first[:, :C], first[:, C:])

    # Warn ONCE per process, not per rollout / member / window.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(params=_params())
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_warm_start_uses_last_h_frames(monkeypatch) -> None:
    """A 3-frame warm start seeds the H = 2 buffer with frames 1 and 2."""
    monkeypatch.setattr(fm_mod, "_REPEAT_SEEDING_WARNED", False)
    model = _history_model(2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model(state=_multi_frame_state(3), params=_params())

    first = model.model.inputs[0]
    assert first.shape[1] == 2 * C
    # Oldest first: [t=1 u,v,w, t=2 u,v,w]; frame t channel c holds 100t + c.
    expected = torch.tensor(
        [100.0 * t + c for t in (1, 2) for c in range(C)],
        dtype=first.dtype,
    )
    got = first[0, :, 0, 0, 0]
    torch.testing.assert_close(got, expected)


def test_warm_start_shorter_than_history_pads_with_oldest(monkeypatch) -> None:
    """Two frames for an H = 3 network: the oldest frame is repeated."""
    monkeypatch.setattr(fm_mod, "_REPEAT_SEEDING_WARNED", False)
    model = _history_model(3)

    with pytest.warns(RuntimeWarning, match="num_history_steps=3"):
        model(state=_multi_frame_state(2), params=_params())

    first = model.model.inputs[0]
    assert first.shape[1] == 3 * C
    expected = torch.tensor(
        [100.0 * t + c for t in (0, 0, 1) for c in range(C)],
        dtype=first.dtype,
    )
    torch.testing.assert_close(first[0, :, 0, 0, 0], expected)


# -- the ring buffer ---------------------------------------------------------


def test_rollout_chunk_shifts_history_and_emits_prediction() -> None:
    """Step k's input is ``cat(step k-1's input[:, C:], step k-1's prediction)``."""
    model = _history_model(2)
    out = model(state=_multi_frame_state(4), params=_params())

    net = model.model
    n_internal, _ = model._output_schedule()
    assert len(net.inputs) == n_internal >= 2
    for k in range(1, n_internal):
        expected = torch.cat([net.inputs[k - 1][:, C:], net.outputs[k - 1]], dim=1)
        torch.testing.assert_close(net.inputs[k], expected)

    # The emitted frames are the PREDICTIONS, not the (wider) buffer.
    assert out.sizes["time"] == len(net.outputs)
    for j, var in enumerate(STATE_VARS):
        np.testing.assert_allclose(
            out[var].isel(time=-1).values,
            net.outputs[-1][0, j].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_history_rollout_emit_schedule_is_unchanged() -> None:
    """Substepping is orthogonal to history: same internal/emitted step counts."""
    plain = _make_model(output_frequency=1.0, trained_output_frequency=0.5)
    history = _history_model(3, output_frequency=1.0, trained_output_frequency=0.5)
    assert history._output_schedule() == plain._output_schedule()

    out = history(state=_multi_frame_state(5), params=_params())
    assert out.sizes["time"] == 3
    assert len(history.model.inputs) == 6


# -- ensemble ----------------------------------------------------------------


def test_ensemble_history_warm_start(tmp_path, monkeypatch) -> None:
    """An H = 2 ensemble warm-starts every member from its last two frames."""
    monkeypatch.setattr(fm_mod, "_REPEAT_SEEDING_WARNED", False)
    make_stub = _ensemble_stub_factory(tmp_path)
    _patch_backend(monkeypatch, make_stub)

    template = _history_model(2, spinup_forward_model=make_stub())
    ensemble = NeuralSurrogateEnsembleForwardModel(template, ensemble_size=3)

    warm = xr.concat(
        [_multi_frame_state(3) for _ in range(3)], dim="ensemble", join="override"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = ensemble.run_ensemble(state=warm, params=_params())

    assert out.sizes["ensemble"] == 3
    assert out.sizes["time"] == 3
    for v in STATE_VARS:
        assert out[v].dims == ("ensemble", "time", "z", "y", "x")
    # One batched pass over the 3 members, each fed H * C channels.
    assert template.model.inputs[0].shape[:2] == (3, 2 * C)


def test_ensemble_history_cold_start_repeats_spinup_frame(
    tmp_path, monkeypatch
) -> None:
    """A cold start still runs: the single spin-up frame fills the buffer."""
    monkeypatch.setattr(fm_mod, "_REPEAT_SEEDING_WARNED", False)
    make_stub = _ensemble_stub_factory(tmp_path)
    _patch_backend(monkeypatch, make_stub)

    template = _history_model(2, spinup_forward_model=make_stub())
    ensemble = NeuralSurrogateEnsembleForwardModel(template, ensemble_size=2)

    with pytest.warns(RuntimeWarning, match="num_history_steps=2"):
        out = ensemble.run_ensemble(params=_params())

    assert out.sizes["ensemble"] == 2
    assert out.sizes["time"] == 3
    for member in ensemble.ensemble_forward_models:
        assert member.spinup_forward_model.calls == 1


# -- trained-config plumbing -------------------------------------------------


def test_num_history_steps_read_from_model_dir(
    tmp_path, surrogate_model_dir_factory
) -> None:
    """H is read off the built network, which the saved config configures.

    The only test here that uses a *real* architecture (``UNetConvNeXt`` with
    ``num_history_steps=2``) rather than the recording stub, so the config ->
    instantiate -> ``self.num_history_steps`` plumbing is checked end to end.
    """
    model_dir = surrogate_model_dir_factory(
        tmp_path,
        domain={
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "bounds": [[0.0, NX], [0.0, NY], [0.0, NZ]],
        },
        time={"simulation_time": 3.0, "output_frequency": 1.0, "spinup_time": 0.0},
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        num_history_steps=2,
    )
    model = fm_mod.NeuralSurrogateForwardModel(
        spinup_forward_model=_StubSpinup(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
        simulation_time=3.0,
        output_frequency=1.0,
        model_dir=model_dir,
    )
    assert model.num_history_steps == 2
    assert model.n_state_channels == len(STATE_VARS)

    out = model(state=_multi_frame_state(3), params=_params())
    assert out.sizes["time"] == 3


# -- ESMDA seeding -----------------------------------------------------------


def test_write_initial_state_files_writes_history_window(
    tmp_path: pathlib.Path,
) -> None:
    """H = 3 writes the last three frames per member, cycling samples as before."""
    root = _write_training_data(tmp_path / "td", n_samples=2, t_len=5)
    state_files, _ = list_split_samples(root, "train")
    out = write_initial_state_files(
        state_files, n_members=3, out_dir=tmp_path / "ic", num_history_steps=3
    )

    for i, expected_sample in enumerate([0, 1, 0]):
        with xr.open_dataset(out / f"state_{i}.nc") as ds:
            assert ds.sizes["time"] == 3
            # Sample s frame t holds s + t; the last three frames are t = 2,3,4.
            np.testing.assert_allclose(
                ds["u"].max(dim=("zt", "yt", "xt")).values,
                [expected_sample + 2.0, expected_sample + 3.0, expected_sample + 4.0],
            )


def test_write_initial_state_files_rejects_short_trajectory(
    tmp_path: pathlib.Path,
) -> None:
    root = _write_training_data(tmp_path / "td", n_samples=1, t_len=2)
    state_files, _ = list_split_samples(root, "train")
    with pytest.raises(ValueError, match="too few to seed num_history_steps=4"):
        write_initial_state_files(
            state_files, n_members=1, out_dir=tmp_path / "ic", num_history_steps=4
        )
