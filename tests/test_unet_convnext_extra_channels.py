"""Back-compat + extra-channel tests for :class:`UNetConvNeXt`.

The ``extra_in_channels`` / ``extra`` widening must be a strict superset: with
the default ``extra_in_channels=0`` the state-dict keys and forward output are
byte-identical to a model built before the option existed; with
``extra_in_channels>0`` the ``extra`` tensor runs and a wrong channel count
raises.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import UNetConvNeXt

ARCH = dict(base_channels=8, channel_mults=[1, 2], depths=[1, 1])
GRID = (8, 8, 8)


def _inputs(b=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(b, 3, *GRID, generator=g)
    params = torch.randn(b, 2, generator=g)
    geometry = torch.ones(b, 1, *GRID)
    return state, params, geometry


def test_state_dict_keys_identical_at_zero_extra():
    torch.manual_seed(0)
    baseline = UNetConvNeXt(3, 2, **ARCH)
    torch.manual_seed(0)
    widened = UNetConvNeXt(3, 2, extra_in_channels=0, **ARCH)
    assert set(baseline.state_dict().keys()) == set(widened.state_dict().keys())
    # stem in-channels unchanged
    assert baseline.stem.weight.shape == widened.stem.weight.shape


def test_forward_identical_at_zero_extra():
    torch.manual_seed(0)
    baseline = UNetConvNeXt(3, 2, **ARCH).eval()
    torch.manual_seed(0)
    widened = UNetConvNeXt(3, 2, extra_in_channels=0, **ARCH).eval()
    state, params, geometry = _inputs()
    with torch.no_grad():
        a = baseline(state, params, geometry)
        b = widened(state, params, geometry, extra=None)
    assert torch.allclose(a, b, atol=0.0)


def test_extra_channels_run():
    extra_c = 4
    model = UNetConvNeXt(3, 2, extra_in_channels=extra_c, **ARCH).eval()
    state, params, geometry = _inputs()
    extra = torch.randn(1, extra_c, *GRID)
    out = model(state, params, geometry, extra=extra)
    assert out.shape == state.shape
    # stem widened by exactly extra_c
    assert model.stem.weight.shape[1] == 3 + 1 + extra_c


def test_wrong_extra_channel_count_raises():
    model = UNetConvNeXt(3, 2, extra_in_channels=4, **ARCH).eval()
    state, params, geometry = _inputs()
    with pytest.raises(ValueError):
        model(state, params, geometry, extra=torch.randn(1, 3, *GRID))


def test_extra_required_when_widened():
    model = UNetConvNeXt(3, 2, extra_in_channels=4, **ARCH).eval()
    state, params, geometry = _inputs()
    with pytest.raises(ValueError):
        model(state, params, geometry, extra=None)


def test_extra_forbidden_when_not_widened():
    model = UNetConvNeXt(3, 2, extra_in_channels=0, **ARCH).eval()
    state, params, geometry = _inputs()
    with pytest.raises(ValueError):
        model(state, params, geometry, extra=torch.randn(1, 1, *GRID))
