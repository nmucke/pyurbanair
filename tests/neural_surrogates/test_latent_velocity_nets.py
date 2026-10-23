"""Full-reach velocity nets of ``latent_velocity_adaln`` (unit level, CPU).

What is locked down here:

* **No-op when absent** -- ``AdaLNVelocityNet(head_hidden=None)`` has the
  unchanged state-dict keys and output of the plain net.
* **Zero at init** -- the head variant (Design A) and
  ``AdaLNBottleneckVelocityNet`` (Design B) output exactly ``0``, and their
  zero-initialised output layers receive gradients on the first backward.
* **Full reach** -- with ``H < D``, the outputs of one token over many random
  contexts (other tokens + conditioning) span all ``D`` directions for A and B,
  but at most ``H`` for the plain net.
* **Checkpointing** -- ``use_checkpoint`` leaves outputs and gradients unchanged.
* **Parameter counts** of the plan configs at ``D = 1536`` (printed, loosely
  bounded).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("timm")

from neural_surrogates.architectures.latent_velocity_adaln import (  # noqa: E402
    AdaLNBottleneckVelocityNet,
    AdaLNVelocityNet,
)

D, COND, GEOM, H, FILM = 48, 6, 3, 16, 8
GRID = (1, 2, 2)


def _plain(**kwargs):
    return AdaLNVelocityNet(
        D, COND, GEOM, n_layers=2, num_heads=2, hidden_size=H, mlp_ratio=2,
        film_hidden=FILM, **kwargs,
    )  # fmt: skip


def _head(**kwargs):
    return _plain(head_hidden=2 * D, head_layers=2, **kwargs)


def _bottleneck(**kwargs):
    return AdaLNBottleneckVelocityNet(
        D, COND, GEOM, n_layers=2, num_heads=2, hidden_size=H, mlp_hidden=2 * D,
        film_hidden=FILM, **kwargs,
    )  # fmt: skip


def _inputs(b: int = 2, dtype=torch.float32):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, D, *GRID, generator=g, dtype=dtype)
    cond = torch.randn(b, COND, generator=g, dtype=dtype)
    geom = torch.randn(b, GEOM, *GRID, generator=g, dtype=dtype)
    return x, cond, geom


def _randomize(net: torch.nn.Module) -> torch.nn.Module:
    """Give every parameter (incl. the zero-init ones) a random value."""
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return net


def test_head_absent_keeps_the_plain_net():
    torch.manual_seed(0)
    plain = _plain()
    torch.manual_seed(0)
    absent = _plain(head_hidden=None)
    assert absent.head is None
    assert list(absent.state_dict()) == list(plain.state_dict())
    absent.load_state_dict(plain.state_dict(), strict=True)
    _randomize(plain)
    _randomize(absent)
    x, cond, geom = _inputs()
    assert torch.equal(absent(x, cond, geom), plain(x, cond, geom))
    # the head only adds keys under ``head.``
    extra = set(_head().state_dict()) - set(plain.state_dict())
    assert extra and all(k.startswith("head.") for k in extra)


@pytest.mark.parametrize("build", [_head, _bottleneck], ids=["head", "bottleneck"])
def test_zero_at_init_and_output_layers_get_gradients(build):
    torch.manual_seed(0)
    net = build()
    x, cond, geom = _inputs()
    out = net(x, cond, geom)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0
    torch.nn.functional.mse_loss(out, torch.randn_like(out)).backward()
    assert torch.count_nonzero(net.out_proj.weight.grad) > 0
    if isinstance(net, AdaLNVelocityNet):
        assert net.head is not None
        assert torch.count_nonzero(net.head.out_proj.weight.grad) > 0


def _context_rank(net: torch.nn.Module, n: int = 3 * D) -> int:
    """Rank of token 0's output over ``n`` random contexts (others + cond)."""
    net = _randomize(net.double()).eval()
    x, cond, geom = _inputs(b=n, dtype=torch.float64)
    x[:, :, 0, 0, 0] = x[0, :, 0, 0, 0]  # token 0 itself fixed
    geom[:] = geom[0]
    with torch.no_grad():
        out = net(x, cond, geom)[:, :, 0, 0, 0]
    s = torch.linalg.svdvals(out - out.mean(dim=0))
    return int((s > 1e-8 * s[0]).sum())


def test_context_reaches_every_direction():
    assert _context_rank(_plain()) <= H < D
    assert _context_rank(_head()) == D
    assert _context_rank(_bottleneck()) == D


@pytest.mark.parametrize("build", [_head, _bottleneck], ids=["head", "bottleneck"])
def test_checkpointing_is_transparent(build):
    torch.manual_seed(0)
    ref = _randomize(build())
    ckpt = _randomize(build(use_checkpoint=True))
    x, cond, geom = _inputs()
    target = torch.randn_like(x)
    outs = []
    for net in (ref, ckpt):
        net.train()
        out = net(x, cond, geom)
        torch.nn.functional.mse_loss(out, target).backward()
        outs.append(out.detach())
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)
    for (name, p), q in zip(ref.named_parameters(), ckpt.parameters()):
        torch.testing.assert_close(p.grad, q.grad, msg=name)


def test_plan_config_parameter_counts():
    """Configs A and B of the plan at ``D = 1536`` (cond 12*2+64, geometry 256)."""
    d, cond, geom = 1536, 12 * 2 + 64, 256
    a = AdaLNVelocityNet(
        d, cond, geom, n_layers=6, num_heads=8, hidden_size=320, mlp_ratio=4,
        film_hidden=256, head_hidden=1536, head_layers=2,
    )  # fmt: skip
    b = AdaLNBottleneckVelocityNet(
        d, cond, geom, n_layers=4, num_heads=8, hidden_size=256, mlp_hidden=512,
        film_hidden=256,
    )  # fmt: skip
    counts = {
        name: sum(p.numel() for p in net.parameters())
        for name, net in [("A", a), ("B", b)]
    }
    assert a.head is not None
    counts["A_head"] = sum(p.numel() for p in a.head.parameters())
    print(f"parameter counts at D={d}: {counts}")
    for name in ("A", "B"):
        assert 10e6 < counts[name] < 50e6, (name, counts[name])
