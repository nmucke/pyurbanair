"""Regression tests for review finding H2: the vendored naive-attention path
must honour the configured ``num_heads``.

Upstream ``LLMLayer`` called ``AttentionBlock(dim, dim, num_heads, ...)`` while
``AttentionBlock``'s signature is ``AttentionBlock(dim, num_heads=8,
qkv_bias=False, ...)``. The second positional ``dim`` therefore bound to
``num_heads`` (giving ``head_dim = 1``) and the intended ``num_heads`` bound to
``qkv_bias`` (an ``int`` where a ``bool`` was expected). The local fix binds by
keyword; these tests assert the attention blocks report the *configured* head
count and that ``qkv_bias`` is a genuine boolean.

Gated with ``importorskip`` on the vendored Tadpole runtime deps so the module
is skipped (not errored) where they are absent. CPU-only and tiny.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from neural_surrogates.architectures._tadpole.architecture.downstream.llm import (  # noqa: E402
    AttentionBlock,
    LLMLayer,
    SequentialModel,
)


def _attn_blocks(module):
    return [m for m in module.modules() if isinstance(m, AttentionBlock)]


def test_llmlayer_naive_uses_configured_num_heads():
    """The pre-fix bug bound ``dim`` to ``num_heads``; assert the configured
    count wins so ``head_dim`` is ``dim // num_heads`` (not 1)."""
    dim, num_heads = 144, 8
    layer = LLMLayer(dim, dim * 4, num_heads, attention_method="naive")

    assert isinstance(layer.attn, AttentionBlock)
    # Pre-fix this was 144 (== dim, i.e. 144 heads of dimension 1).
    assert layer.attn.num_heads == num_heads
    assert dim % layer.attn.num_heads == 0
    assert dim // layer.attn.num_heads == dim // num_heads  # head_dim == 18, not 1


def test_attentionblock_qkv_bias_is_bool_not_int():
    """Pre-fix, ``num_heads`` (an int) leaked into the ``qkv_bias`` slot, so the
    qkv Linear picked up a bias by accident. Default (no explicit bias) must be
    biasless."""
    layer = LLMLayer(144, 576, 8, attention_method="naive")
    # qkv_bias defaults to False -> the Linear has no bias parameter.
    assert layer.attn.qkv.bias is None


@pytest.mark.parametrize("num_heads", [1, 4, 8])
def test_sequentialmodel_propagates_num_heads(num_heads):
    """Mirror how ``ParamConditionedSubnetwork`` builds the subnetwork (see
    ``tadpole_stepper.ParamConditionedSubnetwork``): ``SequentialModel`` with
    ``attention_method="naive"``. Every attention block must report the
    configured head count."""
    hidden_size = 144
    model = SequentialModel(
        in_dim=8,
        n_layers=3,
        attention_method="naive",
        num_heads=num_heads,
        hidden_size=hidden_size,
        mlp_ratio=4,
        init_zero_proj=True,
    )
    blocks = _attn_blocks(model)
    assert len(blocks) == 3
    for blk in blocks:
        assert blk.num_heads == num_heads
        assert hidden_size % blk.num_heads == 0
        # qkv_bias defaults False; a leaked int would have created a bias tensor.
        assert blk.qkv.bias is None


def test_subnetwork_num_heads_matches_config_via_stepper():
    """End-to-end through the real construction path: build the stepper's
    subnetwork the way ``_SUBNET_SIZES`` drives it and assert the naive
    attention blocks honour the table's ``num_heads`` (8 for size 'S')."""
    pytest.importorskip("diffusers")
    pytest.importorskip("timm")
    from neural_surrogates.architectures.tadpole_stepper import (
        _SUBNET_SIZES,
        ParamConditionedSubnetwork,
    )

    cfg = _SUBNET_SIZES["S"]
    sub = ParamConditionedSubnetwork(
        in_dim=8,
        n_params=0,
        n_layers=cfg["n_layers"],
        num_heads=cfg["num_heads"],
        hidden_size=cfg["hidden_size"],
        param_conditioning="none",
    )
    blocks = _attn_blocks(sub.seqmodel)
    assert len(blocks) == cfg["n_layers"]
    for blk in blocks:
        assert blk.num_heads == cfg["num_heads"]
        assert blk.num_heads != cfg["hidden_size"]  # the pre-fix (head_dim=1) value
        assert blk.qkv.bias is None
