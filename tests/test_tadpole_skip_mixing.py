"""State alignment, immediate learning, and DFT spatial/export contracts."""

# mypy: disallow-untyped-decorators=false

from collections.abc import Iterator
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates.architectures.tadpole_skip_mixing import TadpoleSkipMixing
from neural_surrogates.architectures.tadpole_stepper import TadpoleTimeStepper


@pytest.fixture(autouse=True)
def _threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def _model(mode: str, **kwargs: Any) -> TadpoleTimeStepper:
    model: TadpoleTimeStepper = TadpoleTimeStepper(
        n_state_channels=2,
        n_params=0,
        spatial_mode=mode,
        encoder_crop_size=16,
        subnetwork=None,
        **kwargs,
    ).eval()
    # A pretrained decoder must respond to its inputs, unlike its zero-init head.
    torch.nn.init.normal_(
        model.dft.decoder.transformer_decoder.transformer.final_layer.out_proj.weight,
        std=0.03,
    )
    return model


@pytest.mark.parametrize("mode", ["local", "global", "halo"])
@pytest.mark.parametrize("branch", [False, True])
def test_zero_init_gradients_cross_state_influence_and_reload(
    mode: str, branch: bool
) -> None:
    options = dict(
        encode_geometry=not branch,
        geometry_branch={"width": 4} if branch else None,
        skip_mixing={"width": 8, "levels": [1, 2, 4, 8]},
    )
    model = _model(mode, **options)
    assert model.skip_mixing is not None
    state = torch.randn(2, 2, 17, 16, 16)
    geom = torch.ones(2, 17, 16, 16)
    geom[:, 5:8, 4:7, 3:6] = 0
    with torch.no_grad():
        initial = model(state, None, geom)
        torch.testing.assert_close(
            initial, model._ae_reference_recon(state, geom), rtol=0, atol=0
        )
        baseline = _model(
            mode, encode_geometry=not branch, geometry_branch=options["geometry_branch"]
        )
        baseline.load_state_dict(
            {
                k: v
                for k, v in model.state_dict().items()
                if not k.startswith("skip_mixing.")
            },
            strict=True,
        )
        torch.testing.assert_close(
            initial, baseline(state, None, geom), atol=5e-5, rtol=2e-4
        )

    # Freeze the existing DFT: even with gamma=0 every selected mixer can learn.
    for parameter in model.dft.parameters():
        parameter.requires_grad_(False)
    loss = (model(state, None, geom) - torch.randn_like(state)).square().mean()
    loss.backward()
    for adapter in model.skip_mixing.adapters.values():
        grad = adapter.output_proj.weight.grad
        assert grad is not None and grad.abs().sum() > 0
        assert torch.isfinite(grad).all()
    optimizer = torch.optim.SGD(model.skip_mixing.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    # The first projection starts learning once the final projection opens.
    model(state, None, geom).square().mean().backward()
    assert all(
        a.input_proj.weight.grad.abs().sum() > 0
        for a in model.skip_mixing.adapters.values()
    )

    changed = state.clone()
    changed[0, 1] += torch.randn_like(changed[0, 1])
    with torch.no_grad():
        result = model(state, None, geom)
        perturbed = model(changed, None, geom)
        assert (result[0, 0] - perturbed[0, 0]).abs().max() > 0
        torch.testing.assert_close(result[1], perturbed[1], rtol=0, atol=0)
        assert torch.count_nonzero(result * (1 - geom.unsqueeze(1))) == 0
        model.dft.max_internal_batchsize = 1
        torch.testing.assert_close(
            model(state, None, geom), result, atol=5e-5, rtol=2e-4
        )
        latent, skips = model.encode(state, geom)
        feats = model._geom_branch_kwargs(state, geom, None).get("geom_feats")
        decoded = model.decode(latent, skips, feats).reshape(2, -1, 32, 16, 16)
        torch.testing.assert_close(
            decoded[:, :2, :17] * geom.unsqueeze(1), result, atol=5e-5, rtol=2e-4
        )
        rebuilt = _model(mode, **options)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(
            rebuilt(state, None, geom), result, atol=5e-5, rtol=2e-4
        )


def test_mixer_excludes_geometry_and_preserves_batch_alignment() -> None:
    mixing = TadpoleSkipMixing(2, 3, [4, 4, 4, 4], width=8)
    skips = [[torch.randn(6, 4, 2, 2, 2) for _ in range(2)] for _ in range(2)]
    for adapter in mixing.adapters.values():
        torch.nn.init.normal_(adapter.output_proj.weight)
    result = mixing(skips)
    altered = [[t.clone() for t in group] for group in skips]
    for group in altered:
        for t in group:
            t[2::3] += 100  # Geometry in both samples.
            t[3:5] += torch.randn_like(t[3:5])  # States in sample 1 only.
    changed = mixing(altered)
    assert result[0] == [None, None]
    for before, after in zip(result[1], changed[1]):
        assert before is not None and after is not None
        assert torch.count_nonzero(before[2::3]) == 0
        torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=0)
        assert (before[3:5] - after[3:5]).abs().max() > 0


@pytest.mark.parametrize(
    "options",
    [
        {"width": 0},
        {"width": True},
        {"levels": []},
        {"levels": [3]},
        {"levels": [4, 4]},
        {"levels": [True]},
    ],
)
def test_invalid_mixer_settings(options: dict) -> None:
    with pytest.raises(ValueError, match="skip_mixing"):
        TadpoleSkipMixing(2, 3, [4, 4, 4, 4], **options)
