"""Vendored subset of the ``tadpole`` package (Apache-2.0).

Source: https://github.com/tum-pbs/Tadpole @ 4232e6982a0bf7a6765f24af29d98f42413e4dec
(Liu, Koehler, Holzschuh, Thuerey — "Tadpole: Autoencoders as Foundation Models
for 3D PDEs", arXiv:2605.15284).

Only the autoencoder path is vendored — ``model/autoencoder.py`` plus the
``architecture/p3d`` encoder/decoder it depends on, ``utils.load_weights`` and
the (optional) ``architecture/discriminator`` for the GAN extension. This mirrors
the ``_upt/`` vendoring already in this package.

Why vendored rather than a ``pip``/``git`` dependency (decided per plan 02 §1):
upstream's ``requirements.txt`` pulls in ``torchfsm`` (its *online data-generation*
dep, unused here), which in turn requires ``vape4d`` (a volume renderer) — an
unnecessary, resolution-risky chain for a path that never imports it. The files
here import only ``torch``/``einops``/``diffusers``/``timm``/``numpy`` (all already
present in the dev env) and are byte-for-byte upstream except for one edit in
``model/autoencoder.py``: the ``GIFt`` import (upstream's integer-rank LoRA
fine-tuning library, used only by the ``*_ft_state=<int>`` branch we never take —
we drive LoRA through HF PEFT) is made optional so the module imports without it.

Keeping the exact ``architecture/model/utils`` subtree layout means every internal
relative import resolves unchanged, and the encoder/decoder ``state_dict`` keys
stay identical to upstream — so the HF ``thuerey-group/Tadpole`` pretrained
weights still load via ``weight_encoder``/``weight_decoder``.
"""

from __future__ import annotations

from .model.autoencoder import TadpoleAutoencoder
from .model.dft import TadpoleDFT

__all__ = ["TadpoleAutoencoder", "TadpoleDFT"]
