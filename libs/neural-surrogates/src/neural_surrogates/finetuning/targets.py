"""Per-architecture LoRA ``target_modules`` presets.

PEFT selects which leaf modules to adapt via ``LoraConfig.target_modules``,
matched against each ``named_modules()`` name with ``re.fullmatch`` (regex) or by
name-suffix (explicit list). This module holds the regex presets we ship per
architecture family plus the resolution logic the fine-tune script uses:

    explicit ``target_modules`` (config) > named ``preset`` > error.

The P3D presets below were derived by enumerating ``P3D(...).net.named_modules()``
(see ``docs/neural_surrogate_plans/01_lora_finetuning.md`` §0). The module tree
(``p3d_surrogate``'s ``P3D_S/B/L``) is fixed, so the suffixes are stable:

    net.model_impl.{encoder_level_N,decoder_level_N,latent}.blocks.N
        .attn.qkv                     # fused QKV projection (windowed attention)
        .attn.pos_emb_funct.cpb_mlp.* # continuous relative-position-bias MLP
        .mlp.fc1 / .mlp.fc2           # transformer block feed-forward
        .adain_2.linear               # per-block adaLN modulation
    net.model_impl.{encoder,decoder}.blocks.N.mlp_scale_bias  # outer adaLN
    net.model_impl.{down0_1,down1_2}.body.0                   # downsample Conv3d
    net.model_impl.reduce_chan_level{0,1}                     # 1x1 fuse Conv3d

Excluded from every preset: ``attn.pos_emb_funct.cpb_mlp.*`` (a tiny,
geometry-independent positional-bias net), the sinusoidal time/param embedders,
and the ``param_to_scalar`` head (it lives on the wrapper, not ``net`` -- train it
fully via ``modules_to_save`` instead, per the plan).
"""

from __future__ import annotations

from torch import nn

# Transformer-block Linears present in every P3D stage, plus the outer adaLN.
_P3D_ATTENTION_SUFFIX = r"attn\.qkv|mlp\.fc1|mlp\.fc2|adain_\d+\.linear|mlp_scale_bias"
# Downsampling Conv3d stems (3x3x3). The 1x1x1 ``reduce_chan_level*`` convs are
# deliberately excluded: peft (0.19) merges a 1x1x1 Conv3d LoRA with the Conv2d
# squeeze path and crashes in ``get_delta_weight`` (verified), so they cannot be
# safely folded into the merged ``weights.pt`` ESMDA loads. Target one explicitly
# via ``lora.target_modules`` only if you never call ``merge_to_state_dict``.
_P3D_CONV_SUFFIX = r"down\d+_\d+\.body\.\d+"

# preset name -> regex over full module names (``re.fullmatch``); ``None`` is the
# "adapt every Linear/Conv3d leaf" sentinel resolved by enumeration.
P3D_PRESETS: dict[str, str | None] = {
    "attention": rf".*({_P3D_ATTENTION_SUFFIX})",
    "attention+conv": rf".*({_P3D_ATTENTION_SUFFIX}|{_P3D_CONV_SUFFIX})",
    "all": None,
}

# Architecture family -> its preset table. Keyed by the base module's class name
# so the fine-tune script needs no import of the concrete architecture class.
_PRESETS_BY_FAMILY: dict[str, dict[str, str | None]] = {
    "P3D": P3D_PRESETS,
}

# Module types LoRA can adapt (mirrors what peft's lora tuner dispatches on for
# our architectures). Used to enumerate the "all" preset.
_ADAPTABLE_TYPES = (nn.Linear, nn.Conv3d)


def all_adaptable_module_names(
    model: nn.Module, *, exclude: tuple[str, ...] = ("param_to_scalar",)
) -> list[str]:
    """Every LoRA-adaptable ``nn.Linear`` / ``nn.Conv3d`` leaf, minus ``exclude``.

    Used for the ``all`` preset and as the generic fallback for architectures
    without a preset table. ``exclude`` drops modules by name suffix (default: the
    P3D ``param_to_scalar`` head, trained fully via ``modules_to_save``).

    Two conv classes are skipped so the ``all`` preset always **merges** cleanly
    (the merged ``weights.pt`` is the load-bearing ESMDA artifact):

    * **Grouped convs** (``groups != 1``) -- PEFT's ``lora.Conv3d`` requires the
      LoRA rank to be divisible by ``groups``, so a depthwise conv (e.g.
      ``UNetConvNeXt``'s ConvNeXt blocks) cannot be adapted at an arbitrary rank
      and raises mid-injection.
    * **1x1x1 convs** -- peft (0.19) merges a 1x1x1 Conv3d LoRA via the Conv2d
      squeeze path and crashes in ``get_delta_weight`` (verified on P3D's
      ``reduce_chan_level*``), so the adapter can never be folded back in.

    P3D's stems are all ``groups == 1`` with 3x3x3 kernels, so nothing is dropped
    there. Target such a conv explicitly (with a divisible rank, and never merging)
    via ``lora.target_modules`` if you really need it.
    """
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, _ADAPTABLE_TYPES):
            continue
        if isinstance(module, nn.Conv3d) and (
            module.groups != 1 or all(k == 1 for k in module.kernel_size)
        ):
            continue
        if any(name == e or name.endswith("." + e) for e in exclude):
            continue
        names.append(name)
    return names


def resolve_target_modules(
    model: nn.Module,
    *,
    preset: str | None = None,
    target_modules: str | list[str] | None = None,
    exclude: tuple[str, ...] = ("param_to_scalar",),
) -> str | list[str]:
    """Pick the LoRA ``target_modules`` for ``model``.

    Resolution order (matches the plan): an explicit ``target_modules`` (regex
    string or name list) always wins; otherwise the named ``preset`` is looked up
    in the architecture family's table; the ``all`` preset (and any preset on an
    architecture without a table) enumerates every adaptable leaf. Raises on an
    unknown named preset for an architecture that has a preset table.
    """
    if target_modules is not None:
        return target_modules
    if preset is None:
        raise ValueError(
            "resolve_target_modules needs either an explicit target_modules or a "
            "preset name."
        )

    family = type(model).__name__
    table = _PRESETS_BY_FAMILY.get(family)
    if table is None:
        # No preset table for this architecture: only the universal "all" preset
        # is meaningful; anything else must be given as an explicit regex.
        if preset == "all":
            return all_adaptable_module_names(model, exclude=exclude)
        raise ValueError(
            f"No LoRA target preset {preset!r} for architecture {family!r}. "
            "Pass an explicit lora.target_modules regex, or use preset=all."
        )

    if preset not in table:
        raise ValueError(
            f"Unknown target preset {preset!r} for {family!r}; "
            f"choose one of {sorted(table)} or pass an explicit target_modules."
        )
    regex = table[preset]
    if regex is None:  # "all" sentinel -> enumerate
        return all_adaptable_module_names(model, exclude=exclude)
    return regex
