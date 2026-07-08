"""Parameter-efficient fine-tuning (LoRA / PEFT) for neural surrogates.

See ``docs/neural_surrogate_plans/01_lora_finetuning.md``. Public surface:

* :func:`inject_lora` -- wrap an architecture with LoRA adapters (PEFT).
* :func:`merge_to_state_dict` -- fold LoRA into a plain, ESMDA-loadable state dict.
* :func:`save_adapter` / :func:`load_adapter` -- PEFT adapter checkpoint I/O.
* :func:`resolve_target_modules` + :data:`P3D_PRESETS` -- per-architecture
  ``target_modules`` presets and their resolution.

``peft`` is a lazy import inside the submodules, so importing this package does
not pull in the transformers/accelerate stack.
"""

from __future__ import annotations

from neural_surrogates.finetuning.inject import (
    inject_lora,
    load_adapter,
    merge_to_state_dict,
    save_adapter,
)
from neural_surrogates.finetuning.targets import (
    P3D_PRESETS,
    all_adaptable_module_names,
    resolve_target_modules,
)

__all__ = [
    "inject_lora",
    "merge_to_state_dict",
    "save_adapter",
    "load_adapter",
    "resolve_target_modules",
    "all_adaptable_module_names",
    "P3D_PRESETS",
]
