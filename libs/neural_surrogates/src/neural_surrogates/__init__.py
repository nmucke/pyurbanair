"""Architecture-agnostic neural-surrogate framework for pyurbanair.

This top-level module is import-light **on purpose**: it exposes only the
package version and must NOT pull in JAX / Equinox / Optax / Orbax at import
time. The forward-model and training entry points live in submodules so that
composing a non-surrogate Hydra config never imports the neural-network stack
(the same lazy-import invariant ``pypalm`` relies on; see
``docs/codebase_guide.md`` §7 and ``docs/neural_surrogate_plan.md`` §9).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
