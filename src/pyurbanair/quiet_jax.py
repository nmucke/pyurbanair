"""Silence JAX's CUDA-plugin discovery noise on CPU-fallback machines.

When this ``cuda`` pixi environment is used on a box whose installed
``jax_cuda12_plugin`` / cuDNN don't match the bundled ``jaxlib``, JAX is
harmless but loud: it emits a ``RuntimeWarning`` at import time and a
multi-line ``ERROR`` traceback plus ``INFO``/``WARNING`` lines at first
op while it falls back to CPU. None of this is actionable — the run
proceeds on CPU regardless — so we quiet it to keep the terminal (and
SLURM ``.err`` files) readable.

This must take effect *before* ``import jax``; import it as the very
first import in entry-point scripts::

    import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)
    import jax

On a machine where CUDA is healthy none of these messages fire, so the
suppression is a no-op there.
"""

import logging
import warnings

# Fires during ``import jax`` (jaxlib.plugin_support warns when the
# installed CUDA plugin version doesn't match jaxlib).
warnings.filterwarnings(
    "ignore",
    message=r"JAX plugin .* is installed, but it is not compatible.*",
    category=RuntimeWarning,
)

# Fires at first JAX op during backend initialization: the cuDNN-not-found
# plugin traceback (ERROR), the libtpu probe (INFO), and the "falling back
# to cpu" notice (WARNING). Raising the threshold above ERROR drops all of
# them; genuine fatal JAX problems raise exceptions rather than log here.
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
