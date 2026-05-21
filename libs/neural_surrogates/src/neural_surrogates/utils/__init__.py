"""Framework I/O utilities (xarray <-> tensor, params, schema, registry).

These modules are pure-Python/NumPy/xarray and import-light: they do not pull
in Equinox/Optax/Orbax, so they are safe to import from test helpers and the
data-generation script without dragging in the training stack.
"""
