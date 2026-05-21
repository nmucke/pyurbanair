"""Neural-network architectures behind the ``SurrogateArchitecture`` interface.

Importing this subpackage pulls in JAX/Equinox, so it must only be imported
from the framework's compute paths (forward model, training), never from the
top-level ``neural_surrogates`` package.
"""
