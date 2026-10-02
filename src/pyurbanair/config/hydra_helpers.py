from __future__ import annotations

from typing import Any

from pylbm.utils.warm_start_utils import clean_output_files as clean_lbm_output_files
from pyudales.utils.clean_up_utils import clean_output_dir as clean_udales_output_dir


def _unwrap_forward_model(forward_model: Any) -> Any:
    return (
        forward_model.forward_model
        if hasattr(forward_model, "forward_model")
        else forward_model
    )


def prepare_compile(forward_model: Any, compile: bool) -> None:
    _unwrap_forward_model(forward_model).compile(compile=compile)


def prepare_udales(
    forward_model: Any,
    python_or_matlab: str = "python",
) -> None:
    _unwrap_forward_model(forward_model).run_preprocessing(
        python_or_matlab=python_or_matlab
    )


def prepare_neural_surrogate(
    forward_model: Any,
    spinup_backend: str,
    compile: bool = True,
    python_or_matlab: str = "python",
) -> None:
    """Prepare the surrogate's spin-up backend (compile / preprocess).

    The neural surrogate itself needs no preparation, but the CFD backend it
    uses to bootstrap cold starts does. ``spinup_backend`` selects which
    preparation to run on ``forward_model.spinup_forward_model``.

    When ``spinup_source == "training_data"`` the surrogate never runs a spin-up
    (the assimilation warm-starts every window from provided states), so the CFD
    backend is never invoked and there is nothing to prepare — skip the
    preprocessing/compile entirely. This keeps a
    training-data surrogate (e.g. a pypalm-trained net assimilated with a
    pyudales spin-up template) from running an unused uDALES preprocessing pass.
    The same holds for ``"generative"``: the cold start is sampled from the
    latent generator and the CFD backend is not even built, so there is nothing
    to prepare.
    """
    surrogate = _unwrap_forward_model(forward_model)
    if getattr(surrogate, "spinup_source", None) in ("training_data", "generative"):
        return
    spinup = surrogate.spinup_forward_model
    if spinup_backend == "pyudales":
        spinup.run_preprocessing(python_or_matlab=python_or_matlab)
    elif spinup_backend in ("pylbm", "pypalm"):
        spinup.compile(compile=compile)
    else:
        raise ValueError(
            f"prepare_neural_surrogate: unknown spinup_backend {spinup_backend!r}."
        )


def clean_outputs(model_name: str, forward_model: Any) -> None:
    model = _unwrap_forward_model(forward_model)
    if model_name == "pylbm":
        clean_lbm_output_files(model.dirs)
    elif model_name == "pypalm":
        from pypalm.utils.clean_up_utils import clean_palm_output_dir

        clean_palm_output_dir(model.dirs)
    elif model_name == "pyudales":
        clean_udales_output_dir(model.dirs)
    elif model_name == "neural_surrogate":
        # The surrogate keeps no solver output of its own; its spin-up
        # backend cleans up after each call via BaseForwardModel.__call__.
        return
    else:
        # Previously the else arm fell through to uDALES cleanup; raise instead
        # so an unrecognized backend can't silently get the wrong cleanup
        # (docs/codebase_guide.md §8).
        raise ValueError(f"clean_outputs: unknown model_name {model_name!r}.")


def resolve_parameter_schema(model_name: str) -> tuple[str, ...]:
    """Resolve the ordered parameter names a model consumes.

    Keyed off ``model_name``: ``pressure_gradient_magnitude`` is uDALES-only.
    ``vertical_inflow_exponent`` (power-law shear exponent α) and ``sgs_constant``
    (sub-grid-scale mixing constant) are model-error compensation knobs every
    backend can consume per-member; see docs/archive/esmda_model_error_parameters.md.
    """
    base = (
        "inflow_angle",
        "velocity_magnitude",
        "vertical_inflow_exponent",
        "sgs_constant",
    )
    if model_name == "pyudales":
        return base + (
            "pressure_gradient_magnitude",
            "sgs_bias_b0",
            "sgs_bias_b1",
            "sgs_bias_b2",
        )
    return base
