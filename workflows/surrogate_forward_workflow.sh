#!/usr/bin/env bash
# Run the neural surrogate on a geometry, then draw its figures and its HTML
# viewer: forward_workflow.sh with model=neural_surrogate_generative (a time
# stepper cold-started by a latent generator, decoded with local_blend).
#
#   bash workflows/surrogate_forward_workflow.sh [hydra overrides...]
#
# The grid must have the stepper's training spacing (4 m for the default
# model_weights/dft_b_realistic), e.g. the Barcelona case:
#   bash workflows/surrogate_forward_workflow.sh case=barcelona
# Any STL of your own: the domain is the crop window in the STL's frame, 128 m
# high (nx = x extent / 4, ...):
#   bash workflows/surrogate_forward_workflow.sh \
#       geometry.stl_path=path/to/city.stl \
#       domain.nx=176 domain.ny=176 domain.nz=32 \
#       'domain.bounds=[[0,704],[0,704],[0,128]]'
# Other models (a stepper with another step or spacing also needs
# model.forward_model.trained_output_frequency / trained_domain, see the config):
#   bash workflows/surrogate_forward_workflow.sh case=barcelona \
#       model.stepper_dir=model_weights/<stepper> \
#       model.generator_dir=model_weights/<generator>
#
# The surrogate steps 5 s, so time.output_frequency defaults to 5 here (a
# coarser one is emitted at the nearest step). Overrides only, no flags: they
# follow the workflow's own, so a flag such as --config-dir would end Hydra's
# override list (use hydra.searchpath=[...]).
# Run it inside the dev environment (`pixi shell -e dev`, or `-e cuda` for the GPU).
set -euo pipefail

exec bash "$(dirname "$0")/forward_workflow.sh" \
    model=neural_surrogate_generative time.output_frequency=5.0 "$@"
