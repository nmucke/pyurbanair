I want to simplify my config setup. It is way to nested. I want a setup like this:

- model/ , one file per model, as it is now
  - neural_surrogate
  - pyudales
  - pylbm
  - pupalm
- case/ , one file per case, as it is now
  - xie_and_castro
  - barcelona
- assimilation.yaml, should contain all data assimilation settings that is common for all data assimilation. See description below.
- forward.yaml, should contain all settings that is common for all forward model runs. See description below
- assimilation_settings/ , should contain all additional assimilation settings
  - localization.yaml
  - inflation.yaml
  - state_reduction.yaml
  - ...
- smoothing.yaml, should contain all smoothing specific settings. See descriotion below.
- filtering.yaml, should contain all filtering specific settings. See descriotion below.
- hybrid.yaml, , should contain all hybrid data assimilation specific settings. See descriotion below.
- params/ , largely untouched, with the exception of a common.yaml file:
- paths/ , should contain all paths to where results should end up


# assimilation.yaml

Settings available in the assimilation.yaml file:

- num_windows
- ensemble_size
- seed
- params_to_estimate
- ensemble_save_on_disk
- save_prior_state
- truth_dir
- truth_start_time
- save_forecast_history
- truth_params
- truth_model
- assim_model
- failure


# smoothing.yaml

This will replace the current esmda naming convention. As esmda is simply one type of smoothing model.

Settings only used when running smoothing runs:
- model, should use hydra instantiation
- observation_operator, should use hydra instantiation
- prior_params

Furthermore, it should load all necessary settings from other config files, e.g. the assimilation_settings/ files if specified.

# filtering.yaml

This will be very similar to the smoothing.yaml file.

Settings only used when running filtering runs:
- model, should use hydra instantiation
- observation_operator, should use hydra instantiation
- prior_params

Furthermore, it should load all necessary settings from other config files, e.g. the assimilation_settings/ files if specified.

# hybrid.yaml

This will be very similar to the smoothing.yaml file.

Settings only used when running filtering runs:
- smoothing_model, should use hydra instantiation
- filtering_model, should use hydra instantiation
- smoothing_observation_operator, should use hydra instantiation
- filtering_observation_operator, should use hydra instantiation
- prior_params

Furthermore, it should load all necessary settings from other config files, e.g. the assimilation_settings/ files if specified.


# params/

This folder should fairly similar to the current setup. The main exception is the addition of a common.yaml:

- seconds_per_knot

Potentially other parameters.
