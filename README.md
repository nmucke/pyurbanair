# pyurbanair

This repository provides general simulation and data assimilation tools for the UrbanAIR project.

IMPORTANT: The repository is very much under development! Therefore, may things will change and there are probably, some functionalities that don't work as intended.

IMPORTANT: It only works on MacOS! Linux will be added next!

## Installation

All dependencies and environments are handled via Pixi. Pixi can be installed on Linux and MacOS by running:

```
curl -fsSL https://pixi.sh/install.sh | sh
```

Currently, there is only one functioning environment, the dev environment. When installing the dev environment it automatically creates the environment and installs all necessary dependendencies to compile uDALES and LBM as well as python dependencies. At a later stage, I will implement different environments for different purposes (e.g. pyudales-env) such that it only install the necessary dependencies for that.

The environment is installd and initiated via:

```
pixi shell --environment=dev
```

### LBM specifics

Note that for running the LBM code on MacOSn you have to run the following after initializing the environment:

```
ulimit -s unlimited
```

### uDALES specifics

You need to have Matlab installed to run uDALES code. You also have to provide that path to Matlab program.

## Running the code

There are 4 scripts set up in the `pyurbanair/scripts` folder. You should be able to run then when the dev environment is initialized. The scripts load the examples from the `pyurbanair/examples` folder. Currently, only the Xia and Castro case is set up.

When inside the dev environment, simply run:

```
python scripts/<file_to_run>.py
```

All forward models (pylbm and pyudales) automatically generates a `pyurbanair/.temp` folder where intermediate files are stored.

Currently, each of the forward models require different types of configuration files due to the vastly different structure of LBM and uDALES. I might create a unifying yaml-like config file format later.

## Repository Structure

I chose to go with a mono-repo approach. The repository contains a base project `pyurbanair`, and a series of sub-libraries in the `libs/` folder. The general idea is that everything should be run from the `pyurbanair` project which will load functionalities from the other libraries.

### Libraries

#### pyurbanair

This is the base library. It only contains utils to compute quantities of interest and a base forward model. All the other libraries that introduce forward models should inheret from this base class. This ensures compatibility throughout the entire repo. Furthermore, a series of functionalities are already implemented to ease the implementation of other forward models. For examples, it is set up such that one only has to implement `run_single` when implementing a new forward model. Then, the base class can simulate an ensemble based on an ensemble of parameter inputs bu calling the `run_single` method.

#### data-assimilation

This library contains data assimilation functionalities. Currently, it only contains an observation operator, a base smoothing model (from which future smoothing models will inherit) and an ESMDA model. It is setup such that it is compatible with the other libraries. The library is implemented using Jax.

#### pylbm

This library is a wrapper for Geir Evensens Lattice Boltzmann simulator. First time it is being imported it downloads the repo from github and compiles the code based on the experiment specifications.

IMPORTANT: I have implemented an stl-to-LBM convertion. However, I don't think it is completely correct. So don't trust the outputs from pylbm.

#### pyudales

This library is a wrapper for the uDALES simulator. First time it is being imported it downloads the repo from github and compiles the code based on the experiment specifications.

#### Structure

```
pyurbanair/
├── src/
│   └── pyurbanair/          # Main package
│       ├── base_forward_model.py
│       └── utils/
│           └── state_utils.py
│
├── libs/                     # Library submodules
│   ├── data-assimilation/   # Data assimilation library
│   │   ├── pyproject.toml   # Dependencies for data-assimilation
│   │   └── src/data_assimilation/
│   │       ├── observation_operator.py
│   │       └── smoothing/   # Smoothing algorithms (only ESMDA for now)
│   │
│   ├── pylbm/               # Lattice Boltzmann Method wrapper
│   │   ├── pyproject.toml   # Dependencies for pylbm
│   │   └── src/pylbm/
│   │       ├── forward_model.py
│   │       └── ...
│   │
│   └── pyudales/            # u-dales wrapper
│       ├── pyproject.toml   # Dependencies for pyudales
│       └── src/pyudales/
│           ├── forward_model.py
│           └── ...
│
├── scripts/                 # Main execution scripts
│   ├── main_lbm.py          # Forward run with LBM
│   ├── main_udales.py       # Forward run with uDALES
│   ├── esmda_lbm.py         # ESMDA with LBM
│   └── esmda_udales.py      # ESMDA with uDALES
│
├── examples/                # Example experiments
│   ├── lbm/
│   │   └── experiments/
│   └── udales/
│       └── experiments/
│
├── pyproject.toml           # Project configuration
├── LICENSE
└── .gitmodules              # Git submodules (u-dales, LBM)
```

### Benchmark geometry

A script to generate the geometry in `stl` (as well as other format) for the Xie Castro 2002 Benchmark can be found in `examples/benchmark_geometry/` folder.

By importing `XieCastroBenchmarkGeometry` from `boundary_geometry.py` one can configure and serialize the specific setup. There is also a commandline tool available. The dependencies are available in the `dev` environment. Usage:

```
pixi shell -e dev
python examples/boundary_geometry/boundary_geometry.py --help

```

One example for stl is

```
python examples/boundary_geometry/boundary_geometry.py stl output --num-tiles 3 3
```

For Geir Evensen's Lattice Boltzmann code one can as well configure a fortran file, which needs to be compiled subsequently. To change the base resolution one can provide a refinement factor as well.

```
python examples/boundary_geometry/boundary_geometry.py stl output --resolution 4 --num-tiles 3 3
```

## Data and file types

The main data and files types are netcdf and xarray. All forward models take in parameters as xarrays and out states as xarray. When the model is configures to save the simulation outputs is always stores in netcdf format. This is to ensure compatibility accross libraries.

### State data

States are always provide and output as xarrays. They should have the following format:

```
Dimensions:  (time: 1, zm: 6, yt: 128, xt: 128, zt: 6, ym: 128, xm: 128)
Coordinates:
  * time     (time) float32 4B 50.27
  * zm       (zm) float32 24B 0.0 6.667 13.33 20.0 26.67 33.33
  * yt       (yt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * xt       (xt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * zt       (zt) float32 24B 3.333 10.0 16.67 23.33 30.0 36.67
  * ym       (ym) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
  * xm       (xm) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
Data variables:
    w        (time, zm, yt, xt) float32 393kB 0.0 0.0 0.0 ... -0.0216 -0.01649
    pres     (time, zt, yt, xt) float32 393kB ...
    v        (time, zt, ym, xt) float32 393kB -0.06821 -0.1152 ... 0.5625 0.5629
    u        (time, zt, yt, xm) float32 393kB -0.05016 0.08196 ... 3.14 3.139
```

Note that xt vs xm is udales specific. For pylbm there is only x, y, z. However, time should always be there, even when only one time step is present. For example, if the simulations is set to only output tha last state in a simulation.

Ensembles of states are also in xarray format:

```
Dimensions:  (ensemble: 50, time: 1, zm: 6, yt: 128, xt: 128,
              zt: 6, ym: 128, xm: 128)
Coordinates:
  * time     (time) float32 4B 50.15
  * zm       (zm) float32 24B 0.0 6.667 13.33 20.0 26.67 33.33
  * yt       (yt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * xt       (xt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * zt       (zt) float32 24B 3.333 10.0 16.67 23.33 30.0 36.67
  * ym       (ym) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
  * xm       (xm) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
Dimensions without coordinates: ensemble
Data variables:
    w        (ensemble, time, zm, yt, xt) float32 59MB 0.0 ... -0...
    pres     (ensemble, time, zt, yt, xt) float32 59MB -0.1003 .....
    v        (ensemble, time, zt, ym, xt) float32 59MB -0.38 ... ...
    u        (ensemble, time, zt, yt, xm) float32 59MB 0.3301 ......
```

Here, the format is the same but with an added ensemble dimension.

### Parameter data

Parameters can be provided as argument as an xarray when calling the forward model. The format should be like this:

```
true_params = xarray.Dataset(
    data_vars={
        "inflow_angle": TRUE_ANGLE,
        "velocity_magnitude": TRUE_VELOCITY_MAGNITUDE,
    },
)
```

Currently, only inflows_angle and velocity_magnitude are supported. This will be expanded later.

An ensemble of parameter can be provided in the same manner, but the an added ensemble dimension:

```
params_ensemble = xarray.Dataset(
    data_vars={
        "inflow_angle": ("ensemble", inflow_angle_range),
        "velocity_magnitude": ("ensemble", velocity_magnitude_range),
    },
    coords={"ensemble": jnp.arange(len(inflow_angle_range))},
)
```

Running with this automatically simulates an ensemble. This is set up in the base forward model.

## Development

When adding to the repository, first create a new branch. Then make the changes you want, commit, and then create a pull request.

If you want to add to the repository you should make use of the linting and formatting. These are automatically installed in the dev environment. Simply run:

```
pixi run pre-commit
```

and it will apply formatting and give you "errors" to be fixed. Note that it only applies to files that are staged. Note that somtimes the linting is a bit annoying and gives errors that you don't necessarily want to fix. These errors you can ignore by adding the following after the line in question:

```
# type: ignore[<something>]
```

There is currently no pretection on the main branch! So comitting is possible without passing the pre-commit! Please be mindful before comitting!
