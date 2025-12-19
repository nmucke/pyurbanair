# pyurbanair
This repository provides general simulation and data assimilation tools for the UrbanAIR project.  

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

All forward models (pylbm and pyudales) automaitcally generates a `pyurbanair/.temp` folder where intermediate files are stored. 

## Repository Structure
The repository contains a base project `pyurbanair`, and a series of sub-libraries in the `libs/` folder. The general idea is that everything should be run from the `pyurbanair` projectm which will load functionalities from the other libraries. 

### Libraries

#### pyurbanair
This is the base library. It only contains utils to compute quantities of interest and a base forward model. All the other libraries that introduce forward models should inheret from this base class. This ensures compatibility throughout the entire repo. Furthermore, a series of functionalities are already implemented to ease the implementation of other forward models. For examples, it is set up such that one only has to implement `run_single` when implementing a new forward model. Then, the base class can simulate an ensemble based on an ensemble of parameter inputs bu calling the `run_single` method.  

#### data-assimilation
This library contains data assimilation functionalities. Currently, it only contains an observation operator, a base smoothing model (from which future smoothing models will inherit) and an ESMDA model. It is setup such that it is compatible with the other libraries. The library is implemented using Jax.

#### pylbm
This library is a wrapper for Geir Evensens Lattice Boltzmann simulator. First time it is being imported it downloads the repo from github and compiles the code based on the experiment specifications.

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

## Data and file types
The main data and files types are netcdf and xarray. All forward models take in parameters as xarrays and out states as xarray. When the model is configures to save the simulation outputs is always stores in netcdf format. This is to ensure compatibility accross libraries. 

### State data

### Parameter data

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