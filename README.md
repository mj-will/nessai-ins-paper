# Importance nested sampling with normalising flows

This repository contains code to accompany the paper Importance nested sampling with normalising flows.

## Installation

### Cloning the repository

This repo contains git submodules, so it should be cloned using:

```
git clone --recurse-submodules git@github.com:mj-will/nessai-ins-paper.git
```

or equivalently with HTTPS URL.

### Python packages

The `environment.yml` and `env-spec-file` contain details of all the dependencies to construct a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that contains the relevant Python dependencies. However, custom forks of `nessai`, `bilby` and `bilby_pipe` were used. These are all included in `packages` directory and can be installed with the following commands:

```
cd packages/<name>
pip install .
```

where `<name>` should be substituted for `nessai`, `bilby` and `bilby_pipe`.


## Analysis

The analysis is split into three directories:

* `toy_example`: contains the code for the toy example in Section 3
* `experiments`: contains the analytical likelihood and Rosenbrock tests
* `gw`: contains all of the gravitational-wave tests

Each directory contains instructions for how to run the experiments.
