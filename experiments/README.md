# Analytic and Rosenbrock experiments

This directory contains results for the analytic and Rosenbrock likelihoods.

## Running the analysis

The `Makefile` is used to submit all of the analyses via HTCondor. For example

```
make gaussian
```
will submit the Gaussian runs with `i-nessai` and

```
make baseline_gaussian
```

will submit the Gaussian runs with `nessai`.

All the runs can be submitted at once using `make ins` and `make baselines`

## Algorithm scaling

The results for the "Algorithm scaling" section are produced by running

```
make nlive_comparison
```


## Producing the plots

All the plots are produced in `analytic_results.ipynb` in `nlive_comparison_results.ipynb`.
