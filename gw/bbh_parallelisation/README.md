# BBH Parallelisation experiment

Experiments comparing how `nessai` and `i-nessai` scale for increasing number of cores. Settings for each sampler are configured in `base.ini` and `base_baseline.ini` where baseline refers to standard `nessai`.

To run all of the analyses run the following command:

```bash
make all
```

This will submit all of the jobs to condor.


## Injection details

The injection file is created by running

```bash
python get_injection.py
```

This gets the injection `precessing_injection.json` in `gw/bbh_pp_test` and saves it as `parallelisation_injection.json`. This is done automatically when using the `Makefile`.
