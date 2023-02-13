# BNS analysis

## Injection

The GW190425-like injection was generated using `bilby_pipe` and the `bns_injection.prior` and is saved in `GW190425-like.json`.

## Priors

The priors for the analysis are specified in the individual .ini files. The priors are:

```
prior-dict={
chirp_mass = Uniform(name='chirp_mass', minimum=1.342689302683192, maximum=1.542689302683192, unit='$M_{\odot}$'),
mass_ratio = Uniform(name='mass_ratio', minimum=0.125, maximum=1),
mass_1 = Constraint(name='mass_1', minimum=0.62587375, maximum=1000),
mass_2 = Constraint(name='mass_2', minimum=0.62587375, maximum=1000),
chi_1 = bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.05)),
chi_2 = bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.05)),
luminosity_distance = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1, maximum=400, unit='Mpc'),
dec = Cosine(name='dec'),
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
theta_jn = Sine(name='theta_jn'),
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
phase = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'),
}
```

## ROQs

These analyses make use of the ROQ files available at https://git.ligo.org/lscsoft/ROQ_data. If running on of the LDG, these files should be available at `/home/cbc/ROQ_data/IMRPhenomPv2/`.

## Submitting the jobs

The jobs can be submitted by running

```bash
make <sampler>
```

where `<sampler>` should be one of `{inessai, nessai, dynesty}`

## Analysing the results

The results are analysed in `analyse_bns_results.ipynb`.

**Meta-proposal plot**

The meta-proposal plot is produced separately using:

```
python plot_meta_proposal.py <path/to/result/file/>
```

In this case, the result file must not the merged file, but one of the files labelled `par*`.
