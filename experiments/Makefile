CONFIG = config.mk
include ${CONFIG}

EXE = run_experiment.py
ANALYTIC_MODELS = gaussian gmm_paper


.PHONY : baseline_gaussian
baseline_gaussian:
	python submit_runs.py --name=baseline -e=$(EXE) -n=$(N) \
		--config=baseline_gaussian.ini --output=$(OUTDIR) --dims $(DIMS)

.PHONY : baseline_gmm
baseline_gmm:
	python submit_runs.py --name=baseline -e=$(EXE) -n=$(N) \
		--config=baseline_gmm.ini --output=$(OUTDIR) --dims $(DIMS)

.PHONY : baseline_rosenbrock
baseline_rosenbrock:
	python submit_runs.py --name=baseline -e=$(EXE) -n=$(N_ROSENBROCK) \
		--config=baseline_rosenbrock.ini --output=$(OUTDIR) \
		--dims $(ROSENBROCK_DIMS)

.PHONY : baselines
baselines : baseline_gaussian baseline_gmm baseline_rosenbrock

.PHONY : check_baselines
check_baselines :
	for M in $(ANALYTIC_MODELS) ; do \
		for D in ${DIMS}; do \
			echo "Model $${M} $${D} "; \
			ls $(OUTDIR)/baseline_$${M}_$${D}d/analysis/run*/summary.json | wc -l; \
		done \
	done

.PHONY : gaussian
gaussian :
	python submit_runs.py --name=ins -e=run_experiment.py -n=$(N) \
	--config=gaussian.ini --output=$(OUTDIR) --dims $(DIMS)

.PHONY : gmm
gmm :
	python submit_runs.py --name=ins -e=run_experiment.py -n=$(N) \
	--config=gmm.ini --output=$(OUTDIR) --dims $(DIMS)

.PHONY : rosenbrock
rosenbrock :
	python submit_runs.py --name=ins -e=run_experiment.py -n=$(N_ROSENBROCK) \
	--config=rosenbrock.ini --output=$(OUTDIR) --dims $(ROSENBROCK_DIMS)

.PHONY : ins
ins: gaussian gmm rosenbrock


.PHONY : nlive_comparison
nlive_comparison :
	python submit_nlive_comparison.py --config=nlive_comparison.ini \
	-n $(N_NLIVE) --nlive $(NLIVE) --output=$(OUTDIR) --name=ins_nlive \
	-e=run_experiment.py

..PHONY : clean_baselines
clean_baselines :
	rm -r $(OUTDIR)/baseline*
