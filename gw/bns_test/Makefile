CONFIG = config.mk
include ${CONFIG}

all: ins-nessai nessai dynesty

.PHONY : ins-nessai
inessai :
	bilby_pipe inessai.ini --submit

.PHONY : nessai
nessai :
	bilby_pipe nessai.ini --submit

.PHONY : dynesty
dynesty:
	bilby_pipe dynesty.ini --submit

.PHONY : plot-meta-proposal
plot-meta-proposal:
	python plot_meta_proposal.py \
	$(INESSAI_OUTDIR)/result/inessai_bns_data0_0_analysis_H1L1_par0_result.hdf5
