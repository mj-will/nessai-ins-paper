# Makefile for BBH parallelisation experiment
all : n_pool_1 n_pool_2 n_pool_4 n_pool_8 n_pool_16

parallelisation_injection.json:
	python get_injection.py

.PHONY : n_pool_1
.PHONY : n_pool_2
.PHONY : n_pool_4
.PHONY : n_pool_8
.PHONY : n_pool_16

n_pool_1: parallelisation_injection.json
	bash create_ini.sh 1
	bilby_pipe n_pool_1.ini --submit
	bilby_pipe n_pool_1_baseline.ini --submit

n_pool_2: parallelisation_injection.json
	bash create_ini.sh 2
	bilby_pipe n_pool_2.ini --submit
	bilby_pipe n_pool_2_baseline.ini --submit

n_pool_4: parallelisation_injection.json
	bash create_ini.sh 4
	bilby_pipe n_pool_4.ini --submit
	bilby_pipe n_pool_4_baseline.ini --submit

n_pool_8: parallelisation_injection.json
	bash create_ini.sh 8
	bilby_pipe n_pool_8.ini --submit
	bilby_pipe n_pool_8_baseline.ini --submit

n_pool_16: parallelisation_injection.json
	bash create_ini.sh 16
	bilby_pipe n_pool_16.ini --submit
	bilby_pipe n_pool_16_baseline.ini --submit

clean:
	rm -r n_pool_1*
	rm -r n_pool_2*
	rm -r n_pool_4*
	rm -r n_pool_8*
	rm -r n_pool_16*
