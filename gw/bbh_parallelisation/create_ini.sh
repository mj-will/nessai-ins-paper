#/usr/bin/env bash

# Script to create ini files with n_pool
base_ini="base.ini"
n_pool=$1
label=n_pool_${n_pool}

cp ${base_ini} ${label}.ini
sed -i "s/label.*/label=${label}/g" ${label}.ini
sed -i "s/outdir.*/outdir=${label}/g" ${label}.ini
sed -i "s/request-cpus.*/request-cpus=${n_pool}/g" ${label}.ini
sed -i "s/n_pool=.*/n_pool=${n_pool},/g" ${label}.ini

# Create ini files for the baseline
baseline_ini="base_baseline.ini"
baseline_label=n_pool_${n_pool}_baseline

cp ${baseline_ini} ${baseline_label}.ini
sed -i "s/label.*/label=${baseline_label}/g" ${baseline_label}.ini
sed -i "s/outdir.*/outdir=${baseline_label}/g" ${baseline_label}.ini
sed -i "s/request-cpus.*/request-cpus=${n_pool}/g" ${baseline_label}.ini
sed -i "s/n_pool=.*/n_pool=${n_pool},/g" ${baseline_label}.ini
