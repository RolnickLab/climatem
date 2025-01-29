#!/bin/bash
# run this file in climatem:
# bash scripts/tuning/jk_all_experiments.sh

# run an experiment for each config file
for CONFIG_FILE in $HOME/climatem/scripts/configs/tuning/*.json; do
    echo "Launching experiment with config file: $CONFIG_FILE"
    sbatch $HOME/climatem/scripts/tuning/jk_single_experiment.sh $CONFIG_FILE   
done