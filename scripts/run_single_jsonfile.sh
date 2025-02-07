#!/bin/bash

#SBATCH --job-name=run_single
#SBATCH --output=run_single_output.txt
#SBATCH --error=run_single_error.txt
#SBATCH --partition=long                                                 # Ask for long job
#SBATCH --gpus-per-task=1                                                # Ask for 1 GPU
#SBATCH --cpus-per-task=16                                               # Ask for 2 CPUs
#SBATCH --ntasks-per-node=1                                              # Ask for 2 CPUs
#SBATCH --nodes=1                                                        # Ask for 2 CPUs
#SBATCH --mem=80G                                                       # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                                  # The job will run for 10 hours
#SBATCH -o /network/scratch/j/julien.boussard/slurm_logs/slurm-%j.out  # Write the log on scratch

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10

# 2. Load your environment
source $HOME/causal_model/env_emulator_climatem/bin/activate


# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"


export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

accelerate launch \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process=8 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=1 \
    --num_machines=1 \
    --gpu_ids='all' \
    $HOME/causal_model/climatem/scripts/main_explore_predictions_ar_sparsity_constraint_explore_ensembles_multigpu_accelerate.py --config-path $HOME/causal_model/climatem/configs/single_param_file.json
