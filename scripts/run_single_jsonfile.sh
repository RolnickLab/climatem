#!/bin/bash

#SBATCH --job-name=run_single
#SBATCH --output=/network/scratch/s/sebastian.hickman/slurm_logs/slurm-%j.out  # Write the log on scratch
#SBATCH --gpus-per-task=1                                                      # Ask for 1 GPU
#SBATCH --cpus-per-task=8                                                      # Ask for 2 CPUs
#SBATCH --ntasks-per-node=1                                                    # Ask for 2 CPUs
#SBATCH --nodes=1                                                              # Ask for 2 CPUs
#SBATCH --mem=32G                                                              # Ask for 10 GB of RAM
#SBATCH --time=20:00                                                           # The job will run for 2 hours
#SBATCH --partition=main

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10

# 2. Load your environment
source $HOME/work/new_climatem/climatem/env_emulator_poetry/bin/activate


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
    $HOME/work/new_climatem/climatem/scripts/main_picabu.py --config-path $HOME/work/new_climatem/climatem/configs/single_param_file_test.json


