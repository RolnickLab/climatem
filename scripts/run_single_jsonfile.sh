#!/bin/bash

#SBATCH --job-name=run_single                                           # Set name of job
#SBATCH --output=run_single_output.txt                                  # Set location of output file
#SBATCH --error=run_single_error.txt                                    # Set location of error file
#SBATCH --gpus-per-task=1                                               # Ask for 1 GPU
#SBATCH --cpus-per-task=4                                               # Ask for 4 CPUs
#SBATCH --ntasks-per-node=1                                             # Ask for 4 CPUs
#SBATCH --nodes=1                                                       # Ask for 4 CPUs
#SBATCH --mem=32G                                                       # Ask for 32 GB of RAM
#SBATCH --time=02:00:00                                                 # The job will run for 2 hours
#SBATCH --partition=long                                                # Ask for long partition

# 0. Clear the environment
module purge

# 1. Load the required modules
module load compiler/intel/2021.4.0
module load devel/python/3.10.5_intel_2021.4.0

# 2. Load your environment
source $HOME/my_projects/climatem/env_emulator_climatem/bin/activate
source $HOME/env_climatem/bin/activate


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
    $HOME/climatem/scripts/main_picabu.py --config-path $HOME/climatem/configs/single_param_file.json
