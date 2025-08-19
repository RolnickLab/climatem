#!/bin/bash

#SBATCH --job-name=pnw_S_15_[10,25]                                         # Set name of job
#SBATCH --output=pnw_S_15_[10,25].txt                                  # Set location of output file
#SBATCH --error=pnw_S_15_[10,25]_err.txt                                    # Set location of error file
#SBATCH --gpus-per-task=1                                               # Ask for 1 GPU
#SBATCH --cpus-per-task=8                                               # Ask for 4 CPUs
#SBATCH --ntasks-per-node=1                                             # Ask for 4 CPUs
#SBATCH --nodes=1                                                       # Ask for 4 CPUs
#SBATCH --mem=256G                                                       # Ask for 32 GB of RAM
#SBATCH --time=48:00:00                                                 # The job will run for 2 hours
#SBATCH --partition=long                                                # Ask for long partition

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10


# 2. Load your environment assuming environment is called "env_climatem" in $HOME/env/ (standardized)
source $HOME/env/env_climatem/bin/activate
# 3. Enable expandable allocator to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"


export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

echo "=== calling accelerate"

# Make sure to change program file path to correct dir
accelerate launch \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process=8 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=1 \
    --num_machines=1 \
    --gpu_ids='all' \
    $HOME/dev/climatem/scripts/main_picabu.py --config-path single_param_file_ERA5.json
