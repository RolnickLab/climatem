#!/bin/bash

#SBATCH --job-name=run_single                                           # Set name of job
#SBATCH --output=run_single_output_1.txt                                  # Set location of output file
#SBATCH --error=run_single_error_1.txt                                    # Set location of error file
#SBATCH --gpus-per-task=1                                               # Ask for 1 GPU
#SBATCH --cpus-per-task=8                                               # Ask for 4 CPUs
#SBATCH --ntasks-per-node=1                                             # Ask for 4 CPUs
#SBATCH --nodes=1                                                       # Ask for 4 CPUs
#SBATCH --mem=128G
#SBATCH --time=56:00:00
#SBATCH --partition=long                                                # Ask for long partition

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10

# 2. Load your environment
source "$HOME/env/env_climatem/bin/activate"

# 3. Avoid CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"


export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
# Optional: silence wandb warning if you don't use it
# export WANDB_DISABLED=true

echo "=== calling accelerate"

CONFIG_PATH="single_param_file_chirps.json"
PY_MAIN="$HOME/dev/climatem/scripts/main_picabu.py"

# Sweeps
SPARSITY_LIST=(0.20 0.15)
Z_LATENTS_LIST=(10 5)   # z (second var) = 5 or 10, lsp stays 25

for Z in "${Z_LATENTS_LIST[@]}"; do
  for SP in "${SPARSITY_LIST[@]}"; do
    EXP_OUT="$SCRATCH/results/CHIRPS_DATA_TEST/chirps_[${Z},25]/sp_${SP}"

    # rename job so squeue shows z/sp
    scontrol update JobId=$SLURM_JOB_ID JobName=chirps_W_z${Z}_sp${SP}

    accelerate launch \
      --machine_rank=$SLURM_NODEID \
      --num_cpu_threads_per_process=8 \
      --main_process_ip=$MASTER_ADDR \
      --main_process_port=$MASTER_PORT \
      --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='all' \
      $HOME/dev/climatem/scripts/main_picabu.py \
      --config-path single_param_file_chirps.json \
      --hp exp_params.d_z="[25,${Z}]" \
      --hp optim_params.sparsity_upper_threshold="$SP" \
      --hp exp_params.exp_path="$EXP_OUT"
  done
done
