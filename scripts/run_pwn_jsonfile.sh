#!/bin/bash

#SBATCH --job-name=run_pnw                                   # Base job name (will be updated per run)
#SBATCH --output=run_pnw_output.txt                          # Stdout
#SBATCH --error=run_pnw_err.txt                              # Stderr
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=long

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
# Optional: silence wandb warning if unused
# export WANDB_DISABLED=true

echo "=== calling accelerate"

CONFIG_PATH="single_param_file_ERA5.json"
PY_MAIN="$HOME/dev/climatem/scripts/main_picabu.py"

# Sweeps
SPARSITY_LIST=(0.15 0.20)
Z_LATENTS_LIST=(10 5)   # z500 latents sweep; first var stays at 25

for Z in "${Z_LATENTS_LIST[@]}"; do
  for SP in "${SPARSITY_LIST[@]}"; do
    EXP_OUT="$SCRATCH/results/ERA5_DATA_TEST/pnw_[${Z},25]_fixed/sp_${SP}"

    # Update job name so squeue shows z/sp
    scontrol update JobId=$SLURM_JOB_ID JobName=pnw_z${Z}_sp${SP}

    accelerate launch \
      --machine_rank=$SLURM_NODEID \
      --num_cpu_threads_per_process=8 \
      --main_process_ip=$MASTER_ADDR \
      --main_process_port=$MASTER_PORT \
      --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='all' \
      "$PY_MAIN" \
      --config-path "$CONFIG_PATH" \
      --hp exp_params.d_z="[25,${Z}]" \
      --hp optim_params.sparsity_upper_threshold="$SP" \
      --hp exp_params.exp_path="$EXP_OUT"
  done
done
