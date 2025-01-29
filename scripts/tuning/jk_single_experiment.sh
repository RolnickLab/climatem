#!/bin/bash

#SBATCH --partition=long                                           # Ask for long job
#SBATCH --gpus-per-task=1                                                # Ask for 1 GPU
#SBATCH --cpus-per-task=16     # 16                                          # Ask for 2 CPUs
#SBATCH --ntasks-per-node=2                                              # Ask for 2 CPUs
#SBATCH --nodes=1                                                        # Ask for 2 CPUs
#SBATCH --mem=80G            # 80                                       # Ask for 10 GB of RAM
#SBATCH --time=16:00:00                                                  # The job will run for 10 hours
#SBATCH -o /home/mila/j/julia.kaltenborn/slurm_logs/climatem/slurm-%j.out  # Write the log on scratch

# RUN this file like that (in climatem):
# sbatch scripts/tuning/jk_single_experiment.sh $HOME/climatem/scripts/configs/tuning/default_configs.json

# indicate which configs are running
echo "Running CDSD Experiment with config file: $1"

# 1. Load the required modules
module load OpenSSL/1.1
module --quiet load python/3.10
# fix SSL issues
export SSL_CERT_DIR=/etc/ssl/certs

# 2. Load your environment
source $HOME/climatem/env_emulator_climatem/bin/activate

# create directory if it does not exist yet:
mkdir -p $HOME/climatem/Climateset_DATA/
mkdir -p $HOME/scratch/results/climatem_hyper/

# 3. Copy your dataset on the compute node - I am not sure whether I need to do this at the moment...
# cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
#unset CUDA_VISIBLE_DEVICES

#export NCCL_DEBUG=INFO

export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

# accelerate test

accelerate launch \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process=16 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=2 \
    --num_machines=1 \
    --gpu_ids='all' \
    $HOME/climatem/scripts/main_explore_predictions_ar_sparsity_constraint_explore_ensembles_multigpu_accelerate.py --no-gt --gpu --config-exp-path $1 --exp-path $HOME/scratch/results/climatem_hyper/ --config-path $HOME/climatem/scripts/params/default_params_testing_sparsconst_sbatch_nl_ensembles_relax_single_hilatent.json --lr 0.001 --reg-coeff 0.3 --sparsity-upper-threshold 0.5

