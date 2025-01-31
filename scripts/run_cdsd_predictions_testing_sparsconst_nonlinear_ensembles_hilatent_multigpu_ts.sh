#!/bin/bash

#SBATCH --gpus-per-task=1                                                # Ask for 1 GPU
#SBATCH --cpus-per-task=16                                               # Ask for 2 CPUs
#SBATCH --ntasks-per-node=1                                              # Ask for 2 CPUs
#SBATCH --nodes=1                                                        # Ask for 2 CPUs
#SBATCH --mem=80G                                                       # Ask for 10 GB of RAM
#SBATCH --time=16:00:00                                                  # The job will run for 10 hours
#SBATCH -o /scratch/slurm-%j.out                                        # Write the log on scratch

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10

# 2. Load your environment
source $HOME/work/climatem/env_emu_poetry/bin/activate


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
    $HOME/work/climatem/scripts/main_explore_predictions_ar_sparsity_constraint_explore_ensembles_multigpu_accelerate_no_crps_code_review.py --no-gt --tau 5 --gpu --d-z 90 --d-x 6250 --config-exp-path $HOME/work/climatem/scripts/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all_icosa_picontrol.json --exp-path $HOME/scratch/results/climatem_spectral_no_crps_no_detach_github_close/ --config-path $HOME/work/climatem/scripts/params/default_params_testing_sparsconst_sbatch_nl_ensembles_relax_single_hilatent.json --lr 0.001 --reg-coeff 0.31 --sparsity-upper-threshold 0.5

