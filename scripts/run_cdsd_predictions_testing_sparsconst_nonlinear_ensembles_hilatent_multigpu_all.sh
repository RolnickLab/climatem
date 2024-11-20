#!/bin/bash

#SBATCH --partition=long                                                 # Ask for long job
#SBATCH --gpus-per-task=1                                                # Ask for 1 GPU
#SBATCH --cpus-per-task=16                                              # Ask for 2 CPUs
#SBATCH --ntasks-per-node=2                                              # Ask for 2 CPUs
#SBATCH --nodes=1                                                        # Ask for 2 CPUs
#SBATCH --mem=80G                                                        # Ask for 10 GB of RAM
#SBATCH --time=16:00:00                                                  # The job will run for 10 hours
#SBATCH -o /network/scratch/s/sebastian.hickman/slurm_logs/slurm-%j.out  # Write the log on scratch

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10

# 2. Load your environment
source $HOME/work/climatem/env_emulator_climatem/bin/activate

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

# accelerate test

accelerate launch \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process=8 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=1 \
    --num_machines=1 \
    --gpu_ids='all' \
    $HOME/work/climatem/scripts/main_explore_predictions_ar_sparsity_constraint_test_ensembles_multigpu_accelerate.py --no-gt --tau 3 --gpu --d-z 50 --d-x 6250 --config-exp-path $HOME/work/climatem/scripts/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all_icosa.json --exp-path $HOME/scratch/results/test_climatem/ --config-path $HOME/work/climatem/scripts/params/default_params_testing_sparsconst_sbatch_nl_ensembles_relax_single_hilatent.json --lr 0.001 --reg-coeff 0.1 --sparsity-upper-threshold 0.5


#accelerate launch $HOME/work/causalpaca/causal/main_explore_predictions_ar_sparsity_constraint_test_ensembles_multigpu_accelerate.py --no-gt --tau 3 --gpu --d-z 50 --d-x 6250 --config-exp-path $HOME/work/causalpaca/causal/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all.json --exp-path $HOME/scratch/results/test_multigpu/ --config-path $HOME/work/causalpaca/causal/params/default_params_testing_sparsconst_sbatch_nl_ensembles_relax_single_hilatent.json --lr 0.001 --reg-coeff 0.735740171 --sparsity-upper-threshold 0.5

#srun python -u $HOME/work/causalpaca/causal/main_explore_predictions_ar_sparsity_constraint_test_ensembles_multigpu_accelerate.py --no-gt --tau 3 --gpu --d-z 50 --d-x 6250 --config-exp-path $HOME/work/causalpaca/causal/configs/climate_predictions_picontrol_icosa_nonlinear_ensembles_hilatent_all.json --exp-path $HOME/scratch/results/test_multigpu/ --config-path $HOME/work/causalpaca/causal/params/default_params_testing_sparsconst_sbatch_nl_ensembles_relax_single_hilatent.json --lr 0.001 --reg-coeff 0.1 --sparsity-upper-threshold 0.5


# cp $SLURM_TMPDIR/<to_save> /network/scratch/<u>/<username>/

# I should probably do something sensible here where I look at how to move data to $SLURM_TMPDIR and so on.