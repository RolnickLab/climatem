#!/bin/bash

#SBATCH --partition=long                                                 # Ask for long job
#SBATCH --gpus-per-task=1                                                # Ask for 1 GPU
#SBATCH --cpus-per-task=32                                               # Ask for 2 CPUs
#SBATCH --ntasks-per-node=1                                              # Ask for 2 CPUs
#SBATCH --nodes=1                                                        # Ask for 2 CPUs
#SBATCH --mem=80G                                                       # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                                  # The job will run for 10 hours
#SBATCH -o /network/scratch/s/sebastian.hickman/slurm_logs/slurm-%j.out  # Write the log on scratch

# 0. Clear the environment
module purge

# 1. Load the required modules
module --quiet load python/3.10

# 2. Load your environment
source $HOME/causal_model/env_emu_climateem/bin/activate

# 3. Copy your dataset on the compute node - I am not sure whether I need to do this at the moment...
# cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
# -u is for unbuffered output

python -u $HOME/causal_model/climatem/notebooks/rollout_pf_rejection.py 

echo "Job finished."

# cp $SLURM_TMPDIR/<to_save> /network/scratch/<u>/<username>/

# I should probably do something sensible here where I look at how to move data to $SLURM_TMPDIR and so on.
