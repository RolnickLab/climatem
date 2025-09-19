#!/bin/bash

#SBATCH --job-name=combine_netcdfs
#SBATCH --output=combine_output.txt
#SBATCH --error=combine_error.txt
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=long

module purge
module load python/3.10

source $HOME/env/env_climatem_experimental/bin/activate

LOG_DIR="$HOME/scratch/dev/climatem/logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/combine_netcdfs_$(date +%Y%m%d_%H%M%S).log"

echo "Running combine_netcdfs.py..." | tee -a $LOG_FILE

# Explicitly print Python version and working directory
which python | tee -a $LOG_FILE
python --version | tee -a $LOG_FILE
pwd | tee -a $LOG_FILE

# Run the script
python $SCRATCH/dev/climatem/combine_netcdf.py 2>&1 | tee -a $LOG_FILE

echo "Done." | tee -a $LOG_FILE
