#!/bin/bash

#SBATCH --job-name=download_data                                # Job name
#SBATCH --output=download_output.txt                           # Output log
#SBATCH --error=download_error.txt                             # Error log
#SBATCH --cpus-per-task=32                                     # More CPUs for parallel downloads
#SBATCH --ntasks=1                                             # Single task, but utilizes multiple CPUs
#SBATCH --nodes=1                                              # One node
#SBATCH --mem=128G                                             # Adjust memory as needed
#SBATCH --time=12:00:00                                        # Increase time for large downloads
#SBATCH --partition=long                                       # Use long partition (or check if a data-specific node is available)

# 0. Clear the environment
module purge

# 1. Load required modules
module --quiet load python/3.10

# 2. Activate the Python environment
source $HOME/env/env_climatem_experimental/bin/activate

# 3. Set up logging
LOG_DIR="$HOME/scratch/dev/climatem/logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/era5_download_$(date +%Y%m%d_%H%M%S).log"

# 4. Check disk space before downloading
echo "=== Checking disk space before downloading ===" | tee -a $LOG_FILE
df -h | tee -a $LOG_FILE

# 5. Set up environment variables (modify as needed)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# 6. Run the data download script
echo "=== Starting ERA5 data download ===" | tee -a $LOG_FILE

python $SCRATCH/dev/climatem/era_5_download.py --t2m_pnw

# 7. Check disk space after downloading
echo "=== Checking disk space after downloading ===" | tee -a $LOG_FILE
df -h | tee -a $LOG_FILE

echo "=== Download process completed ===" | tee -a $LOG_FILE
