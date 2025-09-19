#!/bin/bash

#SBATCH --job-name=mask_cascadia_combined
#SBATCH --output=mask_cascadia_combined.txt
#SBATCH --error=mask_cascadia_combined.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=long

module purge
module --quiet load miniconda/3
conda activate env_climatem_conda
conda install gdal
# === CONFIGURATION ===
MASK="$SCRATCH/data/ERA5_DATA_TEST/cascadia_bioregion.geojson"
INPUT_FILE="$SCRATCH/data/ERA5_DATA_PROCESSED/combined/t2m_1950_2024_combined.nc"
OUTPUT_FILE="$SCRATCH/data/ERA5_DATA_PROCESSED/combined/t2m_1950_2024_combined_masked.nc"

# === CHECK TOOLS ===
if ! command -v gdalwarp &> /dev/null; then
    echo "gdalwarp not found. Please load GDAL module or install GDAL."
    exit 1
fi

echo "Masking combined file..."
gdalwarp \
    -cutline "$MASK" \
    -crop_to_cutline \
    -of NETCDF \
    "$INPUT_FILE" "$OUTPUT_FILE"

echo "Masked file saved to $OUTPUT_FILE"
