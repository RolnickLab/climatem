#!/bin/bash

# === USAGE CHECK ===
if [ $# -ne 1 ]; then
    echo "Usage: $0 <variable>"
    echo "Example: $0 z500"
    exit 1
fi

# === INPUT ARGUMENT ===
VAR_INPUT=$1

# === CONFIGURATION ===
INPUT_DIR="$SCRATCH/data/ERA5_DATA_TEST"
DAILY_DIR="${INPUT_DIR}/daily"

# === CREATE DAILY DIRECTORY ===
mkdir -p "$DAILY_DIR"

# === MAP INPUT VARIABLE TO FILENAME MATCH ===
case "$VAR_INPUT" in
    sst)
        VAR_PATTERN="sst"
        ;;
    t2m)
        VAR_PATTERN="t2m"
        ;;
    z500)
        VAR_PATTERN="z500"
        ;;
    *)
        echo "Unsupported variable: $VAR_INPUT"
        echo "Supported variables: sst, t2m, z500"
        exit 1
        ;;
esac

# === PROCESS FILES ===
for FILE in ${INPUT_DIR}/ERA5_${VAR_PATTERN}_*.nc; do
    BASENAME=$(basename "$FILE")
    echo "Found file: $BASENAME"

    # Extract YEAR using regex
    YEAR=$(echo "$BASENAME" | grep -oE '[0-9]{4}')
    if [[ -z "$YEAR" ]]; then
        echo "Skipping $BASENAME (year not found)"
        continue
    fi

    echo "Processing $VAR_INPUT for year: $YEAR"

    # Create output directory
    YEARLY_DAILY_DIR="${DAILY_DIR}/${VAR_INPUT}/${YEAR}"
    mkdir -p "$YEARLY_DAILY_DIR"

    TMP_DIR=$(mktemp -d)

    # Split into daily files using CDO
    echo "Splitting into true daily files using seldate..."
    cdo showdate "$FILE" | tr ' ' '\n' | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}' > "${TMP_DIR}/dates.txt"

    i=1
    while read -r DATE; do
        PADDED=$(printf "%06d" "$i")
        OUTFILE="${YEARLY_DAILY_DIR}/${VAR_INPUT}_${YEAR}_day_${PADDED}.nc"
        cdo seldate,$DATE "$FILE" "$OUTFILE"
        ((i++))
    done < "${TMP_DIR}/dates.txt"

    rm -r "$TMP_DIR"
    echo "Done with $BASENAME"
done

echo "All years processed for variable: $VAR_INPUT"
