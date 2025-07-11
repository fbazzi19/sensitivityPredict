#!/bin/bash
#SBATCH --job-name=DRUG_MODEL
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --output=logs/model-%A-%a.log
#SBATCH --time=24:00:00          # Time limit

# Read input arguments
DRUG_FILE=$1
OUTPUT_PATH=$2
GDSC_VER=$3
METADATA=$4
BINARY=$5
EMAIL=$6
CONDA_PATH=$7

# Dynamically update --mail-user
#SBATCH --mail-user="$EMAIL"

# Load your environment (e.g., modules, virtual environments)
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate sklearn-env

# Get the line corresponding to the current task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DRUG_FILE")

# Use Python to parse the line into DRUG_NAME and DRUG_ID
PARSED_LINE=$(python3 ./helperScripts/line_parser.py "$LINE")

# Split the output into DRUG_NAME and DRUG_ID
DRUG_NAME=$(echo "$PARSED_LINE" | sed -n '1p')
DRUG_ID=$(echo "$PARSED_LINE" | sed -n '2p')

# Run the Python script
python3 ./workflow.py -gV "${GDSC_VER}" -dOI "$DRUG_NAME" -dID "$DRUG_ID" -oP "$OUTPUT_PATH" -m "$METADATA" -b "$BINARY"
