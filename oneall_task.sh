#!/bin/bash
#SBATCH --job-name=MODEL_EVAL
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
EMAIL=$4
CONDA_PATH=$5

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

# Replace spaces and commas in DRUG_NAME with a dash
DRUG_NAME=$(echo "$DRUG_NAME" | sed 's/[ ,/]/-/g')

# Run the Python script
python3 ./one_v_all.py -m "${OUTPUT_PATH}models/GDSC${GDSC_VER}_${DRUG_NAME}_${DRUG_ID}_elastnet_model.pkl" -g "${OUTPUT_PATH}model_genes/GDSC${GDSC_VER}_${DRUG_NAME}_${DRUG_ID}_model_genes.csv" -oP "$OUTPUT_PATH"
