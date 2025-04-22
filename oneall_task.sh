#!/bin/bash
#SBATCH --job-name=MODEL_EVAL
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fateema.bazzi@external.fht.org
#SBATCH --output=logs/model-%A-%a.log
#SBATCH --time=24:00:00          # Time limit



# Load your environment (e.g., modules, virtual environments)
source /home/fateema.bazzi/miniconda3/etc/profile.d/conda.sh
conda activate sklearn-env

# Read input arguments
DRUG_FILE=$1
INPUT_PATH=$2
OUTPUT_PATH=$3
GDSC_VER=$4

# Get the line corresponding to the current task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DRUG_FILE")

# Split the line into drug_name and drug_id
IFS=',' read -r DRUG_NAME DRUG_ID <<< "$LINE"

# Remove spaces from DRUG_NAME
DRUG_NAME=$(echo "$DRUG_NAME" | tr -d ' ')

# Run the Python script
python3 ./one_v_all.py -m "${OUTPUT_PATH}models/GDSC${GDSC_VER}_${DRUG_NAME}_${DRUG_ID}_elastnet_model.pkl" -g "${OUTPUT_PATH}model_genes/GDSC${GDSC_VER}_${DRUG_NAME}_${DRUG_ID}_model_genes.csv" -rF "${INPUT_PATH}rnaseq_latest.csv.gz" -dF "${INPUT_PATH}GDSC${GDSC_VER}_fitted_dose_response_27Oct23.xlsx" -mlF "${INPUT_PATH}model_list_latest.csv.gz" -oP "$OUTPUT_PATH"
