#!/bin/bash
#SBATCH --job-name=DRUG_MODEL
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

# Run the Python script
python3 ./workflow.py "${INPUT_PATH}rnaseq_all_data_20220624.csv" "${INPUT_PATH}${GDSC_VER}_fitted_dose_response_27Oct23.xlsx" "${INPUT_PATH}model_list_20241120.csv" -dOI "$DRUG_NAME" -dID "$DRUG_ID" -oP "$OUTPUT_PATH"
