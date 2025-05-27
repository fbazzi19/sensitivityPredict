#User params
OUTPUT_PATH=$1
GDSC_VER=$2
EMAIL=$3
CONDA_PATH=$4

# Load your environment (e.g., modules, virtual environments)
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate sklearn-env

#script to get each drug
python3 ./helperScripts/extract_drugs.py -gV "${GDSC_VER}" -oP "${OUTPUT_PATH}"
DRUG_FILE="${OUTPUT_PATH}drugs_list_${GDSC_VER}.txt"
#number of drugs
NUM_ROWS=$(wc -l < "$DRUG_FILE")
#sbatch call
sbatch  --array=1-"$NUM_ROWS"%50 ./oneall_task.sh "$DRUG_FILE" "$OUTPUT_PATH" "$GDSC_VER" "$EMAIL" "$CONDA_PATH"
