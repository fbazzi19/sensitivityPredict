#User params
INPUT_PATH=$1
OUTPUT_PATH=$2
GDSC_VER=$3

# Load your environment (e.g., modules, virtual environments)
source /home/fateema.bazzi/miniconda3/etc/profile.d/conda.sh
conda activate sklearn-env

#script to get each drug
python3 ./helperScripts/extract_drugs.py -d "${INPUT_PATH}GDSC${GDSC_VER}_fitted_dose_response_27Oct23.xlsx" -oP "${OUTPUT_PATH}"
DRUG_FILE="${OUTPUT_PATH}drugs_list.txt"
#number of drugs
NUM_ROWS=$(wc -l < "$DRUG_FILE")
#sbatch call
sbatch  --array=1-"$NUM_ROWS"%50 ./oneall_task.sh "$DRUG_FILE" "$INPUT_PATH" "$OUTPUT_PATH" "$GDSC_VER"