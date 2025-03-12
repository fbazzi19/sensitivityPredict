#User params
INPUT_PATH=$1
OUTPUT_PATH=$2
GDSC_VER=$3

#script to get each drug
python3 ./helperScripts/extract_drugs.py -d "${INPUT_PATH}${GDSC_VER}_fitted_dose_response_27Oct23" -oP "${OUTPUT_PATH}"
DRUG_FILE="${OUTPUT_PATH}drugs_list.txt"
#number of drugs
NUM_ROWS=$(wc -l < "$DRUG_FILE")
#sbatch call
sbatch  --array=1-"$NUM_ROWS"%50 ./model_task.sh "$DRUG_FILE" "$INPUT_PATH" "$OUTPUT_PATH" "$GDSC_VER"

