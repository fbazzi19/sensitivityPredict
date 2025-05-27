import pandas as pd
import os
import argparse

from dataLoadIn import dataLoad

if __name__=="__main__":
    #take in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-oP", "--outputPath", help="Location to store drug list", type=str, required=True)
    parser.add_argument("-gV", "--gdscVer", help="Which GDSC version", type=int, default=2)
    args = parser.parse_args()

    #ensure gdsc version is valid
    if(args.gdscVer!=1 and args.gdscVer!=2):
        sys.exit("Invalid GDSC version. Specify 1 or 2.")
    #ensure output path is valid
    if(args.outputPath[-1]!="/"):
        sys.exit("The output path is invalid (should end in '/').")

    #load in data
    if(args.gdscVer==2):
        drugdata=dataLoad("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx", "drug")
    else:
        drugdata=dataLoad("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx", "drug")

    #find each unique drug name and id pair
    drugdata_subset = drugdata[['DRUG_NAME', 'DRUG_ID']].drop_duplicates()

    #output file name
    task_file = args.outputPath+"drugs_list_"+str(args.gdscVerVersion)+".txt"

    # Ensure the outpath directory exists
    os.makedirs(args.outputPath, exist_ok=True)

    #write drugs to file
    with open(task_file, "w") as f:
        for _, row in drugdata_subset.iterrows():
            f.write(f"{row['DRUG_NAME']},{row['DRUG_ID']}\n")

    print(f"Task file {task_file} created with {len(drugdata_subset)} tasks.")
