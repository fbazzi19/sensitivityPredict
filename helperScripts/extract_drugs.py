import pandas as pd
import os
import argparse

if __name__=="__main__":
    #take in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--drugdata", required=True, help="path/to/drugdata.xlsx", type=str)
    parser.add_argument("-oP", "--outputPath", help="Location to store drug list", type=str, required=True)
    args = parser.parse_args()

    #find each unique drug name and id pair
    drugdata=pd.read_excel(args.drugdata)
    drugdata_subset = drugdata[['DRUG_NAME', 'DRUG_ID']].drop_duplicates()
    #drugdata_subset['DRUG_NAME']=drugdata_subset['DRUG_NAME'].str.replace(" ", "")
    #output file name
    task_file = args.outputPath+"drugs_list.txt"

    # Ensure the outpath directory exists
    os.makedirs(args.outputPath, exist_ok=True)

    #write drugs to file
    with open(task_file, "w") as f:
        for _, row in drugdata_subset.iterrows():
            f.write(f"{row['DRUG_NAME']},{row['DRUG_ID']}\n")

    print(f"Task file {task_file} created with {len(drugdata_subset)} tasks.")
