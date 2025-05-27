#library imports
import os
import pandas as pd
import numpy as np
import sklearn
import scipy
import nbformat
import itertools
import math
import sys
import argparse
import joblib

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

from helperScripts.data_preproc import preproc
from helperScripts.dataLoadIn import dataLoad

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    #arguments
    parser.add_argument("-m","--model", help="path/to/model.pkl", type=str, required=True)
    parser.add_argument("-g","--genes", help="path/to/genes.csv", type=str, required=True)
    parser.add_argument("-oP","--outputPath", help="Location to store any output", type=str, required=True)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    #check to make sure the files are of the correct format
    if args.genes.split(".")[-1] != "csv":
        sys.exit("The list of genes should be a csv file")
    if args.model.split(".")[-1] != "pkl":
        sys.exit("The model should be a pkl file")
    if(args.gdscVer!=1 and args.gdscVer!=2):
        sys.exit("Invalid GDSC version. Specify 1 or 2.")

    #ensure output path is a path
    if(args.outputPath[-1]!="/"):
        sys.exit("The output path is invalid (should end in '/').")

    return args

if __name__=="__main__":
    #load in user specified parameters
    # Parse the arguments
    args = parseArguments()

    #load in model
    model = joblib.load(args.model)
    #load in model feature names
    genes = pd.read_csv(args.genes)

    #load in basal transcription data
    rnaseq = rnaseq = dataLoad("https://cog.sanger.ac.uk/cmp/download/rnaseq_latest.csv.gz", "rna")
    #load in IC50 data
    drugdata_one=dataLoad("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx", "drug")
    drugdata_two=dataLoad("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx", "drug")

    #name of drug model is for
    model_drug_name=args.model[0:[pos for pos, char in enumerate(args.model) if char == "_"][-2]]
    model_drug_name=model_drug_name.split("/")[-1]
    
    #set random seed for reproducibility
    np.random.seed(42)

    # Ensure the outpath directory exists
    outpath=args.outputPath+"one_all_metrics/"
    os.makedirs(outpath, exist_ok=True)

    #for each drug, preprocess the data (not binary) and use the model on resulting data
    #train and test data not separated, just predict for all the data
    #note r2, rmse, and pcorr for each drug
    metrics=[]
    drug_names=[]
    
    for drugdata in [drugdata_one, drugdata_two]:
        drugs=drugdata['DRUG_NAME'].unique()
        for doi in drugs:
            #retrieve all unique drug_ids associated with specified drug name
            drugdata_subset=drugdata[drugdata['DRUG_NAME']==doi]
            drug_ids=drugdata_subset['DRUG_ID'].unique()
            for did in drug_ids:
                X_train, X_test, y_train, y_test, y_scaler, new_doi = preproc(rnaseq, drugdata, 
                                                                                doi, did, binary=0, 
                                                                                visuals=0, outpath=outpath, 
                                                                                dM=0, metadata=False, 
                                                                                genes=genes)

                #predict y values
                y_pred=model.predict(X_test)

                #un-normalize the values
                y_pred_reshaped = y_pred.reshape(-1, 1)
                y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
                y_pred_unnorm = y_pred_unnorm.flatten()
                #r2 score
                r2 = r2_score(y_test.values.ravel(), y_pred_unnorm)
                #rmse
                rmse_un=root_mean_squared_error(y_test.values.ravel(), y_pred_unnorm)

                #calculate pearson correlation
                correlation = np.corrcoef(y_test.values.ravel(), y_pred_unnorm)[0, 1]
                metrics.append([r2, rmse_un, correlation])
                drug_names.append(new_doi)

    metrics_df=pd.DataFrame(metrics, index=drug_names, columns=['R2', 'RMSE', 'Pearson Correlation'])
    #send metrics to csv to make visuals with R
    metrics_df.to_csv(outpath+''+model_drug_name+'_oneall_metrics.csv')
    
    print("Process completed for drug model "+ model_drug_name)
