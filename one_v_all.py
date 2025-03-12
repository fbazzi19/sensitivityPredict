#library imports
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import sklearn
import scipy
import nbformat
import itertools
import math
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import sys
import argparse
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.manifold import TSNE
from scipy.integrate import simpson
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, classification_report, precision_recall_curve, auc, f1_score, root_mean_squared_error

from data_preproc import preproc

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model", help="path/to/model.pkl", type=str)
    parser.add_argument("genes", help="path/to/genes.csv", type=str)
    parser.add_argument("rnafile", help="path/to/rnaseqfile.csv", type=str)
    parser.add_argument("drugfile", help="path/to/drugfile.xlsx", type=str)
    parser.add_argument("modelListfile", help="path/to/modelListfile.csv", type=str)


    # Optional arguments
    parser.add_argument("-v", "--visuals", help="Option to produce visuals", type=int, default=0)
    parser.add_argument("-oP", "--outputPath", help="Location to store any output", type=str, default="./")

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    #check to make sure the files are of the correct format
    if args.genes.split(".")[-1] != "csv":
        sys.exit("The second parameter should be a csv file")

    if args.rnafile.split(".")[-1] != "csv":
        sys.exit("The third parameter should be a csv file")

    if args.drugfile.split(".")[-1] != "xlsx":
        sys.exit("The fourth parameter should be a xlsx file")
    
    if args.modelListfile.split(".")[-1] != "csv":
        sys.exit("The fifth parameter should be a csv file")

    if args.model.split(".")[-1] != "pkl":
        sys.exit("The first parameter should be a pkl file")

    #TODO add other checks:
    #vP needs to be a file path, not a file

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
    rnaseq = pd.read_csv(args.rnafile)
    #load in IC50 data
    drugdata=pd.read_excel(args.drugfile)
    #load in cancer type data
    cancertypes=pd.read_csv(args.modelListfile)
    cancertypes=cancertypes[['model_id', 'cancer_type']]

    #name of drug model is for
    model_drug_name=args.model[0:[pos for pos, char in enumerate(args.model) if char == "_"][-2]]
    model_drug_name=model_drug_name.split("/")[-1]
    
    #set random seed for reproducibility
    np.random.seed(42)

    # Ensure the outpath directory exists
    os.makedirs(args.outputPath, exist_ok=True)

    #TODO: for each drug, preprocess the data (not binary) and use the model on resulting data
    #train and test data not separated, just predict for all the data
    #note r2, rmse, and pcorr for each drug
    metrics=[]
    drug_names=[]
    drugs=drugdata['DRUG_NAME'].unique()
    for doi in drugs:
        #retrieve all unique drug_ids associated with specified drug name
        drugdata_subset=drugdata[drugdata['DRUG_NAME']==doi]
        drug_ids=drugdata_subset['DRUG_ID'].unique()
        for did in drug_ids:
            X_train, X_test, y_train, y_test, y_scaler, new_doi = preproc(rnaseq, drugdata, 
                                                                            cancertypes, doi, 
                                                                            did, binary=0, visuals=0,
                                                                            outpath=args.outputPath, 
                                                                            dM=0, genes=genes)
            X_all=pd.concat([X_train, X_test], axis=0)
            y_all=pd.concat([y_train, y_test], axis=0)

            #predict y values
            y_pred=model.predict(X_all)

            #r2 score
            r2 = r2_score(y_all.values.ravel(), y_pred)

            #un-normalize the values
            y_all_unnorm= y_scaler.inverse_transform(y_all)
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
            y_pred_unnorm = y_pred_unnorm.flatten()
            #rmse
            rmse_un=root_mean_squared_error(y_all_unnorm, y_pred_unnorm)

            #calculate pearson correlation
            correlation = np.corrcoef(y_all_unnorm.ravel(), y_pred_unnorm)[0, 1]

            metrics.append([r2, rmse_un, correlation])
            drug_names.append(new_doi)

            
    metrics_df=pd.DataFrame(metrics, index=drug_names, columns=['R2', 'RMSE', 'Pearson Correlation'])
    #send metrics to csv to try making figures with R
    metrics_df.to_csv(args.outputPath+model_drug_name+'_oneall_metrics.csv')
    
    print("Hooray! :)")