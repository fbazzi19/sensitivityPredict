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
import multiprocessing

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
from multiprocessing import Pool, cpu_count
from threadpoolctl import threadpool_limits

from helperScripts.data_preproc import preproc

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    #arguments
    parser.add_argument("-oP","--outputPath", help="Location to store any output", type=str, required=True)
    parser.add_argument("-mP", "--modelPath", help="Location of models", type=str, required=True)
    parser.add_argument("-rF","--rnaFile", help="path/to/rnaseqfile.csv", type=str, required=True)
    parser.add_argument("-dF","--drugFile", help="path/to/drugfile.xlsx", type=str, required=True)
    parser.add_argument("-mlF", "--modelListFile", help="path/to/modelListfile.csv", type=str, required=True)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    #check to make sure the files are of the correct format
    #if args.rnaFile.split(".")[-1] != "csv":
    #    sys.exit("The gene expression data should be a csv file")
    if args.drugFile.split(".")[-1] != "xlsx":
        sys.exit("The drug sensitivity data should be a xlsx file")
    #if args.modelListFile.split(".")[-1] != "csv":
    #    sys.exit("The cell line data should be a csv file")
    #ensure output path is a path
    if(args.outputPath[-1]!="/"):
        sys.exit("The output path is invalid (should end in '/').")
    if(args.modelPath[-1]!="/"):
        sys.exit("The model path is invalid (should end in '/').")

    return args

if __name__=="__main__":
    #load in user specified parameters
    # Parse the arguments
    args = parseArguments()
    #load in basal transcription data
    rnaseq = pd.read_csv(args.rnaFile)
    #load in IC50 data
    drugdata=pd.read_excel(args.drugFile)
    #load in cancer type data
    cancertypes=pd.read_csv(args.modelListFile)
    cancertypes=cancertypes[['model_id', 'cancer_type']]
    
    #set random seed for reproducibility
    np.random.seed(42)

    # Ensure the outpath directory exists
    outpath=args.outputPath+"all_metrics/"
    os.makedirs(outpath, exist_ok=True)

    metrics=[]
    drug_names=[]
    drugs=drugdata['DRUG_NAME'].unique()
    for doi in drugs:
        #retrieve all unique drug_ids associated with specified drug name
        drugdata_subset=drugdata[drugdata['DRUG_NAME']==doi]
        drug_ids=drugdata_subset['DRUG_ID'].unique()
        for did in drug_ids:
            print("starting drug "+doi+" "+str(did))
            doinospace=doi.replace(" ", "")
            model=joblib.load(args.modelPath+"GDSC2_"+doinospace+"_"+str(did)+"_elastnet_model.pkl")
            X_train, X_test, y_train, y_test, y_scaler, new_doi = preproc(rnaseq, drugdata, cancertypes, 
                                                                            doi, did, binary=0,
                                                                            visuals=0, outpath=outpath, dM=0,
                                                                            metadata=False, genes=None)

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
    #send metrics to csv to try making figures with R
    metrics_df.to_csv(outpath+'all_metrics.csv')
    
    print("Hooray! :)")