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
import sys
import argparse

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.manifold import TSNE
from scipy.integrate import simpson
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from data_preproc import preproc
from class_mods import classificationModels
from reg_mods import regressionModels

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("rnafile", help="path/to/rnaseqfile.csv", type=str)
    parser.add_argument("drugfile", help="path/to/drugfile.xlsx", type=str)
    parser.add_argument("modelListfile", help="path/to/modelListfile.csv", type=str)


    # Optional arguments
    parser.add_argument("-dOI", "--drugOfInterest", help="Drug of Interest", type=str, default="")
    parser.add_argument("-b", "--binary", help="Option to have a binary y", type=int, default=0)
    parser.add_argument("-v", "--visuals", help="Option to produce visuals", type=int, default=0)
    parser.add_argument("-oP", "--outputPath", help="Location to store any output", type=str, default="./")
    parser.add_argument("-dM", "--developerMode", help="Whether to look at optional developer functions", type=int, default=0)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    #check to make sure the files are of the correct format
    if args.rnafile.split(".")[-1] != "csv":
        sys.exit("The first parameter should be a csv file")

    if args.drugfile.split(".")[-1] != "xlsx":
        sys.exit("The second parameter should be a xlsx file")
    
    if args.modelListfile.split(".")[-1] != "csv":
        sys.exit("The third parameter should be a csv file")

    #TODO add other checks:
    #if in developer mode, have to have visuals on and only look at one drug
    if (args.developerMode and not args.visuals):
        sys.exit("Developer mode requires you have visuals on")
    if (args.developerMode and args.drugOfInterest==""):
        sys.exit("Developer mode requires the specification of one drug")
    if (args.visuals and args.drugOfInterest==""):
        sys.exit("Visuals are only available for a single drug run")
    #vP needs to be a file path, not a file

    return args


if __name__=="__main__":
    #load in user specified parameters
    # Parse the arguments
    args = parseArguments()


    #load in basal transcription data
    rnaseq = pd.read_csv(args.rnafile)
    #load in IC50 data
    drugdata=pd.read_excel(args.drugfile)
    #load in cancer type data
    cancertypes=pd.read_csv(args.modelListfile)
    cancertypes=cancertypes[['model_id', 'cancer_type']]
    
    #set random seed for reproducibility
    np.random.seed(42)

    # Ensure the outpath directory exists
    os.makedirs(args.outputPath, exist_ok=True)

    #preprocess data
    #option 1: one drug is specified
    if args.drugOfInterest !="":
        #retrieve all unique drug_ids associated with specified drug name
        drugdata_subset=drugdata[drugdata['DRUG_NAME']==args.drugOfInterest]
        drug_ids=drugdata_subset['DRUG_ID'].unique()
        for did in drug_ids:
            X_train, X_test, y_train, y_test, y_scaler, new_doi = preproc(rnaseq, drugdata, cancertypes, args.drugOfInterest, did, args.binary, args.visuals, args.outputPath, args.developerMode)
            y_train.to_csv(args.outputPath+new_doi+'_y_train_set.csv')
            if(args.binary):
                print(classificationModels(X_train, X_test, y_train, y_test, new_doi, args.visuals, args.outputPath))
            else:
                print(regressionModels(X_train, X_test, y_train, y_test, y_scaler, new_doi, args.visuals, args.outputPath))
        
        print("yay")
    else: #TODO: adjust what is returned and how. instead of pdfs, pickled models?
        drugs=drugdata['DRUG_NAME'].unique()
        for doi in drugs:
            #retrieve all unique drug_ids associated with specified drug name
            drugdata_subset=drugdata[drugdata['DRUG_NAME']==doi]
            drug_ids=drugdata_subset['DRUG_ID'].unique()
            for did in drug_ids:
                X_train, X_test, y_train, y_test, y_scaler, new_doi = preproc(rnaseq, drugdata, cancertypes, doi, did, args.binary, args.visuals, args.outputPath, args.developerMode)
                if(args.binary):
                    print(classificationModels(X_train, X_test, y_train, y_test, new_doi, args.visuals, args.outputPath))
                else:
                    print(regressionModels(X_train, X_test, y_train, y_test, y_scaler, new_doi, args.visuals, args.outputPath))
            
        print("Hooray! :)")