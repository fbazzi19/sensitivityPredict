#library imports
import os
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from multiprocessing import Pool

from ./helperScripts/data_preproc import preproc
from ./modelScripts/class_mods import classificationModels
from ./modelScripts/reg_mods import regressionModels

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument("-rf","--rnafile", help="path/to/rnaseqfile.csv", type=str, required=True)
    parser.add_argument("-df","--drugfile", help="path/to/drugfile.xlsx", type=str, required=True)
    parser.add_argument("-mf","--modelListfile", help="path/to/modelListfile.csv", type=str, required=True)
    parser.add_argument("-dOI", "--drugOfInterest", help="Drug of Interest", type=str, required=True)
    parser.add_argument("-dID", "--drugID", help="ID of Drug", type=str, required=True)
        parser.add_argument("-oP", "--outputPath", help="Location to store any output", type=str, required=True)


    # Optional arguments
    parser.add_argument("-b", "--binary", help="Option to have a binary y", type=int, default=0)
    parser.add_argument("-v", "--visuals", help="Option to produce visuals", type=int, default=0)
    parser.add_argument("-dM", "--developerMode", help="Whether to look at optional developer functions", type=int, default=0)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    #check to make sure the files are of the correct format
    if args.rnafile.split(".")[-1] != "csv":
        sys.exit("The gene expression file should be a csv file")

    if args.drugfile.split(".")[-1] != "xlsx":
        sys.exit("The second drug sensitivity file should be a xlsx file")
    
    if args.modelListfile.split(".")[-1] != "csv":
        sys.exit("The cell line information file should be a csv file")

    #if in developer mode, have to have visuals on and only look at one drug
    if (args.developerMode and not args.visuals):
        sys.exit("Developer mode requires you have visuals on")
    #ensure output path is a path
    if(args.outputPath[-1]!="/")
        sys.exit("The output path is invalid (should end in '/').")
    #ensure 0/1 value for binary, visuals, and developer mode
    if(args.binary!=0 and args.binary!=1):
        sys.exit("Invalid binary parameter. Specify 0 (false) or 1 (true).")
    if(args.visuals!=0 and args.visuals!=1):
        sys.exit("Invalid visuals parameter. Specify 0 (false) or 1 (true).")
    if(args.developerMode!=0 and args.developerMode!=1):
        sys.exit("Invalid developer mode parameter. Specify 0 (false) or 1 (true).")

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
    X_train, X_test, y_train, y_test, y_scaler, new_doi = preproc(rnaseq, drugdata, cancertypes, args.drugOfInterest, args.drugID, args.binary, args.visuals, args.outputPath, args.developerMode)
    if(args.visuals):
        y_train.to_csv(args.outputPath+new_doi+'_y_train_set.csv')
    if(args.binary):
        print(classificationModels(X_train, X_test, y_train, y_test, new_doi, args.visuals, args.outputPath))
    else:
        print(regressionModels(X_train, X_test, y_train, y_test, y_scaler, new_doi, args.visuals, args.outputPath))
 
    print("Model Created Successfully")
