#library imports
import os
import os.path as path
import sys
import pandas as pd
import numpy as np
import sklearn
import scipy
import scipy.stats as stats
import nbformat
import itertools
import math
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from scipy.integrate import simpson
from scipy.stats import norm, shapiro, lognorm
from sklearn.preprocessing import StandardScaler
from filelock import FileLock
from helperScripts.lobico import lobico_calc
from helperScripts.figures import ic50_violin_plot, ic50_bin_sens_plot, ic50_distb_hist




def d_mode(drugdata, pdf=None, z=1.96, t=0.05):
    #calculate and display variance of the drugs if the user requests visuals
    #first calculate the variance of the ic50 of each drug
    drugs=drugdata['DRUG_NAME'].unique()
    vars=np.zeros(len(drugs))
    for i in range(len(drugs)):
        vars[i]=np.var(drugdata[drugdata['DRUG_NAME']==drugs[i]]['LN_IC50'])
    vars=pd.DataFrame(vars, index=drugs, columns=['Variation'])

    #produce the histogram
    plt.figure(figsize=(20, 20))
    sb.histplot(data=vars, x='Variation')
    plt.title("Variation of Drug IC50", fontsize=25)
    plt.xlabel("Variation", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    pdf.savefig()
    plt.close()


    sensitivities = np.zeros(shape=(drugdata.shape[0],1))
    drugs=drugdata['DRUG_NAME'].unique()
    thresholds = np.zeros(shape=(len(drugs),1))

    for d in range(len(drugs)):
        y=drugdata[drugdata['DRUG_NAME']==drugs[d]]
        threshold, distb=lobico_calc(y)
        thresholds[d] = threshold

        #calculate binary sensitivity
        idxs=(drugdata['DRUG_NAME']==drugs[d])
        #print(sensitivities[idxs].shape)
        sensitivities[idxs]=((y['LN_IC50']<threshold)*1).values.reshape(-1, 1) #*1 converts bool into 0/1

    drugdatacopy=drugdata
    drugdatacopy.insert(2, 'SENSITIVITY', sensitivities, True) 
    ic50_violin_plot(drugdatacopy, thresholds, len(thresholds), pdf)

    # Use only the first 20 rows of the data
    subset_data = drugdatacopy[drugdatacopy['DRUG_NAME'].isin(drugdatacopy['DRUG_NAME'].unique()[:20])]
    subset_thresholds = thresholds[0:20]
    ic50_violin_plot(subset_data, thresholds, len(subset_thresholds), pdf)

    return 0

def xPreproc(rnaseq):
    #drop columns not needed
    rnaseq=rnaseq.drop_duplicates(subset=['model_id', 'gene_symbol'])
    rnaseq=rnaseq[['model_id', 'gene_symbol', 'rsem_fpkm']]#rsem_tm
    
    #remove rows where there isn't an value
    rnaseq=rnaseq.dropna(subset=['rsem_fpkm'])

    #remove genes that aren't present in every cell line
    num_cell_lines=len(rnaseq['model_id'].unique())
    valcounts=rnaseq['gene_symbol'].value_counts()
    rnaseq = rnaseq[rnaseq['gene_symbol'].isin(valcounts[valcounts == num_cell_lines].index)]

    #sort by model id and gene symbol
    rnaseq=rnaseq.sort_values(by=['model_id', 'gene_symbol'])

    #restructuring the data into a matrix of fpkm values
    models=list(rnaseq['model_id'].unique())
    first_model = models[0]
    num_genes = len(rnaseq[rnaseq['model_id'] == first_model])
    gene_symbols = rnaseq[rnaseq['model_id'] == first_model]['gene_symbol'].values
    
    X=np.zeros(shape=(len(rnaseq['model_id'].unique()), num_genes))
    for i in range(len(models)):
        start=0+(i*num_genes)
        stop=num_genes+(i*num_genes)
        X[i]=rnaseq['rsem_fpkm'][rnaseq.index[range(start, stop)]]

    #creating a pandas data frame with the correct row and column names
    X_pd=pd.DataFrame(X, columns=gene_symbols, index=models)

    '''
    #Converting fpkm values into Z scores
    X_pd = X_pd + 1e-6
    X_pd=np.log2(X_pd)
    X_pd=X_pd.apply(lambda x: stats.zscore(x), axis=0)
    X_pd = X_pd.fillna(0)
    '''

    return X_pd

def yPreproc(drugdata, doi, did, visuals, pdf=None):
    """Adds Sensitivity and properly formats y data

    Parameters
    ----------
    drugdata : pandas dataframe
        IC50 values for each drug
    doi : str
        drug of interest
    visuals : int
        whether we want to produce visuals
    pdf
        file to save figures in
    
    Output
    ------
    y : pandas dataframe
        IC50s binary sensitivity for cell lines with the drug of interest
    
    """
    #subset of drugdata that only looks at the doi
    y=drugdata[drugdata['DRUG_NAME']==doi]
    #further subset that only looks at the did
    y=y[y['DRUG_ID']==int(did)]
    #get the drug name with no spaces
    doinospace=doi.replace(" ", "")
    #adjust drug name to include dataset and id
    y['DRUG_NAME'] = y['DATASET']+ "_"+ doinospace+ "_"+y['DRUG_ID'].astype(str)
    new_drug_name=y['DRUG_NAME'].iloc[0]
    #drop the id and dataset columns
    y=y.drop(columns=['DRUG_ID', 'DATASET'])

    #calculate binary threshold
    threshold, distb = lobico_calc(y)

    #visual of distribution of upsampled ic50 values with binary threshold
    if(visuals):
        plt.figure(figsize=(20, 20))
        sb.histplot(data=distb, label='_nolegend_')
        plt.title("Distribution of Upsampled IC50 Values", fontsize=25)
        plt.ylabel("Count", fontsize=20)
        plt.axvline(x=threshold, color="red", label="Threshold")
        plt.legend(fontsize=18, title='Legend', title_fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        pdf.savefig()
        plt.close()

    #calculate binary sensitivity
    sensitivity=np.zeros(shape=(y.shape[0],1))
    sensitivity=(y['LN_IC50']<threshold)*1 #*1 converts bool into 0/1
    #add sensitivity back to y
    y.insert(2, 'SENSITIVITY', sensitivity, True)

    #Visualize the IC50s colored by sensitivity
    if (visuals):
        y_sorted=y.sort_values('LN_IC50')
        #change 0 and 1 into meaningful labels for the figure
        y_sorted['SENSITIVITY']= y_sorted['SENSITIVITY'].map(str)
        for i in range(len(sensitivity)):
            if y_sorted.iloc[i, 2]=='0':
                y_sorted.iloc[i,2]="Resistant"
            else:
                y_sorted.iloc[i,2]="Sensitive"

        ic50_bin_sens_plot(y, y_sorted, pdf)

    return y, new_drug_name

def final_y(y, binary):
    """Final clean up of y data
    
    Parameters
    ----------
    y : pandas dataframe
        IC50 and binary sensitivity of doi
    binary : int
        whether the prediction will be discrete or continuous
    
    Output
    ------
    y : pandas dataframe
        the cleaned up y
    """
    y=y.drop(columns=['DRUG_NAME', 'RMSE', 'MIN_CONC', 'MAX_CONC'])
    if (binary):
        y=y.drop(columns='LN_IC50')
    else:
        y=y.drop(columns='SENSITIVITY')
    
    y.index=y['SANGER_MODEL_ID']
    y=y.drop(columns='SANGER_MODEL_ID')
    y=y.sort_index()    

    return y


def preproc(rnaseq, drugdata, cancertypes, doi, did, binary, visuals, outpath, dM, metadata=False, genes=None):
    """Formats and cleans the data to be used in machine learning models

    Parameters
    ----------
    rnaseq : pandas dataframe
        transcriptional signatures
    drugdata : pandas dataframe
        IC50 values
    cancertypes : pandas dataframe
        cancer type data
    doi : str
        drug of interest
    binary : int
        whether y should be binary
    visuals : int
        whether figures should be produced
    outvis : str
        path to save visuals pdf to
    dM : int
        whether to call developer mode

    Output
    ------
    Xtrain : pandas dataframe
    Xtest : pandas dataframe
    ytrain : pandas dataframe
    ytest : pandas datarame
    """    
    #if the user requested visuals, open the pdf
    if(visuals):
        doinospace=doi.replace(" ", "")
        dataset=drugdata['DATASET'].iloc[0]
        new_name=dataset+"_"+doinospace+"_"+did

        visfile=os.path.join(outpath, f"{new_name}.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(visfile)
    else:
        pdf = None

    #X preprocessing
    X_pd=xPreproc(rnaseq)

    #y preprocessing
    #drop unnecessary columns
    drugdata=drugdata[['DRUG_NAME', 'SANGER_MODEL_ID', 'RMSE', 'MIN_CONC', 'MAX_CONC', 'LN_IC50', 'DATASET', 'DRUG_ID']]

    #sort data by drug and cell line
    drugdata=drugdata.sort_values(by=['DRUG_NAME', 'SANGER_MODEL_ID'])

    #compute threshold for sensitivity using LOBICO method
    y, new_drug_name=yPreproc(drugdata, doi, did, visuals, pdf)

    #call the developer function
    if (dM):
        d_mode(drugdata, pdf)

    #remove rows from y not present in x and rows from x not present in y
    y=y[y['SANGER_MODEL_ID'].isin(X_pd.index)]
    X_pd=X_pd[X_pd.index.isin(y['SANGER_MODEL_ID'])]

    #final dropping of columns and sorting of y based on the type of model
    y=final_y(y, binary)
    
    #intersect cancer types with model ids from y
    cancertypes=cancertypes[cancertypes['model_id'].isin(y.index)]
    
    #if genes given as input, don't do filtering by variance and just include those genes
    #if genes not given, do filtering and autosave
    if genes is None:
        gene_vars = X_pd.var(axis=0)
        threshold = gene_vars.quantile(0.10) #Alternatively, absolute threshold of 0.1
        X_pd = X_pd.loc[:, gene_vars > threshold]
        genes=list(X_pd)
        genes=pd.DataFrame(genes)
        os.makedirs(outpath+"/model_genes/", exist_ok=True)
        genes.to_csv(outpath+"/model_genes/"+new_drug_name+'_model_genes.csv')
    else:
        X_pd=X_pd[genes['0']]


    #visualize the distribution of cancer types
    if (visuals):
        xlabsct = ["" for x in range(len(cancertypes.value_counts('cancer_type')))]
        for i in range(len(cancertypes.value_counts('cancer_type'))):
            if cancertypes.value_counts('cancer_type').iloc[i]>30:
                #add cancer type string to list/array,
                xlabsct[i]=cancertypes.value_counts('cancer_type').index[i]
                #else leave an empty string

        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=cancertypes, x='cancer_type')
        plt.xticks(xlabsct, rotation=30, ha='right')
        plt.xlabel('Cancer Types', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title("Model Cancer Types", fontsize=25)
        pdf.savefig(bbox_inches='tight')
        plt.close()



    if (not binary):
        #split the data so representative of overall distribution
        y=y.squeeze()
        y_binned = pd.qcut(y, q=10, labels=False)  # Quantile-based binning into 10 bins
        #split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X_pd, y, test_size=0.2, random_state=42, stratify=y_binned)
        y_test=y_test.to_frame()
        y_train=y_train.to_frame()

        #use a standard scaler
        y_train_unscaled=y_train
        y_scaler = StandardScaler()
        # Reshape y_train to 2D before fitting
        y_train_scaled = y_scaler.fit_transform(y_train).ravel()
        # Ensure the transformed data retains the same structure
        y_train = pd.DataFrame(y_train_scaled, columns=['LN_IC50'], index=y_train.index)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_pd, y, test_size=0.2, random_state=42, stratify=y)

        y_scaler=None

    #Converting fpkm values into Z scores
    X_train = X_train + 1e-6
    X_test = X_test + 1e-6
    X_train=np.log2(X_train)
    X_test=np.log2(X_test)
    # Compute training set mean and std for normalization
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)

    # Normalize train and test sets
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    # Replace inf and NaN
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    #gather metadata if necessary
    if (metadata):
        # Define paths
        metadata_file = os.path.join(outpath, 'metadata.csv')
        lock_file = metadata_file + '.lock'

        # Create a lock for the metadata file
        lock = FileLock(lock_file)


        with lock:
            # Check if the metadata file exists; if not, initialize the dataframe
            if os.path.exists(metadata_file):
                metadf = pd.read_csv(metadata_file, header=0)
            else:
                columns = ['drug','total_cell_lines','total_genes','total_ic50_var',
                            'train_cell_lines','train_ic50_var',
                            'test_cell_lines','test_ic50_var']
                metadf = pd.DataFrame(columns=columns)
            #metadata values
            tot_cell_lines=X_pd.shape[0]
            tot_genes=X_pd.shape[1]
            tot_var=pd.concat([y_train_unscaled, y_test], axis=0).var()
            tot_var=tot_var.iloc[0]
            train_cell_lines=X_train.shape[0]
            train_var=y_train_unscaled.var()
            train_var=train_var.iloc[0]
            test_cell_lines=X_test.shape[0]
            test_var=y_test.var()
            test_var=test_var.iloc[0]
            
            s = pd.DataFrame({'drug': [new_drug_name], 'total_cell_lines': [tot_cell_lines],
                            'total_genes': [tot_genes], 'total_ic50_var': [tot_var], 
                            'train_cell_lines': [train_cell_lines], 'train_ic50_var': [train_var],
                            'test_cell_lines': [test_cell_lines], 'test_ic50_var': [test_var]})
            metadf=pd.concat([metadf, s], ignore_index=True)
            metadf.to_csv(outpath+'metadata.csv', index=False)

    #distribution of test and train cancer types
    if(dM):
        ct=cancertypes
        ct.index=ct['model_id']
        ct.drop(columns=['model_id'])
        ct_tr=ct[ct.index.isin(X_train.index)]
        ct_ts=ct[ct.index.isin(X_test.index)]
        ct_tr=ct_tr.assign(dataset=['Train']*len(ct_tr))
        ct_ts=ct_ts.assign(dataset=['Test']*len(ct_ts))
        # Combine the datasets into a single DataFrame
        combined_ct = pd.concat([ct_tr, ct_ts])

        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=combined_ct, x='cancer_type', hue='dataset', multiple='dodge')
        plt.xticks(xlabsct, rotation=30, ha='right')
        plt.xlabel('Cancer Types', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title("Model Cancer Types", fontsize=25)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')
        plt.close()

    if(visuals and not binary):
        #display the range of ic50 values (normalized)
        #in the test and training sets
        ic50_distb_hist(y_train, pdf, test=False)
        
        ic50_distb_hist(y_test, pdf, test=True)

    #close the pdf of visuals if it was produced
    if(visuals):
        pdf.close()

    return X_train, X_test, y_train, y_test, y_scaler, new_drug_name

