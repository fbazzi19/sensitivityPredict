#library imports
import os
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
from lobico import lobico_calc



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
    
    #TODO: add some statistics related to this plot, like quartile or something
    plt.figure(figsize=(20, 20))
    sb.violinplot(x='DRUG_NAME', y='LN_IC50', data= drugdatacopy)#, inner='point') 
    # Add horizontal lines
    for i in range(len(thresholds)):
        # Draw a horizontal line
        plt.hlines(y=thresholds[i], xmin=i - 0.4, xmax=i + 0.4, color='red', linewidth=2)

    plt.title("IC50 Distributions", fontsize=25)
    plt.xticks([])
    plt.xlabel("Drugs", fontsize=20)
    plt.ylabel("ln(IC50)", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    pdf.savefig()
    plt.close()

    # Use only the first 20 rows of the data
    subset_data = drugdatacopy[drugdatacopy['DRUG_NAME'].isin(drugdatacopy['DRUG_NAME'].unique()[:20])]
    subset_thresholds = thresholds[0:20]
    plt.figure(figsize=(20, 20))
    sb.violinplot(x='DRUG_NAME', y='LN_IC50', data= subset_data)#, inner='point') 
    # Add horizontal lines
    for i in range(len(subset_thresholds)):
        # Draw a horizontal line
        plt.hlines(y=thresholds[i], xmin=i - 0.4, xmax=i + 0.4, color='red', linewidth=2)

    plt.title("IC50 Distributions", fontsize=25)
    plt.xticks([])
    plt.xlabel("Drugs", fontsize=20)
    plt.ylabel("ln(IC50)", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    pdf.savefig()
    plt.close()

    return 0

def xPreproc(rnaseq):
    #drop columns not needed
    rnaseq=rnaseq[['model_id', 'gene_symbol', 'fpkm']]
    
    #remove genes that aren't present in every cell line
    num_cell_lines=len(rnaseq['model_id'].unique())
    valcounts=rnaseq['gene_symbol'].value_counts()
    rnaseq = rnaseq[rnaseq['gene_symbol'].isin(valcounts[valcounts == num_cell_lines].index)]


    #sort by model id and gene symbol
    rnaseq=rnaseq.sort_values(by=['model_id', 'gene_symbol'])

    #restructuring the data into a matrix of tpm values
    models=list(rnaseq['model_id'].unique())
    first_model = models[0]
    num_genes = len(rnaseq[rnaseq['model_id'] == first_model])
    gene_symbols = rnaseq[rnaseq['model_id'] == first_model]['gene_symbol'].values
    
    X=np.zeros(shape=(len(rnaseq['model_id'].unique()), num_genes))
    for i in range(len(models)):
        start=0+(i*num_genes)
        stop=num_genes+(i*num_genes)
        X[i]=rnaseq['fpkm'][rnaseq.index[range(start, stop)]]


    #creating a pandas data frame with the correct row and column names
    X_pd=pd.DataFrame(X, columns=gene_symbols, index=models)

    #TODO: convert TPM/fpkm into z-scores? if so, remove normalization
    X_pd = X_pd + 1e-6
    X_pd=np.log2(X_pd)
    X_pd=X_pd.apply(lambda x: stats.zscore(x), axis=0)


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
    y=y[y['DRUG_ID']==did]
    #adjust drug name to include dataset and id
    y['DRUG_NAME'] = y['DATASET']+ "_"+ y['DRUG_NAME']+ "_"+y['DRUG_ID'].astype(str)
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
        plt.figure(figsize=(20, 20))
        sb.scatterplot(data=y_sorted, x='SANGER_MODEL_ID', y='LN_IC50', hue='SENSITIVITY')
        plt.title("IC50 Values Colored by Sensitivity", fontsize=25)
        #since there is more than one min and max concentration for each drug, here I take the one that 
        #occurs the most in the data
        plt.axhline(y=np.log(y['MIN_CONC'].value_counts().idxmax()), linestyle="dashed", color="red",
                        label='_nolegend_')
        plt.axhline(y=np.log(y['MAX_CONC'].value_counts().idxmax()), linestyle="dashed", color="red",
                        label='_nolegend_')
        plt.xticks([])
        plt.xlabel("Cell Lines", fontsize=20)
        plt.ylabel("ln(IC50)", fontsize=20)
        plt.legend(fontsize=18, title='Sensitivity', title_fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        pdf.savefig()
        plt.close()

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


def preproc(rnaseq, drugdata, cancertypes, doi, did, binary, visuals, outpath, dM):
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
        new_name=dataset+"_"+doinospace+"_"+did.astype(str)

        visfile=os.path.join(outpath, f"{new_name}.pdf")
        #plt.style.use('seaborn')
        pdf = matplotlib.backends.backend_pdf.PdfPages(visfile)
    else:
        pdf = None

    #X preprocessing
    X_pd=xPreproc(rnaseq)

    #y preprocessing
    #drop unnecessary columns
    drugdata=drugdata[['DRUG_NAME', 'SANGER_MODEL_ID', 'RMSE', 'MIN_CONC', 'MAX_CONC', 'LN_IC50', 'DATASET', 'DRUG_ID']]
    #'DATASET', 'DRUG_ID',

    #sort data by drug and cell line
    drugdata=drugdata.sort_values(by=['DRUG_NAME', 'SANGER_MODEL_ID'])

    #compute threshold for sensitivity using LOBICO method
    y, new_drug_name=yPreproc(drugdata, doi, did, visuals, pdf)

    #call the developer function
    if (dM):
        d_mode(drugdata, pdf)

    #remove rows from y not present in x and rows from x not present in y
    y=y[y['SANGER_MODEL_ID'].isin(rnaseq['model_id'])]
    X_pd=X_pd[X_pd.index.isin(y['SANGER_MODEL_ID'])]

    #final dropping of columns and sorting of y based on the type of model
    y=final_y(y, binary)
    #print(y.value_counts('SANGER_MODEL_ID')[y.value_counts('SANGER_MODEL_ID')>1])
    
    #Should not be needed anymore
    # remove any duplicate cell lines from y
    #mask = y.index.duplicated(keep='last')
    #y = y[~mask]

    
    #intersect cancer types with model ids from y
    cancertypes=cancertypes[cancertypes['model_id'].isin(y.index)]

    #TODO: Apply variance filtering to the TPM values (before normalization)
    gene_vars = X_pd.var(axis=0)
    threshold = gene_vars.quantile(0.10) #TODO: also try with quantile filtering
    X_pd = X_pd.loc[:, gene_vars > threshold]

    #Normalize TPM Values using log2 method
    #X_pd=np.log2(X_pd+1)
    #x_cols=list(X_pd)
    #print(X_pd.head())
    #x_scaler = StandardScaler()
    #x_scaler.fit(X_pd)
    #X_pd=x_scaler.transform(X_pd)
    #X_pd=pd.DataFrame(X_pd, columns=x_cols, index=y.index)
    #print(X_pd.head())


    #Normalize IC50 Values, if not binary
    if (not binary):
        #use a standard scaler
        #print(y.head())
        y_scaler = StandardScaler()
        y_scaler.fit(y)
        y=y_scaler.transform(y)
        y=pd.DataFrame(y, columns=['LN_IC50'], index=X_pd.index)
        #print(y.head())
    else:
        y_scaler=None
    

    #visualize the distribution of cancer types
    if (visuals):
        #cancertypes.value_counts('cancer_type').index[0]
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

    #split the data so cancer types are representative of overall distribution
    #split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y, stratify=cancertypes['cancer_type'], test_size=0.2, random_state=42)

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
        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=y_train, x='LN_IC50')
        #plt.xlabel('Cancer Types', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        #plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title("IC50 Values in Training Set", fontsize=25)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=y_test, x='LN_IC50')
        #plt.xlabel('Cancer Types', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        #plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title("IC50 Values in Training Set", fontsize=25)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    #close the pdf of visuals if it was produced
    if(visuals):
        pdf.close()

    return X_train, X_test, y_train, y_test, y_scaler, new_drug_name

