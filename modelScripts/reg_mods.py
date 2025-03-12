#library imports
import os
import pandas as pd
import numpy as np
import sklearn
import math
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MaxAbsScaler
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, classification_report, precision_recall_curve, auc, f1_score, root_mean_squared_error
from ../helperScripts/cross_val import cross_val
from ../helperScripts/shuffle_eval import shuffle_eval
from ../helperScripts/figures import scatt_plot, reg_txt_pg

def linreg(X_train, X_test, y_train, y_test, y_scaler, visuals, pdf=None):
    model=LinearRegression()
    #print(y_train)

    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=0, y_scaler=y_scaler)
    # Calculate average and standard deviation of metrics
    rmse_values = [metric['rmse'] for metric in metrics]
    pearson_values = [metric['pearson_corr'] for metric in metrics]
    avg_rmse_val = np.mean(rmse_values)
    std_rmse_val = np.std(rmse_values)
    avg_rmse="Average RMSE over 50 splits: "+str(avg_rmse_val)
    std_rmse="Standard Deviation of RMSE: "+str(std_rmse_val)
    avg_pcorr_val = np.mean(pearson_values)
    std_pcorr_val = np.std(pearson_values)
    avg_pcorr="Average Pearson Correlation over 50 splits: "+str(avg_pcorr_val)
    std_pcorr="Standard Deviation of Pearson Correlation: "+str(std_pcorr_val)

    #fit the model
    model.fit(X_train, y_train.values.ravel())
    #the coefficients
    coefs= model.coef_
    #predict the values
    y_pred=model.predict(X_test)
    #r2 score
    r2 = r2_score(y_test.values.ravel(), y_pred)
    r2_str="R2 score: "+str(r2)

    #un-normalize the values
    y_test_unnorm= y_scaler.inverse_transform(y_test)
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
    y_pred_unnorm = y_pred_unnorm.flatten()
    #rmse
    rmse_un=root_mean_squared_error(y_test_unnorm, y_pred_unnorm)
    rmse_str="RMSE: "+str(rmse_un)

    #calculate pearson correlation
    correlation = np.corrcoef(y_test_unnorm.ravel(), y_pred_unnorm)[0, 1]
    pcorr_str = "Pearson correlation: "+ str(correlation)

    if(visuals):
        # Add a page for print statements (text)
        reg_txt_pg(avg_rmse, std_rmse, avg_pcorr, std_pcorr, r2_str, rmse_str, pcorr_str, pdf, "Linear Regression")

        scatt_plot(y_test.values.ravel(), y_pred, pdf, normalized=True)

        scatt_plot(y_test_unnorm, y_pred_unnorm, pdf, normalized=False)

    return model

def elastnet(X_train, X_test, y_train, y_test, y_scaler, visuals, doi, outpath, pdf=None):
    model=ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42, selection='random')
    param_grid = {'alpha': [0.0001, 0.001,0.01,0.1,1,10,100],
                'l1_ratio': np.linspace(0.01, 1, 10),
                'max_iter': [10000],
                'random_state': [42],
                'selection': ['random']} #not using cyclic because it doesnt converge even at 50000
    
    #create a scorer and conduct cross validation
    scorer=sklearn.metrics.make_scorer(root_mean_squared_error, greater_is_better=False)
    best_params=cross_val(X_train, y_train, model, param_grid, scorer, binary=0, modeltype="Elastic Net")
    model=ElasticNet(**best_params)

    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=0, y_scaler=y_scaler)
    # Calculate average and standard deviation of metrics
    rmse_values = [metric['rmse'] for metric in metrics]
    pearson_values = [metric['pearson_corr'] for metric in metrics]
    avg_rmse_val = np.mean(rmse_values)
    std_rmse_val = np.std(rmse_values)
    avg_rmse="Average RMSE over 50 splits: "+str(avg_rmse_val)
    std_rmse="Standard Deviation of RMSE: "+str(std_rmse_val)
    avg_pcorr_val = np.mean(pearson_values)
    std_pcorr_val = np.std(pearson_values)
    avg_pcorr="Average Pearson Correlation over 50 splits: "+str(avg_pcorr_val)
    std_pcorr="Standard Deviation of Pearson Correlation: "+str(std_pcorr_val)

    model.fit(X_train, y_train.values.ravel())
    if(visuals):
        #get the coefficients of the model
        coefs=model.coef_
        coefs_named=pd.Series(data=coefs, index=X_test.columns.tolist(), name="Coefficients")
        # Sort the Series by absolute value
        coefs_named = coefs_named.reindex(coefs_named.abs().sort_values(ascending=False).index)
        coefs_named = coefs_named[coefs_named!=0] 
        coefs_named.to_csv(outpath+doi+'_top_coefs.csv')

    #predict y values for test data
    y_pred=model.predict(X_test)

    #r2 score
    r2 = r2_score(y_test.values.ravel(), y_pred)
    r2_str="R2 score: "+str(r2)

    #un-normalize the values
    y_test_unnorm= y_scaler.inverse_transform(y_test)
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
    y_pred_unnorm = y_pred_unnorm.flatten()
    #rmse
    rmse_un=root_mean_squared_error(y_test_unnorm, y_pred_unnorm)
    rmse_str="RMSE: "+str(rmse_un)

    #calculate pearson correlation
    correlation = np.corrcoef(y_test_unnorm.ravel(), y_pred_unnorm)[0, 1]
    pcorr_str = "Pearson correlation: "+ str(correlation)

    if(visuals):
        # Add a page for print statements (text)
        reg_txt_pg(avg_rmse, std_rmse, avg_pcorr, std_pcorr, r2_str, rmse_str, pcorr_str, pdf, "Elastic Net Regression")

        scatt_plot(y_test.values.ravel(), y_pred, pdf, normalized=True)

        scatt_plot(y_test_unnorm, y_pred_unnorm, pdf, normalized=False)

    return model

def regressionModels(X_train, X_test, y_train, y_test, y_scaler, doi, visuals, outpath):
    pdf=None
    if (visuals):
        #open a pdf to place text results and (optional) visuals into
        outfile=outpath+doi+"_regression.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)

    linreg_model=linreg(X_train, X_test, y_train, y_test, y_scaler, visuals, pdf)
    elastnet_model=elastnet(X_train, X_test, y_train, y_test, y_scaler, visuals, doi, outpath, pdf)

    #pickle the model
    os.makedirs(outpath+"/models/", exist_ok=True)
    joblib.dump(elastnet_model, outpath+"/models/"+doi+"_elastnet_model.pkl") 

    if(visuals):
        pdf.close()

    return "Completed Regression Models"