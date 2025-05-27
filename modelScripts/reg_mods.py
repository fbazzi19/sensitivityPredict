#library imports
import os
import sys
import pandas as pd
import numpy as np
import sklearn
import math
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import joblib

from filelock import FileLock
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from helperScripts.cross_val import cross_val
from helperScripts.shuffle_eval import shuffle_eval
from helperScripts.figures import scatt_plot, reg_txt_pg

def linreg(X_train, X_test, y_train, y_test, y_scaler, visuals, pdf=None):
    """Create and evaluate linear regression model
    
    Parameters
    ----------
    X_train : pandas dataframe
        X training data
    X_test : pandas dataframe
        X testing data
    y_train : pandas dataframe
        y training data
    y_test : pandas dataframe
        y testing data
    y_scaler : 
        scaler previously used to transform IC50 values
    visuals : int
        whether to create graphics
    pdf : 
        pdf to print visuals to
    
    Output
    ------
    model : 
        the created logistic regression model
    r2 : float
        r2 produced by model on testing data
    mse : float
        mse produced by model on testing data
    rmse : float
        rmse produced by model on testing data
    correlation : float
        Pearson correlation produced by model on testing data
    """
    model=LinearRegression()

    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=0, y_scaler=y_scaler)
    # Calculate average and standard deviation of metrics
    mse_values = [metric['mse']for metric in metrics]
    rmse_values = [metric['rmse'] for metric in metrics]
    pearson_values = [metric['pearson_corr'] for metric in metrics]
    avg_mse_val = np.mean(mse_values)
    std_mse_val = np.std(mse_values)
    avg_mse="Average MSE over 50 splits: "+str(avg_mse_val)
    std_mse="Standard Deviation of MSE: "+str(std_mse_val)
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

    #un-normalize the values
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
    y_pred_unnorm = y_pred_unnorm.flatten()

    #r2 score
    r2 = r2_score(y_test.values.ravel(), y_pred_unnorm)
    r2_str="R2 score: "+str(r2)
    #mse
    mse=mean_squared_error(y_test.values.ravel(), y_pred_unnorm)
    mse_str="MSE: "+str(mse)
    #rmse
    rmse=root_mean_squared_error(y_test.values.ravel(), y_pred_unnorm)
    rmse_str="RMSE: "+str(rmse)
    #calculate pearson correlation
    correlation = np.corrcoef(y_test.values.ravel(), y_pred_unnorm)[0, 1]
    pcorr_str = "Pearson correlation: "+ str(correlation)

    if(visuals):
        # Add a page for print statements (text)
        reg_txt_pg(avg_mse, std_mse, avg_rmse, std_rmse, avg_pcorr, std_pcorr, r2_str, mse_str, rmse_str, pcorr_str, pdf, "Linear Regression")

        scatt_plot(y_test.values.ravel(), y_pred_unnorm, pdf)

    return model, r2, mse, rmse, correlation

def elastnet(X_train, X_test, y_train, y_test, y_scaler, visuals, doi, outpath, pdf=None):
    """Create and evaluate elastic net model
    
    Parameters
    ----------
    X_train : pandas dataframe
        X training data
    X_test : pandas dataframe
        X testing data
    y_train : pandas dataframe
        y training data
    y_test : pandas dataframe
        y testing data
    y_scaler : 
        scaler previously used to transform IC50 values
    visuals : int
        whether to create graphics
    doi : str
        drug of interest, in updated formatting
    outpath : str
        directory to store outputs to
    pdf : 
        pdf to print visuals to
    
    Output
    ------
    model : 
        the created logistic regression model
    r2 : float
        r2 produced by model on testing data
    mse : float
        mse produced by model on testing data
    rmse : float
        rmse produced by model on testing data
    correlation : float
        Pearson correlation produced by model on testing data
    """
    model=ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42, selection='random')
    param_grid = {'alpha': [1e-4, 0.001,0.01,0.1,1, 10, 100],
                'l1_ratio': np.linspace(0.01, 1, 10),
                'max_iter': [10000],
                'random_state': [42],
                'selection': ['random']} #not using cyclic because it doesnt converge even at 50000
    
    #create a scorer and conduct cross validation
    scorer=sklearn.metrics.make_scorer(mean_squared_error, greater_is_better=False)
    best_params=cross_val(X_train, y_train, model, param_grid, scorer, binary=0, modeltype="Elastic Net")
    model=ElasticNet(**best_params)

    #if the model has no non-zero coefficients, perform cross validation again with adjusted alpha
    model.fit(X_train, y_train.values.ravel())
    if(np.count_nonzero(model.coef_)==0):
        model=ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42, selection='random')
        param_grid = {'alpha': [1e-4, 0.001,0.01,0.1], #don't allow high values of alpha
                    'l1_ratio': np.linspace(0.01, 1, 10),
                    'max_iter': [10000],
                    'random_state': [42],
                    'selection': ['random']} #not using cyclic because it doesnt converge even at 50000
        
        #create a scorer and conduct cross validation
        scorer=sklearn.metrics.make_scorer(mean_squared_error, greater_is_better=False)
        best_params=cross_val(X_train, y_train, model, param_grid, scorer, binary=0, modeltype="Elastic Net")
        model=ElasticNet(**best_params)
    
    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=0, y_scaler=y_scaler)

    # Calculate average and standard deviation of metrics
    mse_values = [metric['mse']for metric in metrics]
    rmse_values = [metric['rmse'] for metric in metrics]
    pearson_values = [metric['pearson_corr'] for metric in metrics]
    avg_mse_val = np.mean(mse_values)
    std_mse_val = np.std(mse_values)
    avg_mse="Average MSE over 50 splits: "+str(avg_mse_val)
    std_mse="Standard Deviation of MSE: "+str(std_mse_val)
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

    #get the coefficients of the model
    coefs=model.coef_
    coefs_named=pd.Series(data=coefs, index=X_test.columns.tolist(), name="Coefficients")
    # Sort the Series by absolute value
    coefs_named = coefs_named.reindex(coefs_named.abs().sort_values(ascending=False).index)
    coefs_named = coefs_named[coefs_named!=0] 
    #save to csv
    os.makedirs(outpath+"/model_coefs/", exist_ok=True)
    coefs_named.to_csv(outpath+"/model_coefs/"+doi+'_top_coefs.csv')

    #predict y values for test data
    y_pred=model.predict(X_test)
    
    #un-normalize the values
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
    y_pred_unnorm = y_pred_unnorm.flatten()

    #r2 score
    r2 = r2_score(y_test.values.ravel(), y_pred_unnorm)
    r2_str="R2 score: "+str(r2)
    #mse
    mse=mean_squared_error(y_test.values.ravel(), y_pred_unnorm)
    mse_str="MSE: "+str(mse)
    #rmse
    rmse=root_mean_squared_error(y_test.values.ravel(), y_pred_unnorm)
    rmse_str="RMSE: "+str(rmse)

    #calculate pearson correlation
    correlation = np.corrcoef(y_test.values.ravel(), y_pred_unnorm)[0, 1]
    pcorr_str = "Pearson correlation: "+ str(correlation)

    if(visuals):
        # Add a page for print statements (text)
        reg_txt_pg(avg_mse, std_mse, avg_rmse, std_rmse, avg_pcorr, std_pcorr, r2_str, mse_str, rmse_str, pcorr_str, pdf, "Elastic Net Regression")

        scatt_plot(y_test.values.ravel(), y_pred_unnorm, pdf)

    return model, r2, mse, rmse, correlation

def regressionModels(X_train, X_test, y_train, y_test, y_scaler, doi, visuals, outpath):
    """Create and evaluate regression models
    
    Parameters
    ----------
    X_train : pandas dataframe
        X training data
    X_test : pandas dataframe
        X testing data
    y_train : pandas dataframe
        y training data
    y_test : pandas dataframe
        y testing data
    y_scaler : 
        scaler previously used to transform IC50 values
    doi : str
        drug of interest in updated formatting
    visuals : int
        whether to create graphics
    outpath : str
        directory to store output to
    """
    pdf=None
    if (visuals):
        #open a pdf to place text results and (optional) visuals into
        outfile=outpath+doi+"_regression.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)

    linreg_model, linreg_r2, linreg_mse, linreg_rmse, linreg_pcorr=linreg(X_train, X_test, y_train, y_test, y_scaler, visuals, pdf)
    elastnet_model, elastnet_r2, elastnet_mse, elastnet_rmse, elastnet_pcorr=elastnet(X_train, X_test, y_train, y_test, y_scaler, visuals, doi, outpath, pdf)

    #add the metrics to dataframe
    # Define paths
    allmetrics_file = os.path.join(outpath, 'all_regression_metrics.csv')
    lock_file = allmetrics_file + '.lock'

    # Create a lock for the all metrics file
    lock = FileLock(lock_file)

    with lock:
        # Check if the metrics file exists; if not, initialize the dataframe
        if os.path.exists(allmetrics_file):
            all_metrics = pd.read_csv(allmetrics_file, header=0)
        else:
            columns = ['Model','Drug','R2','MSE', 'RMSE','Pearson Correlation']
            all_metrics = pd.DataFrame(columns=columns)
        #metrics values
        linrow = pd.DataFrame({'Model': ["linear"], 'Drug': [doi],
                        'R2': [linreg_r2], 'MSE': [linreg_mse], 
                        'RMSE': [linreg_rmse], 'Pearson Correlation': [linreg_pcorr]})
        elastrow = pd.DataFrame({'Model': ["elastic net"], 'Drug': [doi],
                        'R2': [elastnet_r2], 'MSE': [elastnet_mse],
                        'RMSE': [elastnet_rmse], 'Pearson Correlation': [elastnet_pcorr]})
        all_metrics=pd.concat([all_metrics, linrow], ignore_index=True)
        all_metrics=pd.concat([all_metrics, elastrow], ignore_index=True)
        all_metrics.to_csv(outpath+'all_regression_metrics.csv', index=False)

    #pickle the model
    os.makedirs(outpath+"/models/", exist_ok=True)
    joblib.dump(elastnet_model, outpath+"/models/"+doi+"_elastnet_model.pkl") 

    if(visuals):
        pdf.close()

    return 0
