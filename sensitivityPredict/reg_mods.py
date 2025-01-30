#library imports
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

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.manifold import TSNE
from scipy.integrate import simpson
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, classification_report, precision_recall_curve, auc, f1_score, root_mean_squared_error

def linreg(X_train, X_test, y_train, y_test, y_scaler, pdf, visuals):
    model=LinearRegression()
    #fit the model
    model.fit(X_train, y_train.values.ravel())
    #the coefficients
    coefs= model.coef_
    #predict the values
    y_pred=model.predict(X_test)
    #r2 score
    r2 = r2_score(y_test.values.ravel(), y_pred)
    #rmse
    rmse=root_mean_squared_error(y_test.values.ravel(), y_pred)

    #un-normalize the values
    y_test_unnorm= y_scaler.inverse_transform(y_test)
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
    y_pred_unnorm = y_pred_unnorm.flatten()
    #r2 score
    r2_un = r2_score(y_test_unnorm, y_pred_unnorm)
    #rmse
    rmse_un=root_mean_squared_error(y_test_unnorm, y_pred_unnorm)


    #TODO: adjust for this model
    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Linear Regression", "R2 score: "+str(r2),
            "RMSE: "+str(rmse),
            "Unnormalized R2: "+str(r2_un),
            "Unnormalized RMSE: "+str(rmse_un)]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)

    if(visuals):
        # Create a scatter plot
        plt.figure(figsize=(20, 20))
        plt.scatter(y_test.values.ravel(), y_pred, color='blue', alpha=0.5)

        # Plot the diagonal line where y_true = y_pred
        plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], color='red', linestyle='--')

        # Add labels and title
        plt.xlabel('True Values (y_true)', fontsize=20)
        plt.ylabel('Predicted Values (y_pred)', fontsize=20)
        plt.title('True vs. Predicted Values', fontsize=25)

        # Show the plot
        plt.grid(True)
        pdf.savefig( bbox_inches='tight')

        # Create a scatter plot
        plt.figure(figsize=(20, 20))
        plt.scatter(y_test_unnorm, y_pred_unnorm, color='blue', alpha=0.5)

        # Plot the diagonal line where y_true = y_pred
        plt.plot([min(y_test_unnorm), max(y_test_unnorm)], [min(y_test_unnorm), max(y_test_unnorm)], color='red', linestyle='--')

        # Add labels and title
        plt.xlabel('True Values (y_true)', fontsize=20)
        plt.ylabel('Predicted Values (y_pred)', fontsize=20)
        plt.title('True vs. Predicted Values (Unnormalized)', fontsize=25)

        # Show the plot
        plt.grid(True)
        pdf.savefig( bbox_inches='tight')

    return 0

def elastnet(X_train, X_test, y_train, y_test, y_scaler, pdf, visuals, doi, outpath):
    model=ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000, random_state=42, selection='random')
    param_grid = {'alpha': [0.001,0.01,0.1,1,10,100],
                'l1_ratio': np.linspace(0.01, 1, 5),
                'max_iter': [10000],
                'random_state': [42],
                'selection': ['random']}
    
    print("pre cv")
    cv = GridSearchCV(model, param_grid,
                            cv=5, scoring=sklearn.metrics.make_scorer(root_mean_squared_error, greater_is_better=False), n_jobs=-1)#, verbose=2)
    cv=cv.fit(X_train, y_train.values.ravel())
    print("post cv")

    #get the best model (already fitted)
    model=cv.best_estimator_
    #get the coefficients of the model
    coefs=model.coef_
    coefs_named=pd.Series(data=coefs, index=X_test.columns.tolist(), name="Coefficients")
    # Sort the Series by absolute value
    coefs_named = coefs_named.reindex(coefs_named.abs().sort_values(ascending=False).index)
    coefs_named = coefs_named[0:100]
    coefs_named.to_csv(outpath+doi+'_top_coefs.csv')

    #predict y values for test data
    y_pred=model.predict(X_test)

    #r2 score
    r2 = r2_score(y_test.values.ravel(), y_pred)
    #rmse
    rmse=root_mean_squared_error(y_test.values.ravel(), y_pred)

    #un-normalize the values
    y_test_unnorm= y_scaler.inverse_transform(y_test)
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
    y_pred_unnorm = y_pred_unnorm.flatten()
    #r2 score
    r2_un = r2_score(y_test_unnorm, y_pred_unnorm)
    #rmse
    rmse_un=root_mean_squared_error(y_test_unnorm, y_pred_unnorm)


    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Elastic Net Regression", "R2 score: "+str(r2),
            "RMSE: "+str(rmse),
            "Unnormalized R2: "+str(r2_un),
            "Unnormalized RMSE: "+str(rmse_un),
            "\nCoefficients:\n" + coefs_named.to_string()]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)

    if(visuals):
        # Create a scatter plot
        plt.figure(figsize=(20, 20))
        plt.scatter(y_test.values.ravel(), y_pred, color='blue', alpha=0.5)

        # Plot the diagonal line where y_true = y_pred
        plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], color='red', linestyle='--')

        # Add labels and title
        plt.xlabel('True Values (y_true)', fontsize=20)
        plt.ylabel('Predicted Values (y_pred)', fontsize=20)
        plt.title('True vs. Predicted Values', fontsize=25)

        # Show the plot
        plt.grid(True)
        pdf.savefig( bbox_inches='tight')

        # Create a scatter plot
        plt.figure(figsize=(20, 20))
        plt.scatter(y_test_unnorm, y_pred_unnorm, color='blue', alpha=0.5)

        # Plot the diagonal line where y_true = y_pred
        plt.plot([min(y_test_unnorm), max(y_test_unnorm)], [min(y_test_unnorm), max(y_test_unnorm)], color='red', linestyle='--')

        # Add labels and title
        plt.xlabel('True Values (y_true)', fontsize=20)
        plt.ylabel('Predicted Values (y_pred)', fontsize=20)
        plt.title('True vs. Predicted Values (Unnormalized)', fontsize=25)

        # Show the plot
        plt.grid(True)
        pdf.savefig( bbox_inches='tight')  


        #plot coefficient values
        plt.figure(figsize=(30, 30))
        plt.bar(coefs_named.index[:20], coefs_named[:20])
        plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Add a horizontal line at 0 for reference
        plt.title('Elastic Net Coefficients', fontsize=30)
        plt.xticks(rotation=30, ha='right')
        plt.xlabel('Genes', fontsize=25) 
        plt.ylabel('Coefficients', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=20) 
        pdf.savefig( bbox_inches='tight')  

    return 0

def regressionModels(X_train, X_test, y_train, y_test, y_scaler, doi, visuals, outpath):
    #open a pdf to place text results and (optional) visuals into
    doinospace=doi.replace(" ", "")
    outfile=outpath+doi+"_regression.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)

    linreg(X_train, X_test, y_train, y_test, y_scaler, pdf, visuals)
    elastnet(X_train, X_test, y_train, y_test, y_scaler, pdf, visuals, doi, outpath)

    pdf.close()

    return "<3"