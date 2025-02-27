#library imports
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, classification_report, precision_recall_curve, auc, f1_score, root_mean_squared_error

def scatt_plot(y_test, y_pred, pdf, normalized=False):
    # Create a scatter plot
    plt.figure(figsize=(20, 20))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    # Plot the diagonal line where y_true = y_pred
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    # Add labels and title
    plt.xlabel('True Values (y_true)', fontsize=30)
    plt.ylabel('Predicted Values (y_pred)', fontsize=30)
    if(normalized):
        plt.title('True vs. Predicted Values', fontsize=35)
    else:
        plt.title('True vs. Predicted Values (Unnormalized)', fontsize=35)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    # Show the plot
    plt.grid(True)
    pdf.savefig( bbox_inches='tight')
    plt.close()
    return 0

def ic50_violin_plot(data, thresholds, thresh_len, pdf):
    plt.figure(figsize=(20, 20))
    sb.violinplot(x='DRUG_NAME', y='LN_IC50', data= data)
    # Add horizontal lines
    for i in range(thresh_len):
        # Draw a horizontal line
        plt.hlines(y=thresholds[i], xmin=i - 0.4, xmax=i + 0.4, color='red', linewidth=2)

    plt.title("IC50 Distributions", fontsize=35)
    plt.xticks([])
    plt.xlabel("Drugs", fontsize=30)
    plt.ylabel("ln(IC50)", fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    pdf.savefig()
    plt.close()
    return 0

def ic50_bin_sens_plot(y, y_sorted, pdf):
    plt.figure(figsize=(20, 20))
    sb.scatterplot(data=y_sorted, x='SANGER_MODEL_ID', y='LN_IC50', hue='SENSITIVITY')
    plt.title("IC50 Values Colored by Sensitivity", fontsize=35)
    #since there is more than one min and max concentration for each drug, here I take the one that 
    #occurs the most in the data
    plt.axhline(y=np.log(y['MIN_CONC'].value_counts().idxmax()), linestyle="dashed", color="red",
                    label='_nolegend_')
    plt.axhline(y=np.log(y['MAX_CONC'].value_counts().idxmax()), linestyle="dashed", color="red",
                    label='_nolegend_')
    plt.xticks([])
    plt.xlabel("Cell Lines", fontsize=30)
    plt.ylabel("ln(IC50)", fontsize=30)
    plt.legend(fontsize=18, title='Sensitivity', title_fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    pdf.savefig()
    plt.close()
    return 0

def ic50_distb_hist(y, pdf, test=False):
    #produce the histogram
    plt.figure(figsize=(20, 20))
    sb.histplot(data=y, x='LN_IC50')
    #plt.xlabel('Cancer Types', fontsize=20)
    plt.ylabel('Count', fontsize=30)
    #plt.tick_params(axis='both', which='major', labelsize=16)
    if (test):
        plt.title("IC50 Values in Test Set", fontsize=35)
    else:
        plt.title("IC50 Values in Training Set", fontsize=35)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    return 0

def class_hist(df, pdf, model):
    plt.figure(figsize=(20, 20))
    sb.histplot(data=df, x='Sensitivity', hue='dataset', multiple='dodge')
    plt.title(model+" Classification", fontsize=35)
    plt.xlabel('Sensitivity', fontsize=30)
    plt.ylabel('Count', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    legend = plt.gca().get_legend()
    legend.set_title("Data")
    pdf.savefig( bbox_inches='tight')
    plt.close()
    return 0

def pr_curve_plot(recall, precision, pdf):
    plt.figure(figsize=(20, 20))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    # Add labels and title
    plt.xlabel('Recall', fontsize=30)
    plt.ylabel('Precision', fontsize=30)
    plt.title('Precision-Recall Curve', fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=20) 
    plt.legend(fontsize=20)
    plt.grid()
    pdf.savefig( bbox_inches='tight')
    plt.close()
    return 0

def class_txt_pg(avg_aucpr, std_aucpr, auc_str, acc_str, y_test, y_pred, pdf, model):
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= [model+": ",
        avg_aucpr, std_aucpr,
        auc_str, acc_str,
        classification_report(y_test.values.ravel(), y_pred)]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    return 0

def reg_txt_pg(avg_rmse, std_rmse, avg_pcorr, std_pcorr, r2_str, rmse_str, pcorr_str, pdf, model):
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= [model, 
            avg_rmse,
            std_rmse,
            avg_pcorr,
            std_pcorr,
            r2_str,
            rmse_str,
            pcorr_str]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    return 0