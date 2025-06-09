#library imports
import pandas as pd
import numpy as np
import sklearn
import math
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from sklearn.metrics import classification_report

def scatt_plot(y_test, y_pred, pdf):
    """scatter plot comparing the true values to the predicted values
    
    Parameters
    ----------
    y_test : list
        True IC50 values
    y_pred : list
        Predicted IC50 values
    pdf :
        pdf to print visuals to
    """
    # Create a scatter plot
    plt.figure(figsize=(20, 20))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    # Plot the diagonal line where y_true = y_pred
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    # Add labels and title
    plt.xlabel('True Values (y_true)', fontsize=30)
    plt.ylabel('Predicted Values (y_pred)', fontsize=30)
    plt.title('True vs. Predicted Values', fontsize=35)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    # Show the plot
    plt.grid(True)
    pdf.savefig( bbox_inches='tight')
    plt.close()
    return 0

def ic50_violin_plot(data, thresholds, thresh_len, pdf):
    """Violin plot containing the upsampled IC50 values and where the threshold lies amongst them
    
    Parameters
    ----------
    data : pandas dataframe
        IC50 values for more than one drug
    thresholds : list
        binary thresholds for each drug in data
    thresh_len : int
        amount of thresholds
    pdf :
        pdf to print visuals to
    """
    plt.figure(figsize=(20, 20))
    sb.violinplot(x='DRUG_NAME', y='LN_IC50', data= data)
    # Add horizontal lines
    for i in range(thresh_len):
        # Draw a horizontal line
        plt.hlines(y=thresholds[i], xmin=i - 0.4, xmax=i + 0.4, color='red', linewidth=4)

    plt.title("IC50 Distributions", fontsize=40)
    plt.xticks([])
    plt.xlabel("Drugs", fontsize=35)
    plt.ylabel("ln(IC50)", fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=25)
    pdf.savefig()
    plt.close()
    return 0

def ic50_bin_sens_plot(y, y_sorted, pdf):
    """Plot IC50 values of cell lines for a drug, colored by binary sensitivity label
    
    Parameters
    ----------
    y : pandas dataframe
        IC50 and binary sensitivity of doi
    y_sorted : pandas dataframe
        IC50 and binary sensitivity of doi, sorted by IC50 values
    pdf :
        pdf to print visuals to
    """
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
    """Plots histogram of distribution of IC50 values
    
    Parameters
    ----------
    y : pandas dataframe
        IC50 and binary sensitivity of doi
    pdf :
        pdf to print visuals to
    test : bool
        whether the values come from the test set
    """
    #produce the histogram
    plt.figure(figsize=(20, 20))
    sb.histplot(data=y, x='LN_IC50')
    plt.ylabel('Count', fontsize=30)
    if (test):
        plt.title("IC50 Values in Test Set", fontsize=35)
    else:
        plt.title("IC50 Values in Training Set", fontsize=35)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    return 0

def class_hist(df, pdf, model):
    """Histogram of the classifications in the true and predicted sets
    
    Parameters
    ----------
    df : pandas dataframe
        true and predicted sensitivity classifications
    pdf :
        pdf to print visuals to
    model : str
        model type
    """
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
    """Plot of the AUC-PR curve
    
    Parameters
    ----------
    recall : 
        recall
    precision : 
        precision
    pdf :
        pdf to print visuals to
    """
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
    """Printing of all the classification model metrics as a page
    
    Parameters
    ----------
    avg_aucpr : str
        average AUC-PR from shuffle split evaluation
    std_aucpr : str
        standard deviation of AUC-PR from shuffle split evaluation
    auc_str : str
        calculated AUC-PR
    acc_str : str
        calculated accuracy
    y_test : pandas dataframe
        true sensitivity labels
    y_pred : list
        predicted sensitivity labels
    pdf :
        pdf to print visuals to
    model : str
        model type
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure
    txt= [model+": ",
        avg_aucpr, std_aucpr,
        auc_str, acc_str,
        classification_report(y_test.values.ravel(), y_pred)]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    return 0

def reg_txt_pg(avg_mse, std_mse, avg_rmse, std_rmse, avg_pcorr, std_pcorr, r2_str, mse_str, rmse_str, pcorr_str, pdf, model):
    """Printing of all the regression metrics as a page
    
    Parameters
    ----------
    avg_mse : str
        average mean squared error from shuffle split evaluation
    std_mse : str
        standard deviation of mean squared error from shuffle split evaluation
    avg_rmse : str
        average root mean squared error from shuffle split evaluation
    std_rmse : str
        standard deviation of root mean squared error from shuffle split evaluation
    avg_pcorr : str
        average pearson correlation from shuffle split evaluation
    std_pcorr : str
        standard deviaiton of pearson correlation from shuffle split evaluation
    r2_str : str
        calculated r2
    mse_str : str
        calculated mean squared error
    rmse_str : str
        calculated root mean squared error
    pcorr_str : str
        calculated pearson correlation
    pdf :
        pdf to print visuals to
    model : str
        model type
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure
    txt= [model,
            avg_mse,
            std_mse, 
            avg_rmse,
            std_rmse,
            avg_pcorr,
            std_pcorr,
            r2_str,
            mse_str,
            rmse_str,
            pcorr_str]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    return 0