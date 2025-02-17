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