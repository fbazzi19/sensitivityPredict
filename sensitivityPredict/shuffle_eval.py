#library imports
import pandas as pd
import numpy as np
import sklearn
import math

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, root_mean_squared_error

def shuffle_eval(X, y, model, binary=0, y_scaler=None):
    # ShuffleSplit for 50 iterations
    if (binary):
        ss_eval = StratifiedShuffleSplit(n_splits=50, test_size=0.25, random_state=None)
    else:
        ss_eval = ShuffleSplit(n_splits=50, test_size=0.25, random_state=None)

    # Initialize list to store performance metrics
    metrics = []

    # Evaluate model over 50 splits
    for train_index, test_index in ss_eval.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model with the best parameters
        model.fit(X_train, y_train.values.ravel())
        
        # Evaluate on the split test set
        y_pred = model.predict(X_test)
        if(binary):
            #probabilities
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            # Generate Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)
            # Compute AUC-PR
            metric = auc(recall, precision)
        else:
            #un-normalize the values
            y_test_unnorm= y_scaler.inverse_transform(y_test)
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_pred_unnorm = y_scaler.inverse_transform(y_pred_reshaped)
            y_pred_unnorm = y_pred_unnorm.flatten()
            #rmse
            metric=root_mean_squared_error(y_test_unnorm, y_pred_unnorm)
        
        
        metrics.append(metric)
    return metrics
