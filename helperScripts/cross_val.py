#library imports
import pandas as pd
import numpy as np
import sklearn
import math

from sklearn.model_selection import GridSearchCV

def cross_val(X_train, y_train, model, params, scorer, binary, modeltype):
    """cross validation to determine the best parameters for the model
    
    Parameters
    ----------
    X_train : pandas dataframe
        training gene expression values
    y_train : pandas dataframe
        training IC50 values
    model : model
        un-trained model
    params : dict
        possible values for the model parameters
    scorer : scorer
        scorer to evaluate results
    binary : int
        whether the model is binary
    modelType : str
        the type of model
    
    Output
    ------
    cv.best_params_ : dict
        the parameters that produced the best results
    """
    print("Beginning cross validation for "+modeltype+" model.")
    #assess all parameter combinations based on scorer
    cv = GridSearchCV(model, params,
                            cv=10, scoring=scorer, n_jobs=-1)

    cv=cv.fit(X_train, y_train.values.ravel())
    print("Completed cross validation for "+modeltype+" model.")
    return cv.best_params_