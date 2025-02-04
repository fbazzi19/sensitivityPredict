#library imports
import pandas as pd
import numpy as np
import sklearn
import math

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KernelDensity

def cross_val(X_train, y_train, model, params, scorer, binary, modeltype):
    print("Beginning cross validation for "+modeltype+" model.")
    # Define the cross-validation strategy
    #if (binary):
    #    ss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    #else:
    #    ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)

    cv = GridSearchCV(model, params,
                            cv=10, scoring=scorer, n_jobs=-1)#, verbose=2)
    cv=cv.fit(X_train, y_train.values.ravel())
    print("Completed cross validation for "+modeltype+" model.")
    return cv.best_params_