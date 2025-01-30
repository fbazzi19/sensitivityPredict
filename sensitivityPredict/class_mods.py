#library imports
import pandas as pd
import numpy as np
import sklearn
import scipy
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score


#accuracy= n_correct/n_total
def accuracy(y_pred, y_true):
    return (y_pred==y_true).sum()/len(y_true)

def auc_pr_scorer(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def logreg(X_train, X_test, y_train, y_test, pdf, visuals):
    #define the model
    model=LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=10000, class_weight='balanced', random_state=42)

    #define the range of parameters
    params = {'penalty':['l1','l2'], 'solver':['saga', 'liblinear'], 'C': [0.001, 0.01, 0.1, 1, 10],
            'max_iter':[10000], 'class_weight':['balanced'], 'random_state':[42]}
    #cross validation
    print("pre cv")
    cv = GridSearchCV(model, params, scoring=sklearn.metrics.make_scorer(auc_pr_scorer, greater_is_better=True, response_method="predict_proba"), cv=5, n_jobs=-1)#, verbose=2)
    cv = cv.fit(X_train, y_train.values.ravel())
    print("post cv")

    #get the best model (already fitted)
    model=cv.best_estimator_
    #predict y values for test data
    y_pred=model.predict(X_test)
    #probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)

    # Compute AUC-PR
    auc_pr = auc(recall, precision)

    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Accuracy for Logistic regression classification: "+str(accuracy(y_pred, y_test.values.ravel())),
        classification_report(y_test.values.ravel(), y_pred),
        "AUC-PR: "+str(auc_pr)]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    
    if(visuals):
        # Convert to DataFrames with a consistent format
        test_df = pd.DataFrame({'Sensitivity': y_test.values.ravel(), 'dataset': 'Test'})
        pred_df = pd.DataFrame({'Sensitivity': y_pred, 'dataset': 'Prediction'})

        # Combine the datasets
        combined_df = pd.concat([test_df, pred_df])

        combined_df['Sensitivity']= combined_df['Sensitivity'].map(str)
        for i in range(len(combined_df['Sensitivity'])):
            if combined_df.iloc[i, 0]=='0':
                combined_df.iloc[i,0]="Resistant"
            else:
                combined_df.iloc[i,0]="Sensitive"

        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=combined_df, x='Sensitivity', hue='dataset', multiple='dodge')
        plt.title("Logistic Regression Classification", fontsize=25)
        plt.xlabel('Sensitivity', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')

        # Plot the PR curve
        plt.figure(figsize=(20, 20))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        # Add labels and title
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.title('Precision-Recall Curve', fontsize=25)
        plt.legend()
        plt.grid()
        pdf.savefig( bbox_inches='tight')

    return accuracy(y_pred, y_test.values.ravel())

def lda(X_train, X_test, y_train, y_test, pdf, visuals):
    #define the model
    model=LDA(solver='svd', tol= 0.0001)

    #define the range of parameters
    params = {'solver':['svd'], 'tol': [0.0001, 0.001, 0.01]}
    #cross validation
    print("pre cv")
    cv = GridSearchCV(model, params, scoring=sklearn.metrics.make_scorer(auc_pr_scorer, greater_is_better=True, response_method="predict_proba"), cv=5, n_jobs=-3)#, verbose=2) 
    cv = cv.fit(X_train, y_train.values.ravel())
    print("post cv")

    #get the best model (already fitted)
    model=cv.best_estimator_
    #predict y values for test data
    y_pred=model.predict(X_test)
    #probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)

    # Compute AUC-PR
    auc_pr = auc(recall, precision)

    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Accuracy for linear discriminant analysis classification: "+str(accuracy(y_pred, y_test.values.ravel())),
        classification_report(y_test.values.ravel(), y_pred),
        "AUC-PR: "+str(auc_pr)]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    
    if(visuals):
        # Convert to DataFrames with a consistent format
        test_df = pd.DataFrame({'Sensitivity': y_test.values.ravel(), 'dataset': 'Test'})
        pred_df = pd.DataFrame({'Sensitivity': y_pred, 'dataset': 'Prediction'})

        # Combine the datasets
        combined_df = pd.concat([test_df, pred_df])

        combined_df['Sensitivity']= combined_df['Sensitivity'].map(str)
        for i in range(len(combined_df['Sensitivity'])):
            if combined_df.iloc[i, 0]=='0':
                combined_df.iloc[i,0]="Resistant"
            else:
                combined_df.iloc[i,0]="Sensitive"

        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=combined_df, x='Sensitivity', hue='dataset', multiple='dodge')
        plt.title("Linear Discriminant Analysis Classification", fontsize=25)
        plt.xlabel('Sensitivity', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')

        # Plot the PR curve
        plt.figure(figsize=(20, 20))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        # Add labels and title
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.title('Precision-Recall Curve', fontsize=25)
        plt.legend()
        plt.grid()
        pdf.savefig( bbox_inches='tight')

    return accuracy(y_pred, y_test.values.ravel())

def knn(X_train, X_test, y_train, y_test, pdf, visuals):
    #define the model
    model = KNeighborsClassifier()

    #define the range of parameters
    params = {'n_neighbors': range(5, 20), 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute'],
                'p':[1,2]}

    #cross validation
    print("pre cv")
    cv = GridSearchCV(model, params, scoring=sklearn.metrics.make_scorer(auc_pr_scorer, greater_is_better=True, response_method="predict_proba"), cv=10, n_jobs=-1)#, verbose=2) 
    cv = cv.fit(X_train, y_train.values.ravel())
    print("post cv")

    #get the best model (already fitted)
    model=cv.best_estimator_
    #predict y values for test data
    y_pred=model.predict(X_test)
    #probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)

    # Compute AUC-PR
    auc_pr = auc(recall, precision)

    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Accuracy for k nearest neighbors classification: "+str(accuracy(y_pred, y_test.values.ravel())),
        classification_report(y_test.values.ravel(), y_pred),
        "AUC-PR: "+str(auc_pr)]
    txt = "\n".join(txt)
    ax.text(0.1, 0.9, txt, va='top', ha='left', fontsize=12, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig)
    
    if(visuals):
        # Convert to DataFrames with a consistent format
        test_df = pd.DataFrame({'Sensitivity': y_test.values.ravel(), 'dataset': 'Test'})
        pred_df = pd.DataFrame({'Sensitivity': y_pred, 'dataset': 'Prediction'})

        # Combine the datasets
        combined_df = pd.concat([test_df, pred_df])

        combined_df['Sensitivity']= combined_df['Sensitivity'].map(str)
        for i in range(len(combined_df['Sensitivity'])):
            if combined_df.iloc[i, 0]=='0':
                combined_df.iloc[i,0]="Resistant"
            else:
                combined_df.iloc[i,0]="Sensitive"

        #produce the histogram
        plt.figure(figsize=(20, 20))
        sb.histplot(data=combined_df, x='Sensitivity', hue='dataset', multiple='dodge')
        plt.title("K-Nearest Neighbors Classification", fontsize=25)
        plt.xlabel('Sensitivity', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')

        # Plot the PR curve
        plt.figure(figsize=(20, 20))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        # Add labels and title
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.title('Precision-Recall Curve', fontsize=25)
        plt.legend()
        plt.grid()
        pdf.savefig( bbox_inches='tight')


    return accuracy(y_pred, y_test.values.ravel())

def classificationModels(X_train, X_test, y_train, y_test, doi, visuals, outpath):
    #open a pdf to place text results and (optional) visuals into
    doinospace=doi.replace(" ", "")
    outfile=outpath+doi+"_classification.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)

    accuracies = {'Model': ['Logistic Regression', 'LDA', 'KNN'],
                    'Accuracy': [logreg(X_train, X_test, y_train, y_test, pdf, visuals), lda(X_train, X_test, y_train, y_test, pdf, visuals), 
                               knn(X_train, X_test, y_train, y_test, pdf, visuals)]}
    
    pdf.close()

    return ":)"