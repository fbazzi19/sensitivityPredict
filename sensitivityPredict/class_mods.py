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
from cross_val import cross_val
from shuffle_eval import shuffle_eval


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
    scorer=sklearn.metrics.make_scorer(auc_pr_scorer, greater_is_better=True, response_method="predict_proba")
    best_params=cross_val(X_train, y_train, model, params, scorer, binary=1, modeltype="Logistic Regression")
    model=LogisticRegression(**best_params)

    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=1)
    # Calculate average and standard deviation of metrics
    avg_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    avg_aucpr="Average AUC-PR over 50 splits: "+str(avg_metric)
    std_aucpr="Standard Deviation of AUC-PR: "+str(std_metric)

    #predict y values for test data
    model.fit(X_train, y_train.values.ravel())
    y_pred=model.predict(X_test)
    #probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)
    # Compute AUC-PR
    auc_pr = auc(recall, precision)
    auc_str="AUC-PR: "+str(auc_pr)
    acc_str="Accuracy: "+str(accuracy(y_pred, y_test.values.ravel()))

    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Logistic Regression: ",
        avg_aucpr, std_aucpr,
        auc_str, acc_str,
        classification_report(y_test.values.ravel(), y_pred)]
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
        plt.figure(figsize=(30, 30))
        sb.histplot(data=combined_df, x='Sensitivity', hue='dataset', multiple='dodge')
        plt.title("Logistic Regression Classification", fontsize=30)
        plt.xlabel('Sensitivity', fontsize=25)
        plt.ylabel('Count', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=20)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')

        # Plot the PR curve
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

    return accuracy(y_pred, y_test.values.ravel())

def lda(X_train, X_test, y_train, y_test, pdf, visuals):
    #define the model
    model=LDA(solver='svd', tol= 0.0001)

    #define the range of parameters
    params = {'solver':['svd'], 'tol': [0.0001, 0.001, 0.01]}
    #cross validation
    scorer=sklearn.metrics.make_scorer(auc_pr_scorer, greater_is_better=True, response_method="predict_proba")
    best_params=cross_val(X_train, y_train, model, params, scorer, binary=1, modeltype="LDA")
    model=LDA(**best_params)

    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=1)
    # Calculate average and standard deviation of metrics
    avg_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    avg_aucpr="Average AUC-PR over 50 splits: "+str(avg_metric)
    std_aucpr="Standard Deviation of AUC-PR: "+str(std_metric)

    #predict y values for test data
    model.fit(X_train, y_train.values.ravel())
    y_pred=model.predict(X_test)
    #probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)
    # Compute AUC-PR
    auc_pr = auc(recall, precision)
    auc_str="AUC-PR: "+str(auc_pr)
    acc_str="Accuracy: "+str(accuracy(y_pred, y_test.values.ravel()))

    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["Linear Discriminant Analysis: ",
        avg_aucpr, std_aucpr,
        auc_str, acc_str,
        classification_report(y_test.values.ravel(), y_pred)]
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
        plt.figure(figsize=(30, 30))
        sb.histplot(data=combined_df, x='Sensitivity', hue='dataset', multiple='dodge')
        plt.title("Linear Discriminant Analysis Classification", fontsize=30)
        plt.xlabel('Sensitivity', fontsize=25)
        plt.ylabel('Count', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=20)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')

        # Plot the PR curve
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

    return accuracy(y_pred, y_test.values.ravel())

def knn(X_train, X_test, y_train, y_test, pdf, visuals):
    #define the model
    model = KNeighborsClassifier()

    #define the range of parameters
    params = {'n_neighbors': range(5, 20), 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute'],
                'p':[1,2]}

    #cross validation
    scorer=sklearn.metrics.make_scorer(auc_pr_scorer, greater_is_better=True, response_method="predict_proba")
    best_params=cross_val(X_train, y_train, model, params, scorer, binary=1, modeltype="KNN")
    model=KNeighborsClassifier(**best_params)
    
    #perform shuffled split evaluation
    metrics=shuffle_eval(X_train, y_train, model, binary=1)
    # Calculate average and standard deviation of metrics
    avg_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    avg_aucpr="Average AUC-PR over 50 splits: "+str(avg_metric)
    std_aucpr="Standard Deviation of AUC-PR: "+str(std_metric)

    #predict y values for test data
    model.fit(X_train, y_train.values.ravel())
    y_pred=model.predict(X_test)
    #probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), y_pred_prob)
    # Compute AUC-PR
    auc_pr = auc(recall, precision)
    auc_str="AUC-PR: "+str(auc_pr)
    acc_str="Accuracy: "+str(accuracy(y_pred, y_test.values.ravel()))

    # Add a page for print statements (text)
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off axes for text-only page
    # Add the text to the figure: accuracy
    txt= ["K-nearest Neighbors: ",
        avg_aucpr, std_aucpr,
        auc_str, acc_str,
        classification_report(y_test.values.ravel(), y_pred)]
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
        plt.title("K-Nearest Neighbors Classification", fontsize=30)
        plt.xlabel('Sensitivity', fontsize=25)
        plt.ylabel('Count', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=20)
        legend = plt.gca().get_legend()
        legend.set_title("Data")
        pdf.savefig( bbox_inches='tight')

        # Plot the PR curve
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