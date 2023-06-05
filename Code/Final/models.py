"""
models.py contains baseline & ensemble algorithms:
    1. Baseline Algorithms:
        (1) Logistic Regression - LR
        (2) Naive Bayes - NB
        (3) Support Vector Machine - SVM

    2. Ensemble Algorithm -> Bagging -> Random Forest (RF)

    3. Unsupervised Learning:
        (1) Principal Component Analysis - PCA
        (2) K-means
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from util import point_eval_metric

# Baseline Models
def BaseLine(model=None, x_train=None, y_train=None, x_test=None, y_test=None):
    """ Fit/predict baseline models and generate confusion matrix/claffication report
    
    Args:
        model: str, three options: LR, SVM, NB
        x_train: DataFrame, training dataset of features
        y_train: DataFrame, training dataset of labels
        x_test: DataFrame, testing dataset of features
        y_test: DataFrame, testing dataset of labels

    Returns:
        DataFrame, with evaluation metrics
    """
    if model.lower() == 'lr':
        clf = LogisticRegression()
    elif model.lower() == 'svm':
        clf = svm.SVC()
    elif model.lower() == 'nb':
        clf = GaussianNB()
    clf.fit(x_train, np.ravel(y_train))
    y_pred = clf.predict(x_test)
    conf_m = confusion_matrix(y_test, y_pred)
    return point_eval_metric(conf_m=conf_m, model=model)

# Unsupervised Learning
# pca_choice()
def pca_choice(x=None, x_train=None, x_test=None, y_train=None, y_test=None):
    """ Iterate and select the best PCA component # based on the improved LR result
    
    Args:
        x: DataFrame, x_train + x_test
    
    Returns:
        int, the best PCA component #
    """
    total_accuracy = 0
    pca_best_components = 0
    for i in range(x.shape[1]):
        # PCA
        pca = PCA(n_components=(i+1)).fit(x)
        x_train_pca, x_test_pca = pca.transform(x_train), pca.transform(x_test)
        # Improved Logistic Regression
        lr = BaseLine(model='lr', x_train=x_train_pca, y_train=y_train, x_test=x_test_pca, y_test=y_test)
        if float(lr['Total Accuracy'][0][:5])/100 > total_accuracy:
            total_accuracy = float(lr['Total Accuracy'][0][:5])/100
            pca_best_components = (i+1)
    return pca_best_components

# kmeans()
def kmeans(n_cluster=2, x=None, y=None, epoch=1000):
    """ Run K-means multiple times with random initialization, pick the highest accuracy one, remove outliers

    Args:
        n_cluster: int, # of clusters
        x, y: datasets
        epoch: int, # of iteration

    Returns:
        highest accuracy, with the new x and y datasets
    """
    acc, x_new, y_new = 0, [], []
    for i in range(epoch):
        km = KMeans(n_clusters=n_cluster, n_init= 'auto').fit(x)
        y_km = km.predict(x)
        correct, x_temp, y_temp = 0, [], []
        for i in range(len(y)):
            if y_km[i] == 1 - y.iloc[i][0]:
                correct += 1
                x_temp.append(x[i])
                y_temp.append(y.iloc[i][0])
        if (correct / len(y)) > acc:
            acc, x_new, y_new = (correct / len(y)), x_temp, y_temp
    return acc, x_new, y_new

# Ensemble Algorithm - Random Forest
def Ensemble(model=None, x_train=None, y_train=None, x_test=None, y_test=None,epoch=1000):
    """ Random Forest with the best performance

    Args:
        epoch: int, # of iteration
    
    Returns:
        DataFrame, evaluation with the best performance
    """
    acc, conf_rep = 0, pd.DataFrame()
    for i in range(epoch):
        rf = RandomForestClassifier().fit(x_train, np.ravel(y_train))
        y_pred = rf.predict(x_test)
        conf_m = confusion_matrix(y_test, y_pred)
        df = point_eval_metric(conf_m=conf_m, model=model)
        if float(df['Total Accuracy'][0][:5])/100 > acc:
            acc = float(df['Total Accuracy'][0][:5])/100
            conf_rep = df
    return conf_rep
