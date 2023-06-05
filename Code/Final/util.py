"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. data_prep: Replace missing values with median/mean by category & then normalize
    3. data_split: Split dataset into training, validation and testing
    4. point_eval_metric: Given confusion matrix, generate point evaluation metrics
"""
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# download_file(url)
def download_file(url=None):
    """ Download the .csv file from the given link and read as dataframe

    Args: 
        url: str
    
    Returns:
        DataFrame
    """
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return pd.read_csv(local_filename)

# data_prep()
def data_prep(df=None, label=None, normalize=True):
    """ Deal with missing values (median/mean) by category, and standardization/normalization
    
    Args:
        df: DataFrame, with unprocessed data
        label: str, label data

    Returns:
        DataFrame, with clean data
    """
    for col in df.columns[(df.columns != label.lower()) & (df.columns != 'preg')]:
        for i in range(2):
            data_prep_helper(df=df, col=col, label=label, diabetes=i)
    if normalize == True:
        for col in df.columns[df.columns != label.lower()]:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def data_prep_helper(df=None, col=None, label=None, diabetes=1):
    """Deal with missing values (median/mean) by category"""
    if df.loc[(df[label] == diabetes), col].median() != 0.:
        df.loc[(df[label] == diabetes) & (df[col] == 0), col] = df.loc[(df[label] == diabetes), col].median() 
    else:
        df.loc[(df[label] == diabetes) & (df[col] == 0), col] = df.loc[(df[label] == diabetes), col].mean() 
    return

# data_split()
def data_split(df=None, label=None, validation=False, train_size=0.7, random_state=42, tensor=False):
    """ Split dataset into training, validation & 
    
    Args:
        df: DataFrame
        label: str, label column name
        validation: boolean, True if a validation set is needed, otherwise False
        train_size: float, size of training dataset, <= 1
        random_state: int, random state, default value as 42
        tensor: boolean, True if need to convert to Tensor, otherwise False

    Returns:
        DataFrames, split
    """
    if validation == False and tensor == False:
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != label], df.iloc[:,df.columns == label], 
                                                            test_size=(1-train_size), random_state=random_state)
        return x_train, x_test, y_train, y_test
    elif validation == True and tensor == True:
        x_train, x_val_te, y_train, y_val_te = train_test_split(df.iloc[:,df.columns != label], df.iloc[:,df.columns == label], 
                                                            test_size=(1-train_size), random_state=random_state)
        x_val, x_test, y_val, y_test = train_test_split(x_val_te, y_val_te, 
                                                            test_size=0.5, random_state=random_state)
        X_train = torch.Tensor(x_train.values)
        X_val = torch.Tensor(x_val.values)
        X_test = torch.Tensor(x_test.values)
        Y_train = torch.Tensor(y_train.values)
        Y_val = torch.Tensor(y_val.values)
        Y_test = torch.Tensor(y_test.values)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

# point_eval_metric()
def point_eval_metric(conf_m=None, model=None):
    """ Given confusion matrix, generate point evaluation metrics

    Args:
        conf_m: confusion matrix
        model: str
    
    Returns:
        DataFrame with info: 
            - model, test_size, prevalence, acc_tot, acc_pos, acc_neg, prec, recall, f1
    """
    if model.lower() == 'lr':
        model = 'LogisticReg'
    elif model.lower() == 'nb':
        model = 'NaiveBayes'
    elif model.lower() == 'svm':
        model = 'SVM'
    elif model.lower() == 'rf':
        model = 'Random Forest'
    else:
        model = model

    tn, fp, fn, tp = conf_m[0][0], conf_m[0][1], conf_m[1][0], conf_m[1][1]
    data =  {'Model': [model],
             'Test Size': [tn + fn + fp + tp],
             'Prevalence': [format((tp + fn) / (tn + fn + fp + tp), '.2%')],
             'Total Accuracy': [format((tp + tn) / (tn + fn + fp + tp), '.2%')],
             'Positive Accuracy': [format(tp / (tp + fn), '.2%')],
             'Negative Accuracy': [format(tn / (tn + fp), '.2%')],
             'Precision': [format(tp / (tp+fp), '.2%')],
             'Recall': [format(tp / (tp+fn), '.2%')],
             'F1-Score': [format(2*((tp / (tp+fp)) * (tp / (tp+fn))) / ((tp / (tp+fp)) + (tp / (tp+fn))), '.2%')]
            }
    
    return pd.DataFrame.from_dict(data)