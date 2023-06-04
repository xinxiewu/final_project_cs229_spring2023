'''
__main__.py contains the workflow to run all programs
'''
import seaborn as sns
import matplotlib.pyplot as plt
from util import *
from supervised_learning import *

def main(fileurl):
    # Step 1: Data Preparation & EDA
    # 1.1 Download dataset from Github & read as DataFrame
    df = download_file(fileurl)
    # 1.2 Missing value with median/mean & normalization
    diabetes = data_prep(df=df, label='diabetes', normalize=True)
    # 1.3 Correlation heatmap export
    sns.heatmap(diabetes.corr(), annot = True)
    plt.savefig('heatmap_correlation.png')
    # 1.4 Data Split
    x_train, x_test, y_train, y_test = data_split(df=diabetes, label='diabetes')
    x_train_nn, x_val_nn, x_test_nn, y_train_nn, y_val_nn, y_test_nn = data_split(df=diabetes, label='diabetes'
                                                                                  ,validation=True, train_size=0.8
                                                                                  ,tensor=True)
    
    # Step 2: Baseline Models

    # Step 3: Ensemble Algorithm -> Bagging -> Random Forest

    # Step 4: Unsupervised Learning (PCA + K-means) + Improved Logistic Regression

    # Step 5: Deep Learning: NN + CNN

    return

if __name__ == '__main__':
    main(fileurl='https://raw.githubusercontent.com/xinxiewu/datasets/main/pima_indians_diabetes.csv')