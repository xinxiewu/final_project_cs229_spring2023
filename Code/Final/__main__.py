'''
__main__.py contains the workflow to run all sub-programs
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from util import *
from models import *
from deep_learning import *

def main(fileurl=None, km_epoch=None, rf_epoch=None, nn_epoch=None, nn_batch_size=None):
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
    
    # Step 2: Baseline Models - Confusion Matrix & Classification Report
    # 2.1 Logistic Regression
    lr_base = BaseLine(model='lr', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # 2.2 Support Vector Machine
    svm_base = BaseLine(model='svm', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # 2.3 Naive Bayes
    nb_base = BaseLine(model='nb', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    res_df = pd.concat([lr_base, svm_base, nb_base])

    # Step 3: Unsupervised Learning (PCA + K-means) + Improved Logistic Regression
    x = pd.concat([x_train, x_test])
    y = pd.concat([y_train, y_test])
    # 3.1 PCA
    pca_n = pca_choice(x=x, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    pca = PCA(n_components=pca_n).fit(x)
    x_pca, x_train_pca, x_test_pca = pca.transform(x), pca.transform(x_train), pca.transform(x_test)
    # 3.2 K-means
    acc, x_new, y_new = kmeans(n_cluster=2, x=x_pca, y=y, epoch=km_epoch)
    # 3.3 Improved Logistic Regression
    # 3.3.1 PCA Only
    lr_pca = BaseLine(model='lr', x_train=x_train_pca, y_train=y_train, x_test=x_test_pca, y_test=y_test)
    lr_pca['Model'] = f"PCA-{pca_n} LR"
    # 3.3.2 K-means + PCA
    x_train_k, x_test_k, y_train_k, y_test_k = train_test_split(x_new, y_new, test_size=0.3, random_state=42)
    lr_pca_km = BaseLine(model='lr', x_train=x_train_k, y_train=y_train_k, x_test=x_test_k, y_test=y_test_k)
    lr_pca_km['Model'] = f"PCA({pca_n})+KM({len(y_new)}, {format(acc, '.2%')}) LR"
    res_df = pd.concat([res_df, lr_pca, lr_pca_km])

    # Step 4: Ensemble Algorithm -> Bagging -> Random Forest
    rf = Ensemble(model='rf', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,epoch=rf_epoch)
    res_df = pd.concat([res_df, rf])

    # Step 5: Deep Learning: ANN + CNN
    train_loader = DataLoader(TensorDataset(x_train_nn, y_train_nn), batch_size=nn_batch_size, shuffle=True)
    # 5.1 ANN - based on changing # of neurons & layers
    dim1s, dim2s, dim3s, dim4s = [5, 10, 32, 64, 88], [3, 7, 16, 32, 50], [4, 9, 20, 40, 70], [2, 5, 10, 20, 40]
    conv1, conv2 = [8, 16, 32, 64], [16, 32, 64, 88]
    final_loss_2, final_loss_3, final_loss_4, final_loss_cnn = [], [], [], []
    # 5.1.1 2-hidden layers
    for i in range(len(dim1s)):
        final_loss, val_df, test_df = training_nn(hidden_layer=2, epochs=nn_epoch, learning_rate=0.01, 
                                                  train_loader=train_loader, dim1=dim1s[i], dim2=dim2s[i],
                                                  x_val=x_val_nn, y_val=y_val_nn, x_test=x_test_nn, y_test=y_test_nn)
        
        res_df = pd.concat([res_df, val_df, test_df])
        final_loss_2.append(final_loss)

    # 5.1.2 3-hidden layers
    for i in range(len(dim1s)):
        final_loss, val_df, test_df = training_nn(hidden_layer=3, epochs=nn_epoch, learning_rate=0.01, 
                                                  train_loader=train_loader, dim1=dim1s[i], dim2=dim2s[i], dim3=dim3s[i],
                                                  x_val=x_val_nn, y_val=y_val_nn, x_test=x_test_nn, y_test=y_test_nn)
        
        res_df = pd.concat([res_df, val_df, test_df])
        final_loss_3.append(final_loss)

    # 5.1.3 4-hidden layers
    for i in range(len(dim1s)):
        final_loss, val_df, test_df = training_nn(hidden_layer=4, epochs=nn_epoch, learning_rate=0.01, 
                                                  train_loader=train_loader, dim1=dim1s[i], dim2=dim2s[i], dim3=dim3s[i], dim4=dim4s[i],
                                                  x_val=x_val_nn, y_val=y_val_nn, x_test=x_test_nn, y_test=y_test_nn)
        
        res_df = pd.concat([res_df, val_df, test_df])
        final_loss_4.append(final_loss)

    # 5.2 CNN
    for i in range(len(conv1)):
        final_loss, val_df, test_df = training_nn(hidden_layer=999, epochs=nn_epoch, learning_rate=0.01, 
                                                  train_loader=train_loader, dim1=10, dim2=7, conv1=conv1[i], conv2=conv2[i],
                                                  x_val=x_val_nn, y_val=y_val_nn, x_test=x_test_nn, y_test=y_test_nn)
    
        res_df = pd.concat([res_df, val_df, test_df])
        final_loss_cnn.append(final_loss)

    # Step 6: Export Results
    res_df.to_csv('model_results.csv', index=False)

    return

if __name__ == '__main__':
    main (fileurl = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/pima_indians_diabetes.csv',
          km_epoch = 1000,
          rf_epoch = 1000,
          nn_epoch = 500,
          nn_batch_size = 10
         )