"""
deep_learning.py contains neural networks:
    1. Multilayer Perceptron (MLP): This one doesn't have convolutional layer
    2. Convolutional Neural Networks (CNN)
    3. Training process
"""
from torch import nn
import torch.optim as optim
from util import *

# ANN/MLP, without convolutional layer
# 2 Hidden Layers
class MyANN_2(nn.Module):
    def __init__(self, input_features, dim1, dim2, output_features):
        super(MyANN_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, dim1, bias=False), nn.ReLU(),
            nn.Linear(dim1, dim2), nn.ReLU(),
            nn.Linear(dim2, output_features), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# 3 Hidden Layers
class MyANN_3(nn.Module):
    def __init__(self, input_features, dim1, dim2, dim3, output_features):
        super(MyANN_3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, dim1, bias=False), nn.ReLU(),
            nn.Linear(dim1, dim2), nn.ReLU(),
            nn.Linear(dim2, dim3), nn.ReLU(),
            nn.Linear(dim3, output_features), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# 4 Hidden Layers
class MyANN_4(nn.Module):
    def __init__(self, input_features, dim1, dim2, dim3, dim4, output_features):
        super(MyANN_4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, dim1, bias=False), nn.ReLU(),
            nn.Linear(dim1, dim2), nn.ReLU(),
            nn.Linear(dim2, dim3), nn.ReLU(),
            nn.Linear(dim3, dim4), nn.ReLU(),
            nn.Linear(dim4, output_features), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# CNN
class MyCNN(nn.Module):
    def __init__(self, dim1, dim2, conv1, conv2, output_features):
        super(MyCNN, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(1, conv1, 2), nn.ReLU(),
            nn.MaxPool1d(2), nn.Conv1d(conv1, conv2, 2),
            nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(),  
        )
        self.model2 = nn.Sequential(
            nn.Linear(conv2, dim1), nn.ReLU(),
            nn.Linear(dim1, dim2), nn.ReLU(),
            nn.Linear(dim2, output_features), nn.Sigmoid()
        )

    def forward(self, input):
        input = input.reshape(-1, 1, 8)   
        x = self.model1(input)
        x = self.model2(x)
        return x    

# training_nn()
def training_nn(hidden_layer=None, epochs=500, learning_rate=0.01, loss_func='BCE', optimizer_para = 'Adam',
                train_loader=None, input_features=8, output_features=1, dim1=None, dim2=None, dim3=None, dim4=None,
                conv1=None, conv2=None,
                x_val=None, y_val=None, x_test=None, y_test=None):
    """ Training process

    Args:
        hidden_layer: # of hidden layers
        train_loader: DataLoader for Pytorch
    Returns:
        Loss by epoch, validation evaluation, testing evaluation
    """
    # Initialize model
    if hidden_layer == 2:
        network = MyANN_2(input_features, dim1, dim2, output_features)
        model = f"2-Layer NN {dim1, dim2}"
    elif hidden_layer == 3:
        network = MyANN_3(input_features, dim1, dim2, dim3, output_features)
        model = f"3-Layer NN {dim1, dim2, dim3}"
    elif hidden_layer == 4:
        network = MyANN_4(input_features, dim1, dim2, dim3, dim4, output_features)
        model = f"4-Layer NN {dim1, dim2, dim3, dim4}"
    elif hidden_layer == 999:
        network = MyCNN(dim1, dim2, conv1, conv2, output_features)
        model = f"2-Layer CNN {dim1, dim2} with conv {conv1, conv2}"

    # Define loss function & optimizer
    if loss_func.upper() == 'BCE':
        loss_function = nn.BCELoss()
    if optimizer_para.upper() == 'ADAM':
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Training
    final_loss = []
    for epoch in range(epochs):
        running_loss = 0.
        for data in train_loader:
            inputs, outputs = data
            optimizer.zero_grad()
            predictions = network(inputs)
            loss = loss_function(predictions, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_loss.append(running_loss/len(train_loader))

    # Validation/Testing Accuracy
    pred_val = [1 if i >= 0.5 else 0 for i in network(x_val).detach().numpy()]
    val_df = point_eval_metric(conf_m=training_nn_helper(pred=pred_val, truth=y_val), model=f"{model} - Val")
    pred_test = [1 if i >= 0.5 else 0 for i in network(x_test).detach().numpy()]
    test_df = point_eval_metric(conf_m=training_nn_helper(pred=pred_test, truth=y_test), model=f"{model} - Test")

    return final_loss, val_df, test_df

def training_nn_helper(pred=None, truth=None):
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(pred)):
        if truth[i] == 0 and pred[i] == 0:
            tn += 1
        elif truth[i] == 0 and pred[i] == 1:
            fp += 1
        elif truth[i] == 1 and pred[i] == 0:
            fn += 1
        elif truth[i] == 1 and pred[i] == 1:
            tp += 1
    return [[tn, fp],[fn, tp]]
