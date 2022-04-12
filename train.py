import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LSTM_model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_model, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        # Define Model Architecture
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # LSTM
        self.Tanh = nn.Tanh()
        self.out = nn.Linear(256, num_classes)  # fully connected last layer

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0))  # LSTM with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # Reshaping the data for next layer
        out = self.Tanh(hn)
        out = self.out(out)  # Final Layer
        return out


def train_LSTM():
    # model parameters
    learning_rate = 0.00001
    num_epochs = 1000 # Number of times we want to iterate
    batch_size = 256
    display_step = 10
    seq_len = 500 # input size of features
    n_hidden = 256  # number of features in hidden state
    n_classes = 2
    num_layers = 1  # number of stacked lstm layers

    # Load data
    x_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/train_small.csv", header=None)
    y_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/label_small.csv", header=None)

    # Transform data + Scale
    mm = MinMaxScaler()
    ss = StandardScaler()
    #X_ss = ss.fit_transform(x_train)
    #y_mm = mm.fit_transform(y_train)

    # Delete
    X_ss = x_train.to_numpy()
    y_mm = y_train.to_numpy()

    # TODO: Implement training + testing split
    x_train = X_ss[:150, :]
    y_train = y_mm[:150, :]
    x_test = X_ss[150:, :]
    y_test = y_mm[150:, :]

    # convert to tensors
    x_train_tensors = Variable(torch.Tensor(x_train))
    y_train_tensors = Variable(torch.Tensor(y_train))
    x_test_tensors = Variable(torch.Tensor(x_test))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # reshaping to rows, position, features
    X_train_tensors_final = torch.reshape(x_train_tensors, (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(x_test_tensors, (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

    # Create Model
    lstm = LSTM_model(n_classes, seq_len, n_hidden, num_layers, X_train_tensors_final.shape[1])  # our lstm class

    # Define the Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensors_final)  # forward pass
        optimizer.zero_grad()  # calculate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    print("Pred")
    train_predict = lstm(X_test_tensors_final)  # forward pass
    print(train_predict.data.numpy())

train_LSTM()
