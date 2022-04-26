import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
import CGR_generator as cgr
import utils


class CNN_model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(CNN_model, self).__init__()
        self.num_classes = num_classes  # number of classes - output layer
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state

        # Model Architecture
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 4 * 4, 10)
        )

    def forward(self, input_x):
        input_x = self.cnn_layers(input_x)
        input_x = input_x.view(input_x.size(0), -1)
        input_x = self.linear_layers(input_x)
        return input_x


class LSTM_model(nn.Module):
    """
    LSTM model class, builds RNN model with the following parameters
    """
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        """
        :param num_classes: Number of output classes. In context of problem, this is always 2 (Virus and Non-Virus)
        :param input_size: Size of string/sequence being passed in. In context of problem, this is always 500 (bp)
        :param hidden_size: Number of hidden layers
        :param num_layers: Number of stacked LSTM units
        """
        super(LSTM_model, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state

        # Define Model Architecture
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # LSTM
        self.Tanh = nn.Tanh()
        self.out = nn.Linear(256, num_classes)  # fully connected last layer

    def forward(self, x):
        """
        Forward Pass used for training and prediction
        :param x: Sequence of length 500 bp
        :return: Returns outputs of final layer. Of size [256,2] #TODO: double check this size
        """
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0))  # LSTM with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # Reshaping the data for next layer
        out = self.Tanh(hn)
        out = self.out(out)  # Final Layer
        return out


def train_LSTM(x_train, y_train):
    # model parameters
    learning_rate = 0.00001
    num_epochs = 1000
    batch_size = 256
    seq_len = 500
    n_hidden = 256
    n_classes = 2
    num_layers = 1

    """"
    # Transform data + Scale
    # TODO: perform data visualization to see if scalers are needed
    mm = MinMaxScaler()
    ss = StandardScaler()
    #X_ss = ss.fit_transform(x_train)
    #y_mm = mm.fit_transform(y_train)
    """

    # Convert data into tensors
    x_train_tensors = Variable(torch.Tensor(x_train))
    y_train_tensors = Variable(torch.Tensor(y_train))

    # Reshaping to [rows, position, features]
    X_train_tensors_final = torch.reshape(x_train_tensors, (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))

    # Create RNN Model
    lstm = LSTM_model(n_classes, seq_len, n_hidden, num_layers)  # our lstm class

    # Define the Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model for specified number of epochs
    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensors_final)  # forward pass
        optimizer.zero_grad()  # calculate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop

        # Print at set interval
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # Test after training
    # TODO: Save Model
    return lstm


def train_CNN(x_train, y_train):
    """
    :param x_train:
    :param y_train:
    :return:
    """

    # model parameters
    learning_rate = 0.0001
    num_epochs = 1000
    mo = 0.9
    batch_size = 256
    seq_len = 500
    n_hidden = 256
    n_classes = 2

    # generate data- TODO: make this into function
    x_train_mat = np.ones([10, 16, 16])
    for i in range(10):
        x_train_mat[i] = cgr.FCGR(''.join(utils.num_to_str(x_train[i])), 4)
    y_train = y_train[:10, 0]

    # Tensor Stuff
    train_x = x_train_mat.reshape(10, 1, 16, 16)
    train_x = train_x.astype(np.float32)
    train_x = torch.from_numpy(train_x)

    train_y = torch.from_numpy(y_train)

    # Model
    cnn = CNN_model(n_classes, seq_len, n_hidden)  # our lstm class
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=mo)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        outputs = cnn(train_x)  # forward pass
        optimizer.zero_grad()  # calculate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, train_y)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop

        # Print at set interval
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    print('Finished Training')

    # todo: save model
    """
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    """

    return cnn


def test_model(model, x_test, y_test):
    x_test_tensors = Variable(torch.Tensor(x_test))
    y_test_tensors = Variable(torch.Tensor(y_test))
    X_test_tensors_final = torch.reshape(x_test_tensors, (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

    print("Prediction")
    train_predict = model(X_test_tensors_final)  # forward pass
    return train_predict.data.numpy()


def test_model_cnn(model, x_test, y_test):

    x_test_mat = np.ones([10, 16, 16])
    for i in range(10):
        x_test_mat[i] = cgr.FCGR(''.join(utils.num_to_str(x_test[i])), 4)
    y_train = y_test[:10, 0]

    # Tensor Stuff
    test_x = x_test_mat.reshape(10, 1, 16, 16)
    test_x = test_x.astype(np.float32)
    test_x = torch.from_numpy(test_x)

    train_y = torch.from_numpy(y_train)
    print("Prediction")
    train_predict = model(test_x)  # forward pass
    return train_predict.data.numpy()

###############


# Load data
x = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/train_small.csv", header=None)
y = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/label_small.csv", header=None)

# Transform data from dataframe to numpy array
X_ss = x.to_numpy()
y_mm = y.to_numpy()

# Testing/ Training split:
# #TODO: Implement training + testing split
Xtrain = X_ss[:150, :]
Ytrain = y_mm[:150, :]
Xtest = X_ss[150:, :]
Ytest = y_mm[150:, :]

#m = train_LSTM(Xtrain, Ytrain)
#out = test_model(m, Xtest, Ytest)
#print(np.shape(out))
m2 = train_CNN(Xtrain, Ytrain)
ans=test_model_cnn(m2, Xtest, Ytest)
print(ans)
