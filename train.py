import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd

def main():
    # Load data
    x_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/rnn_train.csv")
    y_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/label_train.csv", usecols=[0])

    # TODO: Implement training + testing split

    # model parameters
    learning_rate = 0.00001
    training_iters = 1000
    batch_size = 256
    display_step = 10
    seq_max_len = 500
    n_hidden = 256  # number of units in RNN cell
    n_classes = 2

    # convert to tensors
    x_train_tensors = Variable(torch.Tensor(x_train))
    y_train_tensors = Variable(torch.Tensor(y_train))


main()
