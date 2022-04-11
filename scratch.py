import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from torch import nn
import torch
from torch.autograd import Variable
import pandas as pd


"""def forward(self, x, prev_state):
    embed = self.embedding(x)
    output, state = self.lstm(embed, prev_state)
    logits = self.fc(output)
    return logits, state"""
x_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/train_small.csv", header=None)
y_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/label_small.csv", usecols=[0],header=None)

print(len(x_train))
print(len(y_train))

