import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    # Load data
    X_train = np.loadtxt(open("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/rnn_train.csv", "rb"), delimiter=",", skiprows=0)
    y_train = np.loadtxt(open("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/label_train.csv", "rb"), delimiter=",", skiprows=0)

    # model parameters
    learning_rate = 0.00001
    training_iters = 1000
    batch_size = 256
    display_step = 10
    seq_max_len = 500
    n_hidden = 256
    n_classes = 2



main()
