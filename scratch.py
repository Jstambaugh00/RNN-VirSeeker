import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from torch import nn
import torch
from torch.autograd import Variable


def forward(self, x, prev_state):
    embed = self.embedding(x)
    output, state = self.lstm(embed, prev_state)
    logits = self.fc(output)
    return logits, state


