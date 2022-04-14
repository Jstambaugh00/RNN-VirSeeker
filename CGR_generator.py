import collections
from matplotlib import cm
import math
import pandas as pd
import pylab
import utils


x_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/train_small.csv", header=None)
data = ''.join(utils.num_to_str(x_train[1]))


def count_kmers(sequence, k):
    d = collections.defaultdict(int)
    for i in range(len(data) - (k - 1)):
        d[sequence[i:i + k]] += 1
    for key in d.keys():
        if "N" in key:
            del d[key]
    return d


def probabilities(kmer_count, k):
    probabilities = collections.defaultdict(float)
    N = len(data)
    for key, value in kmer_count.items():
        probabilities[key] = float(value) / (N - k + 1)
    return probabilities


def chaos_game_representation(probabilities, k):
    array_size = int(math.sqrt(4 ** k))
    chaos = []
    for i in range(array_size):
        chaos.append([0] * array_size)

    maxx = array_size
    maxy = array_size
    posx = 1
    posy = 1
    for key, value in probabilities.items():
        for char in key:
            if char == "T":
                posx += maxx / 2
            elif char == "C":
                posy += maxy / 2
            elif char == "G":
                posx += maxx / 2
                posy += maxy / 2
            maxx = maxx / 2
            maxy /= 2
        chaos[int(posy) - 1][int(posx) - 1] = value
        maxx = array_size
        maxy = array_size
        posx = 1
        posy = 1

    return chaos


f4 = count_kmers(data, 4)
f4_prob = probabilities(f4, 4)


chaos_k4 = chaos_game_representation(f4_prob, 4)
pylab.title('Chaos game representation for 4-mers')
pylab.imshow(chaos_k4, interpolation='nearest', cmap=cm.gray_r)
pylab.show()