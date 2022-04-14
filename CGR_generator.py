import collections
from matplotlib import cm
import math
import pandas as pd
import pylab
import utils


def count_kmers(sequence, k):
    """"
    Function explanation
    :param sequence: sequence that is being transformed into FCGR
    :param k: size of kmer
    """
    d = collections.defaultdict(int)
    for i in range(len(sequence) - (k - 1)):
        d[sequence[i:i + k]] += 1
    for key in d.keys():
        if "N" in key:
            del d[key]
    return d


def probabilities(kmer_count, k, n):
    """"
    Function explanation
    :param kmer_count:
    :param k: size of kmers
    :param n: length of sequence
    :return:
    """
    probs = collections.defaultdict(float)
    for key, value in kmer_count.items():
        probs[key] = float(value) / (n - k + 1)
    return probs


def FCGR(seq, k):
    """
    Function does X
    :param k:
    :return:
    """
    # Get Count of Kmers
    kmer_counts = count_kmers(seq, k)
    # Get Probability
    kmer_prob = probabilities(kmer_counts, k, len(seq))

    # Initialize matrices
    array_size = int(math.sqrt(4 ** k))  # array size - depends on K
    chaos = []

    # Create chaos empty chaos matrix
    for i in range(array_size):
        chaos.append([0] * array_size)

    # loop for all bases
    for key, value in kmer_prob.items():
        maxX = array_size
        maxY = array_size
        posX = 1
        posY = 1

        # Calculate position in the array
        for char in key:
            if char == "T":
                posX += maxX / 2
            elif char == "C":
                posY += maxY / 2
            elif char == "G":
                posX += maxX / 2
                posY += maxY / 2
            maxX = maxX / 2
            maxY /= 2

        # Save value into Array
        chaos[int(posY) - 1][int(posX) - 1] = value

    return chaos


def sample():
    disp = True
    # Load data
    x_train = pd.read_csv("/Users/jacobstambaugh/Documents/RNN-VirSeeker/data/train_small.csv", header=None)
    data = ''.join(utils.num_to_str(x_train[1]))

    chaos_k4 = FCGR(data, 4)
    print(chaos_k4)

    if disp:
        pylab.title('Chaos game representation for K-mers')
        pylab.imshow(chaos_k4, interpolation='nearest', cmap=cm.gray_r)
        pylab.show()
