"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')

    # Initialize model.

    # Train model via gradient descent.

    # Save trained model to 'trained_weights.np'

    # Load trained model from 'trained_weights.np'

    # Try all other methods: forward, backward, classify, compute accuracy
    A,T = read_dataset(path_to_dataset_folder='data/trainset',index_filename='indexing.txt')
    logistic = LogisticModel(16,'gaussian')
    logistic.fit(T,A,0.001,2000)
    predict = logistic.classify(A)
    print(predict.shape)
    print(logistic.forward(A).shape)
    print(logistic.backward(T,A).shape)
    loss = T - predict
    n = loss.shape[0]
    l = np.count_nonzero(loss)
    accuracy = (n - l)/n
    print(logistic.W)
    print(accuracy)
    #logistic.save_model('codefromscratch'+'/'+'trained_weights.np') 

