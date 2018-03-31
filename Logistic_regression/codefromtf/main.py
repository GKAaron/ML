"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')

    # Initialize model.

    # Build TensorFlow training graph
    
    # Train model via gradient descent.

    # Compute classification accuracy based on the return of the "fit" method
    
    A, T = read_dataset_tf('data/trainset','indexing.txt')
    logistic = LogisticModel_TF(16)
    logistic.build_graph(0.001)
    predict=logistic.fit(T,A,1000)
    predict[predict<0.5] = 0
    predict[predict>0.5] = 1
    loss = T - predict
    accuracy = 1-np.count_nonzero(loss)/T.shape[0]
    print(accuracy)

    
if __name__ == '__main__':
    tf.app.run()
