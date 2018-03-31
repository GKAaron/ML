"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    s = np.concatenate((processed_dataset[0],processed_dataset[1]),1)
    n = s.shape[0]
    for i in range(num_steps):
        if shuffle == True:
            np.random.shuffle(s)
        x_all = s[:, :-1]
        y_all = s[:, -1:]
        if n%batch_size == 0:
            batch = np.random.randint(n/batch_size)
        else:
            batch = np.random.randint(int(n/batch_size)+1)
        j = min((batch+1)*batch_size,n)    
        x = x_all[batch*batch_size:j]
        y = y_all[batch*batch_size:j]
        update_step(x, y, model, learning_rate)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    n = x_batch.shape[0]
    f = model.forward(x_batch)
    total_grad = model.backward(f, y_batch)
    model.w = model.w - learning_rate*total_grad
    


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    x = processed_dataset[0]
    b = np.ones((x.shape[0],1))
    x = np.concatenate((x,b),1)
    y = processed_dataset[1]
    I = np.identity(x.shape[1])
    model.w = np.linalg.inv(x.T.dot(x)+model.w_decay_factor*I).dot(x.T).dot(y)


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x = processed_dataset[0]
    n = x.shape[0]
    y = processed_dataset[1]
    f = model.forward(x)
    loss = model.total_loss(f, y)
    return loss
