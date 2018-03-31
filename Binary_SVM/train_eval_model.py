"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    x = data['image']
    y = data['label']
    n = x.shape[0]
    m = np.concatenate((x,y),1)
    for i in range(num_steps):
        if shuffle:
            np.random.shuffle(m)
        x = m[:,:-1]
        y = m[:,-1:]
        if n%batch_size ==0:
            batch = np.random.randint(n/batch_size)
        else :
            batch = np.random.randint(int(n/batch_size)+1)
        x_batch = x[batch_size*batch:min(batch_size*(batch+1),n)]
        y_batch = y[batch_size*batch:min(batch_size*(batch+1),n)]
        update_step(x_batch,y_batch,model,learning_rate)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    f = model.forward(x_batch)
    gradient = model.backward(f,y_batch)
    model.w = model.w - learning_rate*gradient


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    model.w = z[0:model.ndims+1]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    w_b = np.identity(model.ndims+1)*model.w_decay_factor
    ep1 = np.zeros((model.ndims+1,data['image'].shape[0]))
    ep2 = np.zeros((data['image'].shape[0],model.ndims+1+data['image'].shape[0]))
    P = np.concatenate((np.concatenate((w_b,ep1),1),ep2),0)
    ep1 = np.zeros((model.ndims+1,1))
    ep2 = np.ones((data['image'].shape[0],1))
    q = np.concatenate((ep1,ep2),0)
    y = data['label']
    bias = np.ones((data['image'].shape[0],1))
    x = np.concatenate((data['image'],bias),1)
    y_x = -1*y*x
    ep3 = -1*np.identity(x.shape[0])
    ep4 = np.zeros_like(y_x)
    g1 = np.concatenate((y_x,ep3),1)
    g2 = np.concatenate((ep4,ep3),1)
    G = np.concatenate((g1,g2),0)
    h1 = -1*np.ones((x.shape[0],1))
    h2 = np.zeros((x.shape[0],1))
    h = np.concatenate((h1,h2),0)
    # Implementation here.
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    f = model.forward(data['image'])
    loss = model.total_loss(f,data['label'])
    y_predict = model.predict(f)
    y = y_predict - data['label']
    y = np.count_nonzero(y)
    acc = (y_predict.shape[0]-y)/y_predict.shape[0]
    return loss, acc
