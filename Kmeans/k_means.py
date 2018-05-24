from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
df = pd.read_csv('data/data/iris.data',usecols=[0,1,2,3])
data = df.values
# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def reassign(c):
    dis =  np.empty((data.shape[0],k),dtype='float64')
    for i in range(k):
        dis[:,i] = np.sum(np.square(data-c[i]),axis=1)
    lab = np.argmin(dis,axis=1)
    return lab

def recenter(lab,c):
    new_c = np.empty(c.shape,dtype='float64')
    for i in range(k):
        new_c[i] = np.mean(data[lab==i],axis=0)
    return new_c

def k_means(C):
    # Write your code here!
    new_c = np.array(C)
    err = 1
    while err>10e-3:
        old_c = new_c
        lab = reassign(old_c)
        new_c = recenter(lab,old_c)
        err = np.sum(np.abs(new_c-old_c))
    C_final = new_c
    return C_final









