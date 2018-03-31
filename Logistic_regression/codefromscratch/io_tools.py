"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...] 
                             where yi is +1/-1, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    f1 = open(path_to_dataset_folder+'/'+index_filename,'r')
    label = f1.read().splitlines()
    y = []
    x = []
    for l in label:
        line = l.split()
        y.append(int(line[0]))
        f2 = open(path_to_dataset_folder+'/'+line[1],'r')
        fea = [1]
        feature = f2.read().splitlines()
        feature = list(map(float,feature[0].split()))
        fea.extend(feature)
        x.append(fea)
        f2.close()
    T = np.array(y)
    A = np.array(x)
    return A,T
