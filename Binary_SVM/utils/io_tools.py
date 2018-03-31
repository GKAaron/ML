"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    f1 = open(data_txt_file,'r')
    raw = f1.read().splitlines()
    x = []
    y = []
    for l in raw:
        line = l.split(',')
        p = os.path.join(image_data_path,line[0]+'.jpg')
        image = io.imread(p)
        y.append(line[1])
        x.append(image)
    image = np.array(x)
    label = np.array(y,'int32').reshape((-1,1))
    data['image'] = image
    data['label'] = label
    return data
