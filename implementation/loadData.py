from astropy.io import fits
import os
import numpy as np
from numpy import random


def loadFITS(dir_path, number_of_files, first_index=0):
    """Returns a numpy.ndarray containing image data. In particular, 'data' is an array containing 'number_of_files'
    image-arrays in the following format: (height, width, color_depth). The first file in 'dir_path' can be specified 
    with 'first_index'."""
    # read file names in namelist
    filename_list = []
    for file in os.listdir(dir_path):
        if file.endswith(".fits"):
            filename_list.append(file)
    # read image header from the first image in dir
    sample = fits.open(dir_path + "/" + filename_list[0])
    image_shape = sample[0].data.shape
    pixel_type = sample[0].data.dtype.name
    # declare output ndarray
    last_index = min(first_index + number_of_files, len(filename_list))
    shape = tuple([last_index - first_index]) + image_shape
    data = np.empty(shape=shape,dtype=pixel_type)
    # sequentially read image data into data array
    for i in range(first_index, last_index):
        with fits.open(dir_path + "/" + filename_list[i]) as hdul:
            data[i-first_index] = hdul[0].data
    # transpose image according to format (height, width, color_depth)
    return np.transpose(data, (0,3,2,1))

def loadData(dir_path, number_of_files, first_index=0):
    """Returns a tuple of 2 numpy.ndarrays (x_data, y_data).
    Each is not longer than 'number_of_files'.
    'x_data' contains 3-dimensional numpy.ndarrays
    representing the input image. 'y_data' contains
    1-dimensional numpy.ndarrays representing the unit vector corresponding 
    to the correct output of the model."""
    x_data = loadFITS(dir_path, number_of_files, first_index)
    y_data = np.ones((len(x_data), 1))
    if "nonlens" in dir_path:
        for i in range(len(x_data)):
            y_data[i, 0] = 0
    return (x_data, y_data)

def createTrainingSet(number_of_files, dir_path, first_index=0):
    """Return a tuple (x_train, y_train) containing 'number_of_files' 
    FITS from each directory."""
    x1_train, y1_train = loadData(dir_path + "/lens_1", number_of_files, first_index)
    x2_train, y2_train = loadData(dir_path + "/lens_2", number_of_files, first_index)
    x3_train, y3_train = loadData(dir_path + "/lens_3", number_of_files, first_index)
    x4_train, y4_train = loadData(dir_path + "/lens_4", number_of_files, first_index)
    x5_train, y5_train = loadData(dir_path + "/nonlens_1", number_of_files, first_index)
    x6_train, y6_train = loadData(dir_path + "/nonlens_2", number_of_files, first_index)
    x7_train, y7_train = loadData(dir_path + "/nonlens_3", number_of_files, first_index)
    x8_train, y8_train = loadData(dir_path + "/nonlens_4", number_of_files, first_index)

    x3_train = np.rot90(x3_train, k=1, axes=(1,2))
    x4_train = np.rot90(x4_train, k=1, axes=(1,2))
    x5_train = np.rot90(x5_train, k=2, axes=(1,2))
    x6_train = np.rot90(x6_train, k=2, axes=(1,2))
    x7_train = np.rot90(x7_train, k=3, axes=(1,2))
    x8_train = np.rot90(x8_train, k=3, axes=(1,2))

    x_train = np.concatenate([x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train])
    y_train = np.concatenate([y1_train, y2_train, y3_train, y4_train, y5_train, y6_train, y7_train, y8_train])

    #shuffle the training set
    indices = random.permutation(x_train.shape[0])
    x_train = x_train[indices]
    y_train = y_train[indices]

    return x_train, y_train