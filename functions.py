import numpy as np
from scipy.io import wavfile
import os
from tqdm import tqdm


def import_data(n_train, n_test, label_names):
    """imports data from label folders"""
    """
    n_train: number of training examples of each label
    real n_train = n_train * n_labels
    n_test: number of test examples
    label_names: label names
    """
    """returns: X_train, y_train, X_test, y_test, fs 
    (y vals: onehot notation required in forward processing)"""
    X_train = np.empty((0, 16000))  # arbitrary assumming that fs = 16kHz (vstack does't work)
    y_train = np.array([])
    X_test = np.empty((0, 16000))
    y_test = np.array([])
    for label_i in tqdm(range(len(label_names))):
        fnames = os.listdir('data/' + label_names[label_i] + '/')
        if n_train + n_test > len(fnames):
            return -1
        else:
            for file_i in range(0, n_train + n_test):
                data = np.array([])
                fs, data = wavfile.read(
                    'data/' + label_names[label_i] + '/' + fnames[file_i])
                if len(data) > fs:
                    data = data[0:fs]
                elif len(data) < fs:
                    for data_i in range(len(data), fs):
                        data = np.append(data, 0)
                data = np.transpose(data)
                if file_i in range(0, n_train):
                    X_train = np.vstack((X_train, data))
                    y_train = np.hstack((y_train, label_names[label_i]))
                elif file_i in range(n_train, n_train + n_test):
                    X_test = np.vstack((X_test, data))
                    y_test = np.hstack((y_test, label_names[label_i]))

    return X_train, y_train, X_test, y_test, fs


def one_hot(labels, label_names):
    """returns labels in onehot notation"""
    y_onehot = np.zeros((len(labels), len(label_names)))
    for i in range(len(labels)):
        for j in range(len(label_names)):
            if labels[i] == label_names[j]:
                y_onehot[i, j] = 1           
    return y_onehot


def standardize(input_data):
    sigma = np.std(input_data, axis=1)
    size = input_data.shape[0]
    sigma = sigma.reshape((size, 1))
    mean_val = np.mean(input_data, axis=1)
    mean_val = mean_val.reshape((size, 1))
    return (input_data-mean_val[:, None]) / sigma[:, None]


def y_to_num(y, label_names):
    y_num = np.array([])
    for i in range(len(y)):
        for j in range(len(label_names)):
            if y[i] == label_names[j]:
                y_num = np.append(y_num, j)
    return y_num

