import idx2numpy
import numpy as np

def load_data():
    X_train_raw = idx2numpy.convert_from_file(r'C:\Users\prith\Downloads\gzip\gzip\emnist-letters-train-images-idx3-ubyte\emnist-letters-train-images-idx3-ubyte')
    Y_train_raw = idx2numpy.convert_from_file(r'C:\Users\prith\Downloads\gzip\gzip\emnist-letters-train-labels-idx1-ubyte\emnist-letters-train-labels-idx1-ubyte')
    X_test_raw  = idx2numpy.convert_from_file(r'C:\Users\prith\Downloads\gzip\gzip\emnist-letters-test-images-idx3-ubyte\emnist-letters-test-images-idx3-ubyte')
    Y_test_raw  = idx2numpy.convert_from_file(r'C:\Users\prith\Downloads\gzip\gzip\emnist-letters-test-labels-idx1-ubyte\emnist-letters-test-labels-idx1-ubyte')

    X_flat = X_train_raw.reshape(X_train_raw.shape[0], 784)
    data = np.column_stack([Y_train_raw, X_flat])
    data = np.array(data)

    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    Y_train = Y_train - 1
    Y_dev = Y_dev - 1

    return X_train, Y_train, X_dev, Y_dev