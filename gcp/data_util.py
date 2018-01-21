import numpy as np

def filter_low(matrix, threshold):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if(val < threshold):
                matrix[i][j] = 0
    return matrix

def expand_classes(X, y, nclasses, start):
    new_y = []
    for i, cl in enumerate(y):
        arr = np.zeros(nclasses)
        if(any(X[i, :])):
            arr[cl - start - 1] = 1
        else:
            arr[nclasses - 1] = 1
        new_y.append(arr)
    return np.array(new_y)