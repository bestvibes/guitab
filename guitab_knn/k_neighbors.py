from __future__ import print_function

import numpy as np
import pandas as pd
import sys

from data_util import filter_low, expand_classes
from knn import knn_predict

default_x_threshold = 0.5
start_class = 40
end_class = 89
num_classes = end_class - start_class + 1
num_classes += 1 # for empty note
K = 15

# Training data
df = pd.read_csv("data2.csv", header=None)
df = df.sample(frac=1, random_state=50)
TARGET = [df.columns[-1]]
FEATS = [c for c in df.columns if c != TARGET[0]]

X_train = filter_low(df[FEATS][:int(round(0.8*df.shape[0]))].values, default_x_threshold)
y_train = expand_classes(X_train, df[TARGET][:int(round(0.8*df.shape[0]))].values, num_classes, start_class)

# Test data
X_test = filter_low(df[FEATS][int(round(0.8*df.shape[0])):].values, default_x_threshold)
y_test = expand_classes(X_test, df[TARGET][int(round(0.8*df.shape[0])):].values, num_classes, start_class)

y_pred = knn_predict(K, num_classes, X_train, y_train, X_test)

num_tests = len(X_test)
num_right = 0.
for i, pred in enumerate(y_pred):
    if pred == np.argmax(y_test[i]):
            num_right += 1
print("Accuracy:", num_right / num_tests)
    