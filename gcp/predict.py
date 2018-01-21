from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import sys

from midi import midi_to_label
from data_util import filter_low, expand_classes
from knn import knn_predict

default_x_threshold = 0.5
start_class = 40
end_class = 89
num_classes = end_class - start_class + 1
num_classes += 1 # for empty note
K = 15

# Training data
df = pd.read_csv("../data2.csv", header=None)
df = df.sample(frac=1, random_state=50)
TARGET = [df.columns[-1]]
FEATS = [c for c in df.columns if c != TARGET[0]]

X_train = filter_low(df[FEATS].values, default_x_threshold)
y_train = expand_classes(X_train, df[TARGET].values, num_classes, start_class)

# Test data
df = pd.read_csv(sys.argv[1], header=None)
X_test = filter_low(df[df.columns].values, default_x_threshold)

y_pred = knn_predict(K, num_classes, X_train, y_train, X_test)
map(lambda pred: print(midi_to_label(pred + start_class)), y_pred)
        