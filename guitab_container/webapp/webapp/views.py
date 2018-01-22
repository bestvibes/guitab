from __future__ import print_function
from django.http import HttpResponse

from importlib import import_module
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import os
import csv
import ast

##### MIDI_TO_LABEL

strings = [40, 45, 50, 55, 60, 65];

def midi_to_label(midi):
    if (midi < strings[0]):
        raise ValueError("Note " + note + " is not playable on a guitar in standard tuning.")

    idealString = 0
    for string, string_midi in enumerate(strings):
        if (midi < string_midi):
            break
        idealString = string

    label = [-1, -1, -1, -1, -1, -1]
    label[idealString] = midi - strings[idealString];
    return label

######

##### DATA_UTIL

def filter_low(matrix, threshold):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if(float(val) < threshold):
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

#######

####### KNN

def knn_predict(K, num_classes, X_train, y_train, X_test):
	# tf Graph Input
	xtr = tf.placeholder("float", [None, 255])
	ytr = tf.placeholder("float", [None, num_classes])
	xte = tf.placeholder("float", [255])

	# Euclidean Distance
	distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1)))
	# Prediction: Get min distance neighbors
	values, indices = tf.nn.top_k(distance, k=K, sorted=False)

	nearest_neighbors = []
	for i in range(K):
	    nearest_neighbors.append(tf.argmax(ytr[indices[i]], 0))

	neighbors_tensor = tf.stack(nearest_neighbors)
	y, idx, count = tf.unique_with_counts(neighbors_tensor)
	pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:
	    # Run the initializer
	    sess.run(init)

	    outp = []
	    num_tests = len(X_test)

	    # loop over test data
	    for i in range(num_tests):
	        # Get nearest neighbor
	        nn_index = sess.run(pred, feed_dict={xtr: X_train, ytr: y_train, xte: X_test[i, :]})
	        #print("Test", i, "Prediction:", nn_index)
	        outp.append(nn_index)
	    return outp


#########


default_x_threshold = 0.5
start_class = 40
end_class = 89
num_classes = end_class - start_class + 1
num_classes += 1 # for empty note
K = 15

def index(request):
	if request.method == 'POST':
		data = request.POST

		if 'input' in data:
			print(data['input'])
			inputData = ast.literal_eval(data['input'])

			# Training data
			df = pd.read_csv("./webapp/data2.csv", header=None)
			df = df.sample(frac=1, random_state=50)
			TARGET = [df.columns[-1]]
			FEATS = [c for c in df.columns if c != TARGET[0]]

			X_train = filter_low(df[FEATS].values, default_x_threshold)
			y_train = expand_classes(X_train, df[TARGET].values, num_classes, start_class)

			# Test data
			with open("input.csv", 'a') as csv_file:
				writer = csv.writer(csv_file, delimiter=',')
				for item in inputData:
					print(item)
					writer.writerow(item)

			df = pd.read_csv("input.csv", header=None)

			os.remove("input.csv")
			X_test = filter_low(df[df.columns].values, default_x_threshold)

			y_pred = knn_predict(K, num_classes, X_train, y_train, X_test)
			y_pred = list(map(lambda x: x + start_class, y_pred))

			return HttpResponse(str(y_pred))
		return HttpResponse("No, Post!")

	return HttpResponse("Hello, world!")