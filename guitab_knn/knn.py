from __future__ import print_function

import numpy as np
import tensorflow as tf

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