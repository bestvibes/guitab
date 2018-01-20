import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.1
batch_size = 128
display_step = 50

# Network Parameters
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
num_input = 31 # 31 fft bins
num_classes = 1 # midi note

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def nn_train(get_data, filenames):
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for filename in filenames:
            X_train, y_train = get_data(filename)
            X_train = X_train[:len(X_train)*3/4]
            y_train = np.asarray(y_train).reshape(1, 1)
            for (step, m) in enumerate(X_train):
                # Run optimization op (backprop)
                x_train = np.asarray(m).astype(np.float32).reshape(1, 31)
                sess.run(train_op, feed_dict={X: x_train, Y: y_train})

                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_train,
                                                                         Y: y_train})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        saver.save(sess, './train_save/model')
        print("Optimization Finished!")

def nn_eval(get_data, filenames):
    # Start training
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './train_save/model')
        for filename in filenames:
            X_test, y_test = get_data(filename)
            X_test = X_test[len(X_test)*3/4:]
            y_test = np.asarray(y_test).reshape(1, 1)
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={X: X_test,
                                              Y: y_test}))