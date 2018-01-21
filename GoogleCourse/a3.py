from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

image_size = 28
num_labels = 10
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    global_step = tf.Variable(0)  # count the number of steps taken.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    tf_l2norm_ratio = tf.constant(0.000001)

    # Variables.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, 1024]))
    weights2 = tf.Variable(tf.truncated_normal([1024, 300]))
    weights3 = tf.Variable(tf.truncated_normal([300, 50]))
    weights4 = tf.Variable(tf.truncated_normal([50, num_labels]))
    biases = tf.Variable(tf.zeros([1024]))
    biases2 = tf.Variable(tf.zeros([300]))
    biases3 = tf.Variable(tf.zeros([50]))
    biases4 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    hidden = tf.matmul(tf_train_dataset, weights) + biases
    hidden_val = tf.matmul(tf_valid_dataset, weights) + biases
    hidden_test = tf.matmul(tf_test_dataset, weights) + biases
    relu = tf.nn.relu(hidden)
    relu_dropout = tf.nn.dropout(relu, 0.5)
    relu_val = tf.nn.relu(hidden_val)
    relu_test = tf.nn.relu(hidden_test)
    hidden2 = tf.matmul(relu_dropout, weights2) + biases2
    hidden2_val = tf.matmul(relu_val, weights2) + biases2
    hidden2_test = tf.matmul(relu_test, weights2) + biases2
    relu2 = tf.nn.relu(hidden2)
    relu2_dropout = tf.nn.dropout(relu2, 0.5)
    relu2_val = tf.nn.relu(hidden2_val)
    relu2_test = tf.nn.relu(hidden2_test)
    hidden3 = tf.matmul(relu2_dropout, weights3) + biases3
    hidden3_val = tf.matmul(relu2_val, weights3) + biases3
    hidden3_test = tf.matmul(relu2_test, weights3) + biases3
    relu3 = tf.nn.relu(hidden3)
    relu3_dropout = tf.nn.dropout(relu3, 0.5)
    relu3_val = tf.nn.relu(hidden3_val)
    relu3_test = tf.nn.relu(hidden3_test)
    logits = tf.matmul(relu3_dropout, weights4) + biases4
    logits_val = tf.matmul(relu3_val, weights4) + biases4
    logits_test = tf.matmul(relu3_test, weights4) + biases4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    l2_loss = loss + tf_l2norm_ratio*(tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3) + tf.nn.l2_loss(weights4))

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.001, global_step, 10000, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(logits_val)
    test_prediction = tf.nn.softmax(logits_test)


num_steps = 30001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))