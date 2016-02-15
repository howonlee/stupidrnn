
#
# from https://github.com/nlintz/TensorFlow-Tutorials
#

import tensorflow as tf
import numpy as np
import input_data
import time


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

num_nets = 5
num_hiddens = 100
input_dim = 784

X0 = tf.placeholder("float", [None, input_dim])
Xs = [X0]
for x in xrange(num_nets):
    locals()["X" + str(x+1)] = tf.placeholder("float", [None, num_hiddens + input_dim])
    # more than a bit precious, I'm afraid
    Xs.append(locals()["X" + str(x+1)])

Y = tf.placeholder("float", [None, 10])

w_hs = [tf.Variable(tf.random_normal([input_dim, num_hiddens], stddev=0.01))]
for x in xrange(num_nets):
    w_hs.append(tf.Variable(tf.random_normal([num_hiddens + input_dim, num_hiddens], stddev=0.01)))

w_os = [tf.Variable(tf.random_normal([num_hiddens, 10], stddev=0.01))]
for x in xrange(num_nets):
    w_os.append(tf.Variable(tf.random_normal([num_hiddens, 10], stddev=0.01)))

hs = [tf.nn.sigmoid(tf.matmul(Xs[idx], w_h)) for idx, w_h in enumerate(w_hs)]
py_xs = [tf.matmul(h, w_os[x]) for x, h in enumerate(hs)]

costs = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) for py_x in py_xs]
train_ops = [tf.train.GradientDescentOptimizer(0.05).minimize(cost) for cost in costs]
predict_ops = [tf.argmax(py_x, 1) for py_x in py_xs]

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

curr_trX = trX[:]
curr_teX = teX[:]
for net_idx, curr_train_ops in enumerate(train_ops):
    print "=================="
    print "net : ", net_idx
    print "=================="
    te_fd = {Y: teY}
    te_fd[locals()["X" + str(net_idx)]] = curr_teX
    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            tr_fd = {Y: trY[start:end]}
            tr_fd[locals()["X" + str(net_idx)]] = curr_trX[start:end]
            sess.run(curr_train_ops, feed_dict=tr_fd)
        curr_acc = np.mean(np.argmax(teY, axis=1) ==
                           sess.run(predict_ops[net_idx], feed_dict=te_fd))
        print i, curr_acc, time.clock()
    total_tr_fd = {Y: trY[:]}
    total_tr_fd[locals()["X" + str(net_idx)]] = curr_trX[:]
    curr_trX = np.hstack((trX, hs[net_idx].eval(session=sess, feed_dict=total_tr_fd)))
    curr_teX = np.hstack((teX, hs[net_idx].eval(session=sess, feed_dict=te_fd)))
