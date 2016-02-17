
#
# from https://github.com/nlintz/TensorFlow-Tutorials
# and heavily, heavily modified
#

import tensorflow as tf
import numpy as np
import numpy.random as npr
import time
import sys
import operator as op


num_nets = 5
num_hiddens = 150
num_epochs = 1
num_overall_epochs = 5
minibatch_size = 128


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    """ Stolen from some version of TF docs """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

with open("corpus.txt") as corpus_file:
    chars = list(corpus_file.read())
    print len(chars)
    chars = chars[:50000]
    vocab_size = len(set(chars))
    char_to_idx = {char: idx for idx, char in enumerate(list(set(chars)))}
    idx_to_char = {idx: char for idx, char in enumerate(list(set(chars)))}
    train_len = ((9 * len(chars)) // 10)
    all_data = []
    for fst, snd in zip(chars, chars[1:]):
        all_data.append((char_to_idx[fst], char_to_idx[snd]))
    train_list = all_data[:train_len]
    test_list = all_data[train_len:]

    train_xs = np.array(map(op.itemgetter(0), train_list))
    trXs = dense_to_one_hot(train_xs, vocab_size)

    train_ys = np.array(map(op.itemgetter(1), train_list))
    trYs = dense_to_one_hot(train_ys, vocab_size)

    test_xs = np.array(map(op.itemgetter(0), test_list))
    teXs = dense_to_one_hot(test_xs, vocab_size)

    test_ys = np.array(map(op.itemgetter(1), test_list))
    teYs = dense_to_one_hot(test_ys, vocab_size)
print "finished processing corpus"

input_dim = vocab_size
output_dim = vocab_size

X0 = tf.placeholder("float", [None, input_dim])
Xs = [X0]
for x in xrange(num_nets):
    locals()["X" + str(x+1)] =\
        tf.placeholder("float", [None, num_hiddens + input_dim])
    # more than a bit precious, I'm afraid
    Xs.append(locals()["X" + str(x+1)])

Y = tf.placeholder("float", [None, output_dim])

w_hs = [tf.Variable(tf.random_normal([input_dim, num_hiddens], stddev=0.01))]
for x in xrange(num_nets):
    extremal_val = np.sqrt(6) / np.sqrt(num_hiddens + input_dim + num_hiddens)
    w_hs.append(
        tf.Variable(
            tf.random_uniform(
                [num_hiddens + input_dim, num_hiddens],
                minval=-extremal_val,
                maxval=extremal_val
                )
            )
        )

w_os = [tf.Variable(tf.random_normal([num_hiddens, output_dim], stddev=0.01))]
for x in xrange(num_nets):
    extremal_val = np.sqrt(6) / np.sqrt(num_hiddens + output_dim)
    w_os.append(
        tf.Variable(
            tf.random_uniform(
                [num_hiddens, output_dim],
                minval=-extremal_val,
                maxval=extremal_val
            )
        )
    )

hs = [tf.nn.tanh(
    tf.matmul(Xs[idx], w_h))
    for idx, w_h in enumerate(w_hs)]
py_xs = [tf.matmul(h, w_os[x]) for x, h in enumerate(hs)]

costs = [tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    for py_x in py_xs]

train_ops = [tf.train.AdamOptimizer().minimize(cost)
             for cost in costs]
predict_ops = [tf.argmax(py_x, 1) for py_x in py_xs]

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


def sample(sess, seeds, n, vocab_size, idx_to_char):
    for x in xrange(n):
        curr_feed_dict = {globals()["Y"]: np.zeros((1, vocab_size))}
        for net_idx in xrange(num_nets):
            datum = np.zeros((1, vocab_size))
            datum[0, seeds[net_idx]] = 1.0
            if net_idx == 0:
                next_datum = datum
            else:
                next_datum = np.hstack((datum, new_h))
            curr_feed_dict[globals()["X" + str(net_idx)]] = next_datum
            new_h = hs[net_idx].eval(session=sess, feed_dict=curr_feed_dict)
        # remember, we don't use that last net
        # yes, it's idiotic
        y = py_xs[-2].eval(session=sess, feed_dict=curr_feed_dict)
        p = np.exp(y) / np.sum(np.exp(y))
        curr_sample_idx = npr.choice(range(vocab_size), p=p.ravel())
        sys.stdout.write(idx_to_char[curr_sample_idx])
        sys.stdout.flush()
        seeds.append(curr_sample_idx)
        seeds.pop(0)

for overall_epoch in xrange(num_overall_epochs):
    print "=================="
    print "overall epoch : ", overall_epoch, " / ", num_overall_epochs
    print "=================="
    curr_trX = trXs[:]
    curr_teX = teXs[:]
    for net_idx, curr_train_ops in enumerate(train_ops):
        # the last "net" doesn't work, don't feel like debugging
        if net_idx == len(train_ops) - 1:
            break
        print "=================="
        print "net : ", net_idx, " / ", num_nets
        print "=================="
        te_fd = {Y: teYs[net_idx:]}
        te_fd[locals()["X" + str(net_idx)]] = curr_teX[:]
        for i in range(num_epochs):
            for start, end in zip(range(0, len(curr_trX), minibatch_size), range(minibatch_size, len(curr_trX), minibatch_size)):
                tr_fd = {Y: trYs[start+net_idx:end+net_idx]}
                tr_fd[locals()["X" + str(net_idx)]] = curr_trX[start:end]
                sess.run(curr_train_ops, feed_dict=tr_fd)
            # use prediction accuracy because I can't be bothered to do perplexity properly right now
            curr_acc = np.mean(np.argmax(teYs[net_idx:], axis=1) ==
                               sess.run(predict_ops[net_idx], feed_dict=te_fd))
            print i, " / ", num_epochs, " || ",  curr_acc, time.clock()
        total_tr_fd = {Y: trYs[net_idx:]}
        total_tr_fd[locals()["X" + str(net_idx)]] = curr_trX[:]
        if net_idx < (num_nets-1):
            curr_trX = np.hstack((trXs[net_idx+1:], hs[net_idx].eval(session=sess, feed_dict=total_tr_fd)[:-1]))
            curr_teX = np.hstack((teXs[net_idx+1:], hs[net_idx].eval(session=sess, feed_dict=te_fd)[:-1]))

seeds = [char_to_idx[char] for char in chars[:num_nets+1]]
# and merrily use our global state this way...?
sample(sess, seeds, 400, vocab_size, idx_to_char)
