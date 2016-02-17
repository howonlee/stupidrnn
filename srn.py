
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


def onehot(idx, vocab_size):
    arr = np.zeros(vocab_size)
    arr[idx] = 1.0
    return arr

num_nets = 20
num_hiddens = 200
num_epochs = 30
minibatch_size = 500


def make_data_arr(data_list, vocab_size):
    arr = np.zeros((len(data_list), vocab_size))
    for idx, datum in enumerate(data_list):
        arr[idx, datum] = 1.0
    return arr

with open("corpus.txt") as corpus_file:
    chars = list(corpus_file.read().lower())
    print len(chars)
    chars = chars[:200000]
    vocab_size = len(set(chars))
    char_to_idx = {char: idx for idx, char in enumerate(list(set(chars)))}
    idx_to_char = {idx: char for idx, char in enumerate(list(set(chars)))}
    trXs, teXs, trYs, teYs = [], [], [], []
    train_len = ((19 * len(chars)) // 20)
    for net_idx in xrange(num_nets):
        print net_idx
        all_data = []
        for fst, snd in zip(chars, chars[1:]):
            all_data.append((char_to_idx[fst], char_to_idx[snd]))
        train_list = all_data[:train_len][net_idx:]
        test_list = all_data[train_len:][net_idx:]

        train_xs = map(op.itemgetter(0), train_list)
        trXs.append(make_data_arr(train_xs, vocab_size))

        train_ys = map(op.itemgetter(1), train_list)
        trYs.append(make_data_arr(train_ys, vocab_size))

        test_xs = map(op.itemgetter(0), test_list)
        teXs.append(make_data_arr(test_xs, vocab_size))

        test_ys = map(op.itemgetter(1), test_list)
        teYs.append(make_data_arr(test_ys, vocab_size))
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

w_hs = [tf.Variable(tf.random_normal([input_dim, num_hiddens], stddev=0.0001))]
for x in xrange(num_nets):
    w_hs.append(
        tf.Variable(
            tf.random_normal(
                [num_hiddens + input_dim, num_hiddens], stddev=0.0001
                )
            )
        )

bs = [tf.Variable(tf.random_normal([num_hiddens], stddev=0.0001))]
for x in xrange(num_nets):
    bs.append(
        tf.Variable(tf.random_normal([num_hiddens], stddev=0.0001)))

b_os = [tf.Variable(tf.random_normal([output_dim], stddev=0.0001))]
for x in xrange(num_nets):
    b_os.append(
        tf.Variable(tf.random_normal([output_dim], stddev=0.0001)))

w_os = [tf.Variable(tf.random_normal([num_hiddens, output_dim], stddev=0.0001))]
for x in xrange(num_nets):
    w_os.append(
        tf.Variable(
            tf.random_normal(
                [num_hiddens, output_dim], stddev=0.0001
            )
        )
    )

hs = [tf.nn.tanh(
    tf.matmul(Xs[idx], w_h) + bs[idx])
    for idx, w_h in enumerate(w_hs)]
py_xs = [tf.matmul(h, w_os[x]) + b_os[x] for x, h in enumerate(hs)]

costs = [tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    for py_x in py_xs]

train_ops = [tf.train.GradientDescentOptimizer(1.0).minimize(cost)
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

curr_trX = trXs[0][:]
curr_teX = teXs[0][:]
for net_idx, curr_train_ops in enumerate(train_ops):
    # the last "net" doesn't work, don't feel like debugging
    if net_idx == len(train_ops) - 1:
        break
    print "=================="
    print "net : ", net_idx, " / ", num_nets
    print "=================="
    te_fd = {Y: teYs[net_idx]}
    te_fd[locals()["X" + str(net_idx)]] = curr_teX[:]
    for i in range(num_epochs):
        for start, end in zip(range(0, len(trXs[net_idx]), minibatch_size), range(minibatch_size, len(trXs[net_idx]), minibatch_size)):
            tr_fd = {Y: trYs[net_idx][start:end]}
            tr_fd[locals()["X" + str(net_idx)]] = curr_trX[start:end]
            sess.run(curr_train_ops, feed_dict=tr_fd)
        # use prediction accuracy because I can't be bothered to do perplexity properly right now
        curr_acc = np.mean(np.argmax(teYs[net_idx], axis=1) ==
                           sess.run(predict_ops[net_idx], feed_dict=te_fd))
        print i, " / ", num_epochs, " || ",  curr_acc, time.clock()
    total_tr_fd = {Y: trYs[net_idx]}
    total_tr_fd[locals()["X" + str(net_idx)]] = curr_trX[:]
    if net_idx < (num_nets-1):
        curr_trX = np.hstack((trXs[net_idx+1], hs[net_idx].eval(session=sess, feed_dict=total_tr_fd)[:-1]))
        curr_teX = np.hstack((teXs[net_idx+1], hs[net_idx].eval(session=sess, feed_dict=te_fd)[:-1]))

seeds = [char_to_idx[char] for char in chars[:num_nets+1]]
# and merrily use our global state this way...?
sample(sess, seeds, 2000, vocab_size, idx_to_char)
