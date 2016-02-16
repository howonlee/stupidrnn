
#
# from https://github.com/nlintz/TensorFlow-Tutorials
#

import tensorflow as tf
import numpy as np
import time
import operator as op

def onehot(idx, char_len):
    arr = np.zeros(char_len)
    arr[idx] = 1.0
    return arr

num_nets = 15
num_hiddens = 250
num_epochs = 40

def make_data_arr(data_list, char_len):
    arr = np.zeros((len(data_list), char_len))
    for idx, datum in enumerate(data_list):
        arr[idx, datum] = 1.0
    return arr

with open("corpus.txt") as corpus_file:
    chars = list(corpus_file.read())
    print len(chars)
    chars = chars[:200000]
    char_len = len(set(chars))
    char_to_idx = {char:idx for idx, char in enumerate(list(set(chars)))}
    trXs, teXs, trYs, teYs = [], [], [], []
    for net_idx in xrange(num_nets):
        print net_idx
        all_data = []
        # for fst, snd in zip(chars, chars[1:]):
        for fst, snd in zip(chars, chars[net_idx+1:]):
            all_data.append((char_to_idx[fst], char_to_idx[snd]))
        train_len = (19 * len(all_data)) // 20
        train_list = all_data[:train_len][net_idx:]
        test_list = all_data[train_len:][net_idx:]

        train_xs = map(op.itemgetter(0), train_list)
        trXs.append(make_data_arr(train_xs, char_len))

        train_ys = map(op.itemgetter(1), train_list)
        trYs.append(make_data_arr(train_ys, char_len))

        test_xs = map(op.itemgetter(0), test_list)
        teXs.append(make_data_arr(test_xs, char_len))

        test_ys = map(op.itemgetter(1), test_list)
        teYs.append(make_data_arr(test_ys, char_len))
print "finished processing corpus"

# taken from karpathy's blog post thing and modified:
# https://gist.github.com/karpathy/d4dee566867f8291f086

def sample(seed_ix, n):
    """
    It's going to be a lotta evals up in here
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    curr_h = hs[0].eval(session=sess, feed_dict=total_tr_fd)[2:]))
    for x in xrange(1, num_nets):
        curr_h = hs[0].eval(session=sess, feed_dict=total_tr_fd)[2:]))
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
    return ixes


input_dim = char_len
output_dim = char_len

X0 = tf.placeholder("float", [None, input_dim])
Xs = [X0]
for x in xrange(num_nets):
    locals()["X" + str(x+1)] = tf.placeholder("float", [None, num_hiddens + input_dim])
    # more than a bit precious, I'm afraid
    Xs.append(locals()["X" + str(x+1)])

Y = tf.placeholder("float", [None, output_dim])

w_hs = [tf.Variable(tf.random_normal([input_dim, num_hiddens], stddev=0.01))]
for x in xrange(num_nets):
    w_hs.append(tf.Variable(tf.random_normal([num_hiddens + input_dim, num_hiddens], stddev=0.01)))

w_os = [tf.Variable(tf.random_normal([num_hiddens, output_dim], stddev=0.01))]
for x in xrange(num_nets):
    w_os.append(tf.Variable(tf.random_normal([num_hiddens, output_dim], stddev=0.01)))

hs = [tf.nn.sigmoid(tf.matmul(Xs[idx], w_h)) for idx, w_h in enumerate(w_hs)]
py_xs = [tf.matmul(h, w_os[x]) for x, h in enumerate(hs)]

costs = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) for py_x in py_xs]
train_ops = [tf.train.GradientDescentOptimizer(0.1).minimize(cost) for cost in costs]
predict_ops = [tf.argmax(py_x, 1) for py_x in py_xs]

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

curr_trX = trXs[0][:]
curr_teX = teXs[0][:]
for net_idx, curr_train_ops in enumerate(train_ops):
    print "=================="
    print "net : ", net_idx
    print "=================="
    te_fd = {Y: teYs[net_idx]}
    te_fd[locals()["X" + str(net_idx)]] = curr_teX[:]
    for i in range(num_epochs):
        for start, end in zip(range(0, len(trXs[net_idx]), 128), range(128, len(trXs[net_idx]), 128)):
            tr_fd = {Y: trYs[net_idx][start:end]}
            tr_fd[locals()["X" + str(net_idx)]] = curr_trX[start:end]
            sess.run(curr_train_ops, feed_dict=tr_fd)
        curr_acc = np.mean(np.argmax(teYs[net_idx], axis=1) ==
                           sess.run(predict_ops[net_idx], feed_dict=te_fd))
        print i, curr_acc, time.clock()
        # use prediction accuracy because I can't be bothered
    total_tr_fd = {Y: trYs[net_idx]}
    total_tr_fd[locals()["X" + str(net_idx)]] = curr_trX[:]
    # fix this properly
    if net_idx < (num_nets-1):
        curr_trX = np.hstack((trXs[net_idx+1], hs[net_idx].eval(session=sess, feed_dict=total_tr_fd)[2:]))
        curr_teX = np.hstack((teXs[net_idx+1], hs[net_idx].eval(session=sess, feed_dict=te_fd)[1:]))
