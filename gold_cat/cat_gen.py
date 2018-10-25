import tensorflow as tf
import random
from .common import *
from tensorflow.contrib.layers import *
import sys


l2_reg = 0.001
learning_rate = 0.0001
num_epoches = 500
num_batches = 500
batch_size = 40
display_step = 10
max_delete = 10
max_y_length = max_delete

def create_batch(gold, batch_size):
    ret_x = []
    ret_y = []
    for i in range(batch_size):
        var_gold = create_varied_instance(gold)
        y_length = random.randint(0, max_delete)
        x_length = max_line_num - y_length
        indices = list(range(max_line_num))
        x_indices = sorted(random.sample(indices, x_length))
        y_indices = [ v for v in indices if v not in x_indices]
        X = []
        Y = []
        for j in x_indices:
            X.append(var_gold[j])
        for j in y_indices:
            Y.append(var_gold[j])

        while len(X) < max_line_num:
            X.append([non_line,non_line,non_line,non_line])

        while len(Y) < max_delete:
            Y.append([non_line,non_line,non_line,non_line])
        ret_x.append(X)
        ret_y.append(Y)

    return ret_x, ret_y


def create_encoder():
    x = tf.placeholder(tf.float32, shape=[None, max_line_num, 4])
    with tf.contrib.framework.arg_scope([fully_connected],
                                        activation_fn = tf.nn.relu,
                                        weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                                        weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)):
        flat_in = flatten(x)
        hidden1 = fully_connected(flat_in, 1024)
        hidden2 = fully_connected(hidden1, 1024)
        flat_out = fully_connected(hidden2, max_delete * 4)
        reshaped_out = tf.reshape(flat_out, shape=[-1, max_delete, 4])

    return x,reshaped_out

def main():
    x,logits = create_encoder()
    y = tf.placeholder(tf.float32, shape=[None, max_delete, 4])
    loss = tf.losses.mean_squared_error(y, logits)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([loss] + reg_losses)

    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    gold, train, val, test = separate_data()

    with tf.Session() as sess:
        init.run()
        for epoch in range(num_epoches):
            for b in range(num_batches):
                batch = create_batch(gold, batch_size)
                sess.run(opt, feed_dict={x: batch[0], y:batch[1]})
            if epoch % display_step == 0:
                print('Epoch '+str(epoch))
                batch = create_batch(gold,3)
                print('Loss '+str(sess.run(loss, feed_dict={x: batch[0], y:batch[1]})))
                predicted = sess.run(logits, feed_dict={x:batch[0], y:batch[1]})
                for i in range(3):
                    pic = []
                    before = prune(batch[0][i])
                    draw_lines(before, 'before'+str(epoch)+"_"+str(i))
                    for s in before:
                        pic.append(s)
                    after = prune(predicted[0])
                    for s in after:
                        pic.append(s)
                    draw_lines(pic, 'after'+str(epoch)+"_"+str(i))
                    oracle = prune(batch[1][i])
                    draw_lines(oracle, 'ora'+str(epoch)+"_"+str(i))
                sys.stdout.flush()

if __name__ == '__main__':
    main()