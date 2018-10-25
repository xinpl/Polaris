import tensorflow as tf
import jsonlines
from .common import *
import math
import time
from nnExplain.utils import get_large_tensor_avg
from nnExplain.utils import get_large_tensor_sum
from nnExplain.NNModel import *
import sys

num_conv = 3

batch_size = 32

max_steps = 10000

report_step = 1000

val_size = 10000

test_size = 10000

fc_size = 1024

out_channels = 8

num_fc = 1

use_pool = False

def create_mlp():
    x = tf.placeholder(tf.float32, shape=[None,max_line_num,4])
    y = tf.placeholder(tf.float32, shape=[None,2])
    keep_prob = tf.placeholder(tf.float32)

    conv_input = x
    conv_output = conv_input
    channels = 4
    layer_list = []

    # one-dimension convolution
    for i in range(num_conv):
        filter_shape = [5, channels, out_channels] # window_size: 5, channels: 4, number of features: 8
        bias_shape = [out_channels]
        strides = 1
        w_conv = weight_variable(filter_shape)
        b_conv = bias_variable(bias_shape)
        h_conv = tf.nn.conv1d(conv_input, w_conv, strides, padding = 'SAME')

        layer_list.append(Layer(LayerKind.conv, h_conv, w_conv, b_conv, strides, 'SAME'))

        relu = tf.nn.relu(h_conv + b_conv)

        layer_list.append(Layer(LayerKind.relu, relu))

        if use_pool:

            pooled = tf.nn.pool(relu, [2], 'MAX', 'SAME', strides=[2])

            layer_list.append(Layer(LayerKind.pool, pooled, [2], None, [2], 'SAME'))

            conv_input = pooled
            channels = out_channels
            conv_output = pooled

        else:

            conv_input = relu
            channels = out_channels
            conv_output = relu

    if use_pool:
        aft_conv_size = int(max_line_num / math.pow(2, num_conv)) * channels
    else:
        aft_conv_size = int(max_line_num) * channels


    flattened = tf.reshape(conv_output, [-1, aft_conv_size])

    layer_list.append(Layer(LayerKind.reshape, flattened))

    last_fc_size = aft_conv_size
    dropped = flattened

    for i in range(num_fc):
        w_fc = weight_variable([last_fc_size, fc_size])
        b_fc = bias_variable([fc_size])

        h_fc = tf.matmul(dropped, w_fc) + b_fc

        layer_list.append(Layer(LayerKind.dense, h_fc, w_fc, b_fc ))

        relu = tf.nn.relu(h_fc)

        layer_list.append(Layer(LayerKind.relu, relu))

        dropped = tf.nn.dropout(h_fc, keep_prob)

        layer_list.append(Layer(LayerKind.dropout, dropped))

        last_fc_size = fc_size


    w_out = weight_variable([last_fc_size, 2])
    b_out = bias_variable([2])

    logits = tf.matmul(dropped, w_out) + b_out

    layer_list.append(Layer(LayerKind.dense, logits, w_out, b_out))

    return {'x': x, 'y': y, 'keep': keep_prob, 'logits': logits, 'layers': layer_list}


def main(model_save_path):
    gold, train, val, test = separate_data()
    vars = create_mlp()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=vars['y'], logits=vars['logits']))
    step_size = 0.01
    opt = tf.train.AdadeltaOptimizer(step_size).minimize(loss)
    correct = tf.equal(tf.argmax(vars['y'], 1), tf.argmax(vars['logits'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    confusion = tf.confusion_matrix(tf.argmax(vars['y'], 1), tf.argmax(vars['logits'], 1), 2)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(max_steps):
            print(str(i)+"/"+str(max_steps))
            start_time = time.time()
            # for j in range(int(len(train)/batch_size)):
            batch = create_batch2(batch_size, gold, train, True)
            feed_dict = {vars['x']:batch[0], vars['y']:batch[1], vars['keep']:0.9}
            sess.run(opt, feed_dict=feed_dict)

            # feed_dict[vars['keep']] = 1.0
            # loss_val = sess.run(loss,  feed_dict=feed_dict)
            # print('Loss: '+str(loss_val))
            # acc = sess.run(accuracy, feed_dict=feed_dict)
            # print('accuracy: '+str(acc))

            if i%report_step == 0:
                print('Training one epoch took '+str(time.time() - start_time)+' seconds')
                # training accuracy: choose 10*batches
                train_batch = create_batch2(batch_size*10, gold, train, True)
                train_acc = get_large_tensor_avg(sess, vars['x'], train_batch[0], vars['y'], train_batch[1],
                                                 vars['keep'], 1, accuracy)
                train_conf = get_large_tensor_sum(sess, vars['x'], train_batch[0], vars['y'], train_batch[1],
                                                 vars['keep'], 1, confusion)
                print('train accuracy: '+str(train_acc))
                print('train confusion:\n'+str(train_conf))

                # val_batch = create_batch1(len(val)*2, gold, val)
                val_batch = create_batch2(val_size, gold, val, True)
                val_acc = get_large_tensor_avg(sess, vars['x'], val_batch[0], vars['y'], val_batch[1],
                                                 vars['keep'], 1, accuracy)
                val_conf = get_large_tensor_sum(sess, vars['x'], val_batch[0], vars['y'], val_batch[1],
                                                 vars['keep'], 1, confusion)
                print('val accuracy: '+str(val_acc))
                print('val confusion:\n'+str(val_conf))
                saver.save(sess, model_save_path, global_step=i)



        print('Start testing: ')
        test_batch = create_batch2(test_size, gold, test, True)
        test_acc = get_large_tensor_avg(sess, vars['x'], test_batch[0], vars['y'], test_batch[1],
                                                 vars['keep'], 1, accuracy)
        test_conf = get_large_tensor_sum(sess, vars['x'], test_batch[0], vars['y'], test_batch[1],
                                                 vars['keep'], 1, confusion)
        print('Test accuracy: '+str(test_acc))
        print('Test confusion:\n'+str(test_conf))



if __name__ == '__main__':
    model_save_path = sys.argv[1]
    main(model_save_path)
