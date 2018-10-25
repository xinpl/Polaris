import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import seaborn as sns
import pandas as pd
import numpy as np

from nnExplain.utils import get_large_tensor_sum
from nnExplain.utils import get_large_tensor_avg
from nnExplain.utils import print_f1
from nnExplain.utils import cal_f1

import nnExplain.utils as utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
import tensorflow as tf

from .common import constructNetwork
from .common import convertDateColsToInt
from .common import loadConfig
from .common import focal_loss
from nnExplain.tensorflow_confusion_metrics import tf_confusion_metrics
import time

use_focal_loss = True

def getdummies(df):
    columns = df.columns[df.isnull().any()]
    nan_cols = df[columns]

    df.drop(nan_cols.columns, axis=1, inplace=True)

    cat = df.select_dtypes(include=['object'])
    print("Expand cat data "+str(cat.columns.values)+"\n")
    num = df.drop(cat.columns, axis=1)

    data = pd.DataFrame()
    for i in cat.columns:
        tmp = pd.get_dummies(cat[i], prefix=str(i), drop_first=True)
        data = pd.concat([data, tmp], axis=1)

    df = pd.concat([num,data,nan_cols], axis=1).reset_index(drop=True)
    return df

def printUsage():
    print('Usage: ./mortgage_train.py training_data_path validation_data_path model_path')

def main(train_path, test_path, model_path):
    loadConfig('./config')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("Train data size %d\n"%train_data.shape[0])

    print("test_drivers data size %d\n"%test_data.shape[0])

    print("Raw data loaded successfully.....\n")


    Y_LABEL = 'Default'
    KEYS = [i for i in list(train_data.keys()) if i != Y_LABEL]
    TRAIN_SIZE = train_data.shape[0]
    N_POSITIVE = train_data[Y_LABEL].sum()
    N_INPUT = train_data.shape[1] - 1
    N_CLASSES = train_data[Y_LABEL].unique().shape[0]
    TEST_SIZE = test_data.shape[0]
    LEARNING_RATE = 0.0001                                              # Learning rate
    TRAINING_EPOCHS = 500                                              # Number of epochs
    BATCH_SIZE = 128                                                  # Batch size
    DISPLAY_STEP = 1                                                   # Display progress each x epochs
    PATIENCE = 50
    WINDOW = 5

    print("Variables loaded successfully...\n")
    print(("Number of predictors \t%s" %(N_INPUT)))
    print(("Number of classes \t%s" %(N_CLASSES)))
    print(("TRAINING_SIZE \t%s" %(TRAIN_SIZE)))
    print(("TESTING_SIZE \t%s"%(TEST_SIZE)))
    print(("Number of positive instances \t%s" %(N_POSITIVE)))
    print("\n")
    print("Metrics displayed:\tPrecision\n")

    raw_data = pd.concat([train_data,test_data], axis = 0)

    date_cols = ['OrDate','FirstPayment']

    raw_data = convertDateColsToInt(raw_data, date_cols)

    obj_cols = raw_data.select_dtypes(include=['object'])

    raw_data = getdummies(raw_data)

    print("Expand categorical features.\n")

    print("After expanding: \n")

    N_INPUT = raw_data.shape[1] - 1
    KEYS = [i for i in list(raw_data.keys()) if i != Y_LABEL]

    paramFile = open(model_path+'.param', 'w')

    for nc in obj_cols:
        paramFile.write(nc+': ')
        for k in KEYS:
            if k.startswith(nc+'_'):
                paramFile.write(k+',,,')
        paramFile.write('\n')

    paramFile.write('\n')

    print(("Number of predictors \t%s" %(N_INPUT)))

    print(KEYS)

    train_data = raw_data[:train_data.shape[0]]

    test_data = raw_data[train_data.shape[0]:]

    X_train = train_data[KEYS].get_values()

    y_train = train_data[Y_LABEL].get_values()

    print(X_train.shape)

    X_mean_train = np.mean(X_train, 0)
    X_std_train = np.std(X_train,0)
    zero_remover = lambda x: x if x >0.0 else 1.0
    vfunc = np.vectorize(zero_remover)
    X_std_train = vfunc(X_std_train)
    X_train = (X_train - X_mean_train)/X_std_train

    paramFile.write('X_mean:\n')
    for mean in X_mean_train:
        paramFile.write(str(mean)+', ')
    paramFile.write('\n')
    paramFile.write('X_std:\n')
    for std in X_std_train:
        paramFile.write(str(std)+', ')
    paramFile.flush()
    paramFile.close()

    print('X_mean: '+str(X_mean_train)+' X_std: '+str(X_std_train)+'\n')

    X_test = test_data[KEYS].get_values()

    y_test = test_data[Y_LABEL].get_values()

    X_test = (X_test - X_mean_train)/ X_std_train

    #------------------------------------------------------------------------------
    # Neural net construction

    # Tf placeholders
    X = tf.placeholder(tf.float32, [None, N_INPUT])

    y = tf.placeholder(tf.int64, [None])

    dropout_keep_prob = tf.placeholder(tf.float32)

    pred, layer_map = constructNetwork(X,dropout_keep_prob,N_INPUT,N_CLASSES)

    # Loss and optimizer
    pos_rate = float(N_POSITIVE)/TRAIN_SIZE
    neg_rate = 1 - pos_rate
    y_float = tf.to_float(y)
    print('Pos: '+str(pos_rate) + ', Neg: '+str(neg_rate))
    e_weights = tf.add(tf.scalar_mul(neg_rate, y_float), tf.scalar_mul(pos_rate, tf.subtract(tf.ones(tf.shape(y_float)), y_float)))
    logits = pred
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)) # softmax loss
    if use_focal_loss:
        cost = tf.reduce_mean(focal_loss(pred,tf.one_hot(y, 2, axis=-1)))
    else:
        cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y, logits = logits, weights = e_weights))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

    # Accuracy
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    correct_prediction = tf.equal(tf.argmax(pred, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion = tf.confusion_matrix(y, tf.argmax(pred,1), 2)

    print("Net built successfully...\n")
    print("Starting training...\n")
    #------------------------------------------------------------------------------
    # Training

    # Launch session
    config=tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    last_f1_reading = None
    window_count = 0

    # Training loop
    for epoch in range(TRAINING_EPOCHS):
        start_time = time.time()
        total_batch = int(X_train.shape[0] / BATCH_SIZE)
        indices = np.arange(0, X_train.shape[0])
        np.random.shuffle(indices)
        # Loop over all batches
        i = 0
        costs = []
        accuracies = []
        train_conf = np.zeros((2,2))
        for batch_indices in np.array_split(indices, total_batch):
            batch_xs = X_train[batch_indices, :]
            batch_ys = y_train[batch_indices]
            # Fit using batched data
            # Display progress
            sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 0.9})

            costs.append(sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.}))
            accuracies.append(sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.}))
            train_conf += sess.run(confusion, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})
            if i%100==0 and total_batch > 100:
                print("%d/%d batches\n"%(i,total_batch))
            i+=1

        end_time = time.time()

        print('Training one epoch takes '+str(end_time-start_time) + ' secs.')

        if epoch % DISPLAY_STEP == 0:
            print(("Epoch: %03d/%03d cost: %.9f" % (epoch, TRAINING_EPOCHS, np.mean(costs))))
            print(("Training accuracy: %.3f" % (np.mean(accuracies))))
            print(train_conf)
            print_f1(train_conf)
            sys.stdout.flush()
            test_acc = get_large_tensor_avg(sess, X, X_test, y, y_test, dropout_keep_prob, 1, accuracy)
            print(("Validation accuracy: %.3f" % (test_acc)))

            test_conf = get_large_tensor_sum(sess, X, X_test, y, y_test, dropout_keep_prob, 1, confusion)
            print(test_conf)
            print_f1(test_conf)

            print('Validation takes '+str(time.time() - end_time)+' secs.')

        if epoch > PATIENCE:
            test_conf = get_large_tensor_sum(sess, X, X_test, y, y_test, dropout_keep_prob, 1, confusion)
            test_f1 = cal_f1(test_conf)
            if last_f1_reading == None:
                last_f1_reading = test_f1
            else:
                if window_count >= WINDOW:
                    if (test_f1 - last_f1_reading > 0.01*last_f1_reading):
                        last_f1_reading = test_f1
                        window_count = 0
                    else:
                        print('f_1 score stopped increasing at '+str(test_f1))
                        break
            window_count += 1

    print ("End of training.\n")

    print("Saving model to "+model_path)

    saver = tf.train.Saver()

    saver.save(sess, model_path+'.ckpt')

    sess.close()
    print("Session closed!")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        printUsage()
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_path = sys.argv[3]
    main(train_path, test_path, model_path)
