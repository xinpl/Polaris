import sys
from collections import OrderedDict

import pandas as pd
import numpy as np
import operator as op
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from nnExplain.utils import Timeout

import tensorflow as tf

from .common import constructNetwork
from .common import constructNetworkWithoutDropout
from .common import convertDateColsToInt
from .common import arrayToText
from .common import constructCleverHansModel
from .common import loadConfig
import nnExplain.genExp
from nnExplain.tensorflow_confusion_metrics import tf_confusion_metrics
from nnExplain.genExp import ExplGenerator
from nnExplain.genExp import Model
from nnExplain.genExp import Layer
from nnExplain.genExp import LayerKind
from nnExplain.genExp import CleverHansModelAdapter
from nnExplain.utils import plotCatCat
from nnExplain.utils import plotCatLin
from nnExplain.utils import plotLinLin
from nnExplain.utils import verbose
import traceback

def feat_name_map(fname):
    fn_map = {'OrInterestRate': 'Interest Rate', 'OrLTV': 'Loan-to- Value', 'DTIRat':'Debt-to-Income',
              'CreditScore':'Credit Score', 'PropertyType': 'Property Type',
              'PropertyType_CP': 'Co-Op',
              'PropertyType_SF': 'Single-Family',
              'PropertyType_CO': 'Condo',
              'PropertyType_MH': 'Manufactured Housing',
              'Other': 'PUD'}
    return fn_map[fname]

def main(test_path, model_path, num_points, max_regions):

    loadConfig('./config')

    test_data = pd.read_csv(test_path)

    print(("test_drivers data size %d\n"%test_data.shape[0]))

    print("Raw data loaded successfully.....\n")

    # Intepret params
    param_path = model_path+'.param'
    param_file = open(param_path)
    lines = param_file.readlines()
    X_mean_train = []
    X_std_train = []
    expandMap = OrderedDict()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '':
            i+=1
            continue
        if line == 'X_mean:':
            i+=1
            line = lines[i].strip()
            X_mean_train = [float(x.strip()) for x in line.split(',') if x.strip()]
            i+=1
            continue

        if line == 'X_std:':
            i+=1
            line = lines[i].strip()
            X_std_train =  [float(x.strip()) for x in line.split(',') if x.strip()]
            i+=1
            continue

        tokens = line.split(':')
        k = tokens[0].strip()
        if len (tokens) == 1:
            expandMap[k] = []
            i+=1
            continue
        v = tokens[1].strip()
        if v == '':
            expandMap[k] = []
        else:
            expandMap[k] = [x.strip() for x in v.split(',,,') if x.strip()]
        i+=1

    param_file.close()

    Y_LABEL = 'Default'
    KEYS = [i for i in list(test_data.keys()) if i != Y_LABEL]
    TEST_SIZE = test_data.shape[0]
    N_POSITIVE = test_data[Y_LABEL].sum()
    N_INPUT = test_data.shape[1] - 1
    N_CLASSES = 2

    print("Variables loaded successfully...\n")
    print(("Number of predictors \t%s" %(N_INPUT)))
    print(("Number of classes \t%s" %(N_CLASSES)))
    print(("TESTING_SIZE \t%s"%(TEST_SIZE)))
    print(("Number of positive instances \t%s" %(N_POSITIVE)))
    print("\n")
    print("Metrics displayed:\tPrecision\n")

    date_cols = ['OrDate','FirstPayment']

    test_data = convertDateColsToInt(test_data, date_cols)

    print("Start expanding the test data: ")
    nan_cols = test_data[test_data.columns[test_data.isnull().any()]]

    test_data.drop(nan_cols.columns, axis=1, inplace=True)

    cat = test_data[list(expandMap.keys())]
    print(("Expand cat data "+str(cat.columns.values)+"\n"))
    num = test_data.drop(cat.columns, axis=1)

    data = pd.DataFrame()
    for i in cat.columns:
        if len(expandMap[i]) == 0:
            continue
        tmp = pd.DataFrame(0, index = np.arange(test_data.shape[0]), columns = expandMap[i])
        tmp1 = pd.get_dummies(cat[i], prefix=str(i), drop_first=True)
        for col in tmp1.columns:
            if col in tmp.columns:
                tmp[col] = tmp1[col]
        data = pd.concat([data, tmp], axis=1)

    test_data = pd.concat([num,data,nan_cols], axis=1).reset_index(drop=True)

    print("Expand categorical features.\n")

    print("After expanding: \n")

    ori_KEYS = KEYS

    N_INPUT = test_data.shape[1] - 1
    KEYS = [i for i in list(test_data.keys()) if i != Y_LABEL]

    print(("Number of predictors \t%s" %(N_INPUT)))

    print(KEYS)

    X_test = test_data[KEYS].get_values()

    y_test = test_data[Y_LABEL].get_values()

    X_test = (X_test - X_mean_train)/ X_std_train

    #------------------------------------------------------------------------------
    # Neural net construction

    # Tf placeholders
    X = tf.placeholder(tf.float32, [None, N_INPUT])
    y = tf.placeholder(tf.int64, [None])
    dropout_keep_prob = tf.placeholder(tf.float32)

    pred, layerList = constructNetwork(X,dropout_keep_prob,N_INPUT,N_CLASSES)

    # Loss and optimizer
    logits = pred
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)) # softmax loss

    correct_prediction = tf.equal(tf.argmax(pred, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion = tf.confusion_matrix(y, tf.argmax(pred,1), 2)

    print("Net built successfully...\n")
    print("Starting training...\n")
    #------------------------------------------------------------------------------
    # Training

    # Launch session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess, model_path+'.ckpt')

    '''
    print("Testing...\n")
    #------------------------------------------------------------------------------
    # Testing

    #test_acc = sess.run(accuracy, feed_dict={X: X_test, y: y_test, dropout_keep_prob:1.})
    #
    #test_conf = sess.run(confusion, feed_dict={X: X_test, y: y_test, dropout_keep_prob:1.})

    test_conf = np.zeros((2,2))

    indices = np.arange(0, X_test.shape[0])
    for batch_indices in np.array_split(indices, 100):
        batch_xs = X_test[batch_indices, :]
        batch_ys = y_test[batch_indices]

        test_conf += sess.run(confusion, feed_dict = {X:batch_xs, y: batch_ys, dropout_keep_prob:1.})

    accuracy = (test_conf[0][0] + test_conf[1][1])/float(np.sum(test_conf))


    print(("Testing accuracy: %.3f" % accuracy))

    print(test_conf)
    '''

    # Create attack
    print("Attacking...\n")

    for layer in layerList:
        if layer.kind == LayerKind.dense:
            layer.weights = sess.run(layer.weights)
            layer.biases = sess.run(layer.biases)

    explGen = ExplGenerator(Model(layerList, N_INPUT, X_mean_train, X_std_train), X, dropout_keep_prob, logits, sess, KEYS)

    explGen.set_renamer(feat_name_map)

    rad_map = {}
    weight_map = {}

    for k in ori_KEYS:
        if k in expandMap:
            rad_map[k] = 1
            weight_map[k] = 1
        else:
            k_idx = KEYS.index(k);
            rad_map[k] = X_std_train[k_idx]
            weight_map[k] = 1/X_std_train[k_idx]

    desired_features = ['OrInterestRate', 'OrLTV', 'DTIRat', 'CreditScore', 'PropertyType']
    # desired_features = ['DTIRat', 'CreditScore']

    feat_bounds = {'OrInterestRate': (0,10), 'OrLTV': (0,200), 'DTIRat': (1,64), 'CreditScore':(300,850)}

    norm_feat_bounds = {}
    # normalize the feature bounds
    for k,v in feat_bounds.items():
        k_idx = KEYS.index(k)
        nl = (v[0] - X_mean_train[k_idx]) / X_std_train[k_idx]
        nu = (v[1] - X_mean_train[k_idx]) / X_std_train[k_idx]
        norm_feat_bounds[k_idx] = (nl,nu)

    rad_map['OrInterestRate'] = 0.5
    rad_map['OrLTV'] = 2.5
    rad_map['DTIRat'] = 2.5
    rad_map['CreditScore'] = 25
    rad_map['PropertyType'] = 1

    weight_map['OrInterestRate'] = 1/10.0
    weight_map['OrLTV'] = 1/200.0
    weight_map['DTIRat'] = 1/63.0
    weight_map['CreditScore'] = 1/550.0
    weight_map['PropertyType'] = 1.0

    explGen.set_feat_rad(rad_map)
    explGen.set_feat_scale(weight_map)

    ### Check the weights and radiuses set for selected features
    for f in desired_features:
        print('Weight for '+f+': '+str(weight_map[f]))
        print('Raidus for '+f+': '+str(rad_map[f]))
        print()
    ###

    ban_features = [x for x in ori_KEYS if x not in desired_features]

    START_POINT = 0

    for r_x, r_y in zip(X_test,y_test):
        y_p = sess.run(pred, feed_dict = {X: [r_x], y: [r_y], dropout_keep_prob:1.})
        if y_p[0][1] > y_p[0][0]:
            if nnExplain.genExp.NUM_POINTS < START_POINT - 1:
                nnExplain.genExp.NUM_POINTS += 1
                continue

            if nnExplain.genExp.NUM_POINTS == num_points: # TODO: change it after finishing sampling
                print("Done with {} samples.".format(num_points))
                exit(0)
            print("Rejected "+str(nnExplain.genExp.NUM_POINTS+1)+": \n")
            print(r_x)
            print('\n')
            print('Denormalized: \n')
            r_x1 = r_x * X_std_train +X_mean_train
            print(r_x1)
            print('\n')
            expected_y = [1,0]
            try:
                with Timeout(seconds=10 * 60 * 60):
                    explGen.pickBestExplaination2D(r_x, r_x1, expected_y, ori_KEYS, KEYS, expandMap, ban_features, max_regions,
                                               norm_feat_bounds, sample_scale=None)
            except:
                verbose("Unexpected error:" + str(sys.exc_info()[0]), 0)
                traceback.print_exc()
                print('Error')
                print('Proceed to next!')

            sys.stdout.flush()
    sess.close()
    print("Session closed!")


if __name__ == '__main__':
    test_path = sys.argv[1]
    model_path = sys.argv[2]
    num_points = sys.argv[3]
    max_regions = sys.argv[4]
    main(test_path, model_path, int(num_points), int(max_regions))
