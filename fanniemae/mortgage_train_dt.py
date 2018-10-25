import sys
import sklearn as sk
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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
os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
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

def get_num_leaves(tree):
	ret = 0
	for n1,n2 in zip(tree.children_left, tree.children_right):
		if n1 == n2:
			ret += 1
	return ret

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
    clf = tree.DecisionTreeClassifier(min_samples_split = 2)
    clf.fit(X_train,y_train)

    tree_size = clf.tree_.max_depth
    print("Tree size: {}".format(tree_size))

    preds = clf.predict(X_train)
    acc = accuracy_score(y_train,preds)
    f1 = f1_score(y_train,preds)	
    print("Training accuracy: {}, F1: {}".format(acc, f1))
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test,preds)
    f1 = f1_score(y_test,preds)	
    print("Testing accuracy: {}, F1: {}".format(acc, f1))

    train_stats = []
    test_stats = []
    
    for i in range(60):
        clf = tree.DecisionTreeClassifier(min_samples_split = 5, max_depth=i+1)
        clf.fit(X_train,y_train)
        tree_depth = clf.tree_.max_depth
        num_nodes = clf.tree_.node_count
        num_leaves = get_num_leaves(clf.tree_)
        print("Tree depth: {}, # nodes: {}, # leaves: {}".format(tree_depth, num_nodes, num_leaves))
        preds = clf.predict(X_train)
        acc = accuracy_score(y_train, preds)
        f1 = f1_score(y_train, preds)
        train_stats.append((tree_depth, num_nodes, num_leaves, acc, f1))
        print("Training accuracy: {}, F1: {}".format(acc, f1))
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print("Testing accuracy: {}, F1: {}".format(acc, f1))
        test_stats.append((tree_depth, num_nodes, num_leaves, acc, f1))

    tf = open("./mortgage_train.csv", 'w')

    for t in train_stats:
        tf.write("{},{},{},{},{}\n".format(t[0], t[1], t[2], t[3], t[4]))

    tf.flush()
    tf.close()

    tf = open("./mortgage_test.csv", 'w')

    for t in test_stats:
        tf.write("{},{},{},{},{}\n".format(t[0], t[1], t[2], t[3], t[4]))

    tf.flush()
    tf.close()


 

if __name__ == '__main__':
    if len(sys.argv) != 4:
        printUsage()
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_path = sys.argv[3]
    main(train_path, test_path, model_path)
