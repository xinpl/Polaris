from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Dropout
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import nnExplain
import nnExplain.utils
from nnExplain.utils import createModelFromKerasSave
from nnExplain.NNModel import Layer, LayerKind, Model
from nnExplain.genExp import ExplGenerator
from nnExplain.utils import verbose, Timeout
from .common import get_train_val_test, get_train_val_test_multi_classes, get_X_Y
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import traceback
import sys
import copy

def get_mean_std(raw_data_path):
    raw_df = pd.read_csv(raw_data_path, header=None)

    raw_df.drop(raw_df.columns[[5,21]], axis=1, inplace=True)


    n_raw_rows = raw_df.shape[0]

    raw_df_half = raw_df.head(3059)

    means =raw_df_half.mean().as_matrix()
    std = raw_df_half.std().as_matrix()
    return means, std

def main(model_path,num_points=100):
    nnExplain.utils.STITCH_TOLERANCE_DIGITS = 4

    script_dir = os.path.dirname(__file__)
    # model_path = os.path.join(script_dir, "8x100.h5")

    raw_data_path = os.path.join(script_dir, "./data/ml-prove/all-data-raw.csv")

    means,stds = get_mean_std(raw_data_path)

    multi_classes = False

    if multi_classes:
        train, val, test = get_train_val_test_multi_classes()
    else:
        train,val,test = get_train_val_test()

    test_X, test_Y = get_X_Y(test)

    input_length = test_X.shape[1]

    means = means[:input_length]
    stds = stds[:input_length]

    (m, layer_list, X, logits) = createModelFromKerasSave(model_path, input_length)

    dumpy_dropout = tf.placeholder(tf.float32)

    model = Model(layer_list, input_length, means, stds)

    KEYS = ['X'+str(i) for i in range(input_length)]

    sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    explGen = ExplGenerator(model, X, dumpy_dropout, logits, sess, KEYS)

    predictions = sess.run(logits, {X: test_X})

    KEYS[0] = "% Clauses That Are Unit"
    KEYS[1] = "% Clauses That Are Horn"
    KEYS[2] = "% Clauses That Are Ground"
    KEYS[8] = "Avg. Clause Length"
    KEYS[10] = "Avg. Clause Depth"
    rad_map = {KEYS[0]: 0.025, KEYS[1]:0.025, KEYS[2]:0.025, KEYS[8]: 0.25, KEYS[10]: 0.25}
    weight_map = {KEYS[0]: 1/1, KEYS[1]:1/1, KEYS[2]:1/1, KEYS[8]: 1/10.0, KEYS[10]: 1/10}

    feat_bounds = {KEYS[0]: (0,1), KEYS[1]:(0,1), KEYS[2]:(0,1), KEYS[8]: (0,10), KEYS[10]: (0,10)}

    norm_feat_bounds = {}
    # normalize the feature bounds
    for k,v in feat_bounds.items():
        k_idx = KEYS.index(k)
        nl = (v[0] - means[k_idx]) / stds[k_idx]
        nu = (v[1] - means[k_idx]) / stds[k_idx]
        norm_feat_bounds[k_idx] = (nl,nu)


    explGen.set_feat_rad(rad_map)
    explGen.set_feat_scale(weight_map)

    banned_features = [KEYS[i] for i in range(len(KEYS)) if i not in [0,1,2,8,10]]

    Y_label_num = len(test_Y.columns)
    target_labels = [0] * Y_label_num
    # Both in the cases of binary classification and multi-classification, 0 is the desirable label.
    target_labels[0] = 1

    START_POINT = 0

    for (x, p) in zip(test_X.iterrows(), predictions):
        if np.argmax(target_labels) != np.argmax(p):
            if nnExplain.genExp.NUM_POINTS < START_POINT - 1:
                nnExplain.genExp.NUM_POINTS += 1
                continue

            if nnExplain.genExp.NUM_POINTS == num_points:
                print("Done with {} samples.".format(num_points))
                exit(0)
            print("Rejected:\n{}".format(x))
            x_arr = x[1].as_matrix()
            unnormed_x_arr = (x_arr * stds) + means
            print("Unnormalized value:\n{}".format(unnormed_x_arr))
            try:
                with Timeout(seconds=10 * 60 * 60):
                    explGen.pickBestExplaination2D(x_arr, unnormed_x_arr, target_labels, KEYS, KEYS, {}, banned_features, norm_feat_bounds=norm_feat_bounds, sample_scale=None)
            except:
                verbose("Unexpected error:" + str(sys.exc_info()[0]), 0)
                traceback.print_exc()
                print('Error')
                print('Proceed to next!')

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
