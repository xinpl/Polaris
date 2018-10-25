import keras
import numpy as np
import sklearn as sk
import sklearn.metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from .common import *


use_multi_classes = True

if use_multi_classes:
    train,test,val = get_train_val_test_multi_classes()
else:
    train,test,val = get_train_val_test()

train_X, train_Y = get_X_Y(train)
val_X, val_Y = get_X_Y(val)
test_X, test_Y = get_X_Y(test)

def pred_to_cat(pred):
    if use_multi_classes:
        return [1 if p[0] > 0 else 0 for p in pred]
    else:
        return onehot_to_bin(pred)

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        predict1 = pred_to_cat(predict)
        targ1 = pred_to_cat(targ)
        self.f1s=sk.metrics.f1_score(targ1, predict1)
        self.pre=sk.metrics.precision_score(targ1, predict1)
        self.rec=sk.metrics.recall_score(targ1, predict1)
        print('F1 score: {}, precision: {}, recall: {}'.format(self.f1s, self.pre, self.rec))
        return

def try_model(depth, width, lr = 1e-5, patience = 0):
    print("Trying parameters: {}".format((depth, width)))
    num_out = len(train_Y.columns)
    num_in = len(train_X.columns)

    model = Sequential()

    if depth == 0:
        model.add(Dense(units=num_out, activation='softmax', input_dim=num_in, kernel_initializer="he_normal",
                        bias_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01)))
    else:
        model.add(Dense(units=width, activation='relu', input_dim = num_in,  kernel_initializer="he_normal",
                        bias_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01)))

        for i in range(depth-1):
            model.add(Dense(units=width, activation='relu', kernel_initializer="he_normal",
                        bias_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01)))
            model.add(Dropout(0.1))

        model.add(Dense(units=num_out, activation='softmax', kernel_initializer="he_normal",
                        bias_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01)))

    opt = keras.optimizers.Adamax(lr=lr)
    # opt = keras.optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    metrics = Metrics()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=1000000, batch_size=32, callbacks=[metrics, early_stop])

    loss,accuracy = model.evaluate(test_X, test_Y)
    pred = model.predict(test_X)
    pred1 = pred_to_cat(pred)
    test_Y1 = pred_to_cat(test_Y.values)
    f1 = sk.metrics.f1_score(test_Y1, pred1)
    print((loss,accuracy,f1))
    return (loss,accuracy,f1,model)

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    result_map = {}
    max_depth = 10
    max_width = 10 #*10
    for depth in range(max_depth+1):
        for width in range(max_width):
            width = (width+1)*10
            result = try_model(depth, width, 1e-5, 100)
            result_map[(depth, width)] = result
            if depth == 0:
                break
    #
    # for width in range(max_width):
    #     width = (width+1)*10+100
    #     result = try_model(8, width, 1e-5,100)
    #     result_map[(8, width)] = result

    max_f1 = 0
    max_acc = 0

    max_f1_config = None
    max_acc_config = None

    for k, v in result_map.items():
        loss,accuracy,f1,model = v
        if accuracy > max_acc:
            max_acc = accuracy
            max_acc_config = k

        if f1 > max_f1:
            max_f1 = f1
            max_f1_config = k

    print("Best f1: {}, {}".format(max_f1_config, result_map[max_f1_config]))
    print("Best accuracy: {}, {}".format(max_acc_config, result_map[max_acc_config]))

def test():
    loss, accuracy, f1, model = try_model(8,100, 1e-5, 100)
    save_file_name = './8x100.h5'
    if use_multi_classes:
        save_file_name = './8x100_multi.h5'
    model.save(save_file_name)

def test_multi():
    loss, accuracy, f1, model = try_model(8,100, 1e-5, 100)
    save_file_name = './8x100.h5'
    if use_multi_classes:
        save_file_name = './8x100_multi.h5'
    model.save(save_file_name)

def gen_sensivity():
    for d in [1,5,10,20,50,100]:
    	loss, accuracy, f1, model = try_model(d,100, 1e-5, 100)
    	model.save("./{}x100.h5".format(d))


if __name__ == "__main__":
    test_multi()
    # main()
    # gen_sensivity()
