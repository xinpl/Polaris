from .NNModel import Layer, LayerKind
from .utils import verbose
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, InputLayer
import tensorflow as tf


def createModelFromKerasModel(m, input_length):
    X = tf.placeholder(tf.float32, shape=[None, input_length])
    layers = m.layers
    layer_list = []
    last_output = X
    for l in layers:
        if isinstance(l,InputLayer):
            continue
        if isinstance(l,Dense):
            weights = l.get_weights()[0]
            biases = l.get_weights()[1]
            last_output = tf.matmul(last_output, tf.constant(weights))
            last_output = tf.add(last_output, tf.constant(biases))
            layer_list.append(Layer(LayerKind.dense, last_output, weights, biases))
            activation = l.get_config()['activation']
            if activation == 'relu':
                last_output = tf.nn.relu(last_output)
                layer_list.append(Layer(LayerKind.relu, last_output, None, None))
            elif activation == 'softmax':
                verbose("Warning: treating softmax as the output!",0)
        elif isinstance(l, Dropout):
            continue
        else:
            raise ValueError("Cannot handle layer {}!".format(l))
    return (m, layer_list, X, last_output)


def createModelFromKerasSave(path, input_length):
    m = load_model(path)
    return createModelFromKerasModel(m, input_length)
