import tensorflow as tf
from collections import OrderedDict
import sys
from nnExplain.genExp import Model
from nnExplain.genExp import Layer
from nnExplain.genExp import LayerKind
from nnExplain.genExp import CleverHansModelAdapter
from nnExplain.utils import arrayToText

STDDEV = 1e-4
HIDDEN_SIZE = 200     # Number of hidden neurons 256
DEPTH = 1

def loadConfig(path):
   f = open(path)
   for l in f.readlines():
       tokens = l.split('=')
       if tokens[0].strip() == 'HIDDEN_SIZE':
           global HIDDEN_SIZE
           HIDDEN_SIZE = int(tokens[1].strip())
       if tokens[0].strip() == 'DEPTH':
           global DEPTH
           DEPTH = int(tokens[1].strip())

############## Cleverhans model ######################

def constructCleverHansModel(layerList, input_length, sess):
    # instantiate layerlist tensors
    for layer in layerList:
        if layer.kind == LayerKind.dense:
            layer.weights = sess.run(layer.weights)
            layer.biases = sess.run(layer.biases)
    return CleverHansModelAdapter(Model(layerList, input_length))

######################################################


def mlp_n(_X, n, N_INPUT, N_CLASSES, dropout_keep_prob):
    last_layer = _X
    last_layer_size = N_INPUT
    
    layer_list = []

    for l in range(0,n):
        init1 = tf.random_normal_initializer(stddev=STDDEV)
        init2 = tf.random_normal_initializer(stddev=STDDEV)
        #n_weights = tf.Variable(tf.random_normal([last_layer_size, HIDDEN_SIZE],stddev=STDDEV), name = 'weights_layer_'+str(l))
        #n_b = tf.Variable(tf.random_normal([HIDDEN_SIZE]), name = 'biases_layer_'+str(l))
        n_weights = tf.get_variable('weights_layer_'+str(l), [last_layer_size, HIDDEN_SIZE], initializer=init1)
        n_b = tf.get_variable('biases_layer_'+str(l), [HIDDEN_SIZE], initializer=init2)
        last_layer_size = HIDDEN_SIZE
        new_layer_linear = tf.add(tf.matmul(last_layer, n_weights), n_b);
        # For now, put tensors for weights and biases, we will later replace them with real matrices
        layer_list.append(Layer(LayerKind.dense, new_layer_linear, n_weights, n_b))
        new_layer_relu = tf.nn.relu(new_layer_linear)
        layer_list.append(Layer(LayerKind.relu, new_layer_relu, None, None))
        new_layer_dropout = tf.nn.dropout(new_layer_relu,dropout_keep_prob)
        layer_list.append(Layer(LayerKind.dropout, new_layer_dropout, None, None))
        new_layer =  new_layer_dropout
        
        #if l == n-1:
        #    new_layer =  tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(last_layer, n_weights), n_b)), dropout_keep_prob)
        #else:
        #    new_layer = tf.nn.relu(tf.add(tf.matmul(last_layer, n_weights), n_b))
        last_layer = new_layer
    
    out_weights = tf.get_variable('weights_out', [last_layer_size, N_CLASSES])
    out_b = tf.get_variable('biases_out', [N_CLASSES])      
    #out_weights = tf.Variable(tf.random_normal([last_layer_size, N_CLASSES],stddev=STDDEV),name = 'weights_out')
    #out_b = tf.Variable(tf.random_normal([N_CLASSES]),name = 'biases_out')
 
    out = tf.add(tf.matmul(last_layer, out_weights),out_b)
    
    layer_list.append(Layer(LayerKind.dense, out, out_weights, out_b))
        
    return (out,layer_list)

def mlp_n_nodrop(_X, n, N_INPUT, N_CLASSES):
    last_layer = _X
    last_layer_size = N_INPUT

    layer_list = []

    for l in range(0,n):
        #n_weights = tf.Variable(tf.random_normal([last_layer_size, HIDDEN_SIZE],stddev=STDDEV), name = 'weights_layer_'+str(l))
        #n_b = tf.Variable(tf.random_normal([HIDDEN_SIZE]), name = 'biases_layer_'+str(l))
        n_weights = tf.get_variable('weights_layer_'+str(l), [last_layer_size, HIDDEN_SIZE])
        n_b = tf.get_variable('biases_layer_'+str(l), [HIDDEN_SIZE])        
        last_layer_size = HIDDEN_SIZE
        new_layer_linear = tf.add(tf.matmul(last_layer, n_weights), n_b);
        layer_list.append(Layer(LayerKind.dense, new_layer_linear, n_weights, n_b))
        new_layer_relu = tf.nn.relu(new_layer_linear)
        layer_list.append(Layer(LayerKind.relu, new_layer_relu, None, None))
        new_layer =  new_layer_relu
        
        #if l == n-1:
        #    new_layer =  tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(last_layer, n_weights), n_b)), dropout_keep_prob)
        #else:
        #    new_layer = tf.nn.relu(tf.add(tf.matmul(last_layer, n_weights), n_b))
        last_layer = new_layer
    
    out_weights = tf.get_variable('weights_out', [last_layer_size, N_CLASSES])
    out_b = tf.get_variable('biases_out', [N_CLASSES])      
    #out_weights = tf.Variable(tf.random_normal([last_layer_size, N_CLASSES],stddev=STDDEV),name = 'weights_out')
    #out_b = tf.Variable(tf.random_normal([N_CLASSES]),name = 'biases_out')
 
    out = tf.add(tf.matmul(last_layer, out_weights),out_b)
    
    layer_list.append(Layer(LayerKind.dense, out, out_weights, out_b))
        
    return (out,layer_list)

def constructNetwork(X, dropout_keep_prob, N_INPUT, N_CLASSES):
    print("Constructing a neural network with %d hidden layers and %d neurons per layer.\n"%(DEPTH, HIDDEN_SIZE))
    # Build model
    pred = mlp_n(X, DEPTH, N_INPUT,N_CLASSES, dropout_keep_prob)
    return pred

def constructNetworkWithoutDropout(X, N_INPUT, N_CLASSES):
    print("Constructing a neural network with %d hidden layers and %d neurons per layer.\n"%(DEPTH, HIDDEN_SIZE))
    # Build model
    pred = mlp_n_nodrop(X, DEPTH, N_INPUT,N_CLASSES)
    return pred

def dateToInt(date):
    date_tokens = date.split('/')
    month = int(date_tokens[0])
    year = int(date_tokens[1])
    return (year-1900)*12+month

def convertDateColsToInt(df, dateFields):
    for col in dateFields:
        df[col] = df[col].apply(dateToInt)
        df[col] = df[col].astype(int)
    return df


from tensorflow.python.ops import array_ops


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
