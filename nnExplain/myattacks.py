from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import fgm
import tensorflow as tf
import cleverhans.utils as utils
import cleverhans.utils_tf as utils_tf
import collections
from cleverhans.model import Model, CallableModelWrapper

import numpy as np

class MyFastGradientMethod(FastGradientMethod):
    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(FastGradientMethod, self).__init__(model, back, sess)
    
        self.feedable_kwargs = {'eps': np.float32,
                                    'y': np.float32,
                                    'y_target': np.float32,
                                    'clip_min': np.float32,
                                    'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'features', 'pos_grad']
        
        self.prob = {}
    
        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')        
    
    def printTensorValue(self, name, feed_dict):
        tensor = tf.get_default_graph().get_tensor_by_name(name)
        result = self.sess.run(tensor,feed_dict)  
        print(name+': '+str(result))
    
    def getTensor(self, name, feed_dict):
        tensor = tf.get_default_graph().get_tensor_by_name(name)
        return self.sess.run(tensor,feed_dict)            
    
    def getProb(self,hashKey):
        if hashKey not in self.prob:
            x, new_kwargs, x_adv = self.graphs[hashKey]
            self.prob[hashKey] = self.model.get_probs(x)
        return self.prob[hashKey]
        
    
    def generate_adv(self, x_val, maxiter, **kwargs):
        """
        Go along the gradient until adversarial examples are found or the number of iterations have exceeded maxiter
        :param x_val: A NumPy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A NumPy array holding the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')
        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")

        # the set of arguments that are structural properties of the attack
        # if these arguments are different, we must construct a new graph
        fixed = dict((k, v) for k, v in list(kwargs.items())
                     if k in self.structural_kwargs)

        # the set of arguments that are passed as placeholders to the graph
        # on each call, and can change without constructing a new graph
        feedable = dict((k, v) for k, v in list(kwargs.items())
                        if k in self.feedable_kwargs)

        if len(fixed) + len(feedable) < len(kwargs):
            warnings.warn("Supplied extra keyword arguments that are not "
                          "used in the graph computation. They have been "
                          "ignored.")

        if not all(isinstance(value, collections.Hashable)
                   for value in list(fixed.values())):
            # we have received a fixed value that isn't hashable
            # this means we can't cache this graph for later use,
            # and it will have to be discarded later
            hash_key = None
            raise ValueError('Non hashable params!')
        else:
            # create a unique key for this set of fixed paramaters
            hash_key = tuple(sorted(fixed.items()))

        if hash_key not in self.graphs or hash_key is None:
            self.construct_graph(fixed, feedable, x_val, hash_key)

        x, new_kwargs, x_adv = self.graphs[hash_key]

        feed_dict = {}

        for name in feedable:
            feed_dict[new_kwargs[name]] = feedable[name]
        
        preds = self.getProb(hash_key)

        if 'y_target' in new_kwargs:
            expected_labels = new_kwargs['y_target']
        else:
            expected_labels = None

        if expected_labels != None:
            expected_labels = np.argmax(self.sess.run(expected_labels, feed_dict), axis=1)
        
        feed_dict[x] = x_val
        
        old_probs = self.sess.run(preds, feed_dict)
        orig_labs = np.argmax(old_probs, axis=1)
        new_labs_mult = orig_labs.copy()
        adv_examples = x_val.copy()
        last_adv = x_val
        
        iter_count  = 0
        while iter_count < maxiter:
                iter_count+=1
                feed_dict[x] = adv_examples
                new_x_vals = self.sess.run(x_adv, feed_dict)
                feed_dict[x] = new_x_vals
                new_probs = self.sess.run(preds, feed_dict)
                delta = new_x_vals - adv_examples
                #print(delta)
                #print 'gradient: '
                #print self.sess.run(grad, feed_dict)
                #print 'New probaiblity: '
                #print new_probs
                    
                new_labs = np.argmax(new_probs, axis=1)
                if expected_labels == None:
                    I, = np.where(orig_labs == new_labs_mult)
                else:
                    I, = np.where(expected_labels != new_labs_mult)
                if I.size == 0:
                    break
                if np.array_equal(last_adv,new_x_vals):
                    raise ValueError('Gradient 0. Something wrong or hit the corner case.')
                # update labels
                last_adv = new_x_vals
                new_labs_mult[I] = new_labs[I]
                adv_examples[I] = new_x_vals[I]

        if iter_count >= maxiter:
            print("Fail to find an adversarial example!")
            return None

        return adv_examples
        
    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        return myfgm(x, self.model.get_probs(x), y=labels, eps=self.eps,
                   ord=self.ord, features = self.features, clip_min=self.clip_min,
                   clip_max=self.clip_max,
                   targeted=(self.y_target is not None), pos_grad=self.pos_grad)


    def parse_params(self, eps=0.3, ord=np.inf, features = None, y=None, y_target=None,
                     clip_min=None, clip_max=None, pos_grad = False, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.features = features
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.pos_grad = pos_grad

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True


def myfgm(x, preds, y=None, eps=0.3, ord=np.inf, features = None,
        clip_min=None, clip_max=None,
        targeted=False, pos_grad=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if features is None:
        features = np.ones(x.get_shape()[1:])

    features = tf.constant(features, dtype=x.dtype, name = 'feature_clipping')

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x, name = 'adv_gradient')

    grad = tf.multiply(grad, features, name = 'feature_gradient')

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        #red_ind = list(xrange(1, len(x.get_shape())))
        #normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               #reduction_indices=red_ind,
                                               #keep_dims=True)
        grad_shape = tf.shape(grad, name='gradShape')
        second_dim = tf.reduce_prod(grad_shape[1:], name='secDim')
        #second_dim = 28 * 28
        grad1 = tf.reshape(grad, [grad_shape[0], second_dim], name='flattenGrad')
        # test whether certian dimensions would be clipped
        if (clip_min is not None) and (clip_max is not None):
            reshaped_x = tf.reshape(x, tf.shape(grad1))
            normalized_grad1 = eps * tf.sign(grad1)
            hypo_adv_x = reshaped_x + normalized_grad1
            hypo_adv_x1 = tf.clip_by_value(hypo_adv_x, clip_min, clip_max)
            condition = tf.equal(hypo_adv_x,hypo_adv_x1, name = 'condition')
            mask = tf.cast(condition, grad1.dtype, name = 'castmask')
            grad1 = tf.multiply(grad1, mask, name = 'clipped_grad')

        if pos_grad:
            grad2 = tf.clip_by_value(grad1, 0, 10000)
        else:
            grad2 = tf.abs(grad1, name = 'absGrad')
        #print grad1.get_shape()
        max_grad = tf.argmax(grad2, axis = 1, name = 'maxGrad')
        #print max_grad.get_shape()
        normalized_grad = tf.one_hot(max_grad, second_dim, axis = -1, name='oneHotGrad')
        normalized_grad = tf.multiply(normalized_grad, grad1, name = 'maxGrad')
        normalized_grad = tf.reshape(normalized_grad, tf.shape(grad), name='finalGrad')
        normalized_grad = tf.sign(normalized_grad, name='signed_gradient')
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x