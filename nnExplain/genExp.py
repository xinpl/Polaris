from .myattacks import MyFastGradientMethod
import cleverhans
import tensorflow as tf
import numpy as np
import copy
from .NNModel import *
import collections
import matplotlib.pyplot as plt
import random
from . import utils
from .utils import *
import sys
import traceback
import os

NUM_POINTS = 0
DEFAULT_PADDING = 'SAME'
PLOT_ALL_FEAT_PAIRS = True
NUM_SAMPLES = 10
ADV_MAX_REGIONS = 1000
RADIUS_SCALE_FACTOR = 1.0

def default_renamer(name):
    return name

class CleverHansModelAdapter(cleverhans.model.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def fprop(self, x):
        # Construct a new network and a map for each of its layer
        lastLayer = x
        ret = {}
        layerList = self.model.layerList
        for (idx, layer) in enumerate(layerList):
            if layer.kind == LayerKind.dense:
                nweights = tf.constant(layer.weights, tf.float32, name='chweights_' + str(idx))
                nbiases = tf.constant(layer.biases, tf.float32, name='chbiases_' + str(idx))
                newLayer = tf.add(tf.matmul(lastLayer, nweights), nbiases)
            elif layer.kind == LayerKind.relu:
                newLayer = tf.nn.relu(lastLayer, name='chrelu_' + str(idx))
            elif layer.kind == LayerKind.dropout:
                newLayer = tf.nn.dropout(lastLayer, tf.constant(1.0), name='chdropout_' + str(idx))
            elif layer.kind == LayerKind.conv:
                nweights = tf.constant(layer.weights, tf.float32, name='chweights_'+str(idx))
                nbiases = tf.constant(layer.biases, tf.float32, name='chbiases_'+str(idx))
                newLayer = tf.nn.conv2d(lastLayer, nweights, strides=layer.strides, padding=DEFAULT_PADDING) + nbiases
            elif layer.kind == LayerKind.pool:
                pool_size = layer.weights
                strides = layer.biases
                newLayer = tf.nn.max_pool(lastLayer, ksize=pool_size,
                          strides=strides, padding=DEFAULT_PADDING)
            elif layer.kind == LayerKind.reshape:
                newLayer = tf.reshape(lastLayer, layer.weights)
            elif layer.kind == LayerKind.diff_input:
                newLayer = tf.subtract(lastLayer, x)
            elif layer.kind == LayerKind.absolute:
                newLayer = tf.abs(lastLayer)
            else:
                raise ValueError('Unexpected layer type: ' + str(layer.kind))

            if idx == len(layerList) - 1:
                ret['logits'] = newLayer
            else:
                ret['layer' + str(idx)] = newLayer
            lastLayer = newLayer
        return ret


'''
Main class for generating explanations. 
Since we use modified cleverhans for generating adverserial examples, 
the code is somewhat coupled with tensorflow
'''

class ExplGenerator:
    def __init__(self, model, X, dropout, pred, sess, feature_list):
        self.model = model
        self.sess = sess
        self.X = X
        self.pred = pred
        self.dropout = dropout
        self.feature_list = feature_list
        self.feat_rad = None
        self.feat_scale = None
        self.vis_input = None
        self.pos_grad = False
        chModel = CleverHansModelAdapter(self.model)
        self.fgsm = MyFastGradientMethod(chModel, sess=self.sess)
        self.renamer = default_renamer
        self.adv_func = None

    """
    Set a function to rename the features when plotting the graph.
    """
    def set_renamer(self, fun):
        self.renamer = fun

    """
    bound_map is a dict that maps feature name to a radius. That is, the interval for this feature should be 2*radius
    """
    def set_feat_rad(self, rad_map):
        nmap = {}
        for k,v in rad_map.items():
            nmap[k] = v * RADIUS_SCALE_FACTOR
        self.feat_rad = nmap


    """
    Set the function to visualize the input and the generated adversrial input
    """
    def set_vis_input(self, fun):
        self.vis_input = fun

    """
    dist_map is a dict that assigns a weight to a feature when calculating the distance
    """
    def set_feat_scale(self, dist_map):
        self.feat_scale = dist_map

    def getClassConstr(self, relu_record, targeted_label):
        sym_result = self.model.getOutLayerSymFromRecords(relu_record)

        n_classes = len(targeted_label)

        tgt_idx = np.argmax(targeted_label)

        result_A = sym_result.A
        result_B = sym_result.B

        trans_A = result_A.T
        A_t_shape = trans_A.shape
        add_cons_A_T = np.empty([A_t_shape[0] - 1, A_t_shape[1]])
        add_cons_B = np.empty(len(result_B)-1)
        for i in range(0, tgt_idx):
            add_cons_A_T[i] = trans_A[tgt_idx] - trans_A[i]
            add_cons_B[i] = result_B[tgt_idx] - result_B[i]

        for i in range(tgt_idx+1, n_classes):
            add_cons_A_T[i-1] = trans_A[tgt_idx] - trans_A[i]
            add_cons_B[i-1] = result_B[tgt_idx] - result_B[i]
        add_cons_A = add_cons_A_T.T

        ops = []

        for i in range(0, n_classes-1):
            ops.append(ExprOp.ge)

        class_cons = LinearConstraintSet(add_cons_A, add_cons_B, ops)
        return class_cons

    def normalize_cat_map(self, cat_idx_map):
        ret = {}
        for idx, v in cat_idx_map.items():
            nv = (v - self.model.mean[idx])/self.model.std[idx]
            ret[idx] = nv
        return ret

    def sample_points_Lin_Lin(self, x_val, targeted_label, f1, f2, norm_feat_bounds, sample_scale):
        pos_samples = []
        neg_samples = []
        samples = []
        assert (norm_feat_bounds[f1] is not None)
        assert (norm_feat_bounds[f2] is not None)
        lb1, ub1 = norm_feat_bounds[f1]
        lb2, ub2 = norm_feat_bounds[f2]

        for si1 in range(sample_scale):
            for si2 in range(sample_scale):
                x_val_copy = copy.deepcopy(x_val)
                x_val_copy[f1] = lb1 + (ub1 - lb1) / sample_scale * si1
                x_val_copy[f2] = lb2 + (ub2 - lb2) / sample_scale * si2
                samples.append(x_val_copy)

        sample_results = get_large_tensor_batch(self.sess, self.X, samples, None, None, self.dropout, 100, self.pred)

        for s, r in zip(samples, sample_results):
            denorm_s = np.multiply(s, self.model.std)
            denorm_s = np.add(denorm_s, self.model.mean)
            denorm_s = [denorm_s[f1], denorm_s[f2]]
            if np.argmax(r) == np.argmax(targeted_label):
                pos_samples.append(denorm_s)
            else:
                neg_samples.append(denorm_s)
        return pos_samples, neg_samples

    def sample_points_Cat_Lin(self, x_val, expandMap, KEYS, targeted_label, v1, f2, norm_feat_bounds, sample_scale):
        pos_samples = []
        neg_samples = []

        for exp_col1 in expandMap[v1]:
            s = copy.deepcopy(x_val)
            samples = []
            pos_col = []
            neg_col = []
            for exp_col2 in expandMap[v1]:
                col_idx = KEYS.index(exp_col2)
                if exp_col1 != exp_col2:
                    s[col_idx] = (0 - self.model.mean[col_idx]) / self.model.std[col_idx]
                else:
                    s[col_idx] = (1 - self.model.mean[col_idx]) / self.model.std[col_idx]
            for si1 in range(sample_scale):
                s1 = copy.deepcopy(s)
                lb1, ub1 = norm_feat_bounds[f2]
                s1[f2] = lb1 + (ub1 - lb1) / sample_scale * si1
                samples.append(s1)

            results = get_large_tensor_batch(self.sess, self.X, samples, None, None, self.dropout, 100, self.pred)
            for s,r in zip(samples,results):
                s_v2 = s[f2]
                s_v2 = s_v2 * self.model.std[f2] + self.model.mean[f2]
                if np.argmax(r) == np.argmax(targeted_label):
                    pos_col.append(s_v2)
                else:
                    neg_col.append(s_v2)
            pos_samples.append(pos_col)
            neg_samples.append(neg_col)

        s = copy.deepcopy(s)
        for exp_col in expandMap[v1]:
            col_idx = KEYS.index(exp_col)
            s[col_idx] = (0 - self.model.mean[col_idx]) / self.model.std[col_idx]

        lb1, ub1 = norm_feat_bounds[f2]
        for si in range(sample_scale):
            s1 = copy.deepcopy(s)
            s1[f2] = lb1 + (ub1 - lb1) / sample_scale * si
            samples.append(s1)
        pos_col = []
        neg_col = []
        results = get_large_tensor_batch(self.sess, self.X, samples, None, None, self.dropout, 100, self.pred)
        for s, r in zip(samples, results):
            s_v2 = s[f2]
            s_v2 = s_v2 * self.model.std[f2] + self.model.mean[f2]
            if np.argmax(r) == np.argmax(targeted_label):
                pos_col.append(s_v2)
            else:
                neg_col.append(s_v2)
        pos_samples.append(pos_col)
        neg_samples.append(neg_col)

        return pos_samples, neg_samples

    '''
    # x_val: normalized input vector 
    # unorm_x_val: original input without normalization
    # targeted_label: the desired output class. One-hot encoding
    # ori_KEYS: the original feature names before one-hot encoding
    # KEYS: the feature names after one-hot encoding
    # expandMap: a map that contains information about one-hot encoding. It maps a categorical features to its expanded features (in strings)
    # ban_features: set/list of features that are not allowed to change (in strings)
    # norm_feat_bounds: a dictionary that maps a feature index to the feature's lower bound and upper bound (normalized)
    '''
    def pickBestExplaination2D(self, x_val, unnorm_x_val, targetded_label, ori_KEYS, KEYS, expandMap, ban_features = [], num_regions = 100, norm_feat_bounds = {}, sample_scale = None):
        global NUM_POINTS
        NUM_POINTS += 1
        best_dist = None
        feat_pair = None
        start_time = time.time()
        for (i1,v1) in enumerate(ori_KEYS):
            for (i2, v2) in enumerate(ori_KEYS):
                if i1 >= i2:
                    continue
                if v1 in ban_features or v2 in ban_features:
                    continue

                fea_pair_startime = time.time()
                if_cat1 = (v1 in expandMap)
                if_cat2 = (v2 in expandMap)
                fig_name = str(NUM_POINTS)+'_'+v1+'_'+v2

                verbose('Search on features: '+v1+', '+v2, 1)

                if not if_cat1 and not if_cat2:
                    f1 = KEYS.index(v1)
                    f2 = KEYS.index(v2)
                    features = np.zeros(len(x_val))
                    features[f1] = 1
                    features[f2] = 1
                    ret = self.genExplaination(x_val, targetded_label, features, {}, norm_feat_bounds,num_regions)
                    if ret is not None:
                        tri = ret[0]
                        edges = ret[1]
                        ori_point = unnorm_x_val[[f1,f2]]
                        print('Triangle: ' + str(tri))
                        print('Edges: '+str(edges))
                        if PLOT_ALL_FEAT_PAIRS:
                            # Sample points
                            pos_samples = None
                            neg_samples = None
                            if sample_scale is not None:
                                pos_samples, neg_samples = self.sample_points_Lin_Lin(x_val, targetded_label, f1, f2, norm_feat_bounds, sample_scale)
                            plotLinLin(fig_name, [self.renamer(v1), self.renamer(v2)], ori_point, tri, edges, pos_samples, neg_samples)
                        dist, cp = self.get_nearest_box_lin_lin(ori_point, tri, v1, v2)
                        if dist is not None:
                            verbose('Distance: '+str(dist), 1)
                            if best_dist is None or best_dist > dist:
                                best_dist = dist
                                feat_pair = (v1, v2)
                        else:
                            verbose('Rejected for stability', 1)
                    else:
                        verbose("Fail to find a region", 1)

                if if_cat1 and not if_cat2:
                    f2 = KEYS.index(v2)
                    features = np.zeros(len(x_val))
                    features[f2] = 1
                    inters = []
                    cat_names = [ x for x in expandMap[v1]]
                    f2_v = unnorm_x_val[f2]
                    f1_v = None
                    verbose('Number of categories for '+v1+': '+str(len(expandMap[v1])), 1)
                    cat_idx = 0
                    for (exp_col1_idx, exp_col1) in enumerate(expandMap[v1]):
                        ec1 = KEYS.index(exp_col1)
                        if unnorm_x_val[ec1] > 0.1:
                            f1_v = exp_col1_idx
                        cat_idx_map = {}
                        cat_idx_map[ec1] = 1
                        for exp_col2 in expandMap[v1]:
                            if exp_col2 != exp_col1:
                                cat_idx_map[KEYS.index(exp_col2)] = 0
                        cat_idx_map = self.normalize_cat_map(cat_idx_map)
                        cat_stime = time.time()
                        inter = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds)
                        verbose('Time spent for category '+str(cat_idx)+': '+str(time.time()-cat_stime),1)
                        cat_idx += 1

                        inters.append(inter)
                    # handle the none case. Note when we use one-hot encoding, we only create n-1 columns:
                    cat_names.append('Other')
                    cat_idx_map = {}
                    for exp_col in expandMap[v1]:
                        cat_idx_map[KEYS.index(exp_col)] = 0
                    cat_idx_map = self.normalize_cat_map(cat_idx_map)
                    inter = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds, num_regions)

                    inters.append(inter)

                    if f1_v is None:
                        f1_v = len(cat_names) - 1
                    if len(inters) != 0 and any(x is not None for x in inters):
                        print('Intervals: ' + str(inters))
                        ori_point = (f1_v, f2_v)
                        ret_x, ret_y, dist = self.get_nearestbox_cat_lin(ori_point, inters, v1, v2)
                        if dist is not None:
                            verbose('Distance: '+str(dist), 1)
                            if best_dist is None or best_dist > dist:
                                best_dist = dist
                                feat_pair = (v1, v2)
                        else:
                            verbose('Rejected for stability', 1)
                        if PLOT_ALL_FEAT_PAIRS:
                            # Sample points
                            pos_samples = None
                            neg_samples = None
                            if sample_scale is not None:
                                pos_samples,neg_samples = self.sample_points_Cat_Lin(x_val, expandMap, KEYS, targetded_label, v1, f2, norm_feat_bounds, sample_scale)
                            plotCatLin(fig_name, [self.renamer(v1), self.renamer(v2)], cat_names, ori_point, inters, pos_samples, neg_samples)

                if not if_cat1 and if_cat2:
                    f1 = KEYS.index(v1)
                    features = np.zeros(len(x_val))
                    features[f1] = 1
                    inters = []
                    cat_names = [ x for x in expandMap[v2]]
                    f1_v = unnorm_x_val[f1]
                    f2_v = None

                    verbose('Number of categories for '+v2+': '+str(len(expandMap[v2])),1)
                    cat_idx = 0
                    for (exp_col1_idx, exp_col1) in enumerate(expandMap[v2]):
                        ec1 = KEYS.index(exp_col1)
                        if unnorm_x_val[ec1] > 0.1:
                            f2_v = exp_col1_idx
                        cat_idx_map = {}
                        cat_idx_map[ec1] = 1
                        for exp_col2 in expandMap[v2]:
                            if exp_col2 != exp_col1:
                                cat_idx_map[KEYS.index(exp_col2)] = 0
                        cat_idx_map = self.normalize_cat_map(cat_idx_map)
                        cat_stime = time.time()
                        inter = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds, num_regions)
                        verbose('Time spent for category '+str(cat_idx)+': '+str(time.time()-cat_stime), 1)
                        cat_idx += 1

                        inters.append(inter)
                    # handle the none case. Note when we use one-hot encoding, we only create n-1 columns:
                    cat_names.append('Other')
                    cat_idx_map = {}
                    for exp_col in expandMap[v2]:
                        cat_idx_map[KEYS.index(exp_col)] = 0
                    cat_idx_map = self.normalize_cat_map(cat_idx_map)
                    inter = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds, num_regions)

                    inters.append(inter)

                    if f2_v is None:
                        f2_v = len(cat_names) - 1

                    if len(inters) != 0 and any(x is not None for x in inters):
                        ori_point = (f2_v, f1_v)
                        print('Intervals: ' + str(inters))
                        ret_x, ret_y, dist = self.get_nearestbox_cat_lin(ori_point, inters, v2, v1)
                        if dist is not None:
                            verbose('Distance: '+str(dist), 1)
                            if best_dist is None or best_dist > dist:
                                best_dist = dist
                                feat_pair = (v1, v2)
                        else:
                            verbose('Rejected for stability', 1)

                        if PLOT_ALL_FEAT_PAIRS and len(inters) != 0:
                            # Sample points
                            pos_samples = None
                            neg_samples = None
                            if sample_scale is not None:
                                pos_samples, neg_samples = self.sample_points_Cat_Lin(x_val, expandMap, KEYS,
                                                                                      targetded_label, v2, f1,
                                                                                      norm_feat_bounds, sample_scale)
                            plotCatLin(fig_name, [self.renamer(v2), self.renamer(v1)], cat_names, (f2_v, f1_v), inters, pos_samples, neg_samples)

                if if_cat1 and if_cat2:
                    features = np.zeros(len(x_val))
                    points = []
                    f1_v = None
                    f2_v = None
                    cat_names1 = [x for x in expandMap[v1]]
                    cat_names2 = [x for x in expandMap[v2]]
                    for (exp_col1_idx, exp_col1) in enumerate(expandMap[v1]):
                        ec1 = KEYS.index(exp_col1)
                        if unnorm_x_val[ec1] > 0.1:
                            f1_v = exp_col1_idx

                        cat_idx_map = {}
                        cat_idx_map[ec1] = 1
                        for exp_col2 in expandMap[v1]:
                            if exp_col2 != exp_col1:
                                cat_idx_map[KEYS.index(exp_col2)] = 0

                        for (exp_col3_idx, exp_col3) in enumerate(expandMap[v2]):
                            ec3 = KEYS.index(exp_col3)
                            cat_idx_map[ec3] = 1
                            if unnorm_x_val[ec3] > 0.1:
                                f2_v = exp_col3_idx
                            for exp_col4 in expandMap[v2]:
                                if exp_col4 != exp_col3:
                                    cat_idx_map[KEYS.index(exp_col4)] = 0
                            cat_idx_map = self.normalize_cat_map(cat_idx_map)
                            point = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds)
                            if point is not None:
                                points.append([exp_col1_idx, exp_col3_idx])

                        # None case
                        for exp_col3 in expandMap[v2]:
                            cat_idx_map[KEYS.index(exp_col3)] = 0
                        cat_idx_map = self.normalize_cat_map(cat_idx_map)
                        point = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds)
                        if point is not None:
                            points.append([exp_col1_idx, len(expandMap[v2])])

                    # None case for first feature
                    cat_idx_map = {}
                    for exp_col1 in expandMap[v1]:
                        cat_idx_map[KEYS.index(exp_col1)] = 0

                    exp_col1_idx = len(expandMap[v1])

                    for (exp_col3_idx, exp_col3) in enumerate(expandMap[v2]):
                        ec3 = KEYS.index(exp_col3)
                        cat_idx_map[ec3] = 1
                        for exp_col4 in expandMap[v2]:
                            if exp_col4 != exp_col3:
                                cat_idx_map[KEYS.index(exp_col4)] = 0
                        cat_idx_map = self.normalize_cat_map(cat_idx_map)
                        point = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds)
                        if point is not None:
                            points.append([exp_col1_idx, exp_col3_idx])

                    # None case
                    for exp_col3 in expandMap[v2]:
                        cat_idx_map[KEYS.index(exp_col3)] = 0
                    cat_idx_map = self.normalize_cat_map(cat_idx_map)
                    point = self.genExplaination(x_val, targetded_label, features, cat_idx_map, norm_feat_bounds)
                    if point is not None:
                        points.append([exp_col1_idx, len(expandMap[v2])])

                    cat_names1.append('Other')
                    cat_names2.append('Other')

                    if f1_v is None:
                        f1_v = len(cat_names1) - 1

                    if f2_v is None:
                        f2_v = len(cat_names2) - 1
                    if len(points) != 0:
                        print('POINTS: ' + str(points))
                        ori_point = (f1_v, f2_v)
                        dist = self.get_nearestbox_cat_cat(ori_point, points, v1, v2)
                        if dist is not None:
                            verbose('Distance: '+str(dist), 1)
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                feat_pair = (v1, v2)
                        else:
                            print('Rejected for stability')
                        if PLOT_ALL_FEAT_PAIRS:
                            plotCatCat(fig_name, [self.renamer(v1), self.renamer(v2)], cat_names1, cat_names2, ori_point, points)

                verbose('Search time for '+v1+", "+v2+": "+str(time.time() - fea_pair_startime),1)

        verbose('RESULT: point '+str(NUM_POINTS), 0)
        verbose("RESULT: best pair "+str(feat_pair), 0)
        verbose("RESULT: distance "+str(best_dist), 0)
        verbose('RESULT: Total time: '+str(time.time() - start_time), 0)


    def get_nearest_box_lin_lin(self, ori_point, tri: List, feat_name1, feat_name2):
        # convert tri to a set of linear constraints
        tri_A = []
        tri_B = []
        tri_ops = []
        for i in range(len(tri)):
            pa = tri[i]
            pb = tri[(i+1) % len(tri)]
            a_row = [pb[1] - pa[1], pa[0] - pb[0]]
            b_row = pa[0] * (pa[1] - pb[1]) + pa[1] * (pb[0] - pa[0])
            pc = tri[(i+2) % len(tri)]
            cv = np.dot(pc, a_row) + b_row
            if cv >= 0:
                op_row = ExprOp.ge
            else:
                op_row = ExprOp.le
            tri_A.append(a_row)
            tri_B.append(b_row)
            tri_ops.append(op_row)
        tri_cons = LinearConstraintSet(np.transpose(tri_A), tri_B, tri_ops)

        assert(not tri_cons.eval(ori_point))

        assert(tri_cons.solve() is not None)

        tri = tri_cons

        # construct a new linear program
        r1 = self.feat_rad[feat_name1]
        r2 = self.feat_rad[feat_name2]
        w1 = self.feat_scale[feat_name1]
        w2 = self.feat_scale[feat_name2]

        diag1 = tri.shift([r1,r2])
        diag2 = tri.shift([r1,-r2])
        diag3 = tri.shift([-r1,-r2])
        diag4 = tri.shift([-r1,r2])

        cons_base = diag1.extend(diag2).extend(diag3).extend(diag4)

        assert(not cons_base.eval(ori_point))

        ret = None
        dist = None
        # the case where central.x > ori_point.x and central.y > ori_point.y
        loc_constr = LinearConstraintSet(np.transpose([[1,0],[0,1]]),
                                         [-ori_point[0], -ori_point[1]],
                                         [ExprOp.ge, ExprOp.ge])
        case1 = cons_base.extend(loc_constr)
        ret = case1.minimize([w1,w2])
        if ret is not None:
            dist = w1 * (ret[0] - ori_point[0]) + w2 * (ret[1] - ori_point[1])

        # the case where central.x > ori_point.x and central.y < ori_point.y
        loc_constr = LinearConstraintSet(np.transpose([[1,0],[0,1]]),
                                         [-ori_point[0], -ori_point[1]],
                                         [ExprOp.ge, ExprOp.le])
        case2 = cons_base.extend(loc_constr)
        ret1 = case2.minimize([w1,-w2])
        if ret1 is not None:
            dist1 = w1 * (ret1[0] - ori_point[0]) + w2 * (ori_point[1] - ret1[1])
            if dist is None or dist1 < dist:
                dist = dist1
                ret = ret1

        # the case where central.x < ori_point.x and central.y > ori_point.y
        loc_constr = LinearConstraintSet(np.transpose([[1,0],[0,1]]),
                                         [-ori_point[0], -ori_point[1]],
                                         [ExprOp.le, ExprOp.ge])
        case3 = cons_base.extend(loc_constr)
        ret1 = case3.minimize([-w1,w2])
        if ret1 is not None:
            dist1 = w1 * (ori_point[0] - ret1[0]) + w2 * (ret1[1] - ori_point[1])
            if dist is None or dist1 < dist:
                dist = dist1
                ret = ret1

        # the case where central.x < ori_point.x and central.y < ori_point.y
        loc_constr = LinearConstraintSet(np.transpose([[1,0],[0,1]]),
                                         [-ori_point[0], -ori_point[1]],
                                         [ExprOp.le, ExprOp.le])
        case4 = cons_base.extend(loc_constr)
        ret1 = case4.minimize([-w1,-w2])
        if ret1 is not None:
            dist1 = w1 * (ori_point[0] - ret1[0]) + w2 * (ori_point[1] - ret1[1])
            if dist is None or dist1 < dist:
                dist = dist1
                ret = ret1

        return (dist, ret)

    def get_nearestbox_cat_lin(self, ori_point, intervals: List, feat_name1, feat_name2):
        r1 = self.feat_rad[feat_name1]
        r2 = self.feat_rad[feat_name2]
        w1 = self.feat_scale[feat_name1]
        w2 = self.feat_scale[feat_name2]

        ret_dist = None
        ret_y = None
        ret_x = None

        d1 = int(r1 * 2)
        # We need to use an integer linear programming problem to solve the problem
        m = gurobipy.Model('cat linear regions')
        num_cat_vars = len(intervals)
        # Add category vars, each var means whether we choose a given category
        cat_vars = m.addVars(list(range(num_cat_vars)), vtype = gurobipy.GRB.BINARY)
        # y_var of the center point. x_var is represented by whether the given category is chosen
        y_var = m.addVar(-gurobipy.GRB.INFINITY, gurobipy.GRB.INFINITY, 0.0, gurobipy.GRB.CONTINUOUS, 'y')
        # Encode the volume constraint along y, the linear feature
        for (idx, inter) in enumerate(intervals):
            cat_var = cat_vars[idx]
            if inter is None:
                m.addConstr(cat_var, gurobipy.GRB.EQUAL, 0)
                continue
            m.addConstr(cat_var*(inter[0]+r2) - (1-cat_var)*utils.LINEAR_TOLERANCE_SCALE <= y_var)
            m.addConstr(y_var <= cat_var*(inter[1]-r2) + (1-cat_var)*utils.LINEAR_TOLERANCE_SCALE)

        # Encode the volume constraint on number of categories
        num_cat_chosen = 0
        for idx, cat_var in cat_vars.items():
           num_cat_chosen += cat_var

        m.addConstr(num_cat_chosen, gurobipy.GRB.GREATER_EQUAL, d1)

        # Finally add the objective
        # two cases, y is above the original y or below
        # case 1, above:
        case_cons = m.addConstr(y_var >= ori_point[1])
        ori_x_var = cat_vars[ori_point[0]]
        obj = (-ori_x_var) * w1 + y_var * w2
        m.setObjective(obj, gurobipy.GRB.MINIMIZE)

        m.optimize()
        obj_expr = m.getObjective()
        if m.status == gurobipy.GRB.OPTIMAL:
            ret_dist = obj_expr.getValue() + w1 - w2*ori_point[1]
            ret_x = (ori_x_var.x > 0.5)
            ret_y = y_var.x

        # case 2, below:
        m.remove(case_cons)
        m.update()
        case_cons = m.addConstr(y_var <= ori_point[1])
        ori_x_var = cat_vars[ori_point[0]]
        obj = (-ori_x_var) * w1 - y_var*w2
        m.setObjective(obj, gurobipy.GRB.MINIMIZE)
        m.optimize()
        obj_expr = m.getObjective()
        if m.status == gurobipy.GRB.OPTIMAL:
            obj_value = obj_expr.getValue()
            if ret_dist is None or obj_value < ret_dist:
                ret_dist = obj_expr.getValue() + w1 + w2*ori_point[1]
                ret_x = (ori_x_var.x > 0.5)
                ret_y = y_var.x

        return (ret_x, ret_y, ret_dist)

    def get_nearestbox_cat_cat(self, ori_point, points: List, feat_name1, feat_name2):
        r1 = self.feat_rad[feat_name1]
        r2 = self.feat_rad[feat_name2]
        w1 = self.feat_scale[feat_name1]
        w2 = self.feat_scale[feat_name2]
        d1 = int(r1*2)
        d2 = int(r2*2)

        x_to_ys = {}
        for p in points:
            px = p[0]
            py = p[1]
            if px not in x_to_ys:
                x_to_ys[px] = set()
            x_to_ys[px].add(py)

        x_cats = list(x_to_ys.keys())

        m = gurobipy.Model('cat cat regions')
        num_x_cats= len(x_cats)
        # Add category vars, each var means whether we choose a given category
        cat_vars = m.addVars(list(range(num_x_cats)), vtype = gurobipy.GRB.BINARY)
        # Add point vars, each var means whether a given point falls into
        num_points = len(points)
        point_vars =m .addVars(list(range(num_points)), vtype = gurobipy.GRB.BINARY)

        # Encode the constraint about whether a given category is chosen:
        for x_cat, x_cat_var in zip(x_cats, cat_vars):
            for py in x_to_ys[x_cat]:
                # encode the relation that if x_cat is not chosen, none of (x_cat, *) would be chosen
                p_idx = points.index([x_cat, py])
                p_var = point_vars[p_idx]
                m.addConstr(p_var <= x_cat_var)
            # encode the relation that if x_cat is chosen, only (*, y_cat) where y_cat is in x_to_ys[x_cat] should be chosen
            for p, p_var in zip(points, point_vars):
                if p[1] not in x_to_ys[x_cat]:
                    m.addConstr(p_var <= (1 - x_cat_var))

        # Encode the constraint along x:
        x_cat_expr = 0
        for x_cat_var in cat_vars:
            x_cat_var += x_cat_var
        m.addConstr(x_cat_var >= d1)

        # Encode the constraint along y. We do it indirectly, total_num_points >= x * y:
        point_expr = 0
        for p_var in point_vars:
            point_expr += p_var
        m.addConstr(point_expr >= d1 * d2)

        # Finally, minimize the distance
        ori_x_cat_var = m.addVar(vtype = gurobipy.GRB.BINARY)
        ori_y_cat_var = m.addVar(vtype = gurobipy.GRB.BINARY)

        x_matched = False

        for x_cat, x_cat in zip(x_cats, cat_vars):
            if x_cat == ori_point[0]:
                m.addConstr(ori_x_cat_var >= x_cat)
                x_matched = True
                break

        if not x_matched:
            m.addConstr(ori_x_cat_var == 0)

        y_matched = False

        for p, p_var in zip(points, point_vars):
            if p[1] == ori_point[1]:
                y_matched = True
                m.addConstr(ori_y_cat_var,gurobipy.GRB.EQUAL,p_var)

        if not y_matched:
            m.addConstr(ori_y_cat_var, gurobipy.GRB.EQUAL,0)

        obj_expr = -w1 * ori_x_cat_var - w2*ori_y_cat_var

        m.setObjective(obj_expr, gurobipy.GRB.MINIMIZE)

        m.optimize()

        if m.status == gurobipy.GRB.INFEASIBLE:
            return None

        return m.getObjective().getValue()


    def setPosGrad(self):
        self.pos_grad = True

    def denormalize(self, x_val):
        return np.multiply(x_val, self.model.std) + self.model.mean

    def getSeedRegionConstr(self, x_val, targeted_label, bounding_constraints, fixed_feature, fixed_values):
        ori_relu_record = self.model.getReluActivationRecord(x_val)

        visited_relu_record = []

        workList = []
        workList.append(ori_relu_record)

        while len(visited_relu_record) < ADV_MAX_REGIONS and len(workList) > 0:
            head = workList[0]
            visited_relu_record.append(head)
            workList = workList[1:]
            head_region:LinearRegion = self.model.getLinearRegionFromRecord(head)
            class_cons = self.getClassConstr(head, targeted_label)
            head_region.addAddtionalConstraint(class_cons)
            head_region.addAddtionalConstraint(bounding_constraints)
            head_region.fix_vars(fixed_feature, fixed_values)
            verbose('Number of regions checked: '+str(len(visited_relu_record)), 2)
            try:
                if(head_region.check()):
                    return head
            except TimeoutError as e:
                raise(e)
            except Exception as e:
                verbose("Unexpected error:", sys.exc_info()[0], 0)
                print(e)

            for (idx, rr) in enumerate(head):
                for idx1 in range(len(rr.record)):
                    if len(visited_relu_record) + len(workList) < ADV_MAX_REGIONS + 10:
                        rrs1 = copy.copy(head) # type: List[ReluRecord]
                        nRecord = list(rrs1[idx].record)
                        nRecord[idx1] = 1 - nRecord[idx1]
                        rrs1[idx] = ReluRecord(rrs1[idx].idx, tuple(nRecord))
                        # first check if we have visited this region
                        if rrs1 not in visited_relu_record and rrs1 not in workList:
                            workList.append(rrs1)

        return None

        # adv_same_region = ori_region.getLinearConstr().eval(ori_x_val)

        ori_region.addAddtionalConstraint(class_cons)
        # ori_region.addAddtionalConstraint(feat_cons)

        ori_region.addAddtionalConstraint(bounding_box)

        ori_region.fix_vars(fixed_feature, fixed_values)


    def getSeedRegionGrid(self, x_val, targeted_label, bounds, fixed_feature, fixed_values, pred, X, dropout, sess, scale = 1e-2):
        def eval_X(x):
            y = sess.run(pred, feed_dict={X: [x_val], dropout: 1.0})[0]
            return np.argmax(y[0]) == np.argmax(targeted_label)

        x_val1 = copy.deepcopy(x_val)
        for i, v in zip(fixed_feature, fixed_values):
            x_val1[i] = v
        
        feat_length = len(x_val1)

        # store the delta to each feature   
        feat_stack = []

        unfixed_indices = []

        # seed
        for i in range(feat_length):
            if i not in fixed_feature:
                unfixed_indices.append(i)
                feat_stack.append(0)
        
        x_val2 = copy.deepcopy(x_val1)
                
        while True:
            assert(len(feat_stack) == len(unfixed_indices))
            for idx, v in enumerate(feat_stack):
                feat_idx = unfixed_indices[idx]
                x_val2[feat_idx] = x_val1[feat_idx] + v
            if eval_X(x_val2):
                return self.model.getReluActivationRecord(x_val2)
            
            while len(feat_stack) > 0:
                tidx = len(feat_stack) - 1
                tfeat = unfixed_indices[tidx]
                tv = feat_stack[-1]
                poped = False
                if tv > 0:
                    if x_val1[tfeat] - tv < bounds[tidx][0] and x_val1[tfeat] + tv + scale > bounds[tidx][1]:
                        feat_stack.pop()
                        poped = True
                    else:
                        if x_val1[tfeat] -tv >= bounds[tidx][0]:
                            feat_stack[-1] = -tv
                        elif x_val1[tfeat] + tv + scale <= bounds[tidx][1]:
                            feat_stack[-1] = tv + scale
                        else:
                            raise ValueError("Wrong")
                else:
                    if x_val1[tfeat] - tv + scale > bounds[tidx][1] and x_val1[tfeat] +tv - scale < bounds[tidx][0]:
                        feat_stack.pop()
                        poped = True
                    else:
                        if x_val1[tfeat] - tv + scale <= bounds[tidx][1]:
                            feat_stack[-1] = scale - tv
                        elif x_val1[tfeat] + tv - scale >= bounds[tidx][0]:
                            feat_stack[-1] = tv - scale
                        else:
                            raise ValueError("Wrong")
                
                if not poped:
                    break
            
            if len(feat_stack) == 0:
                return None
            
            while len(feat_stack) < len(unfixed_indices):
                feat_stack.append(0)
                    

    def get_adv(self, x_val, targeted_label, features_to_change):
        # chModel = CleverHansModelAdapter(self.model)
        # fgsm = MyFastGradientMethod(chModel, sess=self.sess)

        fgsm = self.fgsm

        # targeted_tensor = tf.placeholder(tf.int64, [None, pred.get_shape()[1]])

        fgsm_params = {'eps': 0.01,
                       'ord': 1,
                       'y_target': np.array([targeted_label]),
                       'features': tuple(features_to_change)
                       }
        if self.pos_grad:
            fgsm_params['pos_grad'] = True
               
        maxIter = 1000
        verbose('Finding a seed ... \n', 1)
        adv_X = fgsm.generate_adv(np.array([x_val]), maxIter, **fgsm_params)
        if adv_X is None:
            return None

        adv_X = adv_X[0]
        return adv_X


    def set_adv_func(self, adv_func):
        self.adv_func = adv_func


    def getSeedRegionAdv(self, x_val, y_val, targeted_label, features_to_change, X, pred, dropout, sess):
        if(np.argmax(y_val) == np.argmax(targeted_label)):
            adv_X = x_val
        else:
            if self.adv_func is not None:
                adv_X = self.adv_func(x_val, targeted_label, features_to_change)
            else:
                adv_X = self.get_adv(x_val, targeted_label, features_to_change)
            if adv_X is None:
                return None
            if dropout is None:
                verbose(('Attack label: ' + str(sess.run(pred, feed_dict={X: [adv_X]}))), 1)
            else:
                verbose(('Attack label: ' + str(sess.run(pred, feed_dict={X: [adv_X], dropout: 1.0}))), 1)
            delta = adv_X - x_val
            clip_func = np.vectorize(lambda x: x if abs(x) > 0.009 else 0.0)
            delta = clip_func(delta)
            adv_X = x_val + delta
            if dropout is None:
                verbose(('Attack label: ' + str(sess.run(pred, feed_dict={X: [adv_X]}))), 1)
            else:
                verbose(('Attack label: ' + str(sess.run(pred, feed_dict={X: [adv_X], dropout: 1.0}))), 1)

            if self.vis_input is not None:
                ori_input = self.denormalize(x_val)
                adv_input = self.denormalize(adv_X)
                delta = (adv_input - ori_input)*10
                fig_id = utils.unique_id_gen()
                self.vis_input(ori_input, 'ori_'+str(fig_id))
                self.vis_input(adv_input, 'adv_'+str(fig_id))
                self.vis_input(delta, 'delta_'+str(fig_id))
            else:
                pass
        return self.model.getReluActivationRecord(adv_X)

    '''
    Suggest what features to change by using fgsm with L0.
    '''
    def suggestFeaturesToChange(self, x_val, targeted_label, norm_feat_bounds={}):
        fgsm = self.fgsm
        features_to_change = [1] * len(x_val)
        clip_min = 100
        clip_max = -100
        for l,u in norm_feat_bounds.values():
            if l < clip_min:
                clip_min = l
            if u > clip_max:
                clip_max = u
        fgsm_params = {'eps': 0.01,
                       'ord': 1,
                       'y_target': np.array([targeted_label]),
                       'features': tuple(features_to_change),
                       'clip_min': clip_min,
                       'clip_max': clip_max
                       }
        if self.pos_grad:
            fgsm_params['pos_grad'] = True
        maxIter = 1000
        verbose('Finding a seed ... \n', 1)
        adv_X = fgsm.generate_adv(np.array([x_val]), maxIter, **fgsm_params)
        if adv_X is None:
            return None

        adv_X = adv_X[0]

        ret = []

        for x, ax in zip(x_val, adv_X):
            if abs(ax - x) > LINEAR_TOLERANCE:
                ret.append(1)
            else:
                ret.append(0)

        return ret


    def genExplaination(self, x_val, targeted_label, features_to_change, features_change_fixed, norm_feat_bounds = None, MAX_NUM_REGIONS = 100, adv_center = False, seed_region_meth = 0):
        """
        Give an explanation on why x_val does not yield the given targted_label or why x_val yields the label. Work with a single row.
        x_val: single normalized instance.
        targeted_label: desired label. One-hot encoding.
        features_to_change: specify the features that are allowed to change. One-hot encoding. This is after flattening the array.
        features_change_fixed: a map that maps a given feature to a fixed value. This is after flattening the array.
        norm_feat_bound: a dict that maps feature indices to features' lower and upper bounds. This is after flattening the array.
        Note the return value is denormalized.
        """
        try:
            flattened_x = np.ndarray.flatten(x_val)
            feat_indices = []
            for (i,f) in enumerate(features_to_change):
                if f > 0.1:
                    feat_indices.append(i)

            verbose('# dimensions: {}'.format(str(len(feat_indices))), 1)

            sess = self.sess
            X = self.X
            pred = self.pred
            dropout = self.dropout

            ori_x_val = x_val
            # x_val = copy.copy(x_val)

            # modify x_val based on features_change_fixed
            for k,v in features_change_fixed.items():
                flattened_x[k] = v

            x_val = np.reshape(flattened_x, np.shape(x_val))

            # check the new label of x_val
            if dropout is None:
                y_val = sess.run(pred, feed_dict={X: [x_val]})
            else:
                y_val = sess.run(pred, feed_dict={X: [x_val], dropout: 1.0})

            if np.count_nonzero(features_to_change) == 0:
                if np.argmax(y_val) == np.argmax(targeted_label):
                    return []
                else:
                    return None

            ############## Before doing the job, let us analyze the network first ##################

            # extend norm_feat_bounds
            for v in feat_indices:
                if v not in norm_feat_bounds:
                    norm_feat_bounds[v] = [-100, 100]

            self.model.analyzeRedundant(flattened_x, feat_indices, norm_feat_bounds)

            ############## Add a bounding box so the region found will always be bounded #####################
            bb_A = np.zeros([len(flattened_x),2*len(feat_indices)])
            for (idx, f_v) in enumerate(feat_indices):
                bb_A[f_v, idx*2] = 1
                bb_A[f_v, idx*2+1] = 1

            bb_B = np.empty([2*len(feat_indices)])
            for idx, v in enumerate(feat_indices):
                # Since this is after normalization, we don't need a very large number
                if v in norm_feat_bounds:
                    bb_B[idx*2] = -norm_feat_bounds[v][1] # upper bound
                    bb_B[idx*2+1] = -norm_feat_bounds[v][0] # lower bound
                else:
                    bb_B[idx*2] = -100
                    bb_B[idx*2+1] = 100

            bb_ops = []
            for idx in range(len(feat_indices)):
                bb_ops.append(ExprOp.le)
                bb_ops.append(ExprOp.ge)

            bounding_box = LinearConstraintSet(bb_A, bb_B, bb_ops)

            ############# Generate feature constraints, it basically fix the values of features that are not allowed to change #########

            proj_X = np.array(flattened_x * self.model.std+self.model.mean)[feat_indices]

            '''
            feat_cons_A = np.identity(len(x_val))
            feat_cons_B = -x_val
    
            for i in feat_indices:
                feat_cons_A[i,i] = 0
                feat_cons_B[i] = 0
    
            ops = []
            for i in range(0,len(x_val)):
                ops.append(ExprOp.eq)
    
            feat_cons = LinearConstraintSet(feat_cons_A, feat_cons_B, ops)
            '''
            fixed_feature = []
            fixed_values = []
            for idx, (if_change, v) in enumerate(zip(features_to_change, flattened_x)):
                if if_change == 0:
                    fixed_feature.append(idx)
                    fixed_values.append(v)

            # # Stupid cleverhans puts normal code inside assert. I cannot disable assertions globaly now.
            # if not bounding_box.eval(adv_X):
            #     print("Bounding box constraint fail on "+str(adv_X))
            #     return None

            ############## End bounding box constraints ########################

            ### find seed region
            if seed_region_meth == 0:
                seed_relu_record = self.getSeedRegionAdv(x_val, y_val, targeted_label, features_to_change, X, pred, dropout, sess)
            elif seed_region_meth == 1:
                seed_relu_record = self.getSeedRegionConstr(x_val, targeted_label, bounding_box, fixed_feature, fixed_values)
            elif seed_region_meth == 2:
                bounds = []
                for v in feat_indices:
                    # Since this is after normalization, we don't need a very large number
                    if v in norm_feat_bounds:
                        bounds.append((norm_feat_bounds[v][0], norm_feat_bounds[v][1])) # lower bound, upper bound
                    else:
                        bounds.append((-10, 10))
                seed_relu_record = self.getSeedRegionGrid(x_val, targeted_label, bounds, fixed_feature, fixed_values, pred, X, dropout, sess)
            else:
                raise ValueError("Unknown adv method: "+str(seed_region_meth))
            ###

            if seed_relu_record is None:
                verbose('Fail to find an initial region!', 1)
                return None

           # Start finding continuous regions that contain desired inputs

            region_list = []
            # Find the region that adv_X falls into
            ori_region = self.model.getLinearRegionFromRecord(seed_relu_record)

            # Note class_cons is specific to a region
            # Generate n_classes - 1 linear constraints to encode the fact that the targeted
            # label has the highest probability
            # r_record = self.model.getReluActivationRecord(adv_X)
            class_cons = self.getClassConstr(seed_relu_record, targeted_label)

            # adv_same_region = ori_region.getLinearConstr().eval(ori_x_val)

            ori_region.addAddtionalConstraint(class_cons)
            # ori_region.addAddtionalConstraint(feat_cons)

            ori_region.addAddtionalConstraint(bounding_box)

            ori_region.fix_vars(fixed_feature, fixed_values)

            # We use reluRecords to mark visited regions. It is cheaper this way
            visited = set()
            visited.add(tuple(ori_region.reluRecords))

            work_list: List[LinearRegion] = [ori_region]

            region_list.append(ori_region)

            verbose('Start searching for the first region.', 1)

            if not ori_region.check():
                verbose('Fail to find the initial region.', 0)
                return None
            
            out_sym = self.model.getOutLayerSymFromInput(x_val)
            # print(np.add(np.matmul(x_val, out_sym.A),out_sym.B))
            # assert(ori_region.getLinearConstr().eval(x_val[feat_indices]) is not True)

            verbose('Found the first region!', 1)

            # tri = self.inferSimplex(feat_indices, simp_region_constr_list)

            # self.showRegionsInfo(simp_region_constr_list, tri, np.array(self.feature_list)[feat_indices], proj_X)

            isDone = False

            while len(work_list) > 0 and not isDone:
                verbose('Going into the worklist algorithm.', 1)
                cur = work_list[0]
                work_list = work_list[1:]
                rrs = cur.reluRecords # type: List[ReluRecord]
                region_time = time.time()
                # Flip one constraint at a time
                for (idx, rr) in enumerate(rrs):
                    if isDone:
                        break
                    for idx1 in range(len(rr.record)):
                        rrs1 = copy.copy(rrs) # type: List[ReluRecord]
                        nRecord = list(rrs1[idx].record)
                        nRecord[idx1] = 1 - nRecord[idx1]
                        rrs1[idx] = ReluRecord(rrs1[idx].idx, tuple(nRecord))
                        # first check if we have visited this region
                        rrs1 = tuple(rrs1)
                        if rrs1 not in visited:
                            bound_time = time.time()
                            visited.add(rrs1)
                            if cur.checkBoundary(idx, idx1):
                                newRegion = self.model.getLinearRegionFromRecord(rrs1)
                                class_cons = self.getClassConstr(rrs1, targeted_label)
                                newRegion.addAddtionalConstraint(class_cons)
                                # newRegion.addAddtionalConstraint(feat_cons)
                                newRegion.addAddtionalConstraint(bounding_box)
                                newRegion.fix_vars(fixed_feature, fixed_values)
                                if newRegion.check():
                                    cur.markInternalSurface(idx, idx1)
                                    newRegion.markInternalSurface(idx, idx1)
                                    # do not forget to add the class constraint and feature contraint
                                    # self.printRegionInfo(newRegion,x_val, feat_indices)
                                    work_list.append(newRegion)
                                    region_list.append(newRegion)
                                    # tri = self.inferSimplex(feat_indices, simp_region_constr_list)
                                    # Infer a triangle area in the current region. We can stop the iteration when found a satisfying triangle.
                                    verbose('Number of regions found: ' + str(len(region_list)), 1)
                                    if len(region_list) >= MAX_NUM_REGIONS:
                                        verbose('Find enough regions stop', 1)
                                        isDone = True
                                        break
                            else:
                                verbose('Checking failed!', 2)
                            verbose('Bound time: {}s'.format(time.time() - bound_time), 2)

                verbose('Region time: {}s.'.format(time.time() - region_time), 2)

            simp_region_constr_list = [r.getLinearConstr() for r in region_list]
            if adv_center:
                if self.adv_func is not None:
                    adv_X = self.adv_func(x_val, targeted_label, features_to_change)
                else:
                    adv_X = self.getAdv(x_val, targeted_label, features_to_change)
                adv_X = np.ndarray.flatten(adv_X)
                # adv_X = self.denormalize(adv_X)
                return self.inferSimplex(feat_indices, simp_region_constr_list, adv_X[feat_indices])
            else:
                return self.inferSimplex(feat_indices, simp_region_constr_list)
        except TimeoutError as e:
            raise e
        except Exception as e:
            print('Error!')
            traceback.print_exc()

            return None


    def simplifyRegion(self, region: LinearRegion, x_val, features):
        constr: LinearConstraintSet = region.getLinearConstr()
        # indices = list(range(len(x_val)))
        # vals = x_val * self.model.std+self.model.mean # denomoramlize vals as they are used later
        # for f in features:
        #     indices.remove(f)
        # vals = np.delete(vals, features)
        # assert(constr.solve() is not None)
        constr1 = constr.denormalize(np.array(self.model.mean)[features], np.array(self.model.std)[features])
        assert(constr1.solve() is not None)
        # constr1 = constr1.fix_and_remove_vars(indices,vals)
        # assert(constr1.solve() is not None)
        # constr1 = constr1.removeRedundant()
        # assert(constr1.solve() is not None)
        return constr1

    def showRegionsInfo(self, region_list: List[LinearConstraintSet], tri, feature_names, projected_X):
        if VERBOSE_LEVEL < 2:
            return
        # assert(len(feature_names) == 2)
        verbose('Regions found so far: ', 2)
        # fig, ax = plt.subplots()
        # plt.xlim((0,100)) # let x be dti
        # plt.ylim((0,1000)) # let y be credit score

        # plt.plot([projected_X[0]], [projected_X[1]], 'ro')

        for r in region_list:
            verbose(r.to_str(feature_names), 2)
            # vertices = r.getVertices()
            # for i in range(len(vertices)):
            #     j = (i+1)%len(vertices)
                # plt.plot([vertices[i][0], vertices[j][0]],
                #          [vertices[i][1], vertices[j][1]], 'b-')

        # for i in range(len(tri)):
        #     j = (i+1)%len(tri)
            # plt.plot([tri[i][0], tri[j][0]],
            #          [tri[i][1], tri[j][1]], 'y-')

        # plt.savefig(str(projected_X[0])+'_'+str(projected_X[1])+'.pdf')


    def inferSimplex(self, features_to_change, region_list:List[LinearConstraintSet], start=None):
        means = np.array(self.model.mean)[features_to_change]
        stds = np.array(self.model.std)[features_to_change]
        if len(features_to_change) == 1:
            ret = self.inferInterval(region_list)
            ret = np.add(np.multiply(ret, stds), means)
            return ret
        if len(features_to_change) == 2:
            vs, es = self.inferTriangle(region_list)
            vs = np.add(np.multiply(vs, stds), means)
            es = [[np.add(np.multiply(e[0], stds), means),  np.multiply(e[1], stds)]for e in es]
            return (vs,es)

        (ubs, lbs) = self.inferBox(region_list, start)
        ubs = np.add(np.multiply(ubs, stds), means)
        lbs = np.add(np.multiply(lbs, stds), means)
        return (ubs,lbs)
        # return self.sample_points(region_list)
        # return self.inferSimplexHighDim(region_list)

       # raise ValueError("Do not support high dimension: "+len(features_to_change))

    def inferInterval(self, region_list:List[LinearConstraintSet]):
        if len(region_list) > 1:
            print('Debug')
        vertices = []
        for r in region_list:
            for v in r.getVertices():
                if v not in vertices:
                    vertices.append(v)
                else:
                    vertices.remove(v)

        if len(vertices) != 2:
            print('Debug')
        assert(len(vertices) == 2)
        if vertices[0] > vertices[1]:
            return [vertices[1], vertices[0]]
        return vertices

    def getMoveLimitStep(self, edges, p0, orth_unit, div = 50):
        move_steps = []
        for e in edges:
            move_step = self.get_intersection(e, p0, orth_unit)
            if move_step is not None:
                move_steps.append(move_step)
        if len(move_steps) == 0:
            return 0

        move_step = None

        for m in move_steps:
            m_size = np.linalg.norm(m)
            if m_size > 0:
                if move_step is None:
                    move_step = m_size
                elif m_size < move_step: # in case the polygon is not convex
                    move_step = m_size
        if move_step is None:
            return 0
        return move_step / div

    def eval_tri_size(self, tri):
        max_x = 0
        max_y = 0
        for i in range(len(tri)):
            p = tri[i]
            q = tri[(i+1)%3]
            cur_x_diff = abs(p[0] - q[0])
            cur_y_diff = abs(p[1] - q[1])
            if cur_x_diff > max_x:
                max_x = cur_x_diff
            if cur_y_diff > max_y:
                max_y = cur_y_diff
        return max_x + max_y

    def is_point_inside(self, p, region_list:List[LinearConstraintSet]):
        for r in region_list:
            if r.eval(p):
                return True
        return False

    def sample_point_region_rejection(self, vertices, region_list:List[LinearConstraintSet]):
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x = np.min(xs)
        max_x = np.max(xs)

        min_y = np.min(ys)
        max_y = np.max(ys)

        while True:
            v = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if self.is_point_inside(v, region_list):
                return np.array(v)

    def sample_point_region_vertices(self, region_list:List[LinearConstraintSet]):
        selected_region = region_list[np.random.randint(0, len(region_list))]
        vertices = selected_region.getVertices()
        v = np.zeros(len(vertices[0]))
        ws = np.random.randint(1, 11, size=len(vertices))
        s = float(np.sum(ws))
        ws = ws / s
        for (v1, w1) in zip(vertices, ws):
            v += np.array(v1) * w1
        return v

    def get_bounding_box(self, region_list: List[LinearConstraintSet]):
        ubs = []
        lbs = []

        for lin_region in region_list:
            u,l = lin_region.get_bounds()
            if len(ubs) == 0:
                assert(len(lbs) == 0)
                ubs = u[:]
                lbs = l[:]
            else:
                for idx, value in enumerate(ubs):
                    if value > ubs[idx]:
                        ubs[idx] = value

                for idx, value in enumerate(lbs):
                    if value < lbs[idx]:
                        lbs[idx] = value
        return ubs, lbs

    def sample_points_region_center(self, region_list: List[LinearConstraintSet], sample_step=1, sample_num=3, clip_min=0, clip_max=255):
        ubs, lbs = self.get_bounding_box(region_list)
        center_point = []

        for u, l in zip(ubs,lbs):
            center_point.append((u+l)/2)

        sampled_points = []

        sampled_points.append(center_point)

        for i in range(len(center_point)):
            for j in [-1, 1]:
                for k in range(sample_num):
                    new_sample = copy.deepcopy(center_point)
                    new_sample[i] = new_sample[i] + (k+1)*j*sample_step
                    new_sample[i] = np.clip(new_sample[i], clip_min, clip_max)
                    sampled_points.append(new_sample)

        ret = []

        for p in sampled_points:
            for r in region_list:
                if r.eval(p):
                    ret.append(p)
                    break

        return ret

    def check_box_containtment(self, center, hw, region_list: List[LinearConstraintSet]):
        control_points = []
        for i in range(len(center)):
            pass


    '''
    Infer a triangle. Limited to 2-D case.
    '''
    def inferTriangle(self, region_list: List[LinearConstraintSet]):
        edges = [] # we use (c_x, c_y) and (a_x, a_y) to represent a line where points on the line can be defined as
                   # p = c +t*a
        verts = []
        for r in region_list:
            vertices = r.getVertices()
            if len(vertices) < 2:
                continue
            # form edges
            for i in range(len(vertices)):
                v = vertices[i]
                if v not in verts:
                    verts.append(v)
                start = vertices[i]
                end = vertices[(i+1)%len(vertices)]
                e = (start, np.round(np.subtract(end, start), decimals=utils.STITCH_TOLERANCE_DIGITS).tolist())
                if e in edges: # if it is contained, it is numerical issue
                    continue
                r_e = (end, np.round(np.subtract(start,end), decimals=utils.STITCH_TOLERANCE_DIGITS).tolist())
                if r_e not in edges:
                    edges.append(e)
                else:
                    edges.remove(r_e)

        final_ret = []

        for xmx in range(NUM_SAMPLES):
            # sample three points in the first region
            ret = []
            n = 3

            while len(ret) < n:
                ret.append(self.sample_point_region_vertices(region_list))
                if len(ret) == 3:
                    if self.test_intersection_triangle(ret[0], ret[1], ret[2], edges):
                        ret = []

            # prepare to walk
            xs = np.array(verts)[:,0]
            ys = np.array(verts)[:,1]
            scale = 50
            step_size = np.min([(np.max(xs) - np.min(xs))/scale, (np.max(ys) - np.min(ys))/scale])

            fig_sequence = 0

            while True:
                enlarged = False

                for i in range(len(ret)):
                    p0 = ret[i]
                    p1 = np.array(ret[(i+1)%len(ret)])
                    p2 = np.array(ret[(i+2)%len(ret)])
                    p1_p2 = p2 - p1
                    p1_p2_unit = p1_p2 / np.linalg.norm(p1_p2)
                    p1_m = p0 - p1
                    orth = p1_m - np.dot(p1_m, p1_p2_unit) * p1_p2_unit
                    if np.linalg.norm((orth)) == 0:
                        break
                    orth_unit = orth / np.linalg.norm(orth)
                    p2_p1_unit = - p1_p2_unit

                    last_move = None

                    while True:
                        # try to move to see if we can enlarge the area
                        step_size1 = self.getMoveLimitStep(edges, p0, orth_unit)
                        step_size1 = np.max([step_size, step_size1])
                        p0_1 = p0 + orth_unit*step_size1
                        if not self.test_intersection_triangle(p0_1, p1, p2, edges):
                            p0 = p0_1
                            last_move = orth_unit
                            enlarged = True
                            continue

                        if not np.array_equal(last_move, p2_p1_unit):
                            step_size1 = self.getMoveLimitStep(edges, p0, p1_p2_unit)
                            step_size1 = np.max([step_size1, step_size])
                            p0_1 = p0 + p1_p2_unit * step_size1
                            if not self.test_intersection_triangle(p0_1, p1, p2, edges):
                                p0 = p0_1
                                last_move = p1_p2_unit
                                continue

                        if not np.array_equal(last_move, p1_p2_unit):
                            step_size1 = self.getMoveLimitStep(edges, p0, p2_p1_unit)
                            step_size1 = np.max([step_size1, step_size])
                            p0_1 = p0 + p2_p1_unit * step_size1
                            if not self.test_intersection_triangle(p0_1, p1, p2, edges):
                                p0 = p0_1
                                last_move = p2_p1_unit
                                continue
                        break

                    if last_move is not None:
                        ret[i] = p0
                if fig_sequence >= 100000:
                    self.display_tri_search(str(fig_sequence)+'_debug', region_list, ret, ['X', 'Y'], [0,0])

                fig_sequence+=1

                if not enlarged:
                    break

            if self.eval_tri_size(ret) > self.eval_tri_size(final_ret):
                final_ret = ret

        return final_ret, edges


    def test_intersection_triangle(self, p1, p2, p3, edges):
        e1 = np.array([p1, p2-p1])
        e2 = np.array([p2, p3-p2])
        e3 = np.array([p3, p1-p3])
        return self.test_intersection(e1, edges) or self.test_intersection(e2, edges) or self.test_intersection(e3, edges)

    """
    test_drivers if edge e intersects any edge in es
    """
    def test_intersection(self, e, es):
        for e1 in es:
            if self.test_intersection_pair(e, e1):
                return True
        return False

    """
    Get the intersection of e and start+x*direction
    """
    def get_intersection(self, e, start, direction):
        rs = self.vec_prod(e[1], direction)
        if rs == 0: # parallel or colinear
            return None
        else:
            u = self.vec_prod(start - e[0], e[1]) / rs
            t = self.vec_prod(start - e[0], direction) / rs

            u = round_num(u,0)
            u = round_num(u,1)
            t = round_num(t,0)
            t = round_num(t,1)

            if t < 0 or t > 1:
                return None
            if u < 0:
                return None
            return u*direction


    """
    Implement the algorithm described at 
    https://stackoverflow.com/questions/563198/whats-the-most-efficent-way-to-calculate-where-two-line-segments-intersect
    """
    def test_intersection_pair(self, e0, e1):
        rs = self.vec_prod(e0[1], e1[1])
        if rs == 0: # parallel or colinear
            qpr = self.vec_prod(e1[0] - e0[0], e0[1])
            if qpr == 0: # colienar
                if e1[0][0] >= e0[0][0] and e1[0][0] <= e0[0][0] + e0[1][0]:
                    return True
                if e1[0][0] + e1[1][0] >= e0[0][0] and e1[0][0] + e1[1][0] <= e0[0][0] + e0[1][0]:
                    return True
                return False
            else: # parallel
                return False
        else:
            u = self.vec_prod(e1[0] - e0[0], e0[1]) / rs
            t = self.vec_prod(e1[0] - e0[0], e1[1]) / rs
            if u >=0 and u <= 1 and t >= 0 and t <= 1:
                return True
            return False

    def vec_prod(self, v1, v2):
        return v1[0]*v2[1] - v1[1] * v2[0]

    def display_tri_search(self, fig_name, region_list: List[LinearConstraintSet], tri, feature_names, projected_X):
        assert(len(feature_names) == 2)
        verbose('Regions found so far: ',2)
        fig, ax = plt.subplots()
        plt.plot([projected_X[0]], [projected_X[1]], 'ro')

        for r in region_list:
            verbose(r.to_str(feature_names), 2)
            vertices = r.getVertices()
            for i in range(len(vertices)):
                j = (i+1)%len(vertices)
                plt.plot([vertices[i][0], vertices[j][0]],
                         [vertices[i][1], vertices[j][1]], 'b-')

        for i in range(len(tri)):
            j = (i+1)%len(tri)
            plt.plot([tri[i][0], tri[j][0]],
                    [tri[i][1], tri[j][1]], 'y-')

        plt.savefig(fig_name+'.pdf')

    def test_box_intersection(self, ub1, lb1, ub2, lb2):
        assert(len(ub1) == len(ub2))
        for i in range(len(ub1)):
            if ub1[i] < lb2[i] or lb1[i] > ub2[i]:
                return False

        return True

    def test_box_intersection_batch(self, ub, lb, ub_lb_list):
        for ub1, lb1 in ub_lb_list:
            if self.test_box_intersection(ub, lb, ub1, lb1):
                return True

        return False

    def test_box_intersection_lp(self, ubs, lbs, regions: List[LinearConstraintSet]):
        for r in regions:
            if r.checkBoxSurfaceIntersection(ubs,lbs):
                return True

        return False

    def sample_point_region_box_rejection(self, r: LinearConstraintSet):
        while True:
            ubs, lbs = r.get_bounds()
            point = []
            for u,l in zip(ubs, lbs):
                point.append(random.uniform(l, u))
            if r.eval(point):
                return point

    def eval_box_volume(self, ubs, lbs):
        ret = 1.0
        for u,l in zip(ubs, lbs):
            ret+= (u - l)
        return ret

    def check_point_containtment(self, p, regions: List[LinearConstraintSet]):
        for r in regions:
            if r.eval(p):
                return True
        return False

    def inferBox(self, region_list: List[LinearConstraintSet], start = None, sample_max = 0, step_scale=500.0):
        bb_ubs, bb_lbs = self.get_bounding_box(region_list)
        step_size = []

        for ub, lb in zip(bb_ubs, bb_lbs):
            # step_size.append((ub-lb)/1000.0)
            # step_size.append((ub-lb)/step_scale)
            step_size.append(1/256.0)
        #
        num_samples = 10
        ret = None
        vol = -1

        assert(self.check_point_containtment(start, region_list))

        for i in range(num_samples):
            verbose('Trying point: '+str(i), 1)
            cur_region = region_list[random.randint(0, len(region_list) - 1)]
            if start is None:
                point = self.sample_point_region_box_rejection(cur_region)
            else:
                if ret is None:
                    point = start
                else:
                    for k in range(100): # try at most 100 times
                        point = copy.deepcopy(start)
                        for j in range(len(point)):
                            point[j] += random.randint(-sample_max, sample_max)
                        if self.check_point_containtment(point, region_list):
                            break
                        else:
                            point = None

            if point is None:
                continue

            lbs = copy.deepcopy(point)
            ubs = copy.deepcopy(point)
            assert(not self.test_box_intersection_lp(ubs, lbs, region_list))
            while True:
                if_grown = False
                #start grow in each direction:
                for j in range(len(step_size)):
                    old_val = lbs[j]
                    lbs[j] = lbs[j] - step_size[j]
                    if self.test_box_intersection_lp(ubs, lbs, region_list):
                        lbs[j] = old_val
                    else:
                        if_grown = True

                    old_val = ubs[j]
                    ubs[j] = ubs[j] + step_size[j]
                    if self.test_box_intersection_lp(ubs, lbs, region_list):
                        ubs[j] = old_val
                    else:
                        if_grown = True
                    if lbs[j] < 0 or ubs[j] > 300:
                        raise ValueError('Something is clearly wrong with inferBoxes ubs: {}, lbs: {}'.format(ubs,lbs))

                if not if_grown:
                    break

            vol1 = self.eval_box_volume(ubs, lbs)
            if ret is None or vol1 > vol:
                vol = vol1
                ret = (ubs, lbs)
            if start is not None and sample_max == 0:
                break

        return ret


    def inferSimplexHighDim(self, region_list: List[LinearConstraintSet]):
        raise ValueError("High dimension not handled yet.")
