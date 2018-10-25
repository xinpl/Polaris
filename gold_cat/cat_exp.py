from .gold_cat_gen_sketch import load_rnn_model, complete_drawing
from .cat_train import create_mlp
from .common import *
import copy
import sys
import nnExplain.genExp as exp
from nnExplain.utils import verbose
from nnExplain.utils import Timeout


def complete_until_satisfy(drawing, gen_model, gen_sess, dis_sess, preds, X, drop):
    drawing1 = convert_abs_to_rel(drawing)
    drawing2 = np.array([[x[0] / 255.0 * 10, x[1] / 255.0 * 10, x[2]] for x in drawing1])
    new_drawing,_ = complete_drawing(gen_sess, gen_model, drawing2)
    new_drawing1 = np.array([[x[0] * 255.0 / 10.0, x[1] * 255.0 / 10.0, x[2]] for x in new_drawing])
    # ret = convert_rel_to_abs(new_drawing1, start_x=drawing[0][0][0], start_y=drawing[0][1][0])

    for i in range(len(new_drawing1) - len(drawing2)):
        drawing_sub = copy.deepcopy(new_drawing1[: len(drawing2)+i+1])
        if drawing_sub[-2][2] == 1:
            continue
        drawing_sub[-1][2] = 1
        drawing_sub = convert_rel_to_abs(drawing_sub, start_x=drawing[0][0][0], start_y=drawing[0][1][0])
        drawing_sub1 = create_instance(drawing_sub)
        # draw_lines(drawing_sub1,'./cmp'+str(i))
        norm_drawing_sub = normalize(drawing_sub1)
        pred = dis_sess.run(preds, feed_dict={X:[norm_drawing_sub], drop:1.0})[0]
        # if pred[1] > pred[0]:
        #     return drawing_sub
    return drawing_sub


def main(model_path, gen_path, num_cat = 200, maximum_del = 6):
    gold = find_gold()
    sample_model, eval_model, gen_sess = load_rnn_model(gen_path)

    g = tf.Graph()

    sess = tf.Session(graph=g)

    with g.as_default():

        model_vars = create_mlp()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        saver.restore(sess, model_path)

        layer_list = model_vars['layers']

        for layer in layer_list:
            if isinstance(layer.weights, tf.Variable):
                layer.weights = sess.run(layer.weights)
            if isinstance(layer.biases, tf.Variable):
                layer.biases = sess.run(layer.biases)

        mean = [0 for i in range(max_line_num*4)]
        dev = [256 for i in range(max_line_num*4)]

        feat_names = []

        for i in range(max_line_num):
            for v in ['start', 'end']:
                for a in ['X', 'Y']:
                    feat_names.append(v+a)

        expGen = exp.ExplGenerator(exp.Model(layer_list, max_line_num*4, mean, dev), model_vars['x'], model_vars['keep'], model_vars['logits'], sess, feat_names)

        for cat_id in range(num_cat):
            verbose('{}th cat'.format(cat_id), 0)
            # Test completion
            gold_var = randomize(gold)

            x0 = create_instance(gold_var)

            print('Drawing the original image.')

            draw_lines(x0,'cat_ori'+str(cat_id))

            pred = sess.run(model_vars['logits'], feed_dict={model_vars['x']:[normalize(x0)], model_vars['keep']: 1.0})

            print(pred)


            gold_var_icmp = del_random_stroke(gold_var, max_del_len=maximum_del)

            # gold_var_icmp = del_random_stroke(gold_var)

            x1 = create_instance(gold_var_icmp)

            verbose('Original image: {}'.format(x1), 0)

            print('Drawing the incomplete image.')

            draw_lines(x1,'del_cat'+str(cat_id))

            pred1 = sess.run(model_vars['logits'], feed_dict={model_vars['x']:[normalize(x1)], model_vars['keep']: 1.0})

            print(pred1)

            comp_drawing = complete_until_satisfy(gold_var_icmp, sample_model, gen_sess, sess, model_vars['logits'], model_vars['x'], model_vars['keep'])

            x2 = create_instance(comp_drawing)

            verbose('Completed image: {}'.format(x2), 0)

            feats_to_change = np.zeros(len(x2)*4)

            change_indices = []
            change_lines = []
            change_line_indices = []

            for idx, l in enumerate(x2):
                if l not in x1:
                    change_lines.append(l)
                    for j in range(4):
                        feats_to_change[idx*4+j] = 1
                        change_indices.append(idx*4+j)
                    change_line_indices.append(idx)

            verbose('Changed indices: {}'.format(change_indices), 0)

            if len(change_indices) > 20:
                verbose('Too many dimensions',0)
                continue

            draw_lines(change_lines, 'cat_change'+str(cat_id))

            norm_x2 = normalize(x2)

            def getAdv(a,b,c):
                return norm_x2

            expGen.set_adv_func(getAdv)

            feat_bounds = {}

            for idx in change_indices:
                feat_bounds[idx] = (0,1)

            start_time = time.time()
            try:
                with Timeout(seconds=60 * 60):
                    ubs, lbs = expGen.genExplaination(norm_x2, [0,1], feats_to_change, {}, feat_bounds, 100, True)
                    verbose('ubs: '+str(ubs), 0)
                    verbose('lbs: '+str(lbs), 0)
                    print('Time spent on analyzing one instance: {}\n'.format(time.time()-start_time))

                    center = [(ub+lb)/2 for (ub, lb) in zip(ubs, lbs)]

                    x3 = copy.deepcopy(x2)
                    for idx, p in zip(change_indices, center):
                        idx1 = int(idx / 4)
                        idx2 = int(idx % 4)
                        x3[idx1][idx2] = p

                    boxes = []

                    for i in range(int(len(center) / 2)):
                        xub = ubs[i*2]
                        xlb = lbs[i*2]
                        yub = ubs[i*2+1]
                        ylb = lbs[i*2+1]
                        b = (xlb, xub, ylb, yub)
                        boxes.append(b)

                    draw_lines_and_boxes(x2, boxes, change_line_indices, 'cat_exp'+str(cat_id))
            except Exception as e:
                verbose("Unexpected error:"+str(sys.exc_info()[0]), 0)
                print(e)
                print('Error')
                print('Proceed to next!')

            sys.stdout.flush()

        sess.close()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])