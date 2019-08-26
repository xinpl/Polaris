import subprocess, re
import matplotlib.pyplot as plt
import numpy as np
import signal
import tensorflow as tf

#TODO clean up the code, import * has some strange features regarding sharing global variables.
# It looks like they create copies inside these files.

LINEAR_TOLERANCE = 1e-05
LINEAR_TOLERANCE_SCALE = 1e+05
STITCH_TOLERANCE_DIGITS = 2
INFINITY = 1e+7
MY_DEBUG = False
VERBOSE_LEVEL = 1
id_seed = 0

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23
def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def dictToList(dic):
    ret = []
    for k,v in dic.items():
        ret.append((k,v))
    return ret


def arrayToText(arr):
    ret = ''
    for ele in arr:
        ret += (str(ele)+'\t\t')
    return ret

def verbose(message, level):
    if VERBOSE_LEVEL >= level:
        print(message)


def round_num(n, desired):
    if desired - LINEAR_TOLERANCE < n < desired + LINEAR_TOLERANCE:
        return desired
    return n

PLOT_SAMPLES_ONLY = False

def plotLinLin(fig_name, feat_names, point, tri, ori_edges = None, sample_pos = None, sample_neg = None):
    if sample_pos is not None and len(sample_pos) + len(sample_neg) > 1e5:
        plt.figure(num=None, figsize=(20,20), dpi=200)
    plt.plot([point[0]], [point[1]], 'ro')

    if not PLOT_SAMPLES_ONLY:
        for i in range(len(tri)):
            j = (i+1)%len(tri)
            plt.plot([tri[i][0], tri[j][0]],
                    [tri[i][1], tri[j][1]], 'b-')

        if ori_edges is not None:
            for e in ori_edges:
                plt.plot([e[0][0], e[0][0]+e[1][0]],
                         [e[0][1], e[0][1]+e[1][1]], 'y--')

    if sample_pos is not None:
        xs = [ele[0] for ele in sample_pos]
        ys = [ele[1] for ele in sample_pos]
        plt.scatter(xs, ys, s=0.00001, c='g', marker=',', alpha=0.5)

    if sample_neg is not None:
        xs = [ele[0] for ele in sample_neg]
        ys = [ele[1] for ele in sample_neg]
        plt.scatter(xs, ys, s=0.00001, c='r', marker=',', alpha=0.5)

    plt.xlabel(feat_names[0])
    plt.ylabel(feat_names[1])

    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    plt.savefig(fig_name+'.pdf', bbox_inches='tight')
    plt.clf()

def plotCatLin(fig_name, feat_names, cat_names, point, inters, pos_samples = None, neg_samples = None):
    if pos_samples is not None and len(pos_samples[0])>100:
        plt.figure(num=None, figsize=(20,20), dpi=100)
    plt.plot([point[0]+1], [point[1]], 'ro')

    if not PLOT_SAMPLES_ONLY:
        for i in range(len(inters)):
            it = inters[i]
            i = i+1
            if it is not None:
                plt.plot([i, i],it, 'b-')

    if pos_samples is not None:
        for i in range(len(inters)):
            samples = pos_samples[i]
            plt.scatter([i]*len(samples), samples, s = 0.0001, c ='g', marker=',', alpha=0.2)

    if neg_samples is not None:
        for i in range(len(inters)):
            samples = neg_samples[i]
            plt.scatter([i] * len(samples), samples, s=0.0001, c='r', marker=',', alpha=0.5)

    plt.xlabel(feat_names[0])
    plt.ylabel(feat_names[1])

    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    x = np.array(list(range(len(cat_names))))
    x = x+1
    plt.xticks(x, cat_names, rotation = 'vertical')
    plt.savefig(fig_name+'.pdf', bbox_inches='tight')
    plt.clf()

def plotCatCat(fig_name, feat_names, cat_names1, cat_names2, point, points):
    plt.plot([point[0]+1], [point[1]+1], 'ro')

    for p in points:
        if p is not None:
            plt.plot([p[0]+1],[p[1]+1], 'bo')

    plt.xlabel(feat_names[0])
    plt.ylabel(feat_names[1])

    x = np.array(list(range(len(cat_names1))))
    x = x+1
    plt.xticks(x, cat_names1, rotation = 'vertical')

    y = np.array(list(range(len(cat_names2))))
    y = y+1
    plt.yticks(y, cat_names2)

    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    plt.savefig(fig_name+'.pdf', bbox_inches='tight')
    plt.clf()

def get_large_tensor_sum(sess, symbol_X, x_val, symbol_Y, y_val, symbol_dropout, drop_out_val, tensor, tensor_shape = None):
    LARGE = 100
    if len(x_val) < LARGE:
        ret = sess.run(tensor, feed_dict={symbol_X: x_val, symbol_Y: y_val, symbol_dropout:drop_out_val})
        return ret
    if tensor_shape == None:
        tensor_shape = tensor.get_shape()
    ret = np.zeros(tensor_shape)
    div = len(x_val) / LARGE

    for x,y in zip(np.array_split(x_val, div), np.array_split(y_val, div)):
        val = sess.run(tensor, feed_dict={symbol_X: x, symbol_Y: y, symbol_dropout:drop_out_val})
        ret += val
    return ret

def get_large_tensor_avg(sess, symbol_X, x_val, symbol_Y, y_val, symbol_dropout, drop_out_val, tensor):
    LARGE = 100
    if len(x_val) < LARGE:
        ret = sess.run(tensor, feed_dict={symbol_X: x_val, symbol_Y: y_val, symbol_dropout:drop_out_val})
        return ret
    ret = np.zeros(tensor.get_shape())
    div = len(x_val) / LARGE

    for x,y in zip(np.array_split(x_val, div), np.array_split(y_val, div)):
        val = sess.run(tensor, feed_dict={symbol_X: x, symbol_Y: y, symbol_dropout:drop_out_val})
        ret += val
    return ret/div

def get_large_tensor_batch(sess, symbol_X, x_val, symbol_Y, y_val, symbol_dropout, drop_out_val, tensor, batch_size = 50):
    if len(x_val) < batch_size:
        ret = sess.run(tensor, feed_dict={symbol_X: x_val, symbol_Y: y_val, symbol_dropout:drop_out_val})
        return ret
    ret = []
    div = len(x_val) / batch_size

    if symbol_Y is not None:
        for x,y in zip(np.array_split(x_val, div), np.array_split(y_val, div)):
            val = sess.run(tensor, feed_dict={symbol_X: x, symbol_Y: y, symbol_dropout:drop_out_val})
            ret.append(val)
    else:
        for x in np.array_split(x_val, div):
            val = sess.run(tensor, feed_dict={symbol_X: x, symbol_dropout:drop_out_val})
            ret.append(val)

    ret1 = []
    for r in ret:
        for r1 in r:
            ret1.append(r1)
    return ret1

def cal_f1(conf_matrix):
    precison = float(conf_matrix[1][1])/(conf_matrix[0][1] + conf_matrix[1][1])
    recall = float(conf_matrix[1][1])/(conf_matrix[1][0] + conf_matrix[1][1])
    f1 = 2/(1/precison+1/recall)
    return f1

def print_f1(conf_matrix):
    precison = float(conf_matrix[1][1])/(conf_matrix[0][1] + conf_matrix[1][1])
    recall = float(conf_matrix[1][1])/(conf_matrix[1][0] + conf_matrix[1][1])
    f1 = 2/(1/precison+1/recall)
    print('Precion: '+str(precison)+', Recall: '+str(recall)+', F1: '+str(f1))

def unique_id_gen():
    global id_seed
    id_seed+=1
    return id_seed

import signal

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)