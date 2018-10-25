import os
from .common import *
import random

cur_dir = os.path.dirname(__file__)
model_dir = os.path.join(cur_dir, './model/gen')

###### helper functions for generative models##############
# import magenta command line tools
from IPython.display import SVG, display
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *
import svgwrite

def load_rnn_model(path):
    g = tf.Graph()
    sess = tf.Session(graph=g)
    [model_params, eval_model_params, sample_model_params] = load_model(path)
    reset_graph()

    with g.as_default():
        sample_model = Model(sample_model_params, reuse=tf.AUTO_REUSE)
        eval_model = Model(eval_model_params, reuse=tf.AUTO_REUSE)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess, path)

    return (sample_model, eval_model, sess)


def preprocess_drawing(drawing):
    ret = to_big_strokes(drawing, max_len=len(drawing))
    return ret

def normalize(drawing):
    return np.array([[x[0]/255.0*10, x[1]/255.0*10, x[2]] for x in drawing])


def complete_drawing(sess, model, drawing, seq_len=250, temperature=1.0, greedy_mode=False, z=None):
  """Samples a sequence from a pre-trained model."""

  drawing = preprocess_drawing(drawing)

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
  if z is None:
    z = np.random.randn(1, model.hps.z_size)  # not used if unconditional

  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  mixture_params = []
  greedy = False
  temp = 1.0

  for i in range(seq_len):

    if not model.hps.conditional:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state
      }
    else:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state,
          model.batch_z: z
      }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                          o_sigma1[0][idx], o_sigma2[0][idx],
                                          o_corr[0][idx], np.sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    # replace strokes[i,:] with the things we have in drawing:
    if i < len(drawing):
        strokes[i][:] = drawing[i][:]

    params = [
        o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
        o_pen[0]
    ]

    mixture_params.append(params)

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)

    # replace prev_x with the thing we have in drawing:
    if i < len(drawing):
        prev_x[0][0] = np.array(drawing[i, :], dtype=np.float32)

    prev_state = next_state

  strokes = to_normal_strokes(strokes)

  return strokes, mixture_params


def main():
    gold_drawing = find_gold()
    sample_model, eval_model, sess = load_rnn_model(model_dir)
    imgs = []
    delete_by_order = False
    for n in range(10):
        input = randomize(gold_drawing)
        in_len = len(input)
        if delete_by_order:
            for i in range(random.randint(1, in_len-1)):
                del input[len(input) - 1]
        else:
            # sample_indices = random.sample(list(range(in_len)), random.randint(1, in_len))
            input = del_random_stroke(input)
        input = convert_abs_to_rel(input)
        input = normalize(input)
        output,_ = complete_drawing(sess, sample_model, input, temperature=1)
        imgs.append([input, [0,n]])
        imgs.append([output, [1,n]])
    svg = make_grid_svg(imgs, grid_space=20)
    draw_strokes(svg, svg_filename='./sketch_cat.svg')


if __name__ == '__main__':
    main()