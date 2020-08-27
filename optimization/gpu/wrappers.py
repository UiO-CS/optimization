import time

import numpy as np
import tensorflow as tf

from optimization.gpu.operators import MRIOperator
from tfwavelets.nodes import dwt2d, idwt2d
from optimization.gpu.algorithms import LASSOGradient, FISTA, PrimalDual
from optimization.gpu.proximal import BPDNFStar, BPDNG

def build_pd_graph(N, wav, levels):
    '''Returns the output node of the Primal-dual algorithm'''

    # Shapes must be set for the wavelet transform to be applicable
    tf_im = tf.compat.v1.placeholder(tf.complex64, shape=[N,N,1], name='image')
    tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

    op = MRIOperator(tf_samp_patt, wav, levels)
    measurements = op.sample(tf_im)

    prox_f_star = BPDNFStar(measurements)
    prox_g = BPDNG()
    alg = PrimalDual(op, prox_f_star, prox_g)


    initial_x = op(measurements, adjoint=True)


    result_coeffs = alg.run(initial_x)

    return result_coeffs

def build_full_pd_graph(N, wav, levels):

    result_coeffs = build_pd_graph(N, wav, levels)

    real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
    imag_idwt = idwt2d(tf.math.imag(result_coeffs), wav, levels)
    node = tf.complex(real_idwt, imag_idwt)

    return node


def run_pd(im, samp_patt, wav, levels, n_iter, eta, sigma=0.5, tau=0.5, theta=1):
    """Perform experiment"""
    N = im.shape[0]
    result_coeffs = build_pd_graph(N, wav, levels)

    real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
    imag_idwt = idwt2d(tf.imag(result_coeffs), wav, levels)
    node = tf.complex(real_idwt, imag_idwt)

    im = np.expand_dims(im, -1).astype(np.complex)
    samp_patt = np.expand_dims(samp_patt, -1).astype(np.bool)

    start = time.time()
    with tf.Session() as sess:
        result = sess.run(node, feed_dict={'image:0': im,
                                           'sampling_pattern:0': samp_patt,
                                           'sigma:0': sigma,
                                           'eta:0': eta,
                                           'tau:0': tau,
                                           'theta:0': theta,
                                           'n_iter:0': n_iter})
    end = time.time()
    print(end-start)
    return np.abs(np.squeeze(result))


def build_fista_graph(N, wav, levels):
    tf_im = tf.compat.v1.placeholder(tf.complex64, shape=[N,N,1], name='image')
    tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

    op = MRIOperator(tf_samp_patt, wav, levels)
    measurements = op.sample(tf_im)

    initial_x = op(measurements, adjoint=True)

    gradient = LASSOGradient(op, measurements)
    alg = FISTA(gradient)

    result_coeffs = alg.run(op(measurements, True))
    real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
    imag_idwt = idwt2d(tf.math.imag(result_coeffs), wav, levels)
    output_node = tf.complex(real_idwt, imag_idwt)


    return output_node


def run_fista(im, samp_patt, wav, levels, n_iter, lam, L=2):
    """Perform experiment"""
    N = im.shape[0]
    tf_result = build_fista_graph(N, wav, levels)

    im = np.expand_dims(im, -1).astype(np.complex)
    samp_patt = np.expand_dims(samp_patt, -1).astype(np.bool)

    start = time.time()
    with tf.Session() as sess:
        result = sess.run(tf_node, feed_dict={'image:0': im,
                                              'sampling_pattern:0': samp_patt,
                                              'L:0': L,
                                              'lambda:0': lam,
                                              'n_iter:0': n_iter})
    end = time.time()
    print(end-start)
    return np.abs(np.squeeze(result))

def run_session(result_node, sess, **kwargs):
    start = time.time()
    result = sess.run(result_node, 
                      feed_dict={k + ':0': v for k, v in kwargs.items()})

    running_time = time.time() - start

    return np.abs(np.squeeze(result)), running_time
