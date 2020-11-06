from abc import ABC, abstractmethod
from .proximal import FISTAProximal, SQLassoProx2, WeightedL1Prox

import tensorflow as tf

import numpy as np
import math

def dtype_to_cdtype(dtype):
    """ Convert tf.float32 or tf.float64 to tf.complex64 or tf.complex128
    """
    if dtype == tf.float32:
        cdtype = tf.complex64
    elif dtype == tf.float64:
        cdtype = tf.complex128
    return cdtype


class Algorithm(ABC):
    """Abstract class for an algorithm."""

    @abstractmethod
    def run(self):
        """Evaluate the algorithm"""
        pass

    def __call__(self, *args):
        return self.run(*args)




class SR_LASSO_Ergodic(Algorithm):

    def __init__(self, op, p_iter, tau, sigma, lam, weights_mat, dtype=tf.float32):
        self.cdtype = dtype_to_cdtype(dtype)
        self.dtype = dtype

        self.op = op
        self.prox1 = WeightedL1Prox(weights_mat, lam*tau, dtype=self.dtype)
        self.prox2 = SQLassoProx2(dtype=self.dtype)
        self.p_iter = p_iter
        self.p_iter_complex = tf.cast(p_iter, self.cdtype);

        # Casting is necessary for types to match in the body
        self.tau = tf.cast(tau, self.cdtype)
        self.sigma = tf.cast(sigma, self.cdtype)

    def body(self, measurements, x_old, y_old, x_sum):

        x_new = self.prox1(x_old - self.tau * self.op(y_old, adjoint=True))
        y_new = self.prox2(y_old + self.sigma * self.op(2*x_new - x_old) - self.sigma*measurements)
        x_sum = x_sum + x_new

        return measurements, x_new, y_new, x_sum

    def run(self, measurements,  initial_x, initial_y=None):
        """Similar as FISTA.body"""

        # Initial values
        x = initial_x
        y = tf.zeros_like(initial_x)
        x_sum = tf.zeros_like(x)

        if initial_y:
            y = initial_y

        measurements, x_result, y_result, x_ergodic = tf.while_loop(lambda *args: True,
                                           self.body,
                                           (measurements, x, y, x_sum),
                                           maximum_iterations=self.p_iter 
                                           )

        x_ergodic = x_ergodic/self.p_iter_complex;

        return x_ergodic

class SR_LASSO_exponential(Algorithm):

    def __init__(self, measurements, x0, op, p_iter, tau, sigma, lam, weights_mat, L_A, eps_0, delta, dtype=tf.float32):
        self.cdtype = dtype_to_cdtype(dtype)
        self.dtype = dtype

        self.measurements = measurements
        self.x0 = x0
        self.op = op
        self.p_iter = p_iter
        self.p_iter_float = tf.cast(p_iter, self.dtype)
        self.tau = tau
        self.sigma = sigma
        self.lam = lam
        self.weights_mat = weights_mat
        self.L_A = L_A 
        self.eps_0 = eps_0 
        self.delta = delta 
        self.inner_it = SR_LASSO_Ergodic(op, p_iter, tau, sigma, lam, weights_mat, dtype)

    def body(self, x, eps_k):

        eps_k_new = math.exp(-1)*(self.delta + eps_k) 
        alpha_k = (eps_k_new*self.p_iter_float) / (2.0*self.L_A);
        alpha_k = tf.cast(alpha_k, self.cdtype);

        x_new = alpha_k*self.inner_it(self.measurements/alpha_k, x/alpha_k)

        return x_new, eps_k_new


    def run(self, n_iter):

        # Initial values
        # get_right_dimensions = self.op.sample(self.measurements, adjoint=True)

        # x0 = tf.zeros_like(get_right_dimensions);

        x_result, eps_n = tf.while_loop(lambda *args: True,
                                           self.body,
                                           (self.x0, self.eps_0),
                                           maximum_iterations=n_iter)

        return x_result
