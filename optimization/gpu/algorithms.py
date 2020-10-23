from abc import ABC, abstractmethod
from .proximal import SQLassoProx2, WeightedL1Prox

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


class SquareRootLASSO(Algorithm):

    def __init__(self, op, prox1, prox2, measurements, tau, sigma, lam, dtype=tf.float32):
        self.cdtype = dtype_to_cdtype(dtype)
        self.dtype = dtype
        self.op = op
        self.prox1 = prox1
        self.prox2 = prox2

        self.measurements = measurements

        # Casting is necessary for types to match in the body
        self.tau = tf.cast(tau, self.cdtype)
        self.sigma = tf.cast(sigma, self.cdtype)

    def body(self, x_old, y_old):

        x_new = self.prox1(x_old - self.tau * self.op(y_old, adjoint=True))
        y_new = self.prox2(y_old + self.sigma * self.op(2*x_new - x_old) - self.sigma*self.measurements)

        return x_new, y_new

    def run(self, initial_x=None, initial_y=None):
        """Similar as FISTA.body"""
        

        # Initial values
        x = initial_x
        y = tf.zeros_like(x)

        if initial_y:
            y = initial_y

        x_result, y_result = tf.while_loop(lambda *args: True,
                                           self.body,
                                           (x, y),
                                           maximum_iterations=tf.compat.v1.placeholder(tf.int32, shape=(), name='n_iter'),
                                           )


        return x_result











