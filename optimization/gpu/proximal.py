from abc import ABC, abstractmethod
# import numpy as np
import tensorflow as tf

class ProximalOperator(ABC):

    @abstractmethod
    def __call__(self, x):
        pass

class SQLassoProx2(ProximalOperator):

    def __init__(self, dtype=tf.float32):
        self.dtype = dtype
        if dtype == tf.float32:
            self.cdtype = tf.complex64
        elif dtype == tf.float64:
            self.cdtype = tf.complex128

    def __call__(self, z):
        tf_norm = tf.compat.v1.norm(z);
        return z * tf.cast(tf.minimum(tf.constant(1.0, dtype=self.dtype), tf.abs(1.0/tf_norm)), self.cdtype);


class WeightedL1Prox(ProximalOperator):
    '''
    Prox function for the weighted l1-norm
    f(x)_i = s * sum_i w_i |x_i|
    
    `s` is a positive real number.

    Computed from the prox function of the l1-norm, assuming that the weights w_i are *strictly* positive.
    '''

    def __init__(self, weights, s, dtype=tf.float32):
        self.weights = weights
        self.s = s
        self.dtype = dtype
        if dtype == tf.float32:
            self.cdtype = tf.complex64
        elif dtype == tf.float64:
            self.cdtype = tf.complex128


    def __call__(self, z):
        return tf.sign(z)*tf.cast(tf.nn.relu(tf.abs(z) - self.weights*self.s), dtype=self.cdtype)



