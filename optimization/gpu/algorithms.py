from abc import ABC, abstractmethod
from .proximal import FISTAProximal

import tensorflow as tf

import numpy as np
from math import sqrt

class LASSOGradient:
    """
    The gradient of || Ay - b ||^2. i.e. 2A*(Ay - b)

    For the problem ||Ay - b||^2 + \lambda || y ||_1
    """

    def __init__(self, op, measurements):
        """
        Initializer

        Arguments
            op: A instance of a LinearOperator, e.g. the A matrix
            measurements: Tensor (3D?)
        """
        self.op = op
        self.measurements = measurements

    def __call__(self, y):
        """Evaluation of the gradient"""
        return tf.complex(2.0, 0.0)*self.op(self.op(y) - self.measurements, adjoint=True)


class Algorithm(ABC):
    """Abstract class for an algorithm."""

    @abstractmethod
    def run(self):
        """Evaluate the algorithm"""
        pass

    def __call__(self, *args):
        return self.run(*args)


class FISTA(Algorithm):
    """The FISTA algorithm

    For ease of use, the proximal operator is set to FISTAProximal and is thus fixed to
    F(z) = f(z) + g(z)
    where
    f(z) = ||Az - b||_2^2 and
    g(x) = lambda * ||z||_1
    """

    def __init__(self, gradient, L, lam):
        """
        Arguments:
            gradient: callable. Gradient of the function
            L: Lipschitz constant
            lam: Lambda parameter in LASSO.
        """
        super().__init__()
        self.L = L
        self.lam = lam
        self.proximal = FISTAProximal(gradient, L, lam)
        self.result = None

    def body(self, x_old, y_old, t_old):
        """
        The iteration step

        Arguments
            x_old: Tensor. x from the previous step
            y_old: Tensor. y from the previous step
            t_old: Tensor. t from the previous step

        Returns:
            Tensors x, y and t after one iteration of Fista
        """

        x = self.proximal(y_old)
        t = (1 + tf.sqrt(1  + 4 * t_old**2))/2.0
        y = x + tf.complex((t_old - 1.0)/t, 0.0) * (x - x_old)

        return x, y, t


    def run(self, n_iter=1000, initial_x=None):
        """
        Runs FISTA

        Arguments:
            n_iter: int. Number of iterations. Default 1000
            initial_x: Tensor. Initial x

        Returns:
            x after n_iter iterations of FISTA
        """
        if initial_x is None and self.result is None:
            raise ValueError('Need an initial value')
        elif self.result != None:
            initial_x = self.result

        x_old = initial_x
        y = x_old
        t_old = 1

        # Iterate until n_iter iteration is reached
        x, y, t = tf.while_loop(lambda *args: True,
                                self.body,
                                (initial_x, initial_x, 1.0),
                                maximum_iterations=n_iter)

        # TODO: Set result
        return x

class PrimalDual(Algorithm):
    """Implementation of the Chambolle--Pock algorithm

    Optimizing F(Ax) + G(x)"""

    def __init__(self, op, prox_f_star, prox_g):
        """
        Arguments
            op: LinearOperator (The matrix A)
            prox_f_star: ProximalOperator. prox_{F*} (* denoting convex conjugate).
                         Example found in proximal module
            prox_g: ProximalOperator. prox_{G}
            theta, tau, sigma, eta: parameters
        """
        self.op = op
        self.prox_f_star = prox_f_star
        self.prox_g = prox_g

        self.theta = tf.placeholder(tf.float32, shape=(), name='theta')
        self.tau = tf.placeholder(tf.float32, shape=(), name='tau')
        self.sigma = tf.placeholder(tf.float32, shape=(), name='sigma')
        self.eta = tf.placeholder(tf.float32, shape=(), name='eta')

        self.prox_f_star.set_parameters(self.theta, self.tau, self.sigma, self.eta)
        self.prox_g.set_parameters(self.theta, self.tau, self.sigma, self.eta)

    def body(self, x_old, x_line, ksi):
        """Similar as FISTA.body"""
        ksi = self.prox_f_star(ksi + tf.cast(self.sigma, tf.complex64)*self.op(x_line))
        x = self.prox_g(x_old - tf.cast(self.tau, tf.complex64)*self.op(ksi, adjoint=True))
        x_line = x + tf.cast(self.theta, tf.complex64)*(x - x_old)

        return x, x_line, ksi

    def run(self, initial_x=None, initial_ksi=None):
        """Similar as FISTA.body"""

        # Initial values
        x = initial_x
        x_line = x
        ksi = tf.zeros_like(x)

        x, x_line, ksi = tf.while_loop(lambda *args: True,
                                       self.body,
                                       (x, x_line, ksi),
                                       maximum_iterations=tf.placeholder(tf.int32, shape=(), name='n_iter'))

        return x
