from abc import ABC, abstractmethod
from .proximal import FISTAProximal

import tensorflow as tf

import numpy as np
from math import sqrt

class LASSOGradient:

    def __init__(self, op, measurements):
        self.op = op
        self.measurements = measurements

    def __call__(self, y):
        return tf.complex(2.0, 0.0)*self.op(self.op(y) - self.measurements, adjoint=True)


class Algorithm(ABC):

    @abstractmethod
    def run(self):
        pass

    def __call__(self, *args):
        return self.run(*args)


class FISTA(Algorithm):

    def __init__(self, gradient, L, lam):
        super().__init__()
        self.L = L
        self.lam = lam
        self.proximal = FISTAProximal(gradient, L, lam)
        self.result = None

    def body(self, x_old, y_old, t_old):

        x = tf.Print(self.proximal(y_old), [], "Hello")
        t = (1 + tf.sqrt(1  + 4 * t_old**2))/2.0
        y = x + tf.complex((t_old - 1.0)/t, 0.0) * (x - x_old)

        return x, y, t


    def run(self, n_iter=1000, initial_x=None):
        if initial_x is None and self.result is None:
            raise ValueError('Need an initial value')
        elif self.result != None:
            initial_x = self.result

        x_old = initial_x
        y = x_old
        t_old = 1

        x, y, t = tf.while_loop(lambda *args: True,
                                self.body,
                                (initial_x, initial_x, 1.0),
                                maximum_iterations=n_iter)

        # TODO: Set result
        return x

class PrimalDual(Algorithm):

    def __init__(self, op, prox_f_star, prox_g, theta, tau, sigma, eta):
        self.op = op
        self.prox_f_star = prox_f_star
        self.prox_g = prox_g
        self.theta = theta
        self.tau = tau
        self.sigma = sigma
        self.eta = eta

    def run(self, n_iter, initial_x=None, initial_ksi=None):
        # TODO: Implement
        pass

        # x_old = initial_x
        # x_line = x_old
        # ksi = np.zeros_like(x_old)

        # for i in range(n_iter):
        #     print(i)
        #     ksi = self.prox_f_star(ksi + self.sigma*self.op(x_line))
        #     x = self.prox_g(x_old - self.tau*self.op(ksi, adjoint=True))
        #     x_line = x + self.theta*(x - x_old)

        #     x_old = x


        # self.result = x
        # return x


