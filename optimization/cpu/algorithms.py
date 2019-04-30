from abc import ABC, abstractmethod
from .proximal import FISTAProximal

import numpy as np
from math import sqrt

class LASSOGradient:

    def __init__(self, op, measurements):
        self.op = op
        self.measurements = measurements

    def __call__(self, y):
        return 2*self.op(self.op(y) - self.measurements, adjoint=True)


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

    def run(self, n_iter=1000, initial_x=None):
        if initial_x is None and self.result is None:
            raise ValueError('Need an initial value')
        elif self.result != None:
            initial_x = self.result

        x_old = initial_x
        y = x_old
        t_old = 1

        for i in range(1,n_iter+1):
            x = self.proximal(y)
            t = (1 + sqrt(1 + 5*t_old**2))/2.0
            y = x + (t_old - 1)/t * (x - x_old)

            x_old = x
            t_old = t

        self.result = x_old
        return x_old

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

        x_old = initial_x
        x_line = x_old
        ksi = np.zeros_like(x_old)

        for i in range(n_iter):
            ksi = self.prox_f_star(ksi + self.sigma*self.op(x_line))
            x = self.prox_g(x_old - self.tau*self.op(ksi, adjoint=True))
            x_line = x + self.theta*(x - x_old)

            x_old = x


        self.result = x
        return x


