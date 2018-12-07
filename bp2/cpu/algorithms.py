from abc import ABC, abstractmethod
from .proximal import FISTAProximal

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
            print(i)
            x = self.proximal(y)
            t = (1 + sqrt(1 + 5*t_old**2))/2.0
            y = x + (t_old - 1)/t * (x - x_old)

            x_old = x
            t_old = t

        self.result = x_old
        return x_old


    def __call__(self, *args):
        return self.run(*args)
