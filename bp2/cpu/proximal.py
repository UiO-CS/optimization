from abc import ABC, abstractmethod
import numpy as np

class ProximalOperator(ABC):

    @abstractmethod
    def __call__(self, x):
        pass



class FISTAProximal(ProximalOperator):
    '''Assumes gradient is 2*op(op(z) - y, adjoint=True)'''

    def __init__(self, gradient, L, lam):
        super().__init__()
        self.gradient = gradient
        self.L = L
        self.lam = lam

    def base_call(self, z):
        return np.sign(z)*np.clip(np.abs(z) - self.lam/self.L, 0, None)

    def __call__(self, y):
        b = y - self.gradient(y)/self.L
        return self.base_call(b)
