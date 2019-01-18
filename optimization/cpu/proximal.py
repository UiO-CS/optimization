from abc import ABC, abstractmethod
import numpy as np

class ProximalOperator(ABC):

    @abstractmethod
    def __call__(self, x):
        pass


# _        _    ____ ____   ___
#| |      / \  / ___/ ___| / _ \
#| |     / _ \ \___ \___ \| | | |
#| |___ / ___ \ ___) |__) | |_| |
#|_____/_/   \_\____/____/ \___/
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



# ____  ____  ____  _   _
#| __ )|  _ \|  _ \| \ | |
#|  _ \| |_) | | | |  \| |
#| |_) |  __/| |_| | |\  |
#|____/|_|   |____/|_| \_|
class BPDNFStar(ProximalOperator):

    def __init__(self, sigma, eta, y):
        self.sigma = sigma
        self.eta = eta
        self.y = y


    def __call__(self, ksi):
        norm_expression = ksi - self.sigma*self.y
        norm_val = np.linalg.norm(norm_expression)
        compare_val = self.eta*self.sigma

        if norm_val < compare_val:
            return np.zeros_like(ksi)

        return (1 - compare_val/norm_val) *norm_expression


class BPDNG(ProximalOperator):

    def __init__(self, tau):
        self.tau = tau

    def __call__(self, z):
        return np.sign(z)*np.clip(np.abs(z) - self.tau, 0, None)

