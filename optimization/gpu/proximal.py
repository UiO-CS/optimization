from abc import ABC, abstractmethod
# import numpy as np
import tensorflow as tf

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
    '''Assumes gradient is 2*op(op(z) - y, adjoint=True)

    Proximal operator for Fista on lasso'''

    def __init__(self, gradient, L, lam):
        super().__init__()
        self.gradient = gradient
        self.L = L
        self.lam = lam

    def base_call(self, z):
        return tf.sign(z)*tf.complex(tf.nn.relu(tf.abs(z) - self.lam/self.L), 0.0)

    def __call__(self, y):
        # TODO: Implement
        pass
        b = y - self.gradient(y)/self.L
        return self.base_call(b)



# ____  ____  ____  _   _
#| __ )|  _ \|  _ \| \ | |
#|  _ \| |_) | | | |  \| |
#| |_) |  __/| |_| | |\  |
#|____/|_|   |____/|_| \_|
class BPDNFStar(ProximalOperator):
    """prox_{F*}

    Where F* is the convex conjugate for when F(ksi) is the function that is 1 when
    ||ksi - y|| <= eta and 0 otherwise

    Equation can be found on p 485 of A mathematical introduction to
    compressive sensing"""

    def __init__(self, measurements):
        # TODO: Type might be wrong
        self.sigma = tf.placeholder(tf.float32, shape=(), name='sigma')
        self.eta = tf.placeholder(tf.float32, shape=(), name='eta')
        self.y = measurements


    def __call__(self, ksi):
        norm_expression = ksi - tf.cast(self.sigma, tf.complex64)*self.y
        norm_val = tf.cast(tf.norm(norm_expression), tf.float32)
        compare_val = self.eta*self.sigma

        result = tf.cond(
            tf.cast(norm_val, tf.float32) < compare_val,
            lambda: tf.zeros_like(ksi),
            lambda: tf.cast(1 - compare_val/norm_val, tf.complex64) * norm_expression
        )

        return result



class BPDNG(ProximalOperator):
    """prox_G when G(z) = ||z||_1

    Equation (15.23) in A mathematical introduction to compressive sensing"""

    def __init__(self, tau):
        self.tau = tau

    def __call__(self, z):
        return tf.sign(z)*tf.complex(tf.nn.relu(tf.abs(z) - self.tau), 0.0)

