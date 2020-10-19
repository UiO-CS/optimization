from abc import ABC, abstractmethod
# import numpy as np
import tensorflow as tf

class ProximalOperator(ABC):

    @abstractmethod
    def __call__(self, x):
        pass



class PDProxOperator(ProximalOperator):
    """Primal-Dual prox operator. Abstract Class for the Primal-Dual algorithm"""

    @abstractmethod
    def set_parameters(self, theta, tau, sigma, eta):
        """Paramters are placeholders to the parameters"""
        pass

class FISTAProxOperator(ProximalOperator):

    @abstractmethod
    def set_parameters(self, L, lam):
        """Paramters are placeholders to the parameters"""
        pass

        

# _        _    ____ ____   ___
#| |      / \  / ___/ ___| / _ \
#| |     / _ \ \___ \___ \| | | |
#| |___ / ___ \ ___) |__) | |_| |
#|_____/_/   \_\____/____/ \___/
class FISTAProximal(ProximalOperator):
    '''Proximal operator for Fista on lasso'''

    def __init__(self, gradient):
        super().__init__()
        self.gradient = gradient
        self.L = None
        self.lam = None

    def base_call(self, z):
        return tf.sign(z)*tf.complex(tf.nn.relu(tf.abs(z) - self.lam/self.L), 0.0)

    def __call__(self, y):
        pass
        b = y - self.gradient(y)/tf.cast(self.L, tf.complex64)
        return self.base_call(b)

    def set_parameters(self, L, lam):
        self.L = L
        self.lam = lam



# ____  ____  ____  _   _
#| __ )|  _ \|  _ \| \ | |
#|  _ \| |_) | | | |  \| |
#| |_) |  __/| |_| | |\  |
#|____/|_|   |____/|_| \_|
class BPDNFStar(PDProxOperator):
    """prox_{F*}

    Where F* is the convex conjugate for when F(ksi) is the function that is 1 when
    ||ksi - y|| <= eta and 0 otherwise

    Equation can be found on p 485 of A mathematical introduction to
    compressive sensing"""

    def __init__(self, measurements, sigma, eta, dtype=tf.float32):
        # TODO: Type might be wrong
        self.sigma = sigma
        self.eta = eta
        self.y = measurements
        self.dtype = dtype;
        if dtype == tf.float32:
            self.cdtype = tf.complex64
        elif dtype == tf.float64:
            self.cdtype = tf.complex128

    def __call__(self, ksi):
        norm_expression = ksi - tf.cast(self.sigma, self.cdtype)*self.y
        norm_val = tf.cast(tf.norm(norm_expression), self.dtype)
        compare_val = self.eta*self.sigma

        result = tf.cond(
            tf.cast(norm_val, self.dtype) < compare_val,
            lambda: tf.zeros_like(ksi),
            lambda: tf.cast(tf.constant(1.0, dtype=self.dtype) - compare_val/norm_val, self.cdtype) * norm_expression
        )

        return result

    def set_parameters(self, theta, tau, sigma, eta):
        self.sigma = sigma
        self.eta = eta


class BPDNG(PDProxOperator):
    """prox_G when G(z) = ||z||_1

    Equation (15.23) in A mathematical introduction to compressive sensing"""

    def __init__(self, tau, dtype=tf.float32):
        self.tau = tau
        self.dtype = dtype;
        
    def __call__(self, z):
        return tf.sign(z)*tf.complex(tf.nn.relu(tf.abs(z) - self.tau), tf.constant(0.0, dtype=self.dtype))

    def set_parameters(self, theta, tau, sigma, eta):
        self.tau = tau



#  ____   ___  _   _   _    ____  _____ ____   ___   ___ _____ 
# / ___| / _ \| | | | / \  |  _ \| ____|  _ \ / _ \ / _ \_   _|
# \___ \| | | | | | |/ _ \ | |_) |  _| | |_) | | | | | | || |  
#  ___) | |_| | |_| / ___ \|  _ <| |___|  _ <| |_| | |_| || |  
# |____/ \__\_\\___/_/   \_\_| \_\_____|_| \_\\___/ \___/ |_|  
#                                                              
#  _        _    ____ ____   ___  
# | |      / \  / ___/ ___| / _ \ 
# | |     / _ \ \___ \___ \| | | |
# | |___ / ___ \ ___) |__) | |_| |
# |_____/_/   \_\____/____/ \___/ 
#                                 

# TODO Very rushed naming and computations
class SQLassoProx1(ProximalOperator):

    def __init__(self, dtype=tf.float32):
        self.tau = None
        self.lam = None
        self.dtype = dtype

    def __call__(self, z):
        return tf.sign(z)*tf.complex(tf.nn.relu(tf.abs(z) - self.tau*self.lam), tf.constant(0.0, dtype=self.dtype))


    def set_parameters(self, tau, lam):
        self.tau = tau
        self.lam = lam


class SQLassoProx2(ProximalOperator):


    def __init__(self, dtype=tf.float32):
        self.dtype = dtype
        if dtype == tf.float32:
            self.cdtype = tf.complex64
        elif dtype == tf.float64:
            self.cdtype = tf.complex128


    def __call__(self, z):
        return z * tf.cast(tf.minimum(tf.constant(1.0, dtype=self.dtype), tf.abs(1.0/tf.norm(z))), self.cdtype);


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


    def __call__(self, z):
        return tf.sign(z)*tf.complex(tf.nn.relu(tf.abs(z) - self.weights*self.s), tf.constant(0.0, dtype=self.dtype))



