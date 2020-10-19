from abc import ABC, abstractmethod
from .proximal import FISTAProximal, SQLassoProx2, WeightedL1Prox

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
        # return self.op(self.op(y) - self.measurements, adjoint=True)
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

    def __init__(self, gradient):
        """
        Arguments:
            gradient: callable. Gradient of the function
            L: Lipschitz constant
            lam: Lambda parameter in LASSO.
        """
        super().__init__()
        self.L = tf.compat.v1.placeholder(tf.float32, shape=(), name='L')
        self.lam = tf.compat.v1.placeholder(tf.float32, shape=(), name='lambda')
        self.proximal = FISTAProximal(gradient)
        self.proximal.set_parameters(self.L, self.lam)

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


    def run(self, initial_x=None):
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
                                maximum_iterations=tf.compat.v1.placeholder(tf.float32,
                                                                  shape=(),
                                                                  name='n_iter'))

        # TODO: Set result
        return x

class PrimalDual(Algorithm):
    """Implementation of the Chambolle--Pock algorithm

    Optimizing F(Ax) + G(x)"""

    def __init__(self, op, prox_f_star, prox_g, theta, tau, sigma, eta, dtype):
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

        #self.theta = tf.compat.v1.placeholder(tf.float32, shape=(), name='theta')
        #self.tau = tf.compat.v1.placeholder(tf.float32, shape=(), name='tau')
        #self.sigma = tf.compat.v1.placeholder(tf.float32, shape=(), name='sigma')
        #self.eta = tf.compat.v1.placeholder(tf.float32, shape=(), name='eta')
        
        self.theta = theta;
        self.tau = tau;
        self.sigma = sigma;
        self.eta = eta;
        self.dtype = dtype;
        self.cdtype = dtype_to_cdtype(dtype);
        #self.prox_f_star.set_parameters(self.theta, self.tau, self.sigma, self.eta)
        #self.prox_g.set_parameters(self.theta, self.tau, self.sigma, self.eta)

    def body(self, x_old, x_line, ksi):
        """Similar as FISTA.body"""
        ksi = self.prox_f_star(ksi + tf.cast(self.sigma, self.cdtype)*self.op(x_line))
        x = self.prox_g(x_old - tf.cast(self.tau, self.cdtype)*self.op(ksi, adjoint=True))
        x_line = x + tf.cast(self.theta, self.cdtype)*(x - x_old)

        return x, x_line, ksi

    def run(self, n_iter, initial_x=None, initial_ksi=None):
        """Similar as FISTA.body"""

        # Initial values
        x = initial_x
        x_line = x
        ksi = tf.zeros_like(x)

        x, x_line, ksi = tf.while_loop(lambda *args: True,
                                       self.body,
                                       (x, x_line, ksi),
                                       maximum_iterations=n_iter)

        return x


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

class SR_LASSO_old(Algorithm):

    def __init__(self, measurements, op, n_iter, tau, sigma, lam, weights_mat, dtype=tf.float32):
        self.cdtype = dtype_to_cdtype(dtype)
        self.dtype = dtype
        self.op = op
        self.prox1 = WeightedL1Prox(weights_mat, lam*tau, dtype=self.dtype)
        self.prox2 = SQLassoProx2(dtype=self.dtype)
        self.n_iter = n_iter;
        self.measurements = measurements

        # Casting is necessary for types to match in the body
        self.tau = tf.cast(tau, self.cdtype)
        self.sigma = tf.cast(sigma, self.cdtype)

    def body(self, x_old, y_old):

        x_new = self.prox1(x_old - self.tau * self.op(y_old, adjoint=True))
        y_new = self.prox2(y_old + self.sigma * self.op(2*x_new - x_old) - self.sigma*self.measurements)

        return x_new, y_new

    def run(self, initial_x, initial_y=None):
        """Similar as FISTA.body"""
        

        # Initial values
        x = initial_x
        y = tf.zeros_like(x)

        if initial_y:
            y = initial_y

        x_result, y_result = tf.while_loop(lambda *args: True,
                                           self.body,
                                           (x, y),
                                           maximum_iterations=self.n_iter,
                                           )


        return x_result

class SR_LASSO_Ergodic(Algorithm):

    def __init__(self, op, p_iter, tau, sigma, lam, weights_mat, dtype=tf.float32):
        self.cdtype = dtype_to_cdtype(dtype)
        self.dtype = dtype
        
        self.op = op
        self.prox1 = WeightedL1Prox(weights_mat, lam*tau, dtype=self.dtype)
        self.prox2 = SQLassoProx2(dtype=self.dtype)
        self.p_iter = p_iter
        self.p_iter_complex = tf.cast(p_iter, self.cdtype);

        # Casting is necessary for types to match in the body
        self.tau = tf.cast(tau, self.cdtype)
        self.sigma = tf.cast(sigma, self.cdtype)

    def body(self, measurements, x_old, y_old, x_sum):

        x_new = self.prox1(x_old - self.tau * self.op(y_old, adjoint=True))
        y_new = self.prox2(y_old + self.sigma * self.op(2*x_new - x_old) - self.sigma*measurements)
        x_sum = x_sum + x_new

        return measurements, x_new, y_new, x_sum

    def run(self, measurements,  initial_x, initial_y=None):
        """Similar as FISTA.body"""
        

        # Initial values
        x = initial_x
        y = tf.zeros_like(x)
        x_sum = tf.zeros_like(x)

        if initial_y:
            y = initial_y

        measurements, x_result, y_result, x_ergodic = tf.while_loop(lambda *args: True,
                                           self.body,
                                           (measurements, x, y, x_sum),
                                           maximum_iterations=self.p_iter #tf.placeholder(tf.int32, shape=(), name='n_iter'),
                                           )

        x_ergodic = x_ergodic/self.p_iter_complex;

        return x_ergodic

class SR_LASSO_exponential(Algorithm):

    def __init__(self, measurements, op, p_iter, tau, sigma, lam, weights_mat, L_A, eps_0, delta, dtype=tf.float32):
        self.cdtype = dtype_to_cdtype(dtype)
        self.dtype = dtype

        self.measurements = measurements
        self.op = op
        self.p_iter = p_iter
        self.p_iter_float = tf.cast(p_iter, self.dtype)
        self.tau = tau
        self.sigma = sigma
        self.lam = lam
        self.weights_mat = weights_mat
        self.L_A = L_A 
        self.eps_0 = eps_0 
        self.delta = delta 
        self.inner_it = SR_LASSO_Ergodic(op, p_iter, tau, sigma, lam, weights_mat, dtype)

    def body(self, x, eps_k):

        eps_k_new = math.exp(-1)*(self.delta + eps_k) 
        alpha_k = (eps_k*self.p_iter_float) / (2*self.L_A);
        alpha_k = tf.cast(alpha_k, self.cdtype);

        # TODO check division with complex numbers in TF.
        x_new = alpha_k*self.inner_it(self.measurements/alpha_k, x/alpha_k, )

        return x_new, eps_k_new


    def run(self, n_iter):

        # Initial values
        get_right_dimensions = self.op.sample(self.measurements, adjoint=True)

        x0 = tf.zeros_like(get_right_dimensions);

        x_result, eps_n = tf.while_loop(lambda *args: True,
                                           self.body,
                                           (x0, self.eps_0),
                                           maximum_iterations=n_iter)

        return x_result
