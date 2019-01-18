from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from tfwavelets.nodes import dwt2d, idwt2d

class LinearOperator(ABC):
    """Abstract class for Linear opeators"""

    @abstractmethod
    def forward(self, x):
        """Computes Ax"""
        pass

    @abstractmethod
    def adjoint(self, x):
        """Computes A* x"""
        pass

    def __call__(self, x, adjoint=False):
        """Convenience method"""
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)

    def sample(self, x):
        """In case one wants to sample differently than with the forward transform e.g.
        if A = PFW*, and we want to sample with PF.
        """
        return self.forward(x)



class MRIOperator(LinearOperator):
    """A = PFW* where
    P: Projection matrix for a sampling pattern
    F: Discrete Fourier transform
    W*: Inverse discrete wavelet transform"""

    def __init__(self, samp_patt, wavelet, levels, shift_samp_patt=True):
        """
        Arguments:
            samp_patt: Tensor of dtype bool. As Tensorflow does not
                       have a fftshift function, the sampling pattern
                       must look accordingly.
            wavelet: Instance of the Wavelet class
            levels:  Number of wavelet levels
            shift_samp_patt: Optional bool. Default True.
                             WARNING: Not used. Should indicate if the
                             sampling pattern is fftshifted or not, in
                             which case 
        """
        super().__init__()

        self.wavelet = wavelet
        self.levels = levels
        self.samp_patt = samp_patt


    def forward(self, x):
        """
        Arguments:
           x: Tensor
        """
        real_idwt = idwt2d(tf.real(x), self.wavelet, self.levels)
        imag_idwt = idwt2d(tf.imag(x), self.wavelet, self.levels)
        result = tf.complex(real_idwt, imag_idwt)

        # TODO: is this right?
        result = tf.transpose(result, [2,0,1])

        # TODO Is this right? Scaling to make FFT unitary
        result = tf.complex(1.0/tf.sqrt(tf.size(result, out_type=tf.float32)), 0.0) * tf.fft2d(result)
        result = tf.transpose(result, [1,2,0])

        # Subsampling
        result = tf.where(self.samp_patt, result, tf.zeros_like(result))
        return result

    def adjoint(self, x):
        """Calculate WF*P"""
        result = tf.where(self.samp_patt, x, tf.zeros_like(x))
        result = tf.transpose(result, [2,0,1]) # [channels, height, width]
        result = tf.complex(tf.sqrt(tf.size(result, out_type=tf.float32)), 0.0) * tf.ifft2d(result)
        result = tf.transpose(result, [1,2,0]) # [height, width, channels]
        real_dwt = dwt2d(tf.real(result), self.wavelet, self.levels)
        imag_dwt = dwt2d(tf.imag(result), self.wavelet, self.levels)
        result = tf.complex(real_dwt, imag_dwt)
        return result

    def sample(self, x):
        """Calculate PFx"""
        result = tf.transpose(x, [2,0,1])
        # TODO Is this right?
        result = tf.complex(1.0/tf.sqrt(tf.size(result, out_type=tf.float32)), 0.0) * tf.fft2d(result)
        result = tf.transpose(result, [1,2,0])
        result = tf.where(self.samp_patt, result, tf.zeros_like(result))
        return result
