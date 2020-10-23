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

    def sample(self, x, adjoint=False):
        """In case one wants to sample differently than with the forward transform e.g.
        if A = PFW*, and we want to sample with PF.
        """
        return self.forward(x)



class MRIOperator(LinearOperator):
    """A = PFW* where
    P: Projection matrix for a sampling pattern
    F: Discrete Fourier transform
    W*: Inverse discrete wavelet transform"""

    def __init__(self, samp_patt, wavelet, levels, dtype=tf.float32, shift_samp_patt=True):
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
        self.dtype = dtype
        self.samp_patt = samp_patt


    def forward(self, x):
        """
        Arguments:
           x: Tensor
        """
        real_idwt = idwt2d(tf.math.real(x), self.wavelet, self.levels)
        imag_idwt = idwt2d(tf.math.imag(x), self.wavelet, self.levels)
        result = tf.dtypes.complex(real_idwt, imag_idwt)

        result = tf.transpose(result, [2,0,1])

        result = tf.dtypes.complex(1.0/tf.sqrt(tf.cast(tf.size(result), self.dtype)), tf.constant(0.0, dtype=self.dtype)) * tf.signal.fft2d(result)
        result = tf.transpose(result, [1,2,0])

        # Subsampling
        result = tf.compat.v1.where_v2(self.samp_patt, result, tf.zeros_like(result))
        return result

    def adjoint(self, x):
        """Calculate WF*P"""
        result = tf.compat.v1.where_v2(self.samp_patt, x, tf.zeros_like(x))
        result = tf.transpose(result, [2,0,1]) # [channels, height, width]
        result = tf.dtypes.complex(tf.sqrt(tf.cast(tf.size(result), self.dtype)), tf.constant(0.0, dtype=self.dtype)) * tf.signal.ifft2d(result)
        result = tf.transpose(result, [1,2,0]) # [height, width, channels]
        real_dwt = dwt2d(tf.math.real(result), self.wavelet, self.levels)
        imag_dwt = dwt2d(tf.math.imag(result), self.wavelet, self.levels)
        result = tf.dtypes.complex(real_dwt, imag_dwt)
        return result

    def sample(self, x, adjoint=False):
        """Calculate PFx"""
        if not adjoint:
            result = tf.compat.v1.transpose(x, perm=[2,0,1], conjugate=False)
            result = tf.dtypes.complex(1.0/tf.sqrt(tf.cast(tf.size(result), self.dtype)), tf.constant(0.0, dtype=self.dtype)) * tf.signal.fft2d(result)
            #result = tf.signal.fft2d(result)
            result = tf.compat.v1.transpose(result, perm=[1,2,0], conjugate=False)
            result = tf.compat.v1.where_v2(self.samp_patt, result, tf.zeros_like(result))
        else:
            result = tf.compat.v1.where_v2(self.samp_patt, x, tf.zeros_like(x))
            result = tf.compat.v1.transpose(result, perm=[2,0,1], conjugate=False) # [channels, height, width]
            result = tf.dtypes.complex(tf.sqrt(tf.cast(tf.size(result), self.dtype)), tf.constant(0.0, dtype=self.dtype)) * tf.signal.ifft2d(result)
            #result = tf.signal.ifft2d(result)
            result = tf.compat.v1.transpose(result, perm=[1,2,0], conjugate=False) # [height, width, channels]
        return result
