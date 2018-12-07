from abc import ABC, abstractmethod

import numpy as np
from .dwt import dwt2, idwt2

class LinearOperator(ABC):

    @abstractmethod
    def forward(self, x):
        """Computes Ax"""
        pass

    @abstractmethod
    def adjoint(self, x):
        """Computes A*x"""
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
    """A = PFW*"""

    def __init__(self, samp_patt, wavelet, levels, shift_samp_patt=True):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.samp_patt = samp_patt
        if shift_samp_patt:
            self.samp_patt = np.fft.fftshift(self.samp_patt)


    def forward(self, x):
        result = idwt2(x, self.wavelet, self.levels)
        result = np.fft.fft2(result, norm='ortho')
        result[~self.samp_patt] = 0
        return result

    def adjoint(self, x):
        result = np.zeros_like(x)
        result[self.samp_patt] = x[self.samp_patt]
        result = np.fft.ifft2(result, norm='ortho')
        result = dwt2(result, self.wavelet, self.levels)
        return result

    def sample(self, x):
        result = np.fft.fft2(x, norm='ortho')
        result[~self.samp_patt] = 0
        return result

