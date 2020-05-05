import pywt

import numpy as np
import matplotlib.pyplot as plt
import pywt.data

def estimate_sparsity(im, wavelet, nres=None, eps=1e-1):
    '''Python implementation of csl_compute_sparsity_of_image from
    https://bitbucket.org/vegarant/cslib.git

    Parameters:
        im: 2d numpy array. Can be complex-valued
        wavelet: Name of wavelet (TODO: or wavelet object?)
        nres: Optional. Number of wavelet levels. Default given by 
              `pywt.dwtn_max_level`
        eps: Tolerance. Everything below eps in abs. value will be treated 
             as zero

    Returns:
        List of sparsity for each level, beginning in the upper left corner, i.e. 
        the low res coefficients
    '''

    im = np.abs(im)
    M = im.max()
    m = im.min()
    im = (im - m)/(M-m)
    
    # Default to max number of levels
    if nres is None:
        nres = pywt.dwtn_max_level(im.shape, wavelet)

    coeffs = pywt.wavedec2(im, wavelet, 'periodization', nres)

    sparsities = [ np.sum(np.abs(coeffs[0]) > eps) ]
    for level in coeffs[1:]:
        # TODO: Could be done more elegantly
        sparsities.append(0)
        for detail in level:
            sparsities[-1] += np.sum(np.abs(detail) > eps)

    return sparsities
