'''Implements wrappers around pywavelets to create multi-level wavelet
transforms

NOTE: What pywavelets call cV and cH is reversed from what is typical. These functions will use
+--------+--------+
|        |        |
|   cA   |   cH   |
|        |        |
+--------+--------+
|        |        |
|   cV   |   cD   |
|        |        |
+--------+--------+

TODO: IDWT is computed by copying the entire array. It might be a good idea to
do this inplace. This can be done by simply removing the z = z.copy() lines,
but beware that this modifies the input array. A workaround might be to just
copy it once by using an inner function.

TODO: It might be a good idea to rewrite these to not be recursive

'''
import pywt
import numpy as np

def dwt2(z, wavelet, levels=1, mode='periodization'):
    if levels == 0:
        return z
    elif levels < 0:
        raise ValueError('levels must be non-negative')

    cA, (cV, cH, cD) = pywt.dwt2(z, wavelet, mode)
    return np.block([[dwt2(cA, wavelet, levels-1, mode), cH], [cV, cD]])

def idwt2(z, wavelet, levels=1, mode='periodization'):
    if levels == 0:
        return z
    elif levels < 0:
        raise ValueError('levels must be non-negative')

    n = z.shape[0]//(2**levels)
    m = 2*n
    cA = z[:n,:n]
    cH = z[:n,n:m]
    cV = z[n:m,:n]
    cD = z[n:m,n:m]

    z = z.copy()
    z[:m, :m] = pywt.idwt2((cA, (cV, cH, cD)), wavelet, mode)

    return idwt2(z, wavelet, levels-1, mode)



def dwt(z, wavelet, levels, mode='periodization'):
    if levels == 0:
        return z
    elif levels < 0:
        raise ValueError('levels must be non-negative')

    cA, cD = pywt.dwt(z, wavelet, 'periodization')

    return np.concatenate((dwt(cA, wavelet, levels-1, mode), cD))

def idwt(z, wavelet, levels, mode='periodization'):
    if levels == 0:
        return z
    elif levels < 0:
        raise ValueError('levels must be non-negative')

    n = z.shape[0]//(2**levels)
    m = 2*n
    z = z.copy()
    z[:m] = pywt.idwt(z[:n], z[n:m], wavelet, mode)

    return idwt(z, wavelet, levels-1, mode)
