import numpy as np

def generate_weight_matrix(N, weights, dtype):
    '''Creates a weight matrix where each weight is put into a wavelet level, 
       with the first corresponding to the low res coeffs
    
    Paramters:
        N: integer, power of 2 width and height of result
        weights: iterable of weight to be put in each layer
        dtype: datatype of outputWhat si a reasonable default
    '''

    result = np.zeros((N,N), dtype=dtype)

    s = np.sum(weights)
    for sj in reversed(weights):
        result[:N,:N] = np.sqrt(s/sj)
        N = N >> 1



    return result
