import numpy as np
import numpy.linalg 
from numba import njit,vectorize, jit
import scipy





@njit
def PolynomialKernel(x,y): 
    return (x.dot(y)+c)**d

@njit
def GaussianKernel(x,y,sig2 = 1):
    return np.exp(-numpy.linalg.norm(x-y)**2/(2*sig2))

@njit
def LinearKernel(x,y): 
    return x.dot(y)



