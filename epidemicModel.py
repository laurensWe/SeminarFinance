# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:22 2016

@author: Sebastiaan Vermeulen

"""

import pandas as pd
import numpy as np
from skmonaco import mcimport

I = pd.DataFrame({'x':{0:0,1:1,2:1,3:0}})
llh = lambda L: likelihood(I,L)
result,error = mcimport(llh, 100, np.random.exponential, nprocs=4 )

def likelihood(I,L):
    I = np.matrix(I)
    return likelihood_per_period(1)

def likelihood_window(I,L):
    """
    t : scalar to determine current time, 0,...,T
    I : state np.matrix of size T*N
    L : parameter np.matrix of size N*N
    """
    I = np.matrix(I)
    (T,N) = I.shape 
    if N**2!=L.size:
        raise ValueError('L moet lengte en breedte N hebben, heeft nu %d'%L.shape[0])
    likelihood = 1
    for t in range(1,T):
        likelihood *= likelihood_per_period(I,L)
    return likelihood
    
def likelihood_per_period(t,I,L):
    likelihood = 1
    (T,N) = I.shape
    if (t < 1) or (t > T):
        raise ValueError('t(%d) moet tussen 1 en T(%d) liggen.'%(t,T))
    # parameters that are of interest
    lambdas =  I[t-1,:]*L
    for idx in range(N):
        likelihood *= p(I[t,idx], I[t-1,idx],  L[idx,idx], lambdas[idx])
    return likelihood
         
def p(i,i_lag,l1,l2):
    if (i_lag == 0):
        if   (i == 1):
            return 1-np.exp(-l2)
        else:# i = 0
            return np.exp(-l2)
    else: # i_lag = 1
        if (i == 1):
            return 1 - np.exp(-l2) + np.exp(-l1)  - l1/(l1+l2)*(1 - np.exp(-l1-l2))
        else:# i = 0 
            return np.exp(-l2)*(1-np.exp(-l1))
        