# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 07:27:35 2016

@author: ms
"""
# preamble
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
from scipy.stats import norm

# using Metropolis-Hastings sampling for mean calculation
def doEpidemicModel(crisis_data, window_size, precision = 1e-3,probability=.95,starting_value=None,info=0,nprocs=4):
    """
    Compute the expectation and variance of the bayes-estimator for the model 
    specified in our paper. The posterior distribution is sampled using a 
    Metropolis MCMC that works through all windows simultaneously. Convergence
    is determined by the average sample variance of all parameters: if the
    average drops below precision, iteration is terminated.
    
    Parameters
    ----------
    crisis_data : numpy matrix, array or dataframe
        The data to optimize over, should be 0's and 1's stored in a array 
        with size ``T*N``, where ``N`` is the number of sectors and ``T`` is 
        the number of observations.
    window_size : scalar
        The size of the rolling window. Should be between ``T`` and ``N``.
    precision : scalar
        The average variance at which to stop iterating.
    probability : scalar
        The probability with which the precision is met.
    starting_value : scalar
        The starting value of the parameters. defaults to a random array.
        
    Returns
    -------
    R0 : numpy array
        the R0 of the data
    mean : numpy array
        the parameters' sample mean 
    variance : numpy array
        the parameters' sample variance. This is not a measure of the 
        parameters' significance! it just gives an impression of how precise
        the estimated mean is.
    count : scalar
        number of sampled datapoints.
    """
    
    # some checks to help with the debugging
    (T,N) = crisis_data.shape
    if (window_size < N) or (window_size > T):
        raise ValueError('The window size should be between the number of se'+\
            'ctors and the number of observations.')
    if starting_value is None:
        starting_value = np.random.uniform(size=(T-window_size,N,N))
    (t,n,m) = starting_value.shape       
    if (t!=T-window_size) or (n!=m) or (n!=N):
        raise ValueError('The supplied starting parameter should have dimens'+\
            'ions [#obs.]*[#sectors]*[#sectors].')
    if (nprocs<0) or (nprocs is not int(nprocs)):
        raise ValueError('The number of procs must be a positive integer.')
        
    # initialize the model
   # x = multiproc.parmap(run(starting_value,info,nprocs,probability,precision,window_size,crisis_data),range(nprocs),nprocs)
    r,mean,variance,count = run(starting_value,info,nprocs,probability,precision,window_size,crisis_data)
    r = pd.DataFrame(r, index=crisis_data.index[window_size:],columns = crisis_data.columns)
    return r,mean,variance,count

    
   
def run(starting_value,info,nprocs,probability,precision,window_size,I):
    # main program that tracks the different steps we wish to follow
    _T,_N = I.shape
    _I = np.array(I)        
    _shape = (_T-window_size,_N,_N)
    _precision = (2*norm.ppf((1-probability)/2)/precision)**2/nprocs
    mean, var, count = optimize(starting_value,_precision,info,_I,_shape,_N,window_size)
    r = R0(mean,info)
    return r, mean, var, count
    
def R0(L,info):
    # R0 is the sum of all infection rates
    r = (np.sum(L,axis=2)-L.diagonal(axis1=1,axis2=2))/L.diagonal(axis1=1,axis2=2)
    if info>0:
        print('done calculating R0')
    return r
    
def optimize(starting_value,precision,info,I,shape,N,window_size):
    starttime = time.time()
    max_var = 1
    counter = 1
    current = starting_value.copy()
    m1 = current.copy()
    m2 = np.power(current,2)
    while (counter < precision*max_var):
        counter += 1
        # sample a new datapoint
        metropolis_hastings_sampler(current,shape,I,N,window_size)
        # update the 1st and 2nd moment and the variance
        m1 += current
        m2 += np.power(current,2)
        if not counter%100:
            max_var = np.max(m2/counter-np.power(m1/counter,2))
        if info>0:
            print('counter: %12d, %12.6f, %12.6f, %12.2f'%(counter,precision,max_var,time.time()-starttime))
    if info>0:
        print('done with optimization')            
    return m1/counter, m2/counter-np.power(m1/counter,2), counter
           
def metropolis_hastings_sampler(current,shape,I,N,window_size):
    """
    In the current implementation the estimates will be correlated through 
    time. To decrease this correlation, the calculation of new proposals 
    and acceptance rates should be moved inside the loop. This is comput-
    ationally expensive however, and possibly the reverse of what we want.
    """
    # propose a new point based on the current point: exponential with mean [current]
    proposal = np.random.uniform(0,1,size=shape)
    # the acceptance rate of the metropolis sampler for this iteration
    acceptance_rate = np.random.uniform()
    # loop over all windows  
    for start in range(1,shape[0]):
        probability = likelihood_per_window(start,proposal[start,:,:],I,N,window_size) \
            / likelihood_per_window(start,current[start,:,:],I,N,window_size) 
        # if the proposal should be accepted, update current. else keep current.
        if probability > acceptance_rate:
            current[start,:,:] = proposal[start,:,:].copy()
        
        
    # start may run from 1 to N-window_size
def likelihood_per_window(start,L,I,N,window_size):
    # nicely format the parameters
    p_i = L.diagonal()
    p_tildes = 1-np.prod(1-L+np.eye(N)*p_i,axis=0)   
    # loop over all times in the window
    pool = Pool()
    likelihoods = pool.map(p,[ (I[start+t,i], I[start+t-1,i],  p_i[i], p_tildes[i]) for i in range(N) for t in range(window_size)])
    pool.close()
    pool.join()
    return np.prod(likelihoods)
    
def p(x):
    if x[0]:
        if x[1]:
            return 1-x[2]
        else:
            return x[2]
    else:
        if x[1]:
            return x[3]*(1-x[2])
        else:
            return 1-x[3]*(1-x[2])
          
def p_(i,i_lag,p_i,p_tilde):
    # probability according to discrete (geometric) distribution
    if i_lag:# = 1
        if i:#(i == 1):
            return 1-p_i
        else:# i = 0 
            return p_i
    else:# (i_lag == 0):
        if i:#  (i == 1):
            return p_tilde*(1-p_i)
        else:# i = 0
            return p_i*p_tilde + 1 - p_tilde

