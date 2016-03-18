# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:22 2016

@author: Sebastiaan Vermeulen

"""
# preamble
import numpy as np
import pandas as pd
from scipy.stats import norm

# using Metropolis-Hastings sampling for mean calculation
def doEpidemicModel(crisis_data,window_size,n_iter,starting_value=None,info=0):
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
    # initialize the model
    return epidemicModel(crisis_data,window_size).optimize(starting_value,n_iter,info)

def R0(L):
    # R0 is the sum of all infection rates
    r = (np.sum(L,axis=2)-L.diagonal(axis1=1,axis2=2))/L.diagonal(axis1=1,axis2=2)
    return r
    
def precision(variance, n_iter, probability=.95):
    return np.max(variance)*norm.ppf(.5+probability/2)/np.sqrt(n_iter)

class epidemicModel(object):
    
    def __init__(self,I,window_size):
        # initialize the class variables
        self.T,self.N = I.shape
        self.I = np.array(I).astype(np.bool)
        self.window_size = window_size
        self.shape = (self.T-self.window_size,self.N,self.N)
        self.likelihoods = np.ones(self.shape[0])*1e-6
    
    def optimize(self,starting_value,n_iter,info):
        counter = 1
        current = starting_value.copy()
        m1 = current.copy()
        m2 = np.power(current,2)
        while (counter < n_iter):
            counter += 1
            if not counter % 100:
                print(counter)
            # sample a new datapoint
            self.metropolis_hastings_sampler(current)
            # update the 1st and 2nd moment and the variance
            m1 += current
            m2 += np.power(current,2)
            
        return np.array([m1/counter, m2/counter-np.power(m1/counter,2)])
           
    def metropolis_hastings_sampler(self,current):
        """
        In the current implementation the estimates will be correlated through 
        time. To decrease this correlation, the calculation of new proposals 
        and acceptance rates should be moved inside the loop. This is comput-
        ationally expensive however, and possibly the reverse of what we want.
        """
        # propose a new point based on the current point: exponential with mean [current]
        proposal = np.random.uniform(0,1,size=self.shape)
        # the acceptance rate of the metropolis sampler for this iteration
        acceptance_rate = np.random.uniform()
        # loop over all windows   
        p_i = proposal.diagonal(axis1=1,axis2=2)    
        p_tildes = 1-np.prod(1-proposal, axis = 1)/(1-p_i)        
        for start in range(1,self.T-self.window_size):
            a = self.likelihoods[start]
            b = self.likelihood_per_window(start,p_i[start,:],p_tildes[start,:])
                
            # if the proposal should be accepted, update current. else keep current.
            if b/a > acceptance_rate:
                current[start,:,:] = proposal[start,:,:]  
                self.likelihoods[start] = b
        
    # start may run from 1 to N-window_size
    def likelihood_per_window(self,start,p_i,p_tildes):
        likelihood  = np.prod(self.I[start-1:start+self.window_size-1,:] \
                    * np.repeat(p_i.reshape((1,p_i.size)),self.window_size,axis=0) \
                    - self.I[start:start+self.window_size,:]) \
                    * np.prod( ~self.I[start-1:start+self.window_size-1,:] \
                    * np.repeat(1-p_i.reshape((1,p_i.size)),self.window_size,axis=0) \
                    * np.repeat(p_tildes.reshape((1,p_tildes.size)),self.window_size,axis=0) \
                    -  ~self.I[start:start+self.window_size,:])
        return np.abs(likelihood)
        
        
        
        
        
        
        
        
        
        