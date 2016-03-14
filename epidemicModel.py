# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:22 2016

@author: Sebastiaan Vermeulen

"""
# preamble

import pandas as pd
import numpy as np

# some mock-data
I = pd.DataFrame({'x':{0:0,1:1,2:1,3:0}})

# using Metropolis-Hastings sampling for mean calculation
def doEpidemicModel(crisis_data, window_size, precision = 1e-3,starting_value=None,info=0):
    """
    Compute the expectation and variance of the bayes-estimator for the model 
    specified in our paper. The posterior distribution is sampled using a 
    Metropolis MCMC that works through all windows simultaneously. Convergence
    is determined by the average sample variance of all parameters: if the
    average drops below precision, iteration is terminated.
    
    Parameters
    ----------
    crisis_data : numpy matrix, array or dataframe
        The data to optimize over, should be 0's and 1's stored in a matrix 
        with size ``T*N``, where ``N`` is the number of sectors and ``T`` is 
        the number of observations.
    window_size : scalar
        The size of the rolling window. Should be between ``T`` and ``N``.
    precision : scalar
        The average variance at which to stop iterating.
    starting_value : scalar
        The starting value of the parameters. defaults to a matrix of ones.
        
    Returns
    -------
    mean : numpy matrix
        the parameters' sample mean 
    variance : numpy matrix
        the parameters' sample variance. This is not a measure of the 
        parameters' significance! it just gives an impression of how precise
        the estimated mean is.
    count : scalar
        number of sampled datapoints.
    """
    (T,N) = crisis_data.shape
    if (window_size < N) or (window_size > T):
        raise ValueError('The window size should be between the number of se'+\
            'ctors and the number of observations.')
    if starting_value is None:
        starting_value = np.matrix(np.ones(shape=(T-window_size,N,N)))
    (t,n,m) = starting_value.shape       
    if (t!=T) or (n!=m) or (n!=N):
        raise ValueError('The supplied starting parameter should have dimens'+\
            'ions [#obs.]*[#sectors]*[n#sectors].')
    model = epidemicModel(crisis_data,window_size,precision)
    return model.run(starting_value,info)

class epidemicModel(object):
    
    def __init__(self,I,window_size,precision):
        self.T,self.N = I.shape
        self.I = np.matrix(I)
        self.window_size = window_size
        self.precision = precision
        self.shape = (self.T-self.window_size,self.N,self.N)

    def run(self,starting_value,info):
        mean, var, count = self.optimize(starting_value,info)
        R0 = self.R0(mean)
        return R0, mean, var, count
    
    def R0(self,L):
        r = np.ndarray(self.T-self.window_size,self.N)
        i = np.ones(self.N)
        for start in range(1,self.T-self.window_size):
            lambdas = i*L[start,:,:]
            p_i = L[start,:,:].diagonal()*(lambdas - L[start,:,:].diagonal())*np.power(lambdas,-2)
            r[start,:] = p_i/(1-p_i)**2
        return r
    
    def optimize(self,starting_value,info):
        error = 1
        counter = 0
        current = starting_value.copy()
        m1 = current.copy()
        m2 = current.copy()**2
        while (error > self.precision):
            self.metropolis_hastings_sampler(current)
            m1 += current
            m2 += np.power(current,2)
            error = np.mean( (m2/counter-np.power(m1/counter,2))/counter )
            if info>0:
                print('counter: %12d, %12.6f'%(counter,error))
                if info>1:
                    print('first moment:\n',m1/counter)
                    if info>2:
                        print('second moment:\n',m2/counter)
                    
        return m1/counter, (m2/counter-np.power(m1/counter,2)), counter
           
    def metropolis_hastings_sampler(self,current):
        proposal = np.multiply(np.random.exponential(size=self.shape), current)
        acceptance_rate = np.random.uniform()
        condition_number = self.exp_joint_pdf(current,proposal)/self.exp_joint_pdf(proposal,current)
        for start in range(1,self.T-self.window_size):
            probability = self.likelihood_per_window(start,proposal[start,:,:]) \
                / self.likelihood_per_window(start,current[start,:,:]) 
            if probability*condition_number > acceptance_rate:
                current[start,:,:] = proposal[start,:,:]
        
    # start may run from 1 to N-window_size
    def likelihood_per_window(self,start,L):
        likelihood = 1
        for t in range(self.window_size):
            likelihood *= self.likelihood_per_period(start+t,L)
        return likelihood
        
    def likelihood_per_period(self,t,L):
        likelihood = 1
        lambdas =  self.I[t-1,:]*np.multiply(L,1-np.eye(self.N)) # all the lambda-tilde's
        for idx in range(self.N):
            likelihood *= self.p(self.I[t,idx], self.I[t-1,idx],  L[idx,idx], lambdas[idx])
        return likelihood
             
    def p(i,i_lag,l1,l2):
        if (i_lag == 0):
            if   (i == 1):
                return 1-np.exp(-l2)
            else:# i = 0
                return np.exp(-l2) 
        else: # i_lag = 1
            if (i == 1):
                return 1 - np.exp(-l2) + np.exp(-l1)  - l2/(l1+l2)*(1 - np.exp(-l1-l2))
            else:# i = 0 
                return np.exp(-l2)*(1-np.exp(-l1))
            
    def exp_joint_pdf(self,x,b):        
        return np.multiply(1/b,np.exp(x/b)).prod()