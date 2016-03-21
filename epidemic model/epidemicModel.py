# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:22 2016

@author: Sebastiaan Vermeulen

"""
# preamble
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot,style
style.use('ggplot')

# using Metropolis-Hastings sampling for mean calculation
def doEpidemicModel(crisis_data,n_iter): 
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
    return epidemicModel(crisis_data).optimize(n_iter)

def R0(L):
    """
    Use this function to get an approximate R0 from the parameter distributions
    """
    r = (np.sum(L,axis=2)-L.diagonal(axis1=1,axis2=2))/L.diagonal(axis1=1,axis2=2)
    return r
    
def precision(variance, n_iter, probability=.95):
    """"
    Use this function to determine what precision you get from n_iters given some estimated variance.
    """
    return np.max(variance)*norm.ppf(.5+probability/2)/np.sqrt(n_iter)
    
def test_data(n,t,without_systemic_risk=True):
    """
    Use this function to generate testdata for the estimator.
    """
    L = np.random.uniform(low=.1, high=.9, size=(n,n))/2
    p = L.diagonal()
    off_diag = L-np.eye(n)*p
    I = np.zeros(shape=(t,n),dtype=bool)
    for i in range(1,t):
        I[i-1,:] = I[i-1,:] | (np.random.uniform(size = (1,n))<.01) # seed random crises
        q = 1-np.exp(np.dot(I[i-1,:],np.log(1-off_diag)))   
        I[i,:] = (I[i-1,:] | ((~I[i-1,:]) & (np.random.uniform(size=n)<q)))&(np.random.uniform(size=n)>p)
    if without_systemic_risk:     
        return L,I
    else:
        return L,np.append(I,np.ones((I.shape[0],1)), axis=1)
          
def test_geom_estimator():
    """
    Use this function to view the spread of parameter estimates in a simulated model.
    """
    pyplot.clf()
    I = np.random.uniform(size=(256,10000,8))<np.arange(.1,.9,.1)
    i = np.sum(I[:-1,:,:] & ~I[1:,:,:], axis=0) / np.sum(I, axis=0)
    p_ = np.mean(i,axis=0)
    for idx in range(i.shape[1]):
        pyplot.hist(i[:,idx], normed=True, bins = 20, alpha=.5)
    pyplot.show()
    return p_


class epidemicModel(object):
    
    def __init__(self,I):
        # initialize the class variables
        self.I = np.array(I,dtype=bool)
        self.p = np.sum(self.I[:-1,:] & ~self.I[1:,:], axis=0) / np.sum(I, axis=0) # number of streak ends / total length of streaks
        self.T,self.N = I.shape
        self.I = np.array(I,dtype=bool)
        self.likelihood = -1e16
        self.current = np.random.uniform(size=(self.N,self.N))*(1-np.eye(self.N))
    
    def optimize(self,n_iter):
        # initialize the optimalization   
        print('Started optimization')
        m1 = self.current.copy()
        m2 = np.power(self.current,2)
        # loop n_iter times
        for counter in range(int(n_iter)):
            if not counter % 10000:
                print('%.0e'%counter)
            self.draw_next_MH_sample()
            m1 += self.current
            m2 += np.power(self.current,2)           
        return np.array([self.p*np.eye(self.N)+m1/counter, m2/counter-np.power(m1/counter,2)])#np.array([m1/counter, m2/counter-np.power(m1/counter,2)])
           
    def draw_next_MH_sample(self):
        # random proposal, with the diagonal elements equal to 0.
        off_diag = np.random.uniform(0,1,size=(self.N,self.N))*(1-np.eye(self.N))
        # 1 - prod(1-p_ij) for all i that are sick and therefore can infect j
        q = 1-np.exp(np.dot(self.I[:-1,:],np.log(1-off_diag)))        
        proposal_likelihood =  self.log_likelihood(q)       
        if proposal_likelihood-self.likelihood > np.log(np.random.uniform()):
            self.current = off_diag
            self.likelihood = proposal_likelihood  
        
    def log_likelihood(self,q):
        ll =  self.I[:-1,:]*(self.p - self.I[1:,:]) + ~self.I[:-1,:]*((1-self.p)*q-~self.I[1:,:])
        np.place(ll, ll==0, 1) # remove the starters of an epidemic period, which happens with p=0 (because no one is sick)
        return np.sum(np.log(np.abs( ll )))
       
        
        
        
        
        
        
        
        
        