# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:22 2016

@author: Sebastiaan Vermeulen

"""
# preamble
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
import sys
from matplotlib import pyplot

# using Metropolis-Hastings sampling for mean calculation
def doEpidemicModel(crisis_data,n_iter): 
    """
    Compute the expectation and variance of the bayes-estimator for the model 
    specified in our paper. The posterior distribution is sampled using a 
    Metropolis MCMC, that simulates a fixed number of datapoints.
        
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
    mean : numpy array
        the parameters' sample mean 
    variance : numpy array
        the parameters' sample variance. This is not a measure of the 
        parameters' significance! it just gives an impression of how precise
        the estimated mean is.
    """
    return epidemicModel(crisis_data).optimize(n_iter)

def R0(L):
    """
    Use this function to get an approximate R0 from the parameter distributions.
    
    Parameters
    ----------
    L : numpy array
        Array of probabilities. the $(i,j)$-th element represents $p_ij$,
        the probability that sector i infects sector j given that sector i is 
        sick. if $i=j$, then $p_ii$ is the probability with which sector i gets
        better in one period.
        
    Returns
    -------
    r : numpy array
        A vector with the `infectiousness' $R_0$ of sector i as the i-th element.
    """
    one = np.ndim(L)-2
    two = one+1
    r = (np.sum(L,axis=two)-L.diagonal(axis1=one,axis2=two))/L.diagonal(axis1=one,axis2=two)
    return r
    
def precision(variance, n_iter, probability=.95):
    """"
    Use this function to determine what precision you get from n_iters given 
    some estimated variance. This precision is based on a normally-distributed
    confidence interval motivated by the law of large numbers.
    """
    return np.max(variance)*norm.ppf(.5+probability/2)/np.sqrt(n_iter)
    
def test_data(n,t,without_systemic_risk=True):
    """
    Use this function to generate testdata for the estimator. The data is 
    generated period by period, where a crisis is randomly started with 
    probability 1% each period for each sector.
    
    Parameters
    ----------
    n : int
        The number of sectors to include in the random data
    t : int
        The number of periods to generate
    without_systemic_risk : boolean, default=True
        If this variable is set to False, then a column of ones is included to
        allow for the estimation of systemic risk. Thus far, tests have shown 
        that estimates appear to become strongly biased if systemic risk is 
        included in this manner.
    
    Returns
    -------
    L : numpy array
        The randomly chosen parameter values. 
    I : numpy array
        The random crisis data for the n sectors.
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
    Use this function to view the spread of ML parameter estimates for the 
    geometric distribution in a simulated model. This function uses a sequencing
    trick; if you generate a sequence of bernouilli variables, then the streaks
    of successes are geometrically distributed.
    """
    pyplot.clf()
    I = np.random.uniform(size=(256,10000,8))<np.arange(.1,.9,.1)
    i = np.sum(I[:-1,:,:] & ~I[1:,:,:], axis=0) / np.sum(I, axis=0)
    p_ = np.mean(i,axis=0)
    for idx in range(i.shape[1]):
        pyplot.hist(i[:,idx], normed=True, bins = 20, alpha=.5)
    pyplot.show()
    return p_

def printout(arr,n_dec=0,newline='\n',delimiter=' & '):
    """
    This function is a convenience to view arrays and make them easier to be copy-
    pasted in Excel/Word files for reporting.
    
    Parameters
    ----------
    arr : numpy array
        The array you wish to print as text
    n_dec : scalar, defaults to 6
        the number of decimals to print
    newline : str, defaults to '\n'
        The newline character to use. Should be '\n' or '\r\n' on Windows/Linux
        and '\r' on Macs.
    delimiter : str, defaults to ' '
        The character(s) used to separate values. Use '\t' for a tab.
    """
    form = '{:.'+str(n_dec)+'f}'
    print( newline.join(map(lambda row: delimiter.join(map(form.format, row)),arr)) )

class epidemicModel(object):
    
    def __init__(self,I):
        # initialize the class variables
        self.I = np.array(I,dtype=bool)
        # calculate the ML estimator for the probability of getting better
        self.p = np.sum(self.I[:-1,:] & ~self.I[1:,:], axis=0) / np.sum(self.I, axis=0) # number of streak ends / total length of streaks
        self.T,self.N = I.shape    
        # initialize the Metropolis sampler with a random starting point and an impossibly low log-likelihood.        
        self.likelihood = -1e16
        self.current = np.random.uniform(size=(self.N,self.N))*(1-np.eye(self.N))
    
    def optimize(self,n_iter):
        # initialize the optimalization   
        print('Started optimization')
        # the first sample moment
        m1 = self.current.copy()
        # the second sample moment
        m2 = np.power(self.current,2)
        # loop n_iter times
        for counter in range(int(n_iter)):
            # keep track of progress, but not too intensely
            if not counter % 10000:
                print('%.4e'%counter)
            # draw a new sample from the distribution
            self.draw_next_MH_sample()
            # update the moments
            m1 += self.current
            # comment this in hopes of performance enhancement
            #m2 += np.power(self.current,2)           
        # return the mean and variance, calculated from the moments
        return np.array([self.p*np.eye(self.N)+m1/counter, m2/counter-np.power(m1/counter,2)])#np.array([m1/counter, m2/counter-np.power(m1/counter,2)])
           
    def draw_next_MH_sample(self):
        # random proposal, with the diagonal elements equal to 0.
        off_diag = np.random.uniform(0,1,size=(self.N,self.N))*(1-np.eye(self.N))
        # q[j] = 1 - prod(1-p_ij) for all i that are sick and therefore can infect j
        q = 1-np.exp(np.dot(self.I[:-1,:],np.log(1-off_diag)))   
        # the log-likelihood of the proposed new sample point
        proposal_likelihood =  self.log_likelihood(q)       
        # if the proposed likelihood is high enough, accept it
        if proposal_likelihood-self.likelihood > np.log(np.random.uniform()):
            self.current = off_diag
            self.likelihood = proposal_likelihood  
        
    def log_likelihood(self,q):
        """
        calculate the likelihood for each moment in time and each sector separately in an array.
        self.p[j] = p_j from the stappenplan, which is the probability of getting better.
        q[j] ] = \tilde p_j from the stappenplan, which is the probability of 
        getting sick, ergo the probability that the minimum of all the geometric
        infection processes that are active at t-1, is equal to 1.
        """
        ll =  self.I[:-1,:]*(self.p - self.I[1:,:]) + ~self.I[:-1,:]*((1-self.p)*q-~self.I[1:,:])
        # if a sector enters crisis while the previous period every sector was healthy, the 
        # calculated probability is 0, while in fact we whish to leave this unspecified in the model.
        # As a practical workaround, replace probability 0(happens only in the aforementioned case) with probability 1.
        np.place(ll, ll==0, 1) # remove the starters of an epidemic period, which happens with p=0 (because no one can infect you)
        # To avoid catastrophic cancellation (we're dealing with small numbers here) calculate and return the log-likelihood.
        return np.sum(np.log(np.abs( ll )))
      
# This if statement is only executed if this file is directly sourced. Basically,
# this specifies which functions to run when we want to debug this module.  
def doe_iets(df,window_size, period,n_iter):
    results = doEpidemicModel(df,n_iter=n_iter)
    fname = 'epidemic model - windowsize %d - period %d - iter %d - %f.npy'%(window_size,period,n_iter,time.time())
    np.save(fname, results)
   
if __name__ == '__main__':
    start = int(sys.argv[1])
    stop = int(sys.argv[2])
    df = pd.read_excel('CrisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,15)})
    df.index = pd.date_range(start='1-1-1952', end='31-12-2015', freq='Q')
    nprocs =1
    n_iter = 1e7
    window_size = 100
    start_time = time.time()
    print('data read succesfull. window size {}, no. of iters {}, running windows {} through to {}.'.format(window_size,n_iter,start,stop))
    for i in [20,40,60,80,100,125]:#np.arange(start,stop):
        print('start window {}. ETA {}.'.format(i, (stop-i)*(time.time() - start_time)/max(1,i) ))
        doe_iets(df.iloc[i:i+window_size+1,:],window_size,i,n_iter)
        
