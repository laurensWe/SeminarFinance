# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:22 2016

@author: Sebastiaan Vermeulen

"""
# preamble
import numpy as np
import pandas as pd
import time
import sys,os

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
         
class epidemicModel(object):
    
    def __init__(self,I):
        # initialize the class variables
        self.I = np.array(I,dtype=bool)
        # calculate the ML estimator for the probability of getting better
        self.p = np.nan_to_num( np.sum(self.I[:-1,:] & ~self.I[1:,:], axis=0) / np.sum(self.I, axis=0) ) # number of streak ends / total length of streaks
        self.T,self.N = I.shape    
        # initialize the Metropolis sampler with a random starting point and an impossibly low log-likelihood.        
        self.likelihood = -1e16
        self.current = np.random.uniform(size=(self.N,self.N)) 
        np.place( self.current, np.eye(self.N), 0)
        self.a = self.I[:-1,:]*(self.p - self.I[1:,:]) -(~self.I[:-1,:] & ~self.I[1:,:])
        self.b = ~self.I[:-1,:]*(1-self.p)
    
    def optimize(self,n_iter):
        # initialize the optimalization   
        print('Started optimization')
        # the first sample moment
        m1 = self.current.copy()
        # loop n_iter times
        for counter in range(int(n_iter)):
            # keep track of progress, but not too intensely
            if not counter % 10000:
                print('%.4e'%counter)
            # draw a new sample from the distribution
            self.draw_next_MH_sample()
            # update the moments
            m1 += self.current        
        # return the mean and variance, calculated from the moments
        return np.array([self.p*np.eye(self.N),m1/counter])
       
    def draw_next_MH_sample(self):
        # random proposal, with the diagonal elements equal to 0.
        off_diag = np.random.uniform(0,1,size=(self.N,self.N))
        np.place( off_diag, np.eye(self.N), 0 )
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
        #ll =  self.I[:-1,:]*(self.p - self.I[1:,:]) + ~self.I[:-1,:]*((1-self.p)*q-~self.I[1:,:])
        # dit wordt
        #ll =  self.I[:-1,:]*(self.p - self.I[1:,:]) -(~self.I[:-1,:] & ~self.I[1:,:]) + ~self.I[:-1,:]*(1-self.p)*q
        # dit wordt
        #self.a = self.I[:-1,:]*(self.p - self.I[1:,:]) -(~self.I[:-1,:] & ~self.I[1:,:])
        #self.b = ~self.I[:-1,:]*(1-self.p)
        ll = self.a + self.b*q
        # if a sector enters crisis while the previous period every sector was healthy, the 
        # calculated probability is 0, while in fact we whish to leave this unspecified in the model.
        # As a practical workaround, replace probability 0(happens only in the aforementioned case) with probability 1.
        np.place(ll, ll==0, 1) # remove the starters of an epidemic period, which happens with p=0 (because no one can infect you)
        # To avoid catastrophic cancellation (we're dealing with small numbers here) calculate and return the log-likelihood.
        return np.sum(np.log(np.abs( ll )))
      
# This if statement is only executed if this file is directly sourced. Basically,
# this specifies which functions to run when we want to debug this module.  
def doe_iets(df,window_size, period,n_iter):
    results = epidemicModel(df).optimize(n_iter)
    fname = 'epidemic model - windowsize %d - period %d - iter %d - %f.npy'%(window_size,period,n_iter,time.time())
    np.save(fname, results)
   
if __name__ == '__main__':
    if len(sys.argv)==3:
        start = int(sys.argv[1])
        stop = int(sys.argv[2])
    else:
        os.chdir(r'C:\Users\ms\OneDrive\Documenten\SeminarFinance\epidemic model\6-sector')
        start = 0
        stop = 5
    df = pd.read_excel('CrisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,6)})
    df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')
    nprocs =1
    n_iter = 1e7
    window_size = 256
    start_time = time.time()
    print('data read succesfull. window size {}, no. of iters {}, running windows {} through to {}.'.format(window_size,n_iter,start,stop))
    for i in np.arange(start,stop):
        print('start window {}. ETA {}.'.format(i, (stop-i)*(time.time() - start_time)/max(1,i) ))
        doe_iets(df.iloc[i:i+window_size+1,:],window_size,i,n_iter)
        