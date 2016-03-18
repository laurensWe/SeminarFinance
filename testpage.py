# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:02:52 2016

@author: Sebas
"""
# preamble
import numpy as np
from scipy.stats import norm
import time

def doEpidemicModel(crisis_data,n_iter):    
    return epidemicModel(crisis_data).optimize(n_iter)

def R0(L):
    r = (np.sum(L,axis=2)-L.diagonal(axis1=1,axis2=2))/L.diagonal(axis1=1,axis2=2)
    return r
    
def precision(variance, n_iter, probability=.95):
    return np.max(variance)*norm.ppf(.5+probability/2)/np.sqrt(n_iter)

class epidemicModel(object):
    
    def __init__(self,I):
        # initialize the class variables
        self.T,self.N = I.shape
        self.I = np.array(I,dtype=bool)
        self.likelihood = -1e16
        self.current = np.random.uniform(size=(self.N,self.N))  
    
    def optimize(self,n_iter):
        # initialize the optimalization       
        m1 = self.current.copy()
        m2 = np.power(self.current,2)
        # loop n_iter times
        for counter in range(int(n_iter)):
            if not counter % 10000:
                print(counter)
            self.draw_next_MH_sample()
            m1 += self.current
            m2 += np.power(self.current,2)
            
        return np.array([m1/counter, m2/counter-np.power(m1/counter,2)])
           
    def draw_next_MH_sample(self):
        # random proposal, we could choose a more dynamic distribution for efficiency
        proposal = np.random.uniform(0,1,size=(self.N,self.N))
        # last row is 'systemic risk'
        proposal[:,-1] = 0 
        # the probability of getting better
        p = proposal.diagonal()
        # all the infection rates
        off_diag = proposal - np.eye(self.N)*p
        # 1 - prod(1-p_ij) for all i that are sick and therefore can infect j
        q = 1-np.exp(np.dot(self.I[:-1,:],np.log(1-off_diag)))        
        # this biatch calculates the probability per period per node and takes the product
        proposal_likelihood =  self.log_likelihood(p,q)
        
        if proposal_likelihood-self.likelihood > np.log(np.random.uniform()):
            self.current = proposal
            self.likelihood = proposal_likelihood  
        
    def log_likelihood(self,p,q):
        return np.sum(np.log(np.abs(self.I[:-1,:]*(p - self.I[1:,:]) + ~self.I[:-1,:]*((1-p)*q-self.I[1:,:]))))
    
def test_data(n,t):
     L = np.random.uniform(size=(n,n))
     p = L.diagonal()
     off_diag = L-np.eye(n)*p
     I = np.zeros(shape=(t,n),dtype=bool)
     for i in range(1,t):
         I[i-1,:] = I[i-1,:] | (np.random.uniform(size = (1,n))<.05) # seed random crises
         q = 1-np.exp(np.dot(I[i-1,:],np.log(1-off_diag)))   
         I[i,:] = (I[i-1,:] | ((~I[i-1,:]) & (np.random.uniform(size=n)<q)))&(np.random.uniform(size=n)>p)
     return L,np.append(I,np.ones((I.shape[0],1)),axis=1)
        
        
if __name__ == '__main__':
    L,I = test_data(5,200)
    t = time.time()
    L_ , v = doEpidemicModel(I,1e6)
    t = time.time()-t
    
        
        