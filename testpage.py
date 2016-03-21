# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:02:52 2016

@author: Sebas
"""
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot,style
style.use('ggplot')

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
    
def test_data(n,t,without_systemic_risk=True):
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
    pyplot.clf()
    I = np.random.uniform(size=(256,10000,8))<np.arange(.1,.9,.1)
    i = np.sum(I[:-1,:,:] & ~I[1:,:,:], axis=0) / np.sum(I, axis=0)
    p_ = np.mean(i,axis=0)
    for idx in range(i.shape[1]):
        pyplot.hist(i[:,idx], normed=True, bins = 20, alpha=.5)
    pyplot.show()
    return p_
    
# run a test when you source this file
if __name__ == '__main__':
    L,I = test_data(5,200)
    #test_geom_estimator()
    L_ , v = doEpidemicModel(I,1e5)
    np.round(np.abs(L-L_)/np.sqrt(v),3)
    t = np.nan_to_num( -np.abs(L-L_)/np.sqrt(v) ) # t statistic of test values
    norm.cdf( t )
    precision(v,1e5)
    I.shape