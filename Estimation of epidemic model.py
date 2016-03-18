# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:09:22 2016

@author: Sebastiaan Vermeulen

"""
from epidemicModel import doEpidemicModel
import numpy as np
import pandas as pd
from multiprocessing import Pool
import time

df = pd.read_excel('crisisPerSector.xlsx',index_col=0)
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')

#%%

n=100
nprocs =1
n_iter = 5e4
x = []

def f(_):
    global df,n,n_iter,nprocs,x
    x.append(doEpidemicModel(df,n,n_iter=n_iter/nprocs,info=0))


 
if __name__ == '__main__':
    while True:
        t = time.time()
        if nprocs == 1:
            results = doEpidemicModel(df,n,n_iter=n_iter,info=0)
        else:
            p = Pool(nprocs)
            p.map(f,range(nprocs))
            p.close()
            p.join()
            results = np.sum(x,axis=0)/nprocs
        t = time.time() - t
        #pd.DataFrame(R0).to_excel('window_'+str(n)+'.xlsx')
        #np.save('mean epidemic model - window %d - iter %d - %f'%(n,n_iter,time.time()), results[0])
        #np.save('variance epidemic model - window %d - iter %d - %f'%(n,n_iter,time.time()), results[1])
        # results = [mean,variance]    
        np.save('epidemic model - window %d - iter %d - %f'%(n,n_iter,time.time()), results)
        