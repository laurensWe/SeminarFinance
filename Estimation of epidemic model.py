# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:09:22 2016

@author: Sebastiaan Vermeulen

"""
from epidemicModel import doEpidemicModel
import numpy as np
from matplotlib import pyplot as plt,style
import pandas as pd
from multiprocessing import Pool
import time
style.use('ggplot')

df = pd.read_excel('crisisPerSector.xlsx',index_col=0)
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')

#%%
results = [0,0]
n=100
nprocs = 4
n_iter = 10e3

def f(_):
    global results, df, n
    x = doEpidemicModel(df,n,n_iter=n_iter/nprocs,info=0)
    results = [x+y for x,y in zip(x,results)]    


 
if __name__ == '__main__':
    t = time.time()
    if nprocs == 1:
        results = doEpidemicModel(df,n,n_iter=n_iter,info=0)
    else:
        p = Pool(nprocs)
        x = p.map(f,range(nprocs))
        p.close()
        p.join()
    t = time.time() - t
    #pd.DataFrame(R0).to_excel('window_'+str(n)+'.xlsx')
    np.save('mean epidemic model - window %d - iter %d - %f'%(n,n_iter,time.time()), results[0])
    np.save('variance epidemic model - window %d - iter %d - %f'%(n,n_iter,time.time()), results[1])
    np.save('epidemic model - window %d - iter %d - %f'%(n,n_iter,time.time()), results)
# time spent @ n_iter = 10e4 for different numbers of processes
# 1 : 171
# 4 : 141
# 8 : \infty

