# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:09:22 2016

@author: Sebastiaan Vermeulen

"""
from epidemicModel import *
import numpy as np
import pandas as pd
from matplotlib import style,pyplot
import time
from ftplib import FTP_TLS
style.use('ggplot')

df = pd.read_excel('crisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,7)})
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')
nprocs =1
n_iter = 1e7
window_size = 100

def upload(fname):
    ftp = FTP_TLS('ftp.servage.net', '3zesp91tTNBV8', 'sbI3cEyWY6pMy8')
    print('connection open. storing')
    ftp.storbinary('STOR '+fname, open(fname, 'rb'))
    print('storing complete. closing connection')
    ftp.quit()

def doe_iets(df,window_size, period,n_iter):
    results = doEpidemicModel(df,n_iter=n_iter)
    fname = 'epidemic model - windowsize %d - period %d - iter %d - %f.npy'%(window_size,period,n_iter,time.time())
    np.save(fname, results)
    
if __name__ == '__main__':
    for i in np.arange(6,25):
        doe_iets(df.iloc[i:i+window_size+1,:],window_size,i,n_iter)

if False:			
    # doe dit of voer zelf doe_iets een keertje uit
    results = np.load('epidemic model - window 1 - iter 10000000 - 1458583174.812032.npy')
    R0(results[0,:,:])
    printout(results[1,:,:])
