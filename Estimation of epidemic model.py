# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:09:22 2016

@author: Sebastiaan Vermeulen

"""
from epidemicModel import doEpidemicModel
import numpy as np
import pandas as pd
import time
from ftplib import FTP_TLS

df = pd.read_excel('crisisPerSector.xlsx',index_col=0)
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')
n=100
nprocs =1
n_iter = 5e4

def upload(fname):
    ftp = FTP_TLS('ftp.servage.net', '3zesp91tTNBV8', 'sbI3cEyWY6pMy8')
    print('connection open. storing')
    ftp.storbinary('STOR '+fname, open(fname, 'rb'))
    print('storing complete. closing connection')
    ftp.quit()
 
if __name__ == '__main__':
    while True:
        results = doEpidemicModel(df,n,n_iter=n_iter,info=0)
        fname = 'epidemic model - window %d - iter %d - %f.npy'%(n,n_iter,time.time())
        np.save(fname, results)
        try:
            upload(fname)
        finally:
            continue