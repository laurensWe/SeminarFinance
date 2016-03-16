# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as tsa
import seaborn as sb
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from matplotlib.mlab import PCA

#%%read data 
file1 = pd.read_excel('Equity growth rates_Fed_correct.xlsx')
file1 = file1.set_index(file1.Time)
mdata =file1[['S&LG', 'FG', 'DF', 'HH&NP', 'W', 'NFB']]
#%% initialize 
#window length
i = 100
n = 1 #number of PC's we use (for PCA I measure)
listje = ['S&LG', 'FG', 'DF', 'HH&NP', 'W', 'NFB']
pcalist = ['1', '2', '3', '4', '5', '6']
j = 0
DGC = pd.Series(np.zeros(255), index=range(0,255))
DGC = pd.DataFrame(data=DGC)
DGCi = pd.DataFrame(np.zeros((255, 6)), index=range(0,255), columns=listje)
SI = pd.Series(np.zeros(255), index=range(0,255))
SI = pd.DataFrame(data=SI)
SIi = pd.DataFrame(np.zeros((255,6)), index=range(0,255), columns=listje)
StabilityVAR = pd.Series(np.zeros(255), index=range(0,255))
StabilityVAR = pd.DataFrame(data=StabilityVAR)
PCAall = pd.DataFrame(np.zeros((255,6)), index=range(0,255), columns=pcalist)

#%% full model
mdata_full=mdata
model_full = tsa.VAR(mdata)
results_full = model_full.fit(5)  
results_full.summary()
results_full.irf(12).plot()
results_full.resid.plot()
results_full.is_stable(verbose=False)

#%% moving window estimation
while (j < 255-i):
    mdata1=mdata[j:(j+i)]
    model = tsa.VAR(mdata1)
    results = model.fit(5)  
    results.summary()
    StabilityVAR.set_value(j+i,0,results.is_stable(verbose=False))
# in this part we calculate the degree of granger causality
    DGC1 = 0
    N = 6.0
    for x in listje:
        DGCihulp=0
        for y in listje:
            if y not in x:
# you test whether x granger causes y                
                if 'fail to reject' in results.test_causality(y, x, kind='f', verbose=False)['conclusion']:
                    DGC1=DGC1+1
                    DGCihulp=DGCihulp+1
        DGCi.set_value(j+i, x, DGCihulp/5.0)          
    divisor = N*(N-1)
    DGC2 = DGC1/divisor
    DGC.set_value(j+i, 0, DGC2)
                        
# in this part we calculate the 1step ahead spillover index
    sigma =results.resid_acov(0)
    p = np.linalg.cholesky(sigma)
#Return the Cholesky decomposition, L * L.H, of the square matrix a, where L is lower-triangular and .H is the conjugate transpose operator (which is the ordinary transpose if a is real-valued). a must be Hermitian (symmetric if real-valued) and positive-definite. Only L is actually returned.
    A = results.orth_ma_rep(10,p)
#Compute Orthogonalized MA coefficient matrices using P matrix such that \Sigma_u = PP^\prime. P defaults to the Cholesky decomposition of \Sigma_u
    #A0 = np.asmatrix(np.compress([1],A, axis=0))
    A0 = np.asmatrix(np.compress([True, False, False, False, False, False, False, False, False, False], A, axis=0))
#take first matrix of A which is A0
    som=0
    for x in range(0, 6):
        somx=0
        for y in range(0,6):
            if x is not y:
                somx = somx+ A0.item(x,y)*A0.item(x,y)
                som = som + A0.item(x,y)*A0.item(x,y)
        SpilloverIndexcontr = somx/np.trace(A0*A0.getT())
        SIi.set_value(j+i, listje[x], SpilloverIndexcontr)
    SpillOverIndex = som/np.trace(A0*A0.getT())
    SI.set_value(j+i, 0, SpillOverIndex)
#spillover index for 1 step ahead is a0,12^2+a0,21^2/trace(A0*A0') if 2 variables
#maybe consider if we also want to look at more steps ahead?
    # in this part we calculate the PCA interconnectedness measure
    results = PCA(mdata1)
    PCAall.set_value(j+i, ['1','2','3','4','5','6'], results.fracs)

    
    j=j+1
DGC.set_index(file1.Time)
DGC[i:255].set_index(file1.Time[i:255]).plot()
SI[i:255].set_index(file1.Time[i:255]).plot()
DGCi[i:255].set_index(file1.Time[i:255]).plot()
SIi[i:255].set_index(file1.Time[i:255]).plot()
StabilityVAR[i:255].set_index(file1.Time[i:255]).plot()
PCAall[i:255].set_index(file1.Time[i:255]).plot()
#%%
SI.to_excel('SpillOverIndex.xlsx')
DGC.to_excel('DGC.xlsx')
DGCi.to_excel('DGCi.xlsx')
SIi.to_excel('SIi.xlsx')
StabilityVAR.to_excel('StabilityVAR.xlsx')
PCAall.to_excel('PCA interconnectedness.xlsx')
