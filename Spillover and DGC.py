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
#%%read data 
file1 = pd.read_excel('Equity growth rates_Fed_correct.xlsx')
file1 = file1.set_index(file1.Time)
mdata =file1[['S&LG', 'FG', 'DF', 'HH&NP', 'W', 'NFB']]
#%% moving window 
#window length
i = 50
listje = ['S&LG', 'FG', 'DF', 'HH&NP', 'W', 'NFB']
j = 0
DGC = pd.Series(np.zeros(255), index=range(0,255))
DGC = pd.DataFrame(data=DGC)
SI = pd.Series(np.zeros(255), index=range(0,255))
SI = pd.DataFrame(data=SI)
while (j < 255-i):
    
    mdata1=mdata[j:(j+i)]
    model = tsa.VAR(mdata1)
    results = model.fit(5)  
    results.summary()
# in this part we calculate the degree of granger causality
    DGC1 = 0
    N = 6.0
    for x in listje:
        for y in listje:
            if y not in x:
                if 'fail to reject' in results.test_causality(x, y, kind='f')['conclusion']:
                    DGC1=DGC1+1
                    divisor = N*(N-1)
                    DGC2 = DGC1/divisor
                    DGC.set_value(j+i, 0, DGC2)
                    
# in this part we calculate the 1step ahead spillover index
    sigma =results.resid_acov(0)
    p = np.linalg.cholesky(sigma)
#Return the Cholesky decomposition, L * L.H, of the square matrix a, where L is lower-triangular and .H is the conjugate transpose operator (which is the ordinary transpose if a is real-valued). a must be Hermitian (symmetric if real-valued) and positive-definite. Only L is actually returned.
    A = results.orth_ma_rep(10,p)
#Compute Orthogonalized MA coefficient matrices using P matrix such that \Sigma_u = PP^\prime. P defaults to the Cholesky decomposition of \Sigma_u
    A0 = np.asmatrix(np.compress([1],A, axis=0))
#take first matrix of A which is A0
    som=0
    for x in range(0, 6):
        for y in range(0,6):
            if x is not y:
                som = som + A0.item(x,y)*A0.item(x,y)
                SpillOverIndex = som/np.trace(A0*A0.getT())
                SI.set_value(j+i, 0, SpillOverIndex)
#spillover index for 1 step ahead is a0,12^2+a0,21^2/trace(A0*A0') if 2 variables
#maybe consider if we also want to look at more steps ahead?
    j=j+1
DGC.set_index(file1.Time)
DGC[i:255].set_index(file1.Time[i:255]).plot()
SI[i:255].set_index(file1.Time[i:255]).plot()
#%%
SI.to_excel('SpillOverIndex.xlsx')
DGC.to_excel('DGC.xlsx')

