# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:36:03 2016

@author: EvaJanssens
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as tsa
import seaborn as sb
import numpy as np
#%%read data 
numberOfLags = 2
file2 = pd.read_excel('Leverages_Fed_correct.xlsx')
file2 = file2.set_index(file2.Time)
mdata2 =file2[['NB', 'HNO', 'STG', 'FG', 'DFS']]
model2 = tsa.VAR(mdata2)
results2 = model2.fit(numberOfLags)  
model2.select_order()
results2.summary()
results2.resid.plot()
results2.resid 
listje1 = ['NB', 'HNO', 'STG', 'FG', 'DFS']
DGC1 = 0
for x in listje1:
    for y in listje1:
        if y not in x:
            if 'fail to reject' in results2.test_causality(x, y, kind='f')['conclusion']:
                DGC1=DGC1+1
print(DGC1)
results2.irf(12).plot()