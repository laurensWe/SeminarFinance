# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:36:01 2016

@author: ms
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as sf
import statsmodels.api as sm
import statsmodels.tsa.api as tsa



dfx = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)
dfy = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)
dfz = pd.read_excel('stability series.xlsx',index_col=0)
dfx.index = dfy.index = dfz.index = pd.date_range(start='31-6-1969', end='30-09-2015', freq='Q')

for col in dfy.columns:
    print(sm.OLS(dfy[col],dfx).fit().summary())
    print(sm.Logit(dfy[col],dfx).fit().summary())
print(tsa.VAR(endog = dfx.merge(pd.DataFrame(dfy['GDP_Growth_Rate']),left_index=True,right_index=True)  ).fit(1).summary())
#    sf.ols(formula = '{} ~ {}'.format(col,'+'.join(dfx.columns) ), data = dfx.merge(dfy[col],left_index=True,right_index=True) ).fit().summary()
    
    
    
dfx.merge(pd.DataFrame(dfy['GDP_Growth_Rate']),left_index=True,right_index=True)