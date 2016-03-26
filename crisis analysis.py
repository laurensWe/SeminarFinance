# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:36:01 2016

@author: ms
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
import statsmodels.formula.api as sf
import statsmodels.api as sm
import statsmodels.tsa.api as tsa

dfx = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)
dfy = pd.read_excel('leverages.xlsx',index_col=0)
dfz = pd.read_excel('stability series.xlsx',index_col=0)
dfx.index = dfy.index = dfz.index = pd.date_range(start='1969-09-30', end='30-09-2015', freq='Q')

IV = sm.OLS(dfy,dfx).fit().fittedvalues

IV.corr()
dfy.corr()

estims = {}
estims['NBER_RECESSIONS'] = {'lev':sm.Logit(dfz['NBER_RECESSIONS'], sm.add_constant(dfy)).fit(), 'iv':sm.Logit(dfz['NBER_RECESSIONS'], sm.add_constant(IV)).fit()}
estims['GDP_Growth_Rate'] = {'lev':sm.OLS(dfz['GDP_Growth_Rate'], sm.add_constant(dfy)).fit(), 'iv':sm.OLS(dfz['GDP_Growth_Rate'], sm.add_constant(IV)).fit()}

ax = pyplot.subplot()

for key in estims.keys():  
    ax = pyplot.plot(dfz[key])
    pyplot.title(key)
    for subkey in estims[key].keys():
        pyplot.plot(estims[key][subkey].fittedvalues, axes = ax,label = key)
    pyplot.legend(axes = ax)
    pyplot.show(axes = ax)
    pyplot.text