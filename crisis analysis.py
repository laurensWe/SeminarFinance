# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:36:01 2016

@author: ms
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot,style
import statsmodels.formula.api as sf
import statsmodels.api as sm
from statsmodels.tsa.stattools import lagmat
import statsmodels.tsa.api as tsa
from pandas.tools.plotting import autocorrelation_plot
style.use('ggplot')

dfx = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)
dfy = pd.read_excel('leverages.xlsx',index_col=0)
dfz = pd.read_excel('stability series.xlsx',index_col=0)
dfx.index = dfy.index = dfz.index = pd.date_range(start='1969-09-30', end='30-09-2015', freq='Q')

IV_model = sm.OLS(dfy,dfx).fit()
IV = IV_model.fittedvalues
IV.columns = list(map(lambda x: x+'_IV', IV.columns))
rem = IV_model.resid
rem.columns = list(map(lambda x: x+'_resid', rem.columns))
sep = IV.join(rem)

np.round(sep.corr(),decimals=1)
IV.corr()
dfy.corr()
dfx.corr()

estims = {}
estims['NBER RECESSIONS'] = {'direct estimation':sm.Logit(dfz['NBER RECESSIONS'], sm.add_constant(sep)).fit(), 'IV estimation':sm.Logit(dfz['NBER RECESSIONS'], sm.add_constant(IV)).fit()}
estims['GDP Growth Rate'] = {'direct estimation':sm.OLS(dfz['GDP Growth Rate'], sm.add_constant(sep)).fit(), 'IV estimation':sm.OLS(dfz['GDP Growth Rate'], sm.add_constant(IV)).fit()}

estims['GDP Growth Rate']['IV estimation'].resid.hist()

tsa.AR(endog=dfz['GDP Growth Rate'], exog=IV).fit(2).resid.hist()

X = IV
X = X.join(pd.DataFrame(lagmat(dfz['GDP Growth Rate'], maxlag=2),columns=['lag_0','lag_1'], index=dfz.index))


#%% VISUALS %%
if False:
    for key in estims.keys():  
        pyplot.clf()
        ax = pyplot.subplot()
        ax.plot(dfz[key])
        pyplot.title(key)
        for subkey in estims[key].keys():
            pyplot.plot(estims[key][subkey].fittedvalues, axes = ax,label = subkey)
        ax.legend(loc=3)
        pyplot.savefig('{}.png'.format(key))
        
    IV.plot(title = 'leverage ~ instruments') 
    dfy.plot(title = 'leverage' )
    rem.plot(title = 'leverage ~ instruments -> residuals')

    ax = pyplot.subplot()
    dfz['GDP Growth Rate'].plot(ax=ax)
    estims['GDP Growth Rate']['IV estimation'].fittedvalues.plot(ax=ax,label='IV fitted values')
    tsa.AR(dfz['GDP Growth Rate']).fit(2).fittedvalues.plot(ax=ax,label='AR fitted values')
    sm.OLS(dfz['GDP Growth Rate'], X).fit().fittedvalues.plot(ax=ax,label='ARX(IV) fitted values')
    ax.legend()