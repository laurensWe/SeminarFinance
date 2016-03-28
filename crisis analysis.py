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
from scipy.stats import chi2
from statsmodels.tsa.stattools import lagmat
from statsmodels.sandbox.tools.tools_pca import pca
import statsmodels.tsa.api as tsa
from pandas.tools.plotting import autocorrelation_plot
style.use('ggplot')

def lltest(model0,modelA):
    stat = 2*(modelA.llf - model0.llf)
    return stat, 1-chi2.cdf(stat, modelA.df_model-model0.df_model )

dfx = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)
dfy = pd.read_excel('leverages.xlsx',index_col=0)
dfz = pd.read_excel('financial stability measures continuous.xlsx',index_col=1).drop('dropme',axis=1)

_,dfy_pca,_,_ = pca(dfy,3)
f =  sm.OLS(dfy_pca,dfx).fit() 
IV = f.fittedvalues
rem = f.resid
IV.columns = list(map(lambda x: x+'_IV', IV.columns))
rem.columns = list(map(lambda x: x+'_resid', rem.columns))
sep = IV.join(rem)



estims = {}
# example categorical
#estims['NBER RECESSIONS'] = {   'direct':sm.Logit(dfz['NBER RECESSIONS'], sm.add_constant(sep)).fit(), 
#                                'IV':sm.Logit(dfz['NBER RECESSIONS'], sm.add_constant(IV)).fit()}
# example continuous  
for col in dfz.columns:
    estims[col] = {     'direct':sm.OLS(dfz[col], sm.add_constant(sep)).fit(), 
                        'IV':    sm.OLS(dfz[col], sm.add_constant(IV)).fit(),
                        'VARX-IV':  sm.OLS(dfz[col], sm.add_constant(IV.join(pd.DataFrame(lagmat(dfz[col], maxlag=2),columns=['lag_0','lag_1'], index=dfz.index)))).fit(),
                        'VARX-direct':  sm.OLS(dfz[col], sm.add_constant(sep.join(pd.DataFrame(lagmat(dfz[col], maxlag=2),columns=['lag_0','lag_1'], index=dfz.index)))).fit(),
                        }

for col in dfz.columns:
    with open(col+'.txt','w') as f:
        # significantie van parameters
        print('significance of residuals in regular OLS')
        print('likelihood ratio:')
        print(lltest(estims[col]['IV'],estims[col]['direct']), file=f)
        print('F-test:')
        print(estims[col]['direct'].f_test(np.array([[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])), file=f)
        print('significance of residuals in VAR-X')
        print('likelihood ratio:')
        print(lltest(estims[col]['VARX-IV'],estims[col]['VARX-direct']), file=f)
        print('F-test:')
        print(estims[col]['VARX-direct'].f_test(np.array([[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0]])), file=f)

        # samenvatting van regressies
        print(estims[col]['IV'].summary(), file=f)
        print(estims[col]['direct'].summary(), file=f)
        print(estims[col]['VARX-IV'].summary(), file=f)
        print(estims[col]['VARX-direct'].summary(), file=f)
        

#%% VISUALS %%
if False:
    np.round(sep.corr()*100)
    IV.corr()
    dfy.corr()
    dfx.corr()    
    
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