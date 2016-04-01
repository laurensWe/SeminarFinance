# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:36:01 2016

@author: ms
"""
import itertools
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

def lltest(key,model0,modelA):
    stat = 2*(modelA.llf - model0.llf)
    return {(key,'stat'):round(stat,4), (key,'prob'):round(1-chi2.cdf(stat, modelA.df_model-model0.df_model),4)}
    
def fresults(key,f_test):
    return {(key,'stat'):round(float(f_test.fvalue),4), (key,'prob'):round(float(f_test.pvalue),4)}
    
dfx = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)[pd.Timestamp('1969-07-01 00:00:00'):pd.Timestamp('2015-07-01 00:00:00')]
dfy = pd.read_excel('leverages.xlsx',index_col=0)[pd.Timestamp('1969-07-01 00:00:00'):pd.Timestamp('2015-07-01 00:00:00')]
dfz = pd.read_excel('financial stability measures continuous.xlsx',index_col=1).drop('dropme',axis=1)[pd.Timestamp('1969-07-01 00:00:00'):pd.Timestamp('2015-07-01 00:00:00')]

_,dfy_pca,_,evecs = pca(dfy,3)
f =  sm.OLS(dfy_pca,dfx,missing='drop').fit() 
IV = f.fittedvalues
rem = f.resid
IV.columns = list(map(lambda x: x+'_IV', IV.columns))
rem.columns = list(map(lambda x: x+'_resid', rem.columns))
sep = IV.join(rem)

#%%

estims = {}
# example categorical
#estims['NBER RECESSIONS'] = {   'direct':sm.Logit(dfz['NBER RECESSIONS'], sm.add_constant(sep)).fit(), 
#                                'IV':sm.Logit(dfz['NBER RECESSIONS'], sm.add_constant(IV)).fit()}
# example continuous  
for col in dfz.columns:
    estims[col] = {     'direct':sm.OLS(dfz[col], sm.add_constant(sep), missing='drop').fit().get_robustcov_results(), 
                        'IV':    sm.OLS(dfz[col], sm.add_constant(IV), missing='drop').fit().get_robustcov_results(),
                        'rem':   sm.OLS(dfz[col], sm.add_constant(rem), missing='drop').fit().get_robustcov_results(),
                        'VARX-IV':    sm.OLS(dfz[col], sm.add_constant(IV.join(pd.DataFrame(lagmat(dfz[col], maxlag=4),columns=['lag_1','lag_2','lag_3','lag_4'], index=dfz.index))), missing='drop').fit().get_robustcov_results(), 
                        'VARX-rem':   sm.OLS(dfz[col], sm.add_constant(rem.join(pd.DataFrame(lagmat(dfz[col], maxlag=4),columns=['lag_1','lag_2','lag_3','lag_4'], index=dfz.index))), missing='drop').fit().get_robustcov_results(),
                        'VARX-direct':sm.OLS(dfz[col], sm.add_constant(sep.join(pd.DataFrame(lagmat(dfz[col], maxlag=4),columns=['lag_1','lag_2','lag_3','lag_4'], index=dfz.index))), missing='drop').fit().get_robustcov_results() }

#%% OUTPUT
def returndict(x):
    return {}
tests = dict(zip(dfz.columns, [{} for _ in dfz.columns]))
for col in dfz.columns:
    list(map(tests[col].update,[
    lltest('likelihood on lags',estims[col]['direct'],estims[col]['VARX-direct']),
    fresults('f-test on lags',estims[col]['VARX-direct'].f_test(np.array([[0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1]]))),
    lltest('likelihood on residuals in ols',estims[col]['IV'],estims[col]['direct']),
    fresults('f-test on residuals in ols',estims[col]['direct'].f_test(np.array([[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]))),
    lltest('likelihood on residuals in var',estims[col]['VARX-IV'],estims[col]['VARX-direct']),
    fresults('f-test on residuals in var',estims[col]['VARX-direct'].f_test(np.array([[0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0]]))),
    lltest('likelihood on iv in ols',estims[col]['rem'],estims[col]['direct']),
    fresults('f-test on iv in ols',estims[col]['direct'].f_test(np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0]]))),
    lltest('likelihood on iv in var',estims[col]['VARX-rem'],estims[col]['VARX-direct']),
    fresults('f-test on iv in var',estims[col]['VARX-direct'].f_test(np.array([[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0]])))]))

    if False:
        with open('results/'+col.replace('/','').replace(':',',')+'.txt','w') as f:
            # samenvatting van regressies
            print(estims[col]['IV'].summary(), file=f)
            print(estims[col]['rem'].summary(), file=f)
            print(estims[col]['direct'].summary(), file=f)
            print(estims[col]['VARX-IV'].summary(), file=f)
            print(estims[col]['VARX-rem'].summary(), file=f)
            print(estims[col]['VARX-direct'].summary(), file=f)
        
pd.DataFrame(tests).to_excel('results/significances of IV leverage interconnectedness stability.xlsx')

#%% SELECTION %%
with open('results/selected results.txt','w') as f:
    print(estims['BCI OECD']['VARX-direct'].summary(), file=f)
    print(estims['CCI OECD']['VARX-direct'].summary(), file=f)
    print(estims['Comm. real estate p (y-o-y %ch)']['VARX-IV'].summary(), file=f)
    print(estims['Debt/gdp']['VARX-IV'].summary(), file=f)
    print(estims['Federal funds effective rate']['VARX-IV'].summary(), file=f)
    print(estims['Financial assets/gdp: nonfin. corp.']['VARX-IV'].summary(), file=f)
    # print(estims['Financial assets/total financial assets: other fin. corp.']['VAR'].summary(), file=f)
    print(estims['GDP GROWTH']['IV'].summary(), file=f)
    print(estims['Inflation']['VARX-IV'].summary(), file=f)
    print(estims['KCFSI']['VARX-direct'].summary(), file=f)
    # print(estims['Return on equity: households']['VAR'].summary(), file=f)
    # print(estims['Total debt/equity: nonfin. corp.']['VAR'].summary(), file=f)
    print(estims['VIX']['VARX-IV'].summary(), file=f)


pd.DataFrame.from_dict({
    'BCI OECD':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['BCI OECD']['VARX-direct'].params[0:6].reshape(3,2,order='F') ).reshape(12,order='F'))),
    'CCI OECD':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['CCI OECD']['VARX-direct'].params[0:6].reshape(3,2,order='F')  ).reshape(12,order='F'))),
    'Comm. real estate p (y-o-y %ch)':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Comm. real estate p (y-o-y %ch)']['VARX-IV'].params[0:3] ))),
    'Debt/gdp':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Debt/gdp']['VARX-IV'].params[0:3] ))),
    'Federal funds effective rate':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Federal funds effective rate']['VARX-IV'].params[0:3] ))),
    'Financial assets/gdp: nonfin. corp.':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Financial assets/gdp: nonfin. corp.']['VARX-IV'].params[0:3] ))),
    'GDP GROWTH':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['GDP GROWTH']['IV'].params[0:3] ))),
    'Inflation':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Inflation']['VARX-IV'].params[0:3] ))),
    'KCFSI':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['KCFSI']['VARX-direct'].params[0:6].reshape(3,2,order='F') ).reshape(12,order='F'))),
    'VIX':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['VIX']['VARX-IV'].params[0:3] )))   }).to_excel('results/selected results.xlsx') 


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
    
for col in dfz.columns:
    pyplot.clf()
    ax = pyplot.subplot()
    estims[col]['direct'].resid.plot(ax=ax,label='direct')
    estims[col]['IV'].resid.plot(ax=ax,label='IV')
    estims[col]['rem'].resid.plot(ax=ax,label='rem' )
    pyplot.title(col)
    pyplot.legend()
    pyplot.savefig(col.replace('/','-').replace(':',',')+'.png')
    pyplot.clf()
    ax = pyplot.subplot()
    
    # TODO deze indexen de goeie lengte maken, oordelen op heteroscedasticiteit enz.
    ax.plot(dfz.index, estims[col]['VARX-direct'].resid, label='VARX-direct')
    ax.plot(dfz.index, estims[col]['VARX-IV'].resid, label='VARX-IV')
    ax.plot(dfz.index, estims[col]['VARX-rem'].resid, label='VARX-rem' )
    pyplot.title(col)
    pyplot.legend()
    pyplot.savefig(col.replace('/','-').replace(':',',')+'-VARX.png')
    
    
    
    
    