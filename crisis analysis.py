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
import os
style.use('ggplot')

def lltest(key,model0,modelA):
    stat = 2*(modelA.llf - model0.llf)
    return {(key,'stat'):round(stat,4), (key,'prob'):round(1-chi2.cdf(stat, modelA.df_model-model0.df_model),4)}
    
def fresults(key,f_test):
    return {(key,'stat'):round(float(f_test.fvalue),4), (key,'prob'):round(float(f_test.pvalue),4)}

# Hier bepaal je over welk tijdsframe je gaat analyseren. Dit is gekozen obv een window van 100.
start,stop = [pd.Timestamp('1969-07-01 00:00:00'),pd.Timestamp('2015-07-01 00:00:00')]

# lees de variabelen waarop we gaan regresseren    
dfx = pd.read_excel('Interconnectednessmeasures.xlsx',index_col=0)[start:stop]
dfy = pd.read_excel('leverages.xlsx',index_col=0)[start:stop]
dfz = pd.read_excel('financial stability measures continuous.xlsx',index_col=1).drop('dropme',axis=1)[start:stop]

# Bereken de principal components, 3 stuks. gebruik evals om te beoordelen hoe veel componenten je wilt.
npc = 3
_,dfy_pca,evals,evecs = pca(np.log(dfy),npc)

# aantal lags wat we straks in de VAR modellen gaan gebruiken
nlags = 4

# Voer OLS uit met alle beschikbare interconnectedness measures op de gekozen principal components.
f =  sm.OLS(dfy_pca,dfx,missing='drop').fit() 

# Sla de gefitte waarden en de residuals in aparte dataframes op. geef ze als colomnamen makkelijk te herkenbare namen. Stop ze ook weer samen voor makkelijke tweede stap regressies
IV = f.fittedvalues
rem = f.resid
IV.columns = list(map(lambda x: x+'_IV', IV.columns))
rem.columns = list(map(lambda x: x+'_resid', rem.columns))
sep = IV.join(rem)

#%% Hier maken we een plot van de gekozen PC's met gefitte waarden en residuals

# zet een plot op van een handig formaat
fig = pyplot.figure(figsize=(16,4))

# itereer over het aantal gekozen PC's
for i in range(npc):
    
    # initialiseer de i-de subplot in een matrix van 1 x npc subplots
    ax = fig.add_subplot(1,npc,i+1)
    
    # plot de lijntjes in het subplot, stel kleur en label in
    ax.plot(IV.iloc[:,i], color = 'red', label='interconnected leverage') 
    ax.plot(rem.iloc[:,i], color = 'grey', label='residual leverage') 
    ax.plot(IV.index, dfy_pca[:,i], color = 'blue', label='principal component') 
    
# verzamel de lijnen en kleuren van de laatste plot (ze zijn immers allemaal hetzelfde)
h,l = ax.get_legend_handles_labels()    

# maak 1 legenda midden bovenin het figuur met de 3 entries naast elkaar 
fig.legend(handles = h, labels = l, loc=9, ncol=3)

# maak mooi en sla op
fig.tight_layout()
fig.savefig('pca-decomposition.png')    

#%%

# verzamel alle mogelijke regressies met robuste errors in een dictionary. deze kun je dan als volgt uitlezen: regressie = estims['stability measure naam']['type regressie']
estims = {}
for col in dfz.columns:
    estims[col] = {     'direct':sm.OLS(dfz[col], sm.add_constant(sep), missing='drop').fit().get_robustcov_results(), 
                        'IV':    sm.OLS(dfz[col], sm.add_constant(IV), missing='drop').fit().get_robustcov_results(),
                        'rem':   sm.OLS(dfz[col], sm.add_constant(rem), missing='drop').fit().get_robustcov_results(),
                        'VARX-IV':    sm.OLS(dfz[col], sm.add_constant(IV.join(pd.DataFrame(lagmat(dfz[col], maxlag=nlags),columns=['lag_1','lag_2','lag_3','lag_4'], index=dfz.index))), missing='drop').fit().get_robustcov_results(), 
                        'VARX-rem':   sm.OLS(dfz[col], sm.add_constant(rem.join(pd.DataFrame(lagmat(dfz[col], maxlag=nlags),columns=['lag_1','lag_2','lag_3','lag_4'], index=dfz.index))), missing='drop').fit().get_robustcov_results(),
                        'VARX-direct':sm.OLS(dfz[col], sm.add_constant(sep.join(pd.DataFrame(lagmat(dfz[col], maxlag=nlags),columns=['lag_1','lag_2','lag_3','lag_4'], index=dfz.index))), missing='drop').fit().get_robustcov_results() }


#%% doe joint significantie test op lags en iv/resid met/zonder lags

# initialiseer de dictionary waar we de tests in gaan opslaan    
tests = dict(zip(dfz.columns, [{} for _ in dfz.columns]))

# stel de restrictiematrices op die we gebruiken voor de f-test
no_lags = np.concatenate( np.zeros([nlags,2*npc]), np.eye(nlags) )
no_resid = np.concatenate( np.zeros([npc,npc]), np.eye(npc), np.zeros([npc,nlags]) )
no_iv = np.concatenate( np.eye(npc), np.zeros([npc, npc+nlags]) )
no_resid_lags = np.concatenate( np.zeros(npc,npc), np.eye(npc) )
no_iv_lags = np.concatenate( np.eye(npc), np.zeros(npc,npc) )

# doe alle tests. De f-test voer je uit door de restrictiematrix op te geven
for col in dfz.columns:
    list(map(tests[col].update,[
    fresults('f-test on lags',estims[col]['VARX-direct'].f_test( no_lags )),
    fresults('f-test on iv in var',estims[col]['VARX-direct'].f_test( no_iv )),
    fresults('f-test on residuals in var',estims[col]['VARX-direct'].f_test( no_resid )),
    fresults('f-test on iv in ols',estims[col]['direct'].f_test( no_iv_lags )),
    fresults('f-test on residuals in ols',estims[col]['direct'].f_test( no_resid_lags )),

# schrijf de resultaten weg naar een excel
pd.DataFrame(tests).to_excel('results/significances of IV leverage interconnectedness stability.xlsx')

#%% Maak een keuze welk model je bij welke stability measure je wilt

# dit is handwerk, sla de regressies die we willen gebruiken handmatig op in een dict. Dat is makkelijker met de marginale effecten berekenen
selection = {
    'BCI OECD':estims['BCI OECD']['VARX-direct'],
    'CCI OECD':estims['CCI OECD']['VARX-direct'],
    'Comm. real estate p (y-o-y %ch)':estims['Comm. real estate p (y-o-y %ch)']['VARX-direct'],
    'log-Debt/gdp':estims['log-Debt/gdp']['VARX-direct'],
    'Federal funds effective rate':estims['Federal funds effective rate']['VARX-IV'],
    'log-Financial assets/gdp: nonfin. corp.':estims['log-Financial assets/gdp: nonfin. corp.']['VARX-direct'],
    'GDP GROWTH':estims['GDP GROWTH']['IV'],
    'Inflation':estims['Inflation']['VARX-IV'],
    'KCFSI':estims['KCFSI']['VARX-direct'],
    'log-Total debt/equity: nonfin. corp.':estims['log-Total debt/equity: nonfin. corp.']['direct']}
    
def doeOpSelectie(fun):
    return dict(zip(selection,map(fun, selection.values())))


#%% SELECTION %%


pd.DataFrame.from_dict({
    'BCI OECD':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['BCI OECD']['VARX-direct'].params[0:6].reshape(3,2,order='F') ).reshape(12,order='F'))),
    'CCI OECD':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['CCI OECD']['VARX-direct'].params[0:6].reshape(3,2,order='F')  ).reshape(12,order='F'))),
    'Comm. real estate p (y-o-y %ch)':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['Comm. real estate p (y-o-y %ch)']['VARX-direct'].params[0:6].reshape(3,2,order='F')  ).reshape(12,order='F'))),
    'log-Debt/gdp':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['log-Debt/gdp']['VARX-direct'].params[0:6].reshape(3,2,order='F')  ).reshape(12,order='F'))),
    'Federal funds effective rate':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Federal funds effective rate']['VARX-IV'].params[0:3] ))),
    'log-Financial assets/gdp: nonfin. corp.':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['log-Financial assets/gdp: nonfin. corp.']['VARX-direct'].params[0:6].reshape(3,2,order='F')  ).reshape(12,order='F'))),
    'GDP GROWTH':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['GDP GROWTH']['IV'].params[0:3] ))),
    'Inflation':dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims['Inflation']['VARX-IV'].params[0:3] ))),
    'KCFSI':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['KCFSI']['VARX-direct'].params[0:6].reshape(3,2,order='F') ).reshape(12,order='F'))),
    'log-Total debt/equity: nonfin. corp.':dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims['log-Total debt/equity: nonfin. corp.']['direct'].params[0:6].reshape(3,2,order='F') ).reshape(12,order='F')))                                       
                                       }).to_excel('results/selected results.xlsx') 

dict(zip(selection,map(lambda x:x.rsquared, selection.values())))

doeOpSelectie(lambda x: x.rsquared)

ax = pyplot.subplot()
doeOpSelectie(lambda x: ax = pyplot.subplot();ax.plot(x.resid);ax.plot(x.fittedvalues))

def safeplot(ax,x,y,*args,**kwargs):
    m = len(x)-len(y)
    return ax.plot(x[m:],y,*args,**kwargs)
        
#%%

fig = pyplot.figure(figsize=(9,9), tight_layout=True); i=0
for key in selection:
    i+=1
    ax = fig.add_subplot(4,3,i,title=key)
    ax2 = ax.twinx()
    safeplot(ax2,dfz.index,selection[key].resid, label='residuals',color='black',alpha=.4)
    safeplot(ax,dfz.index,selection[key].fittedvalues, label='fitted values', color='red')
    dfz[key].plot(ax=ax, color='blue', label='series')
line1, label1 = ax.get_legend_handles_labels()
line2, label2 = ax2.get_legend_handles_labels()
fig.legend(handles = line1+line2, labels = label1+label2,loc=8)   
fig.tight_layout()
pyplot.savefig('fitted-residual-actual/fitted-residual-actual.png')
   
   


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
    
    
    
    
    