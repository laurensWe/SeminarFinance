# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:36:01 2016

@author: ms
"""
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot,style
import statsmodels.api as sm
from statsmodels.tsa.stattools import lagmat
from statsmodels.sandbox.tools.tools_pca import pca
style.use('ggplot')
    
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
no_lags = np.concatenate( [np.zeros([nlags,2*npc+1]), np.eye(nlags)], axis=1 )
no_resid = np.concatenate( [np.zeros([npc,npc+1]), np.eye(npc), np.zeros([npc,nlags])], axis=1 )
no_iv = np.concatenate( [np.zeros([npc,1]), np.eye(npc), np.zeros([npc, npc+nlags])], axis=1 )
no_resid_lags = np.concatenate( [np.zeros([npc,npc+1]), np.eye(npc)], axis=1 )
no_iv_lags = np.concatenate( [np.zeros([npc,1]), np.eye(npc), np.zeros([npc,npc])], axis=1 )

# doe alle tests. De f-test voer je uit door de restrictiematrix op te geven
for col in dfz.columns:
    list(map(tests[col].update,[
    fresults('f-test on lags',estims[col]['VARX-direct'].f_test( no_lags )),
    fresults('f-test on iv in var',estims[col]['VARX-direct'].f_test( no_iv )),
    fresults('f-test on residuals in var',estims[col]['VARX-direct'].f_test( no_resid )),
    fresults('f-test on iv in ols',estims[col]['direct'].f_test( no_iv_lags )),
    fresults('f-test on residuals in ols',estims[col]['direct'].f_test( no_resid_lags ))] ))

# schrijf de resultaten weg naar een excel
pd.DataFrame(tests).to_excel('results/significances of IV leverage interconnectedness stability.xlsx')

#%% Maak een keuze welk model je bij welke stability measure je wilt

# TODO dit is handwerk, sla de regressies die we willen gebruiken handmatig op in een dict. Dat is makkelijker met de marginale effecten berekenen
selection = {
    'BCI OECD':'VARX-direct',
    'CCI OECD':'VARX-direct',
    'Comm. real estate p (y-o-y %ch)':'VARX-direct',
    'log-Debt/gdp':'VARX-direct',
    'Federal funds effective rate':'VARX-IV',
    'log-Financial assets/gdp: nonfin. corp.':'VARX-direct',
    'GDP GROWTH':'IV',
    'Inflation':'VARX-IV',
    'KCFSI':'VARX-direct',
    'log-Total debt/equity: nonfin. corp.':'direct'}
 
#%% Bereken de marginale effecten en de R^2 van de selectie

# deze functie bepaalt wat voor regressie we hadden gekozen, en wat de marginale effecten zijn obv de parameters uit deze regressie
def marginalEffect(series, kind):
    if 'direct' in kind: 
        return dict(zip(itertools.product(dfy.columns,['IV','resid']), np.dot( evecs, estims[series][kind].params[0:2*npc].reshape(npc,2,order='F') ).reshape(4*npc,order='F')))
    elif 'IV' in kind:
        return dict(zip(itertools.product(dfy.columns,['IV']), np.dot( evecs, estims[series][kind].params[0:npc] )))
    else:
        return dict(zip(itertools.product(dfy.columns,['resid']), np.dot( evecs, estims[series][kind].params[0:npc] )))

# hier worden de marginale effecten en de r^2 uitgerekend, samen in een dataframe gestopt en weggeschreven.
pd.DataFrame.from_dict({series:marginalEffect(series,selection[series]) for series in selection}).append(
    pd.DataFrame.from_dict(dict(zip(selection,map(lambda x: {('Rsquared',''):estims[x][selection[x]].rsquared}, selection))))).to_excel('results/selected marginal results.xlsx')
 
#%% plot alle measures, de fitted values en de residuals in 1 figuur

# deze functie zorgt dat je x en y gewoon kunt plotten, ook als y minder datapunten bevat (bijv. bij VAR fitted values)
def safeplot(ax,x,y,*args,**kwargs):
    m = len(x)-len(y)
    return ax.plot(x[m:],y,*args,**kwargs)
  
# doe het plotten.
fig = pyplot.figure(figsize=(15,15), tight_layout=True); i=0
for key in selection:
    i+=1
    # TODO pas het aantal- en de verdeling van de subplots hier aan
    ax = fig.add_subplot(7,2,i,title=key)
    ax2 = ax.twinx()
    safeplot(ax2,dfz.index,estims[key][selection[key]].resid, label='residuals',color='black',alpha=.4)
    safeplot(ax,dfz.index,estims[key][selection[key]].fittedvalues, label='fitted values', color='red')
    dfz[key].plot(ax=ax, color='blue', label='series')
line1, label1 = ax.get_legend_handles_labels()
line2, label2 = ax2.get_legend_handles_labels()
fig.legend(handles = line1+line2, labels = label1+label2,loc=8)   
fig.tight_layout()
pyplot.savefig('fitted-residual-actual/fitted-residual-actual.png')
