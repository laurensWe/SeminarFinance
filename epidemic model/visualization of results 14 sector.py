# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:44:31 2016

@author: ms
"""
import numpy as np
import pandas as pd
from matplotlib import style,pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
import os
os.chdir(r'C:\Users\ms\OneDrive\Documenten\SeminarFinance\epidemic model')
style.use('ggplot')
ws = 100

def diff(df,order=1):
    def frstdiff(df):
        return pd.DataFrame(data = np.array(df)[1:,:]-np.array(df)[:-1,:], index = df.index[1:], columns = df.columns)
    local = df.copy()
    for i in range(order):
        local = frstdiff(local)
    return local

df = pd.read_excel('crisisPerSector14.xlsx',index_col=0,converters={i:bool for i in range(1,17)}, header=[0,1])
df.index = pd.date_range(start='1-1-1952', end='31-12-2015', freq='Q')



#%% crises per sector and total
for sector in df.columns.get_level_values('sector').unique():
    tempdf = df.xs(sector,level='sector',axis=1)
    ncols = len(tempdf.columns)-1
    if ncols>1:
        print('boe')
        plt.figure(figsize=(6,3))
        ax = plt.subplot()
        name = 'crises in subsectors for sector {}'.format(sector)
       # plt.title(name)
        ax.set_ylim([0,ncols])
        ax.set_yticks(np.arange(.5,ncols+.5))
        ax.set_yticklabels(tempdf.columns[1:])
        plt.tight_layout()
        for i in range(ncols):
            col = tempdf.columns[i+1]
            ax.fill_between(tempdf.index,i,i+1, where=tempdf[col], color=['grey'],alpha=1, step='pre')#[color[i]])
        ax.fill_between(tempdf.index,0,ncols,where=tempdf[sector], color=['red'],alpha=.5, step='pre')#[color[i]])
        plt.grid()
        plt.savefig(name+'.png')

#%%

arr = np.array(df)
np.sum(arr[:-1,:] & ~arr[1:,:], axis=0) # number of crises
np.sum(~arr[:-1,:].any(axis=1) & arr[1:,:].T, axis=1) # number of crisis starts
np.sum(df, axis=0) # number of quarters in crisis
np.sum(df, axis=0)/255 # % of quarters in crisis

r.corr()

#%% VISUALS %%

plt.clf()
ax = plt.subplot()
r.plot(ax=ax)
name = '$R_0$ development through time for windowsize '+str(ws)
plt.title(name)
plt.savefig(name)  

plt.clf()
ax = plt.subplot()
m.boxplot(ax=ax, rot=90)
name = 'Spread of estimated parameters $p_ij$ for windowsize '+str(ws)
plt.autoscale(tight=True)
plt.title(name)
plt.savefig(name)

plt.clf()
ax = plt.subplot()
for i in range(36):
    autocorrelation_plot(m.iloc[i,:], ax=ax)#,label=str(m.columns[i]))
name = 'Autocorrelation plot of the parameter estimates for window size '+str(ws)
plt.title(name)
plt.savefig(name)
#plt.legend()

plt.clf()
ax = plt.subplot()
for i in range(6):
    autocorrelation_plot(r.iloc[i,:], ax=ax)#,label=str(m.columns[i]))
name = 'Autocorrelation plot of the $R_0$ for window size '+str(ws)
plt.title(name)
plt.savefig(name)

plt.clf()
diffs = diff(r)
ax = plt.subplot()
for i in range(6):
    autocorrelation_plot(diffs.iloc[i,:], ax=ax)#,label=str(m.columns[i]))
name = 'Autocorrelation plot of the 1st diffs of $R_0$ for window size '+str(ws)
plt.title(name)
plt.savefig(name)


for i in range(6):
    col = df.columns[i]
    plt.clf()
    ax = plt.subplot()
    m.iloc[:,i].plot(ax=ax, color='red', label='$p_{{{}}}$'.format(col))
    ax2 = ax.twinx()
    r.iloc[:,i].plot(ax=ax2, color='green', label='$R_0({})$'.format(col))
    ax.fill_between(df.index,df.iloc[:,0], alpha=.4, step = 'pre')
    line1, label1 = ax.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    ax2.legend(handles = line1+line2, labels = label1+label2)     
    name = '$p_{{{}}}$ and $R_0({})$, with crises periods shaded. window {}'.format(col,col, ws)
    plt.title(name)
    plt.savefig(name+'.png')

# outgoing effect 
for i in range(6):
    plt.clf()
    ax = plt.subplot()
    name = 'Rates of infectiousness of {} at window {}'.format(df.columns[i],ws)
    ax.plot(m.index,mean[:,i,np.array([ i!=j for j in range(6)])])
    ax.fill_between(df.index,df.iloc[:,i], alpha=.4, step = 'pre')
    ax.set_xlim(m.index[0].toordinal(), m.index[-1].toordinal())
    plt.title(name)
    plt.legend(labels=df.columns)
    plt.savefig(name+'.png')
 
#incoming effect
for i in range(6):
    plt.clf()
    ax = plt.subplot()
    name = 'Rates of susceptibility of {} at window {}'.format(df.columns[i],ws)
    ax.fill_between(df.index,df.iloc[:,i], alpha=.4, step = 'pre')
    ax.plot(m.index,mean[:,np.array([ i!=j for j in range(6)]),i])
    ax.set_xlim(m.index[0].toordinal(), m.index[-1].toordinal())
    plt.title(name)
    plt.legend(labels=df.columns)
    plt.savefig(name+'.png')   




#%% crises NBER/BEA
plt.cla()
ax = plt.subplot()
ax.eventplot(positions=dfnber.index[dfnber['NBER_RECESSIONS']], lineoffsets=1, linelengths=2, linewidths=2.5, color=['blue'],alpha=1)#[color[i]])
ax.eventplot(positions=dfnber.index[dfnber['BEA_RECESSIONS']] , lineoffsets=-1,linelengths=2, linewidths=2.5, color=['green'],alpha=1)#[color[i]])
ax.set_ylim([-2,2])
ax.set_yticks([-1,1])
ax.set_yticklabels(['BEA','NBER'])
name = 'crisis periods nber-bea'
plt.title(name)
plt.savefig(name+'.png')


