# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:44:31 2016

@author: ms
"""
import numpy as np
import pandas as pd
from matplotlib import style,pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
style.use('ggplot')
ws = 70

def diff(df,order=1):
    def frstdiff(df):
        return pd.DataFrame(data = np.array(df)[1:,:]-np.array(df)[:-1,:], index = df.index[1:], columns = df.columns)
    local = df.copy()
    for i in range(order):
        local = frstdiff(local)
    return local

df = pd.read_excel('crisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,7)})
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')

mean = np.load('mean for windowsize '+str(ws)+'.npy')
var = np.load('variance for windowsize '+str(ws)+'.npy')
m = pd.read_excel('R0 for windowsize '+str(ws)+'.xlsx')   
r = pd.read_excel('mean for windowsize '+str(ws)+'.xlsx')
v = pd.to_excel('variance for windowsize '+str(ws)+'.xlsx')

plt.clf()
ax = plt.subplot()
r.plot(ax=ax)
name = '$R_0$ development through time for windowsize '+str(ws)
plt.title(name)
plt.savefig(name)  

plt.clf()
ax = plt.subplot()
name = 'Spread of estimated parameters $p_ij$ for windowsize '+str(ws)
plt.title(name)
plt.savefig(name)
m.boxplot(ax=ax, rot=90)

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
 
r.corr()

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
    
    plt.title('$p_{{{}}}$ and $R_0({})$, with crises periods shaded'.format(col,col))
    plt.savefig('rate of getting better and R0 and crises for {}.png'.format(col))


for i in range(6):
    plt.clf()
    ax = plt.subplot()
    name = 'Rates of infectiousness of {}'.format(df.columns[i])
    ax.plot(m.index,mean[:,i,np.array([ i!=j for j in range(6)])])
    ax.fill_between(df.index,df.iloc[:,i], alpha=.4, step = 'pre')
    plt.title(name)
    plt.legend(labels=df.columns)
    plt.savefig(name+'.png')
    

plt.cla()
ax = plt.subplot()
color = ['red','green', 'blue','yellow','purple','magenta']
for i in range(6):
    col = df.columns[i]
    ax.eventplot(positions=df.index[df[col]], lineoffsets=i, linewidths=1.4, color=['darkgrey'])#[color[i]])
ax.set_ylim([-.5,5.5])
ax.set_yticks(list(range(6)))
ax.set_yticklabels(df.columns)
name = 'crisis periods per sector'
plt.title(name)
plt.savefig(name+'.png')