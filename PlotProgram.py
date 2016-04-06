# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:26:01 2016

@author: L.J.R. Weijs
"""

# preamble
import numpy as np
import pandas as pd
from matplotlib import style,pyplot 
from pandas.tools.plotting import autocorrelation_plot
style.use('ggplot')
from matplotlib.mlab import PCA

stwd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\'
spwd = 'Resultaten\\PCA&WA_Analyse\\'

wd = stwd + spwd


#%% this part is for calculation the Principal components of 

#read the interconnectedness measures
was6sec = pd.read_excel(wd+"WA_intercon_6sec.xlsx")
wasPCA = PCA(was6sec)
pd.DataFrame(wasPCA.Y).to_excel(wd+"WAs6Sec.xlsx")

#read the NBER
NBERcrisis = pd.read_excel(wd + "NBER.xlsx")

#%% make beautiful graph with NBER data in between
colors=['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']   
markers = [ '-o', '--', '-v', ':', '-s', '-*']

pca_analyse=PCA(was6sec)
hulp = pd.DataFrame(pca_analyse.Y).set_index(was6sec.index)[0]
ax2=hulp.plot(color='k', label="First principal component)
ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, prop={'size':16})
ax2.fill_between(was6sec.index, 3*NBERcrisis['NBER_Recession'], -3*NBERcrisis['NBER_Recession'], color='m',step='pre', alpha=0.3)

#%% Thick lines for the system 

# initialisation
ax = pyplot.subplot()
    
pd.read_excel(wd + "HHI_14.xlsx").plot(ax=ax, linewidth = .5)
pd.read_excel(wd + "SystemHHI_14.xlsx").plot(ax=ax, linewidth = 4.0, color='lightgrey',style='-o')
pd.read_excel(wd + "SystemHHIWA_14.xlsx").plot(ax=ax, linewidth =4.0, color='grey', style='-^')
ax2= ax.twinx()
pd.read_excel(wd + "SystemHHIPCA_14.xlsx").plot(ax=ax2, linewidth = 4.0, color='black', style='-v')
h,l = ax.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax.legend(h+h2,l+l2,loc=2)

#%% individual lines from an excel file
leverage14Sectors = pd.read_excel(wd + "leverage14_sectors.xlsx")
leverage14Sectors = leverage14Sectors.set_index(leverage14Sectors.Time)
leverage14Sectors.plot(subplots=True, layout=(4,4)) 

#%% The dataseries plot has been formatted in such a way that it can be interpreted with only black ink.
# Also two plots on different axis :D.
# initialisation
interconmeasures = pd.read_excel("WA_intercon_6sec.xlsx")  
R0 = pd.read_excel("R0_6sec.xlsx")   
interconmeasures = interconmeasures.set_index(interconmeasures.Date)
R0 = R0.set_index(R0.Date)          
markers1 = ['-v','-p','-s','-*', '-^']
colors=['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
#ax = interconmeasures.plot(style=markers1, color='grey')
ax=interconmeasures.plot(style=markers1, figsize=(14,6), color=colors)
markers2 = ['-o']
#ax2 = R0.plot(secondary_y=True,ax=ax,linewidth=4.0, style=markers2, color='black')
ax2= ax.twinx()
R0.plot(ax=ax2, color='k', linewidth=3, style=markers2)
#ax.legend(bbox_to_anchor=(1.23, 1), prop={'size':16})
#ax2.legend(bbox_to_anchor=(1.21, 0.65), prop={'size':16})
h1,l1 = ax.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2,l1+l2,loc =2)
ax.legend_.remove()

#pyplot.legend(h1+h2, l1, loc=2)


#%% t


#pyplot.legend(loc=0, prop={'size':11})
#legend(loc=0, prop={'size':11})

#ax = NBERcrisis.plot(kind='area', color='m')
#pd.DataFrame(pca_analyse.Y).plot(secondary_y=True, ax=ax)

#voorbeeld om een reeks kleurtjes en markers mee te geven: 
#colors=['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
#markers = [ '-o', '--', '-v', ':', '-s', '-*']
#data.plot(style=markers, color=colors)

#voorbeeld om legenda naast plaatje te doen met bepaald lettertype: 
#ax.legend(bbox_to_anchor=(1.2, 1.0), prop={'size':20})




#%% adjusted y-axis

#indegree002= pd.read_excel(wd + "InDegree002.xlsx").plot()
#pyplot.ylim(0,1.2)
#indegree2= pd.read_excel(wd + "InDegree02.xlsx").plot()
#pyplot.ylim(0,1.2)
#indegree5= pd.read_excel(wd + "InDegree05.xlsx").plot()
#pyplot.ylim(0,1.2)



#pyplot.
#pyplot.ylim(0,1.2)

#hhi = pd.read_excel(wd + "hhi_6sectors.xlsx").plot()

#Indegree2 = pd.read_excel(wd + "temp2percent.xlsx").plot()
#pyplot.ylim(0,1.2)
#Indegree5 = pd.read_excel(wd + "temp5percent.xlsx").plot()
#pyplot.ylim(0,1.2)

#Systems = pd.read_excel(wd + "systemHHI_InDegree_6sectors.xlsx").plot()
#pyplot.ylim(0,1)

#plot = ggplot(aes(x='Date', y='value', color='variable'), data=hhi) +geom_line()
#fig = plot.draw()

#Systems1 = pd.read_excel(wd + "system1.xlsx").plot()
#pyplot.ylim(0,1)
#Systems2 = pd.read_excel(wd + "system2.xlsx").plot()
#pyplot.ylim(0,1)
#Systems3 = pd.read_excel(wd + "system3.xlsx").plot()
#pyplot.ylim(0,1)


#%%


