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

wd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\Laurens\\Figures\\'

#%% Thick lines for the system 

# initialisation
ax = pyplot.subplot()
    
pd.read_excel(wd + "InDegree0.02_14.xlsx").plot(ax=ax, linewidth = .5)
ax11 = pd.read_excel(wd + "SystemInDegree0.02_14.xlsx").plot(ax=ax, linewidth = 4.0, color='lightgrey',style='-o')
ax12 = pd.read_excel(wd + "SystemInDegreeWA0.02_14.xlsx").plot(ax=ax, linewidth =4.0, color='grey', style='-^')
ax2 = pd.read_excel(wd + "SystemInDegreePCA0.02_14.xlsx").plot(secondary_y=True,ax=ax, linewidth = 4.0, color='black', style='-v')
h,l = ax.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
pyplot.legend(h + h2,l+l2,loc=2)

#%% individual lines from an excel file
leverage14Sectors = pd.read_excel(wd + "leverage14_sectors.xlsx")
leverage14Sectors = leverage14Sectors.set_index(leverage14Sectors.Time)
leverage14Sectors.plot(subplots=True, layout=(4,4)) 

#%% The dataseries plot has been formatted in such a way that it can be interpreted with only black ink.
# Also two plots on different axis :D.
# initialisation
interconmeasures = pd.read_excel(wd + "InterconMeas.xlsx")  
R0 = pd.read_excel(wd + "R0s.xlsx")   
interconmeasures = interconmeasures.set_index(interconmeasures.Date)
R0 = R0.set_index(R0.Date)          
markers1 = ['-v','-p','-s','-*', '-^']
ax = interconmeasures.plot(style=markers1, color='grey')
markers2 = ['-o']
ax2 = R0.plot(secondary_y=True,ax=ax,linewidth=4.0, style=markers2, color='black')
h1,l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
pyplot.legend(h1+h2, l1, loc=2)


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
