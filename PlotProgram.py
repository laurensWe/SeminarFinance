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

wd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\Resultaten\\HHI en in-degree\\14 sectors\\'

#%% Thick lines for the system 

# initialisation
ax = pyplot.subplot()

indiIn = pd.read_excel(wd + "InDegree0_00214.xlsx").plot(ax=ax, linewidth = .5)
sytemIn = pd.read_excel(wd + "SytemInDegree0_02PCA14.xlsx").plot(ax=ax, linewidth = 4.0, color='black')


#%% individual lines from an excel file
leverage14Sectors = pd.read_excel(wd + "leverage14_sectors.xlsx")
leverage14Sectors = leverage14Sectors.set_index(leverage14Sectors.Time)
leverage14Sectors.plot(subplots=True, layout=(4,4)) 

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
