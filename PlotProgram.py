# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:26:01 2016

@author: laure
"""

# preamble
import numpy as np
import pandas as pd
from matplotlib import style,pyplot 
from pandas.tools.plotting import autocorrelation_plot
style.use('ggplot')

wd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\Resultaten\\HHI en in-degree\\'

hhi = pd.read_excel(wd + "hhi_6sectors.xlsx").plot()

Indegree2 = pd.read_excel(wd + "temp2percent.xlsx").plot()
pyplot.ylim(0,1.2)
Indegree5 = pd.read_excel(wd + "temp5percent.xlsx").plot()
pyplot.ylim(0,1.2)

Systems = pd.read_excel(wd + "systemHHI_InDegree_6sectors.xlsx").plot()
pyplot.ylim(0,1)

#plot = ggplot(aes(x='Date', y='value', color='variable'), data=hhi) +geom_line()
#fig = plot.draw()

Systems1 = pd.read_excel(wd + "system1.xlsx").plot()
pyplot.ylim(0,1)
Systems2 = pd.read_excel(wd + "system2.xlsx").plot()
pyplot.ylim(0,1)
Systems3 = pd.read_excel(wd + "system3.xlsx").plot()
pyplot.ylim(0,1)


#%%
