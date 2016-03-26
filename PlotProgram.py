# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:26:01 2016

@author: laure
"""

# preamble
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot,style
style.use('ggplot')
import pandas as pd
import os

wd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\Resultaten\\HHI en in-degree\\'

hhi = pd.read_excel(wd + "hhi_6sectors.xlsx").plot()
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

Indegree2 = pd.read_excel(wd + "temp2percent.xlsx").plot()
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.ylim(0,1.2)
Indegree5 = pd.read_excel(wd + "temp5percent.xlsx").plot()
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.ylim(0,1.2)


#plot = ggplot(aes(x='Date', y='value', color='variable'), data=hhi) +geom_line()
#fig = plot.draw()