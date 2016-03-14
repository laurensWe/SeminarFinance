# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:32:15 2016

@author: laure
"""

# Script for making three dimensional yield curves per instrument.

# intialize packages
import pandas as pd
import os
#import itertools

wd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\Data\\Interconnectedness\\PerSector\\'
#import numpy as np
#Instruments = pd.read_excel(wd+"Instruments.xlsx")
#Time = pd.read_excel(wd+"TimeRange.xlsx")
#%%
# first read all the asset-shares and assets per instrument for all sectors
dataframes = {}
for dirpath,dirnames,filenames in os.walk(wd):
    for filename in filenames:
        if filename[0] != '_':
            dataframes[filename] = pd.read_excel(os.path.join(dirpath,filename)).to_dict()

a = pd.DataFrame(dataframes)

instruments = pd.read_excel(wd + "_Instruments.xlsx")
filesToTake = pd.read_excel(wd + "_FilesToTake.xlsx")
sectors = pd.read_excel(wd + "_Sectors.xlsx")
        
#%% 
horizontalTuples = []
verticalTuples = []
for ints in instruments.columns:
    for sec in sectors.columns:
        horizontalTuples.append((ints,sec))
                
for dates in instruments.index:
    for sec in sectors.columns:   
        verticalTuples.append((dates,sec))        
        
#df = pd.DataFrame(columns=instruments.columns, index=instruments.index)
df = pd.DataFrame(index=verticalTuples,columns=horizontalTuples)
#df = pd.DataFrame()
for dates in instruments.index:    
    for ints in instruments:
        Shares = []
        Assets = []
        for sharefile in filesToTake['Shares']:
            Shares.append(a[sharefile][ints][dates])
        for assetfile in filesToTake['Assets']:    
            Assets.append(a[assetfile][ints][dates])    
        
        simpleMatrix = pd.DataFrame(index=sectors.columns, columns=sectors.columns)
        for share in range(len(Shares)):
            for asset in range(len(Assets)):
                #print('x: ' + str(asset) +',y: ' + str(share)+ ":" + str(Shares[share]*Assets[asset]))
                simpleMatrix.iloc[share, asset] = Shares[share]*Assets[asset]
        for sec1 in sectors.columns:
            for sec2 in sectors.columns:
                df.loc[(dates, sec1), (ints,sec2)] = simpleMatrix.loc[sec1,sec2]
                #df.join(dftemp, how='inner')                
                #df.loc[(ints,sec1),(dates,sec2) = simpleMatrix.loc[sec1,sec2]
                #df.loc[dates,ints] = simpleMatrix.to_dict()



#itertools.combinations(instruments.columns,sectors.columns) 
#list(zip(instruments.columns,sectors.columns))          
                
            
                
                                
                
                
    