# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:28:52 2016

@author: Laurens duhh!!!
En een beetje van Beert
"""

import pandas as pd
import numpy as np
from matplotlib.mlab import PCA
import os

wd = 'C:\\Users\\Beert\\Documents\\SeminarFinance\\Instruments\\'

#%% Initialisation, detemines what data to read.

dataframes = {}
for dirpath,dirnames,filenames in os.walk(wd):
    for filename in filenames:
        if filename[0] != '_':
            dataframes[filename] = pd.read_excel(os.path.join(dirpath,filename)).to_dict()

a = pd.DataFrame(dataframes)

instruments = pd.read_excel(wd + "_Instruments.xlsx")
filesToTake = pd.read_excel(wd + "_FilesToTake.xlsx")
sectors = pd.read_excel(wd + "_Sectors.xlsx")

assetNames = filesToTake['Assets.xlsx']

#%% Calculates the shares of all the sectors and their instruments over time

horizontalTuples = []
verticalTuples = []
for ints in instruments.columns:
    for sec in sectors.columns:
        horizontalTuples.append((ints,sec))

for dates in instruments.index:
    for sec in sectors.columns:
        verticalTuples.append((dates,sec))

Total = pd.DataFrame(0, index=instruments.index, columns=instruments.columns)
for assetfile in filesToTake['Assets.xlsx']:
    Total = Total + pd.read_excel(wd+assetfile, index_col = 0)
    print(assetfile + ': ' + str(len(Total.columns)))
        
    # Debug print line    
    if len(pd.read_excel(wd+assetfile, index_col = 0).columns) != 28:
        print(assetfile)

for i in range(Total.shape[0]):
    for j in range(Total.shape[1]):
        if Total.iloc[i,j] == 0:
            Total.iloc[i,j] = float('inf')

for i in filesToTake.index:
    zero = pd.DataFrame(0, index=instruments.index, columns=instruments.columns)
    asset = zero + pd.read_excel(wd+filesToTake['Assets.xlsx'].iloc[i], index_col = 0)
    share = asset / Total
    share.to_excel(wd+filesToTake['Shares.xlsx'].iloc[i])

df = pd.DataFrame(index=verticalTuples,columns=horizontalTuples)
for dates in instruments.index:    
    for ints in instruments:
        Shares = []
        Assets = []
        for sharefile in filesToTake['Shares.xlsx']:
            Shares.append(a[sharefile][ints][dates])
        for assetfile in filesToTake['Assets.xlsx']:    
            Assets.append(a[assetfile][ints][dates])    
        
        print('ik ga nu de simpleMatrix maken')        
        
        simpleMatrix = pd.DataFrame(index=sectors.columns, columns=sectors.columns)
        for share in range(len(Shares)):
            for asset in range(len(Assets)):
                simpleMatrix.iloc[share, asset] = Shares[share]*Assets[asset]
        for sec1 in sectors.columns:
            for sec2 in sectors.columns:
                df.loc[(dates, sec1), (ints,sec2)] = simpleMatrix.loc[sec1,sec2]

mic = pd.MultiIndex.from_tuples(tuples=df.columns, names=['instrument','sector'])
mii = pd.MultiIndex.from_tuples(tuples=df.index, names=['date','sector'])
df2 = pd.DataFrame(df,index=mii,columns=mic)

#%% Calculate the In Degree interconnectedness measure

print('ik ga nu de inDegrees maken')
inDegree = pd.DataFrame(index=instruments.index, columns=sectors.columns)
systemInDegreeWA = pd.DataFrame(0,columns=['SystemInDegree'],index=instruments.index)
thresholds = [0.002, 0.005, 0.01, 0.02]

for threshold in thresholds:
    for dates in instruments.index:
        tempDate = df2.loc[dates].transpose().sum(level=[1])
        totals = tempDate.sum()
        amountSec = len(totals)
        for secprim in totals.index:
            count = 0
            for secsec in totals.index:
                if tempDate.loc[secprim,secsec]/totals[secprim] > threshold and secsec != secprim:
                    count += 1
                inDegree.loc[dates,secprim] = count / (amountSec - 1)
        #SYSTEM
        #Calculate the weighted average of the System, the weight is dependent from the relative size of the total assets from that specific sector.    
        for sec in totals.index:
            systemInDegreeWA.loc[dates,'SystemInDegree'] = systemInDegreeWA.loc[dates,'SystemInDegree'] + inDegree.loc[dates,sec]*(tempDate[sec].sum()/(Total.loc[dates][Total.loc[dates] != float('inf')].sum()))

    print('System inDegrees with weights from PCA')    
    #SYSTEM

    #calculates the weights based on the first principal component
    matrix = inDegree.as_matrix().astype('float')
    pca = PCA(matrix)
    systemInDegreePCA = pd.DataFrame(matrix*pca.Wt[0]).transpose().sum()
        
    #Write to Excel
    systemInDegreeWA.to_excel(wd + 'SystemInDegreeWA' + str(threshold) + '_' + str(len(Shares)) +'.xlsx')
    pd.DataFrame(systemInDegreePCA).to_excel(wd + 'SytemInDegreePCA' + str(threshold) + '_' + str(len(Shares)) +'.xlsx')
    inDegree.to_excel(wd+'InDegree_'+ str(len(Shares)) +'.xlsx')
        
#%% Calculates the HHI interconnectedness measure

print('ik ga nu de HHI maken')
systemHHIWA = pd.DataFrame(0,columns=['SystemHHI'],index=instruments.index)
systemHHI = pd.DataFrame(columns=['SystemHHI'],index=instruments.index)
hhiIndices = pd.DataFrame(index=instruments.index, columns=sectors.columns)

for dates in instruments.index:
    tempDate = df2.loc[dates].transpose().sum(level=[1])
    totals = tempDate.sum()
    amountSec = len(totals)
    for secprim in totals.index:
        hhiIndex = 0
        for secsec in totals.index:
            if secsec != secprim:
                hhiIndex += tempDate.loc[secprim,secsec]/totals[secprim]
        hhiIndices.loc[dates,secprim] = (hhiIndex - 1/amountSec)/(1-1/amountSec)
    #SYSTEM
    #Calculate the weighted average of the System, the weight is dependent from the relative size of the total assets from that specific sector.    
    for sec in totals.index:
        systemHHIWA.loc[dates,'SystemHHI'] = systemHHIWA.loc[dates,'SystemHHI'] + hhiIndices.loc[dates,sec]*(tempDate[sec].sum()/(Total.loc[dates][Total.loc[dates] != float('inf')].sum()))

#SYSTEM
#calculates the weights based on the first principal component
matrixhhi = hhiIndices.as_matrix().astype('float')
pcahhi = PCA(matrixhhi)
systemHHIPCA  = pd.DataFrame(matrixhhi*pcahhi.Wt[0]).transpose().sum()

for dates in instruments.index:
    systemHHI.loc[dates,'SystemHHI'] = hhiIndices.loc[dates].sum() / amountSec
    
#Write to Excel
systemHHIWA.to_excel(wd + 'SystemHHIWA_'+ str(len(Shares)) +'.xlsx')
pd.DataFrame(systemHHIPCA).to_excel(wd + 'SystemHHIPCA_' + str(len(Shares)) +'.xlsx')
hhiIndices.to_excel(wd+'HHI_'+ str(len(Shares)) +'.xlsx')
systemHHI.to_excel(wd+'SystemHHI_'+ str(len(Shares)) +'.xlsx')