# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:28:52 2016

@author: Laurens duhh!!!
En een beetje van Beert
"""

import pandas as pd
import os

wd = 'C:\\Users\\laure\\SharePoint\\Seminar - Documents\\Data\\Interconnectedness\\14-Sectors\\Indices Models\Instruments\\'

#%%
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

#%%
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
            Total.iloc[i,j] = float('Inf')

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

#%%

print('ik ga nu de inDegrees maken')
inDegree = pd.DataFrame(index=instruments.index, columns=sectors.columns)
threshold = 0.002

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

systemInDegree = pd.DataFrame(columns=['SystemInDegree'],index=instruments.index)

for dates in instruments.index:
    systemInDegree.loc[dates,'SystemInDegree'] = inDegree.loc[dates].sum() / amountSec
    
inDegree.to_excel(wd+'InDegree0_002'+ str(len(Shares)) +'.xlsx')

#%%
print('ik ga nu de HHI maken')
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

systemHHI = pd.DataFrame(columns=['SystemHHI'],index=instruments.index)

for dates in instruments.index:
    systemHHI.loc[dates,'SystemHHI'] = hhiIndices.loc[dates].sum() / amountSec

#%%
hhiIndices.to_excel(wd+'HHI'+ str(len(Shares)) +'.xlsx')