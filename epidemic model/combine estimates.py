import os
import numpy as np
import pandas as pd
import epidemicModel

nsector =6
os.chdir(r'C:\Users\ms\OneDrive\Documenten\SeminarFinance\epidemic model\{}-sector'.format(nsector))

df = pd.read_excel('crisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,nsector)})
if nsector == 6:
    df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')
else:
    df.index = pd.date_range(start='1-1-1952', end='31-12-2015', freq='Q')
ws = 255
length = len(df)-ws+1
nsector = len(df.columns)

def read_all():
    x = {}
    names = []
    for dirpath, dirnames, filenames in os.walk(os.curdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.npy':
                attrs = filename.split(' - ')
                if (len(attrs)>1) and (attrs[1]=='windowsize '+str(ws)):
                    key = int(attrs[2].split(' ')[1])
                    time = float(attrs[4][:-4])
                    if key in x:
                        if time > x[key][0]:
                            x[key] = (time, np.load(os.path.join(dirpath,filename)))
                    else:
                        x[key] = (time , np.load(os.path.join(dirpath,filename)) )
                        names.append(filename)
    
    mean = np.zeros((length,nsector,nsector))
    for key in x:
        mean[key,:,:] = np.sum(x[key][1], axis=0)
    r = epidemicModel.R0(mean)    
    return mean,r,names


if __name__ == '__main__':
    mean,r,names = read_all() 
    
    np.save('mean for windowsize {}'.format(ws) , mean)
    r = pd.DataFrame(data=r, index=df.index[ws-1:], columns = df.columns)
    #m = pd.DataFrame(data=mean.reshape( (length,36), order='F'), columns=pd.MultiIndex.from_tuples([(x,y) for y in df.columns for x in df.columns]), index = r.index)
    #v = pd.DataFrame(data=var.reshape((length,36), order='F'), columns=m.columns,index=m.index)
    r.to_excel('R0 for windowsize {}.xlsx'.format(ws) )   
    #m.to_excel('mean for windowsize '+str(ws)+'.xlsx')
    #v.to_excel('variance for windowsize '+str(ws)+'.xlsx')
    
if False: # dit zijn wat losse statistiekjes enzo, voor debugging vooral
    #[i if np.isnan(np.prod(mean,axis=(1,2))[i]) else None for i in range(157)]
    I = np.array(df)
    np.sum(I[:-1,:] & ~I[1:,:], axis=0) / np.sum(df, axis=0)
    
    np.round(pd.DataFrame(mean[0,:,:],index=df.columns,columns=df.columns),3)
    
    df.loc[I.any(axis=1)[1:] & ~ I.any(axis=1)[:-1],:]
    
    df.loc[ \
    np.concatenate((np.array([False]),  I.any(axis=1)[1:] ),axis=0) & \
    ~np.concatenate((np.array([False]), I.any(axis=1)[:-1]),axis=0),:].sum()
    
    r.plot()    
    
    a = np.load('epidemic model - windowsize 100 - period 0 - iter 10000000 - 1460051751.789101.npy')