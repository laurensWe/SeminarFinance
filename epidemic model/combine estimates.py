import os
import numpy as np
import pandas as pd
import epidemicModel

df = pd.read_excel('crisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,7)})
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')
ws = 100
length = len(df)-ws+1

def read_all():
    x = {}
    names = []
    for dirpath, dirnames, filenames in os.walk(os.curdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.npy':
                attrs = filename.split(' - ')
                if (len(attrs)>1) and (attrs[1]=='windowsize '+str(ws)):
                    x[int(attrs[2].split(' ')[1])] = np.load(os.path.join(dirpath,filename))
                    names.append(filename)
    
    y = np.zeros((2,length,6,6))
    for key in x.keys():
        y[:,key,:,:] = x[key]
    x = y
    mean = x[0,:,:,:]
    var  = x[1,:,:,:]
    r = epidemicModel.R0(mean)    
    return mean,var,r,names



if __name__ == '__main__':
    mean,var,r,names = read_all() 
    
    np.save('mean for windowsize '+str(ws) , mean)
    np.save('variance for windowsize '+str(ws) , var)
    r = pd.DataFrame(data=r, index=df.index[ws-1:], columns = df.columns)
    m = pd.DataFrame(data=mean.reshape( (length,36), order='F'), columns=pd.MultiIndex.from_tuples([(x,y) for y in df.columns for x in df.columns]), index = r.index)
    v = pd.DataFrame(data=var.reshape((length,36), order='F'), columns=m.columns,index=m.index)
    r.to_excel('R0 for windowsize '+str(ws)+'.xlsx')   
    m.to_excel('mean for windowsize '+str(ws)+'.xlsx')
    v.to_excel('variance for windowsize '+str(ws)+'.xlsx')
    