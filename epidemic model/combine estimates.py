import os
import numpy as np
import pandas as pd
from matplotlib import style,pyplot as plt
import epidemicModel
style.use('ggplot')

df = pd.read_excel('crisisPerSector.xlsx',index_col=0,converters={i:bool for i in range(1,7)})
df.index = pd.date_range(start='1-1-1952', end='30-09-2015', freq='Q')

def read_all():
    x = {}
    names = []
    for dirpath, dirnames, filenames in os.walk(os.curdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.npy':
                attrs = filename.split(' - ')
                if attrs[1]=='windowsize 100':
                    x[int(attrs[2].split(' ')[1])] = np.load(os.path.join(dirpath,filename))
                    names.append(filename)
    
    y = np.zeros((2,156,6,6))
    for key in x.keys():
        y[:,key,:,:] = x[key]
    x = y
    mean = x[0,:,:,:]
    var  = x[1,:,:,:]
    r = epidemicModel.R0(mean)    
    return mean,var,r,names
    

if __name__ == '__main__':
    mean,var,r,names = read_all() 
    r = pd.DataFrame(data=r, index=df.index[99:], columns = df.columns)
    ax = plt.subplot()
    r.plot(ax=ax)
    plt.title('$R_0$ development through time')