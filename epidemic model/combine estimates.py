import os
import numpy as np
from matplotlib import style,pyplot as plt
import epidemicModel
style.use('ggplot')

def read_all():
    x = {}
    names = []
    for dirpath, dirnames, filenames in os.walk(os.curdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.npy':
                attrs = filename.split(' - ')
                if attrs[1]=='windowsize 50':
                    x[int(attrs[2].split(' ')[1])] = np.load(os.path.join(dirpath,filename))
                    names.append(filename)
    
    x = np.array([val[1] for val in x.items()])
    mean = x[:,0,:,:]
    var  = x[:,1,:,:]
    r = epidemicModel.R0(mean)    
    return mean,var,r,names
    

if __name__ == '__main__':
    mean,var,r,names = read_all()   
    plt.plot(r)
    x=mean.diagonal(axis1=1,axis2=2)