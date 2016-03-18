import os
import numpy as np
from matplotlib import style,pyplot as plt
import epidemicModel
style.use('ggplot')

def read_all():
    x = []
    counts = []
    for dirpath, dirnames, filenames in os.walk(os.curdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.npy':
                x.append(np.load(os.path.join(dirpath,filename)))
                counts.append(int(filename.split(' - ')[2].split(' ')[1]))
    
    x = np.array(x)
    total = np.sum(counts)
    mean = np.sum(x[:,0,:,:,:]*np.array(counts).reshape((len(counts),1,1,1)),axis=0)/total
    var = np.sum(x[:,1,:,:,:]*np.array(counts).reshape((len(counts),1,1,1))**2,axis=0)/total**2
    delta = epidemicModel.precision(var, n_iter=total)
    r = epidemicModel.R0(mean)    
    return mean,var,delta,r,total,x[-1]
    

if __name__ == '__main__':
    mean,var,delta,r,total,laatste = read_all()