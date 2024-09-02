from scipy.io import loadmat

from config import PDEConfig
import os
import numpy as np


#data=loadmat(os.path.join(PDEConfig.sindpy_data, 'burgers.mat'))
data=loadmat(os.path.join(PDEConfig.sindpy_data, 'burgers_highres2.mat'))
print(data.keys())
t = data['t']
x = data['x']
#print(t.shape, t[1,0]-t[0,0])
#print(x.shape, x[0,1]-x[0,0])

print(t.shape)
print(x.shape)
#print(data['usol'].shape)
print(data['u'].shape)
