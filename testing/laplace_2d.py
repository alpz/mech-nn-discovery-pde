#%%
from scipy.io import loadmat

from config import PDEConfig
import os
import numpy as np

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

from solver.pde_layer import PDEINDLayerEPS
#from solver.ode_layer import ODEINDLayer
#import discovery.basis as B
import ipdb
import extras.logger as logger
import os

from scipy.special import logit
import torch.nn.functional as F
from tqdm import tqdm
import discovery.plot as P




def solve():
    bs = 1
    coord_dims = (32,32,32)
    #coord_dims = (10,10,10)
    # coord, mi_index, range_begin, range_end (inclusive)
    iv_list = [(0,0, [0,0,0],[0,coord_dims[1]-1, coord_dims[2]-1]), 
                (1,0, [1,0,0], [coord_dims[0]-1, 0, coord_dims[2]-1 ]), 
                (2,0, [1,1,0], [coord_dims[0]-1, coord_dims[1]-1, 0 ]), 
                #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
                (0,0, [coord_dims[0]-1,1,1],[coord_dims[0]-1,coord_dims[1]-1, coord_dims[2]-1]), 
                (1,0, [1,coord_dims[1]-1, 1], [coord_dims[0]-2, coord_dims[1]-1, coord_dims[2]-1]),
                (2,0, [1,1, coord_dims[2]-1], [coord_dims[0]-2, coord_dims[1]-2, coord_dims[2]-1])
                ]

    pde = PDEINDLayerEPS(bs=1, coord_dims=coord_dims, order=2, n_ind_dim=1, n_iv=1, 
                        init_index_mi_list=iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)


    t_step_size = 0.1 
    x_step_size = 0.1
    y_step_size = 0.1
    #self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
    #self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

    steps0 = torch.logit(t_step_size*torch.ones(1,1))
    steps1 = torch.logit(x_step_size*torch.ones(1,1))
    steps2 = torch.logit(y_step_size*torch.ones(1,1))

    steps0 = steps0.expand(-1, coord_dims[0]-1)
    steps1 = steps1.expand(-1, coord_dims[1]-1)
    steps2 = steps2.expand(-1, coord_dims[2]-1)
    steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
    steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
    steps2 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
    steps_list = [steps0, steps1, steps2]


    coeffs = torch.zeros((bs, pde.grid_size, pde.n_orders))
    #u, u_t, u_x, u_tt, u_xx
    #u_zz + u_xx + u_yy= 0
    coeffs[..., 4] = 1.
    coeffs[..., 5] = 1.
    coeffs[..., 6] = 1.

    ##up = up.reshape(bs, *self.coord_dims)

    rhs = torch.zeros(bs, *coord_dims)

    #iv
    x_grid = torch.linspace(0, 2*np.pi, coord_dims[0])
    y_grid = torch.linspace(0, 2*np.pi, coord_dims[1])
    z_grid = torch.linspace(0, 2*np.pi, coord_dims[2])

    #xy_steps = x_steps.view(-1, 1).expand(-1, coord_dims[1]-1, coord_dims[2]-1 )
    #yz_steps = y_steps.view(-1,1).expand(coord_dims[0]-1, -1, coord_dims[2]-1)
    #xz_steps = z_steps.view(-1,1).expand(coord_dims[0]-1, coord_dims[1]-1, -1)

    yz = torch.meshgrid(y_grid,z_grid)
    xz = torch.meshgrid(x_grid[1:],z_grid)
    xy = torch.meshgrid(x_grid[1:],y_grid[1:])

    yz_end = torch.meshgrid(y_grid[1:],z_grid[1:])
    xz_end = torch.meshgrid(x_grid[1:-1],z_grid[1:])
    #print(xz_end[0].shape, xz_end[1].shape)
    xy_end = torch.meshgrid(x_grid[1:-1],y_grid[1:-1])
    #print(xy_end[0].shape, xy_end[1].shape)

    iv0 = torch.sin(yz[0] + yz[1]).reshape(-1)
    iv1 = torch.sin(xz[0] + xz[1]).reshape(-1)
    iv2 = torch.sin(xy[0] + xy[1]).reshape(-1)

    iv3 = torch.sin(yz_end[0] + yz_end[1]).reshape(-1)
    iv4 = torch.sin(xz_end[0] + xz_end[1]).reshape(-1)
    iv5 = torch.sin(xy_end[0] + xy_end[1]).reshape(-1)
    
    iv_rhs = torch.cat([iv0,iv1, iv2, iv3, iv4, iv5], dim =0)

    u0,_,eps = pde(coeffs, rhs, iv_rhs, steps_list)
    
    print(eps.max())
    print(u0.shape)
    u0 = u0.reshape(*coord_dims)
    return u0

#%%
u0=solve()
# %%

#plot = plt.pcolormesh(u0, cmap='RdBu', shading='gouraud')
#plot = plt.pcolormesh(u0, cmap='RdBu', shading='gouraud')
#plot = plt.pcolormesh(u0, cmap='viridis', shading='gouraud')

# %%
print(u0.shape)

# %%
