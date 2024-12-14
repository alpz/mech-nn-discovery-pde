#%%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

torch.autograd.set_detect_anomaly(True)

from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

#from solver.pde_layer import PDEINDLayerEPS
#from solver.multigrid import MultigridLayer2 as MultigridLayer
from solver.multigrid import MultigridLayer
#from solver.ode_layer import ODEINDLayer
#import discovery.basis as B
import ipdb
import extras.logger as logger
import os

from scipy.special import logit
import torch.nn.functional as F
from tqdm import tqdm
import discovery.plot as P

#cuda=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solve():
    bs = 1
    coord_dims = (16,16)
    #coord_dims = (32,32)
    #coord_dims = (128,128)
    n_ind_dim=1
    #iv_list = [(0,0, [0,0],[0,coord_dims[1]-2]), 
    #            (1,0, [1,0], [coord_dims[0]-1, 0]), 
    #            #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
    #            (0,0, [coord_dims[0]-1,1],[coord_dims[0]-1,coord_dims[1]-2]), 
    #            #(1,2, [0,0], [self.coord_dims[0]-1, 0]),
    #            #(1,3, [0,0], [self.coord_dims[0]-1, 0])
    #            (1,0, [0,coord_dims[1]-1], [coord_dims[0]-1, coord_dims[1]-1])
    #            ]

    iv_list = [lambda nx, ny: (0,0, [0,0],[0,ny-2]), 
               lambda nx, ny: (1,0, [1,0], [nx-1, 0]), 
                #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
               lambda nx, ny: (0,0, [nx-1,1],[nx-1,ny-2]), 
                #(1,2, [0,0], [self.coord_dims[0]-1, 0]),
                #(1,3, [0,0], [self.coord_dims[0]-1, 0])
               lambda nx,ny: (1,0, [0,ny-1], [nx-1, ny-1])
                ]

    #pde = PDEINDLayerEPS(bs=bs, coord_dims=coord_dims, order=2, n_ind_dim=n_ind_dim, n_iv=1, 
    pde = MultigridLayer(bs=bs, coord_dims=coord_dims, order=2, n_ind_dim=n_ind_dim, n_iv=1, 
                        n_grid=2,
                        init_index_mi_list=iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)


    #t_step_size = 0.1 
    #x_step_size = 0.1

    t_step_size = 2*np.pi/coord_dims[0]
    x_step_size = 2*np.pi/coord_dims[1]
    #self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
    #self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

    steps0 = torch.logit(t_step_size*torch.ones(1,1,1)).to(device)
    steps1 = torch.logit(x_step_size*torch.ones(1,1,1)).to(device)

    steps0 = steps0.expand(-1,n_ind_dim, coord_dims[0]-1)
    steps1 = steps1.expand(-1,n_ind_dim, coord_dims[1]-1)
    steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
    steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
    steps0 = torch.stack([steps0]*bs,dim=0)
    steps1 = torch.stack([steps1]*bs,dim=0)
    steps_list = [steps0, steps1]


    coeffs = torch.zeros((bs, n_ind_dim, pde.grid_size, pde.n_orders), device=device)
    #u, u_t, u_x, u_tt, u_xx
    #u_tt + u_xx = 0
    coeffs[..., 0] = 0.
    coeffs[..., 1] = 0.
    coeffs[..., 2] = 0.
    coeffs[..., 3] = 1.
    #u_x
    coeffs[..., 4] = 1.

    #up = up.reshape(bs, *self.coord_dims)

    #rhs = torch.zeros(bs, n_ind_dim, *coord_dims)
    rhs = torch.zeros(bs, *coord_dims, device=device)
    rhs = nn.Parameter(rhs)
    coeffs = nn.Parameter(coeffs)

    #iv
    x_steps = torch.linspace(0, 2*np.pi, coord_dims[0], device=device)
    y_steps = torch.linspace(0, 2*np.pi, coord_dims[1], device=device)

    x_bc = torch.sin(x_steps) 
    y_bc = torch.sin(y_steps) 
    
    iv0 = x_bc[:-1]
    iv1 = y_bc[1:]
    iv2 = x_bc[1:-1]
    iv3 = y_bc[:]

    #print('iv0', iv0.shape)
    iv_rhs = torch.cat([iv0,iv1, iv2, iv3], dim =-1).to(device)
    iv_rhs = torch.stack([iv_rhs]*bs,dim=0)
    #print('ivrhs', iv_rhs.shape)

    u0,_,eps,out = pde(coeffs, rhs, iv_rhs, steps_list)
    
    #print(eps.max())
    #print(u0.shape)
    u0 = u0.reshape(bs,*coord_dims)

    #l = u0.pow(2).sum()
    #l = u0.mean()
    #l.backward()

    print('mem after',torch.cuda.mem_get_info(), (torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/1e9)
    #u0 = u0.reshape(1,8,8)
    return u0,out

#%%
u0,out=solve()
# %%

u0 = u0.detach().cpu().numpy()
#plot = plt.pcolormesh(u0, cmap='RdBu', shading='gouraud')
#plot = plt.pcolormesh(u0, cmap='RdBu', shading='gouraud')
plot = plt.pcolormesh(u0[0], cmap='viridis', shading='gouraud')

# %%
deltaH = out['deltaH'].cpu().numpy()
deltaup = out['deltaup'].cpu().numpy()
delta = out['delta'].cpu().numpy()
i=0

# %%
plot = plt.pcolormesh(deltaH[:,:,i], cmap='viridis', shading='gouraud')

# %%
plot = plt.pcolormesh(delta[:,:,i], cmap='viridis', shading='gouraud')
# %%
plot = plt.pcolormesh(deltaup[:,:,i], cmap='viridis', shading='gouraud')

# %%
plot = plt.pcolormesh(u0[1], cmap='viridis', shading='gouraud')

# %%
u0
# %%
out['deltaH'].shape
# %%
