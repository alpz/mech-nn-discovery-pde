#%%
from scipy.io import loadmat

from config import PDEConfig
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

#import matplotlib.pyplot as plt
import torch

import torch.optim as optim

#from solver.multigrid import MultigridLayer

from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

#from solver.pde_layer import PDEINDLayerEPS

from solver.pde_layer_dense import PDEDenseLayer
#from solver.ode_layer import ODEINDLayer
#import discovery.basis as B
import ipdb
import extras.logger as logger
import os

from scipy.special import logit
import torch.nn.functional as F
from tqdm import tqdm
#import discovery.plot as P
from sklearn.metrics import mean_squared_error
import net as N


torch.manual_seed(10)
log_dir, run_id = create_log_dir(root='logs/burgers')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

DBL=True
dtype = torch.float64 if DBL else torch.float32
cuda=True
solver_dim=(32,32)
#n_grid=3
batch_size= 10

noise =False
noise_factor = 20
frame_drop_prob = 0.0


#l1 regularization coeff
param_l1 = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L.info(f'Burgers viscous ')
L.info(f'Solver dim {solver_dim} ')


#loss_path = os.path.join(log_dir, 'losses.npy')

class BurgersDataset(Dataset):
    def __init__(self, solver_dim=(32,32)):
        self.down_sample = 1

        data=np.load(os.path.join(PDEConfig.burgers_dir, 'burgers_0.1_256.npy'))
        print(data.shape)

        # step = 0.025, domain length = 20
        self.t_step = 0.025
        self.x_step = 20/data.shape[1]

        print('steps t,x ', self.t_step, self.x_step)

        if noise:
            noise_f=noise_factor/100
            L.info(f'adding {noise_f*100}% noise')
            rmse = np.sqrt(np.mean(data**2))
            data = data + np.random.normal(0, rmse *noise_f, data.shape) 

        self.data = torch.tensor(data, dtype=dtype)#.permute(1,0) 
        print('self ', self.data.shape)

        self.data_dim = self.data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1
        num_x_idx = self.data_dim[1] - self.solver_dim[1] + 1


        self.num_t_idx = num_t_idx//solver_dim[0]  #+ 1
        self.num_x_idx = num_x_idx#//solver_dim[1]  #+ 1

        self.length = self.num_t_idx*self.num_x_idx


    def __len__(self):
        return self.length #self.x_train.shape[0]

    def __getitem__(self, idx):
        (t_idx, x_idx) = np.unravel_index(idx, (self.num_t_idx, self.num_x_idx))

        t_idx = t_idx*solver_dim[0]
        x_idx = x_idx#*solver_dim[1]


        t_step = solver_dim[0]
        x_step = solver_dim[1]


        data = self.data[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

        return data, t_idx, x_idx #, t, x, mask

#%%

ds = BurgersDataset(solver_dim=solver_dim)#.generate()
mask = torch.rand(ds.data.shape[0]) > frame_drop_prob
ds.data = ds.data*mask.unsqueeze(1)
data_all = ds.data.to(device)
mask = mask.to(device)


#%%
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True) 



class Model(nn.Module):
    def __init__(self, bs, solver_dim, steps, device=None, **kwargs):
        super().__init__()

        self.order = 2
        # state dimension
        self.bs = bs
        self.device = device
        self.n_iv=1
        self.n_ind_dim = 1
        self.n_dim = 1

        self.param_in = nn.Parameter(torch.randn(1,64))

        self.coord_dims = solver_dim

        #initial and boundary ranges
        #format nx, ny: (0,0, [0,0],[0,ny-2]) -> (coord to vary, u term index, range initial, range final )
        #Term order is u, u_t, u_tt, u_x, u_xx
        #0 is the index for u (Dirichlet conditions)
        self.iv_list = [lambda nx, ny: (0,0, [0,0],[0,ny-2]),  
                        lambda nx,ny: (1,0, [1,0], [nx-1, 0]), 
                        lambda nx,ny: (1,0, [0,ny-1], [nx-1, ny-1])
                        ]

        #self.pde = MultigridLayer(bs=bs, coord_dims=self.coord_dims, order=2, n_ind_dim=self.n_ind_dim, n_iv=1, 
        #                n_grid=n_grid,
        #                init_index_mi_list=self.iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)

        self.pde = PDEDenseLayer(bs=bs, coord_dims=self.coord_dims, order=2, 
                                n_ind_dim=self.n_dim, n_iv=1,
                                init_index_mi_list=self.iv_list,  n_iv_steps=1, 
                                double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders

        self.rnet1_2d = N.ResNet(out_channels=1, in_channels=1)

        class ParamNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Parameter(torch.randn(1,512)) #*(2/(4*512)))
                self.net = nn.Sequential(
                    nn.Linear(512, 1024),
                    #nn.ELU(),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),

                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 5),
                    #nn.Tanh()
                )
            def forward(self):
                y = self.net(self.input)
                return y

        self.param_net_list = nn.ModuleList() 
        for i in range(3):
            self.param_net_list.append(ParamNet())


        self.t_step_size = ds.t_step 
        self.x_step_size = ds.x_step 
        print('steps ', steps)

        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,1))


    def get_params(self):
        params_list = [net() for net in self.param_net_list]
        params = torch.cat(params_list, dim=0)
        return params


    def get_iv_bc(self, u):
        u1 = u[:,0, :self.coord_dims[1]-2+1]
        u2 = u[:, 1:self.coord_dims[0]-1+1:,0]
        u4 = u[:, 0:self.coord_dims[0]-1+1, self.coord_dims[1]-1]

        ub = torch.cat([u1,u2,u4], dim=-1)

        return ub

    def solve(self, u, up, params):
        bs = u.shape[0]

        steps0 = self.steps0.type_as(params).expand(self.bs, self.coord_dims[0]-1)
        steps1 = self.steps1.type_as(params).expand(self.bs, self.coord_dims[1]-1)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.5)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.5)
        steps_list = [steps0, steps1]


        up = up.reshape(bs, self.pde.grid_size)

        #setup initial and boundary values
        upi = up.reshape(bs, *self.coord_dims)
        iv_rhs = self.get_iv_bc(upi)

        basis = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)
        basis2 = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)
        basis3 = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)

        p = (basis*params[...,0,:]).sum(dim=-1)
        q = (basis2*params[...,1,:]).sum(dim=-1)
        r = (basis3*params[...,2,:]).sum(dim=-1)


        coeffs = torch.zeros((bs, self.pde.grid_size, self.pde.n_orders), device=u.device)
        #u, u_t, u_x, u_tt, u_xx
        #u_t
        coeffs[..., 1] = 1.
        #u_x
        coeffs[..., 2] = p
        #u_xx
        coeffs[..., 4] = q

        rhs = r
        u0,_,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)

        u0 = u0.reshape(u.shape)

        return u0, eps
    
    #def make_patches(self, x):
    #    x_patches = x.unfold(1, self.coord_dims[0], self.coord_dims[0]) 
    #    x_patches = x_patches.unfold(2, self.coord_dims[1], self.coord_dims[1]) 
    #    unfold_shape = x_patches.shape

    #    n_patch_t = x_patches.shape[1]
    #    n_patch_x = x_patches.shape[2]

    #    #x_patches = x_patches.reshape(-1, n_patch_t*n_patch_x, self.pde.grid_size)
    #    x_patches = x_patches.contiguous().view(-1, n_patch_t*n_patch_x, self.pde.grid_size)

    #    return x_patches, unfold_shape

    #def join_patches(self, patches, unfold_shape):
    #    # Reshape back
    #    patches= patches.view(unfold_shape)
    #    n_t = unfold_shape[1] * unfold_shape[-2]
    #    n_x = unfold_shape[2] * unfold_shape[-1]
    #    merged = patches.permute(0,1,3,2,4).contiguous()
    #    merged = merged.view(-1, n_t, n_x)

    #    return merged

    def forward(self, u, t_idx, x_idx):
        bs = u.shape[0]
        ts = u.shape[1]
        #up = self.data_net(u)
        #up = up.reshape(bs, self.pde.grid_size)
        #cin = torch.stack([u,t,x], dim=1)
        #cin = u.unsqueeze(1) #torch.stack([u,t,x], dim=1)
        #cin = u
        #print(cin.shape)

        #up = self.data_conv2d(cin).squeeze(1)
        #up2 = self.data_conv2d2(cin).squeeze(1)


        #cin = u.reshape(bs*ts, 1, solver_dim[1])
        #cin = u#.float()
        cin = data_all.unsqueeze(0)#.float()
        #print(cin.shape)
        #up = self.rnet1(cin.unsqueeze(1)).squeeze(1)
        #up2 = self.rnet2(cin.unsqueeze(1)).squeeze(1)

        #up = u #self.rnet1_2d(cin.unsqueeze(1)).squeeze(1)
        #up = self.rnet1_2d(cin.unsqueeze(1)).squeeze(1)
        up = self.rnet1_2d(cin.unsqueeze(1)).squeeze(1)

        up_list = []
        for i in range(bs):
            _up = up[:, t_idx[i]:t_idx[i]+solver_dim[0], x_idx[i]:x_idx[i]+solver_dim[1]]
            up_list.append(_up)
        up = torch.cat(up_list, dim=0)

        up = up.reshape(bs, *solver_dim)

        u= u.unsqueeze(1)
        up= up.unsqueeze(1)

        params = self.get_params()

        u0, eps = self.solve(u, up, params)

        u0 = u0.squeeze(1)

        return u0, up,params


model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step, ds.x_step), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum =0.9)

if DBL:
    model = model.double()

model=model.to(device)


basis_text = {}
basis_text[0] = "{0:.4f} u_x + {1:.4f} u*u_x+ {2:.4f} u^2*u_x + {3:.4f} u^3*u_x + {4:.4f} u^4*u_x"
basis_text[1] = "{0:.4f} u_xx + {1:.4f} u*u_xx+ {2:.4f} u^2*u_xx + {3:.4f} u^3*u_xx + {4:.4f} u^4*u_xx"
basis_text[2] = "{0:.4f} + {1:.4f} u+ {2:.4f} u^2 + {3:.4f} u^3 + {4:.4f} u^4"
#basis_text[1] = ['u_xx', 'u u_xx', 'u^2 u_xx', 'u^3 u_xx', 'u^4 u_xx']
#basis_text[2] = ['1', 'u', 'u^2', 'u^3', 'u^4']

def print_eq(stdout=False):
    #print learned equation
    xi = model.get_params()
    params = xi.squeeze().detach().cpu().numpy()


    eq_str = "u_t + " + basis_text[0].format(*tuple(params[0])) + "\n" \
            + basis_text[1].format(*tuple(params[1])) + "\n"  \
            + " = "+  basis_text[2].format(*tuple(params[2])) \

    #print(params)
    #return params
    return eq_str
    #return code


def train(nepoch=5000):
    #with tqdm(total=nepoch) as pbar:

    eq_str=print_eq()
    L.info(f'Initial \n{eq_str}\n')
    loss_list = []
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        x_losses = []; var_losses = []; losses = []

        for i, batch_x in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_in, t_idx, x_idx = batch_x
            batch_in = batch_in.double().to(device)
            t_idx = t_idx.to(device)
            x_idx = x_idx.to(device)

            x0, var, params = model(batch_in, t_idx, x_idx)

            mask_list = []
            for i in range(batch_in.shape[0]):
                _mask = mask[t_idx[i]:t_idx[i]+1] #, x_idx[i]:x_idx[i]+solver_dim[1]]
                mask_list.append(_mask.unsqueeze(1))
            dmask = torch.stack(mask_list, dim=0)

            bs = batch_in.shape[0]
            x0 = x0.reshape(bs, -1)
            batch_in = batch_in.reshape(bs, -1)

            var =var.reshape(bs, -1)
            x_loss = (x0*dmask- batch_in).abs().mean(dim=-1)

            var_loss = (var- x0).abs().mean(dim=-1)

            param_loss = params.abs()
            loss = x_loss.mean() + var_loss.mean() + param_l1*param_loss.mean()

            x_losses.append(x_loss.mean().item())
            var_losses.append(var_loss.mean().item())
            losses.append(loss.mean().item())
            
            loss.backward()
            optimizer.step()

            del loss,x_loss, var_loss, param_loss,params

        _x_loss = torch.tensor(x_losses).mean()
        _v_loss = torch.tensor(var_losses).mean()
        mean_loss = torch.tensor(losses).mean()

        loss_list.append(_x_loss.detach().cpu().numpy())

        eq_str=print_eq()
        L.info(f'Learned \n{eq_str}\n')

        L.info(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E} xloss {_x_loss.item():.3E} vloss {_v_loss.item():.3E}')


if __name__ == "__main__":
    train()
    print_eq()
