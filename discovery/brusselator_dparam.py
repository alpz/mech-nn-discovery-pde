#%%
from scipy.io import loadmat

from config import PDEConfig
import os
import numpy as np

import torch
import torch.nn as nn
torch.manual_seed(10)
from torch.nn.parameter import Parameter
import numpy as np

#import matplotlib.pyplot as plt
import torch

import torch.optim as optim

from solver.multigrid import MultigridLayer

from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

#from solver.pde_layer import PDEINDLayerEPS
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


log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

DBL=True
dtype = torch.float64 if DBL else torch.float32
#STEP = 0.001
cuda=True
#T = 2000
#n_step_per_batch = T
#solver_dim=(10,256)
solver_dim=(32,32,32)
#solver_dim=(50,64)
#solver_dim=(32,48)
n_grid=3
batch_size= 4
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1

noise =False

L.info(f'Solver dim {solver_dim} ')


class BrusselatorDataset(Dataset):
    def __init__(self, solver_dim=(32,32)):
        #self.n_step_per_batch=n_step_per_batch
        #self.n_step=n_step

        self.down_sample = 1

        #129,2,64,64
        data=np.load(os.path.join(PDEConfig.brusselator_dir, 'brusselator_01_1en3.npy'))
        print('brusselator shape', data.shape)

        #print(data.keys())
        #t = torch.tensor(np.array(data['t'])).squeeze()
        #x = torch.tensor(np.array(data['x'])).squeeze()

        #self._t = t
        #self._x = x

        #self.t_step = t[1] - t[0]
        #self.x_step = x[1] - x[0]
        self.t_step = 0.01 
        self.x_step = 1
        self.y_step = 1

        #self.t = t.unsqueeze(1).expand(-1, x.shape[0])
        #self.x = x.unsqueeze(0).expand(t.shape[0],-1)

        #print('t x', self.t.shape, self.x.shape)

        ##self.t_subsample = 10
        ##self.x_subsample = 1

        ##self.t_subsample = 32 #10
        ##self.x_subsample = 32 #256


        ##L.info(f'subsample {self.t_subsample}, {self.x_subsample} ')
        ##self.t_subsample =50
        ##self.x_subsample =64

        #print(self.t.shape)
        #print(self.x.shape)
        #print(data['usol'].shape)

        #data = np.real(data['usol'])

        if noise:
            print('adding noise')
            rmse = mean_squared_error(data, np.zeros(data.shape), squared=False)
            # add 20% noise (note the impact on derivatives depends on step size...)
            data = data + np.random.normal(0, rmse / 5.0, data.shape) 

        #permute time, x
        data = torch.tensor(data, dtype=dtype).permute(1,0,2,3) 
        self.u_data = data[0]
        self.v_data = data[1]
        print('u,v ', self.u_data.shape, self.v_data.shape)

        self.data_dim = self.u_data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1
        num_x_idx = self.data_dim[1] #- self.solver_dim[1] + 1
        num_y_idx = self.data_dim[2] #- self.solver_dim[1] + 1


        self.num_t_idx = num_t_idx//solver_dim[0]  #+ 1
        self.num_x_idx = num_x_idx//solver_dim[1]  #+ 1
        self.num_y_idx = num_y_idx//solver_dim[2]  #+ 1

        #if self.t_subsample < self.solver_dim[0]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[0]//self.t_subsample
        #if self.x_subsample < self.solver_dim[1]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[1]//self.x_subsample

        #self.length = self.num_t_idx*self.num_x_idx
        self.length = self.num_t_idx*self.num_x_idx*self.num_y_idx

        ##mask
        #mask = torch.rand_like(self.data)
        ##keep only 80% of data
        #mask = (mask>0.5).double()
        
        #self.data = self.data*mask
        #self.mask = mask
        


    def __len__(self):
        return self.length #self.x_train.shape[0]

    def __getitem__(self, idx):
        #return self.data, self.t, self.x
        ##t_idx = idx//self.num_x_idx
        ##x_idx = idx - t_idx*self.num_x_idx
        (t_idx, x_idx,y_idx) = np.unravel_index(idx, (self.num_t_idx, 
                                                      self.num_x_idx,
                                                      self.num_y_idx))

        t_idx = t_idx*solver_dim[0]
        x_idx = x_idx*solver_dim[1]
        y_idx = y_idx*solver_dim[2]


        t_step = solver_dim[0]
        x_step = solver_dim[1]
        y_step = solver_dim[2]

        u_data = self.u_data[t_idx:t_idx+t_step, 
                             x_idx:x_idx+x_step,
                             y_idx:y_idx+y_step]
        v_data = self.v_data[t_idx:t_idx+t_step, 
                             x_idx:x_idx+x_step,
                             y_idx:y_idx+y_step]


        return u_data, v_data

#%%

ds = BrusselatorDataset(solver_dim=solver_dim)#.generate()

#
##%%
#import matplotlib.pyplot as plt
#u = ds.data.cpu().numpy().T
#t = ds._t.cpu().numpy()
#x = ds._x.cpu().numpy()
#plt.figure(figsize=(10, 4))
#plt.subplot(1, 2, 1)
#plt.pcolormesh(t, x, u)
#plt.xlabel('t', fontsize=16)
#plt.ylabel('x', fontsize=16)
#plt.title(r'$u(x, t)$', fontsize=16)


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
        self.iv_list = [
                        #t=0
                        lambda nt, nx, ny: (0,0, [0,0,0],[0,nx-1, ny-1]), 
                        #nx = 0
                        lambda nt, nx, ny: (1,0, [1,0,0],[nt-1,0, ny-1]), 
                        #ny = 0
                        lambda nt, nx, ny: (2,0, [1,1,0],[nt-1,nx-1, 0]), 

                        #nx = endx
                        lambda nt, nx, ny: (1,0, [1,nx-1,1],[nt-1,nx-1, ny-1]), 
                        lambda nt, nx, ny: (2,0, [1,1,ny-1],[nt-1, nx-2, ny-1]), 

                        ]


        #self.iv_len = self.coord_dims[1]-1 + self.coord_dims[0]-1 + self.coord_dims[1]-2 + self.coord_dims[0]
        #self.iv_len = self.coord_dims[1]-1 + self.coord_dims[0]-1 + self.coord_dims[0]
        #print('iv len', self.iv_len)

        self.n_patches_t = 1 #ds.data.shape[0]//self.coord_dims[0]
        self.n_patches_x = 1 #ds.data.shape[1]//self.coord_dims[1]
        self.n_patches = 1 #self.n_patches_t*self.n_patches_x
        print('num patches ', self.n_patches)


        self.pde = MultigridLayer(bs=2*bs, coord_dims=self.coord_dims, order=2, n_ind_dim=self.n_ind_dim, n_iv=1, 
                        n_grid=n_grid,
                        init_index_mi_list=self.iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders


        self.rnet1 = N.ResNet(out_channels=self.coord_dims[0], in_channels=self.coord_dims[0])
        self.rnet2 = N.ResNet(out_channels=self.coord_dims[0], in_channels=self.coord_dims[2])

        self.param_in = nn.Parameter(torch.randn(1,512))
        self.param_net = nn.Sequential(
            nn.Linear(512, 1024),
            #nn.ELU(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 3),
            #nn.Tanh()
        )
        #self.param_net_out = nn.Linear(1024, 3)

        self.param_in2 = nn.Parameter(torch.randn(1,512))
        self.param_net2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 2),
            #nn.Tanh()
        )


        self.t_step_size = steps[0]
        self.x_step_size = 0.5 #steps[1]
        self.y_step_size = 0.5 #steps[2]
        print('steps ', steps)
        #self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,1))
        self.steps2 = torch.logit(self.y_step_size*torch.ones(1,1))

        self.steps0 = nn.Parameter(self.steps0)
        self.steps1 = nn.Parameter(self.steps1)
        self.steps2 = nn.Parameter(self.steps2)


        #up_coeffs = torch.randn((1, 1, self.num_multiindex), dtype=dtype)
        #self.up_coeffs = nn.Parameter(up_coeffs)

        #self.stepsup0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.stepsup1 = torch.logit(self.x_step_size*torch.ones(2,self.coord_dims[1]-1))

    def get_params(self):
        #params = self.param_net_out(self.param_net(self.param_in))
        #params2 =self.param_net2_out(self.param_net2(self.param_in2))

        u_params = (self.param_net(self.param_in)).squeeze()
        v_params =(self.param_net2(self.param_in2)).squeeze()
        #params = params.reshape(-1,1,2, 3)
        #params = params.reshape(-1,1,2, 2)
        return u_params, v_params

    def get_iv(self, u):
        bs = u.shape[0]
        u1 = u[:,0, :self.coord_dims[1], :self.coord_dims[2]].reshape(bs, -1)
        u2 = u[:,1:self.coord_dims[0], 0, :self.coord_dims[2]].reshape(bs, -1)
        u3 = u[:,1:self.coord_dims[0], 1:self.coord_dims[1], 0].reshape(bs, -1)

        u4 = u[:,1:self.coord_dims[0], -1, 1:self.coord_dims[2]].reshape(bs, -1)
        u5 = u[:,1:self.coord_dims[0], 1:self.coord_dims[1]-1, -1].reshape(bs, -1)

        ub = torch.cat([u1,u2,u3,u4,u5], dim=-1)

        return ub

    def solve(self, u, v, up, vp, params_u, params_v):
        bs = u.shape[0]

        steps0 = self.steps0.type_as(u).expand(2*self.bs, self.coord_dims[0]-1)
        steps1 = self.steps1.type_as(u).expand(2*self.bs, self.coord_dims[1]-1)
        steps2 = self.steps2.type_as(u).expand(2*self.bs, self.coord_dims[2]-1)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.2)
        steps1 = 2*torch.sigmoid(steps1).clip(min=0.005, max=0.55)
        steps2 = 2*torch.sigmoid(steps1).clip(min=0.005, max=0.55)

        steps_list = [steps0, steps1, steps2]

        upi = up.reshape(bs, *self.coord_dims)
        vpi = vp.reshape(bs, *self.coord_dims)
        #upi = upi + up2.reshape(bs, *self.coord_dims)
        #upi = upi/2
        iv_u_rhs = self.get_iv(upi)
        iv_v_rhs = self.get_iv(vpi)


        #basis = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)

        #p = (basis*params[...,0,:]).sum(dim=-1)
        #q = (basis2*params[...,1,:]).sum(dim=-1)


        coeffs_u = torch.zeros((bs, self.pde.grid_size, self.pde.n_orders), device=u.device)
        coeffs_v = torch.zeros((bs, self.pde.grid_size, self.pde.n_orders), device=u.device)

        #u, u_t, u_x, u_y, u_tt, u_xx, u_yy
        #(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)
        #u
        coeffs_u[..., 0] = -1-params_u[0]
        coeffs_v[..., 0] = params_v[0]
        #u_t
        coeffs_u[..., 1] = 1.
        coeffs_v[..., 1] = 1.
        #u_xx
        coeffs_u[..., 5] = params_u[1]
        coeffs_v[..., 5] = params_v[1]
        #u_yy
        coeffs_u[..., 6] = params_u[1]
        coeffs_v[..., 6] = params_v[1]

        #up = up.reshape(bs, *self.coord_dims)

        #rhs_u = torch.zeros(bs, *self.coord_dims, device=u.device)
        rhs_u = params_u[2] + vp*up.pow(2)
        #rhs_v = torch.zeros(bs, *self.coord_dims, device=u.device)
        rhs_v = -vp*up.pow(2)

        coeffs = torch.stack([coeffs_u, coeffs_v], dim=0)
        rhs = torch.stack([rhs_u, rhs_v], dim=0)
        iv_rhs = torch.stack([iv_u_rhs, iv_v_rhs], dim=0)

        u0,_,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)
        #u0_list.append(u0)
        #eps_list.append(eps)

        #u0 = torch.stack(u0_list, dim=1)
        #eps = torch.stack(eps_list, dim=1).max()

        u0 = u0.reshape(2, -1)
        u = u0[0]
        v = u0[1]

        return u, v
    
    def forward(self, u, v):
        bs = u.shape[0]

        # u batch, time, x, y
        up = self.rnet1(u)
        vp = self.rnet2(v)

        params = self.get_params()

        u0, v0 = self.solve(u,v, up, vp, params[0], params[1])

        return u0, v0,up,vp, params
        #return u0, up,eps, params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step, ds.x_step, ds.y_step), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum =0.9)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xu, xv = model.get_params()
    xu = xu.squeeze().detach().cpu().numpy()
    xv = xv.squeeze().detach().cpu().numpy()

    L.info(f'param u\n{xu}')
    L.info(f'param v\n{xv}')
    
    #print(params)
    return xu, xv
    #return code


def train():
    """Optimize and threshold cycle"""

    optimize()


def optimize(nepoch=5000):
    #with tqdm(total=nepoch) as pbar:

    params=print_eq()
    L.info(f'parameters\n{params}')
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        u_losses = []
        v_losses = []
        var_u_losses = []
        var_v_losses = []
        losses = []
        total_loss = 0
        for i, batch_in in enumerate(tqdm(train_loader)):
        #for i, batch_in in enumerate((train_loader)):
            optimizer.zero_grad()
            batch_u, batch_v = batch_in[0], batch_in[1]
            batch_u = batch_u.double().to(device)
            batch_v = batch_v.double().to(device)

            data_shape = batch_u.shape

            #optimizer.zero_grad()
            #x0, steps, eps, var,xi = model(index, batch_in)
            u, v, var_u, var_v, params = model(batch_u, batch_v)


            bs = batch_u.shape[0]
            u = u.reshape(bs, -1)
            v = v.reshape(bs, -1)
            batch_u = batch_u.reshape(bs, -1)
            batch_v = batch_v.reshape(bs, -1)
            var_u =var_u.reshape(bs, -1)
            var_v =var_v.reshape(bs, -1)

            u_loss = (u- batch_u).abs().mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)
            v_loss = (v- batch_v).abs().mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)
            var_u_loss = (var_u- batch_u).abs().mean(dim=-1)
            var_v_loss = (var_v- batch_u).abs().mean(dim=-1)
            #loss = x_loss + var_loss + time_loss
            #param_loss = params.abs()
            #loss = x_loss.mean() + var_loss.mean() #+ 0.01*param_loss.mean()
            #loss = x_loss.mean() + var_loss.mean() + var2_loss.mean() + 0.0001*param_loss.mean()
            #loss = 2*x_loss.mean() + var_loss.mean() + var2_loss.mean() +  0.001*param_loss.mean()
            loss = u_loss.mean() +  v_loss.mean() + var_u_loss.mean() + var_v_loss.mean()
                    #+  0.005*param_loss.mean()

            u_losses.append(u_loss.mean().item())
            v_losses.append(v_loss.mean().item())
            #var_losses.append(var_loss + var2_loss)
            var_u_losses.append(var_u_loss.mean().item())
            var_v_losses.append(var_v_loss.mean().item())
            #var_losses.append(var_loss )
            losses.append(loss)
            total_loss = total_loss + loss
            

            loss.backward()
            optimizer.step()


            #xi = xi.detach().cpu().numpy()
            #alpha = alpha.squeeze().item() #.detach().cpu().numpy()
            #beta = beta.squeeze().item()
        _u_loss = torch.tensor(u_losses).mean().item()
        _v_loss = torch.tensor(v_losses).mean().item()
        _var_u_loss = torch.cat(var_u_losses).mean().item()
        _var_v_loss = torch.cat(var_v_losses).mean().item()

        mean_loss = torch.tensor(losses).mean()

        print_eq()
        #L.info(f'parameters\n{params}')
            #pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E}  xloss {x_loss:.3E} max eps {meps}\n')
        #print(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E}  xloss {_x_loss:.3E} vloss {_v_loss:.3E} max eps {meps}\n')
        L.info(f'run {run_id} epoch {epoch}, loss {mean_loss:.3E}  \
               uloss {_u_loss:.3E} vloss {_v_loss:.3E} \
               var_u_loss {_var_u_loss:.3E} var_v_loss {_var_v_loss:.3E} ')


if __name__ == "__main__":
    train()

    print_eq()

# %%
