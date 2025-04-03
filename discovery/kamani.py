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
#import fno_net as FNO

from solver.pde_layer_dense import PDEDenseLayer


log_dir, run_id = create_log_dir(root='logs_kamani')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

DBL=True
dtype = torch.float64 if DBL else torch.float32
#STEP = 0.001
cuda=True
#T = 2000
#n_step_per_batch = T
#solver_dim=(10,256)
solver_dim=(100,)
#solver_dim=(64,64,64)
#solver_dim=(50,64)
#solver_dim=(32,48)
#n_grid=3
batch_size= 8
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1

noise =False
downsample=2

L.info(f'Kamani equation Rheology')
L.info(f'Solver dim {solver_dim} ')


class KamaniDataset(Dataset):
    def __init__(self, solver_dim=(100,)):
        #self.n_step_per_batch=n_step_per_batch
        #self.n_step=n_step

        self.down_sample = 1

        #129,2,64,64
        #data=np.load(os.path.join(PDEConfig.brusselator_dir, 'brusselator_01_1en3.npy'))
        #u_data=np.load(os.path.join(PDEConfig.ginzburg_dir, 'Ar.npy'))
        #v_data=np.load(os.path.join(PDEConfig.ginzburg_dir, 'Ai.npy'))

        #u_data=np.load(os.path.join(PDEConfig.ginzburg_dir, '256', 'Ar_256_300.npy'))
        #v_data=np.load(os.path.join(PDEConfig.ginzburg_dir, '256', 'Ai_256_300.npy'))
        #uv_data=np.load(os.path.join(PDEConfig.ginzburg_dir, '256', 'A2_256_300.npy'))

        u_data=np.load(os.path.join(PDEConfig.rheology_dir, 'kamani_traj.npy'))
        t_data =np.load(os.path.join(PDEConfig.rheology_dir, 'times.npy'))

        shear_amplitude = np.logspace(-3, 1, 500)

        L.info(f'data shape {u_data.shape}')

        #todo
        self.t_step = t_data[1]-t_data[0]
        L.info(f'time step {self.t_step}')

        u_data = torch.tensor(u_data, dtype=dtype)#.permute(1,0,2,3) 
        t_data = torch.tensor(t_data, dtype=dtype)#.permute(1,0,2,3) 
        amp_data = torch.tensor(shear_amplitude, dtype=dtype)


        self.data_dim = u_data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1

        self.num_t_idx = num_t_idx//solver_dim[0]  #+ 1
        self.num_amp_idx = amp_data.shape[0]

        #if self.t_subsample < self.solver_dim[0]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[0]//self.t_subsample
        #if self.x_subsample < self.solver_dim[1]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[1]//self.x_subsample

        #self.length = self.num_t_idx*self.num_x_idx

        self.u_data = u_data
        self.amp_data = amp_data
        self.t_data = t_data

        self.length = self.num_t_idx*self.num_amp_idx

    def __len__(self):
        return self.length #self.x_train.shape[0]

    def __getitem__(self, idx):
        (amp_idx, t_idx) = np.unravel_index(idx, (self.num_amp_idx, self.num_t_idx))

        t_idx = t_idx*solver_dim[0]
        assert(t_idx <= self.data_dim[1])

        t_step = solver_dim[0]

        u_data = self.u_data[t_idx:t_idx+t_step, amp_idx] 

        t = self.t_data[t_idx:t_idx+t_step]
        amp = self.amp_data[amp_idx]

        # shear rate and derivatives
        shear = amp*torch.sin(t)
        shear_d = amp*torch.cos(t)
        shear_dd = -amp*torch.sin(t)

        return t, u_data, shear, shear_d, shear_dd

#%%

ds = KamaniDataset(solver_dim=solver_dim)#.generate()

# %%
ds.amp_data

# %%
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

        #self.param_in = nn.Parameter(torch.randn(1,64))

        self.coord_dims = solver_dim
        self.n_time = solver_dim[0]

        self.iv_list = [
                        #t=0
                        lambda nt: (0,0, [0],[0]), 
                        #lambda nt: (0,0, [nt-1],[nt-1]), 
                        ]

        #self.pde = MultigridLayer(bs=bs, coord_dims=self.coord_dims, order=2, n_ind_dim=self.n_ind_dim, n_iv=1, 
        #                n_grid=n_grid, evolution=False,
        #                init_index_mi_list=self.iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)

        self.ode = PDEDenseLayer(bs=bs, coord_dims=self.coord_dims, order=2, 
                                n_ind_dim=self.n_dim, n_iv=1,
                                init_index_mi_list=self.iv_list,  n_iv_steps=1, 
                                double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.ode.n_orders


        #self.params_u = nn.Parameter(0.5*torch.randn(1,4))
        class ParamNet(nn.Module):
            def __init__(self, n_out=10):
                super().__init__()
                self.input = nn.Parameter(torch.randn(1,512)) #*(2/(4*512)))
                self.net = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, n_out),
                )
            def forward(self):
                y = self.net(self.input)
                return y

        class TransformNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(self.n_time, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.n_time),
                )
            def forward(self, x):
                y = self.net(x)
                return y

        self.param_net = ParamNet(n_out=9)
        self.param_exp_net = ParamNet(n_out=6)
        self.transform = TransformNet()

        self.t_step_size = ds.t_step #0.05*downsample #steps[0]
        print('step size ', self.t_step_size)
        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,1))

    def get_params(self):
        params = self.params_net()
        params_exp = self.params_exp_net()

        #[-2,2]
        params_exp = 2*torch.tanh(params_exp)
        return [params, params_exp]

    def get_iv(self, u):
        bs = u.shape[0]

        return u[:, 0]

    def get_steps(self, u, t):
        steps0 = self.steps0.type_as(u).expand(self.bs, self.coord_dims[0]-1)
        steps0 = torch.sigmoid(steps0)

        steps_list = [steps0]

        return steps_list

    def solve(self, u, up, vp, params_list, shear_list, steps_list):
        bs = u.shape[0]

        ui = u.reshape(bs, *self.coord_dims)
        upi = up.reshape(bs, *self.coord_dims)
        iv_rhs_u = self.get_iv(upi)


        up = up.reshape(bs, self.n_time)
        u = u.reshape(bs, self.n_time)
        
        up0 = up[:,0]
        #up1 = up[:,1]
        #up2 = up[:,2]

        vp0 = vp[:,0]
        #vp1 = vp[:,1]
        #vp2 = vp[:,2]

        ss = shear_list[0]
        ss_d = shear_list[1]
        ss_dd = shear_list[2]

        pr = params_list[0].reshape(3, -1)
        er = params_list[1].reshape(3, -1)

        #basis1 = torch.stack([torch.ones_like(up0), up0, up0.pow(2), up0.pow(3), vp0, vp0.pow(2), vp0.pow(3), up0*vp0, up0.pow(2)*vp0, up0*vp0.pow(2)], dim=-1)
        basis0 = torch.stack([pr[0,0]*torch.ones_like(ss_d), pr[0,1]*ss_d.abs().pow(er[0,0]), pr[0,2]*ss_d.abs().pow(er[0,1])  ], dim=-1)
        basis1 = torch.stack([pr[1,0]*torch.ones_like(ss_d), pr[1,1]*ss_d.abs().pow(er[1,0]), pr[1,2]*ss_d.abs().pow(er[1,1])  ], dim=-1)
        basis2 = torch.stack([pr[2,0]*torch.ones_like(ss_d), pr[2,1]*ss_d.abs().pow(er[2,0]), pr[2,2]*ss_d.abs().pow(er[2,1])  ], dim=-1)

        #basis2 = torch.stack([torch.ones_like(up1), up1, up1.pow(2), up1.pow(3), vp1, vp1.pow(2), vp1.pow(3), up1*vp1, up1.pow(2)*vp1, up1*vp1.pow(2)], dim=-1)
        #basis3 = torch.stack([torch.ones_like(up2), up2, up2.pow(2), up2.pow(3), vp2, vp2.pow(2), vp2.pow(3), up2*vp2, up2.pow(2)*vp2, up2*vp2.pow(2)], dim=-1)

        p0 = (basis0).sum(dim=-1)
        p1 = (basis1).sum(dim=-1)
        p2 = (basis2).sum(dim=-1)
        #p4 = (basis*params_list[3]).sum(dim=-1)
        #q = (basis2*params[...,1,:]).sum(dim=-1)


        coeffs_u = torch.zeros((bs, self.ode.grid_size, self.ode.n_orders), device=u.device)

        #u
        coeffs_u[..., 0] = 1 #p0 
        #u_t
        coeffs_u[..., 1] = p0 #params_list[3][0]

        rhs_u = p1*ss_d + p2*ss_dd

        u0,_,eps = self.ode(coeffs_u, rhs_u, iv_rhs_u, steps_list)
        u = u0

        return u
    
    def forward(self, t, u, shear, sheard, sheardd):
        bs = u.shape[0]
        ts = solver_dim[0]

        # u batch, time, x, y

        u_in = u_in.reshape(bs*ts, 1, solver_dim[1], solver_dim[2])


        up = self.transform(u_in)

        up = up.reshape(bs,*solver_dim)

        params = self.get_params()
        steps_list = self.get_steps(u, t)

        u0 = self.solve(u,up, params, [shear, sheard, sheardd] , steps_list)

        return { 'u': u0, 'up': up, 'params':params }
        #return u0, up,eps, params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum =0.9)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    tau = 94
    k = 27.93
    n = 0.416
    G = 430
    eta = 23
    true_params = [[tau/G, k/G, eta/G],
                    [tau/G, k/G,0],
                    [tau/G * eta/G, k/G * eta/G, 0]
                    ]
    true_exp = [[-1, n-1], [-1, n-1], [-1, n-1]]

    #print learned params
    params,exps = model.get_params()
    params = params.reshape(3, -1)
    exps = exps.reshape(3, -1)
    params = params.squeeze().detach().cpu().numpy()
    exps = exps.squeeze().detach().cpu().numpy()

    params = params.reshape(3,-1)
    exps = exps.reshape(3,-1)

    L.info(f'param {params}')
    L.info(f'exps {exps}')

    L.info(f'True param {true_params}')
    L.info(f'True exp {true_exp}')
    #print(params)
    #return code


def train():
    """Optimize and threshold cycle"""

    optimize()


def optimize(nepoch=5000):
    #with tqdm(total=nepoch) as pbar:

    print_eq()
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        u_losses = []
        var_u_losses = []
        losses = []
        total_loss = 0
        for i, batch_in in enumerate(tqdm(train_loader)):
        #for i, batch_in in enumerate((train_loader)):
            optimizer.zero_grad()
            batch_t, batch_u, batch_s, batch_sd, batch_sdd = batch_in[0:4]

            batch_u = batch_u.double().to(device)
            batch_t = batch_t.double().to(device)
            batch_s = batch_s.double().to(device)
            batch_sd = batch_sd.double().to(device)
            batch_sdd = batch_sdd.double().to(device)

            data_shape = batch_u.shape

            #optimizer.zero_grad()
            #x0, steps, eps, var,xi = model(index, batch_in)
            ret = model(batch_t, batch_u, batch_s, batch_sd, batch_sdd)

            u = ret['u']
            var_u = ret['up']
            params = ret['params']

            bs = batch_u.shape[0]

            u = u.reshape(bs, -1)
            batch_u = batch_u.reshape(bs, -1)
            var_u =var_u.reshape(bs, -1)

            u_loss = (u- batch_u).abs().mean(dim=-1) 
            var_u_loss = (var_u- u).abs().mean(dim=-1)
            param_loss = torch.stack(params).abs().sum()

            loss = u_loss.mean() +  var_u_loss.mean() 
            #loss = u_loss.mean() +  var_u_loss.mean() 
            #loss = loss + rhs_loss.mean()
            loss = loss +  0.0001*param_loss.mean()

            u_losses.append(u_loss.mean().item())
            var_u_losses.append(var_u_loss.mean().item())
            losses.append(loss.detach().item())
            
            #print('rhs loss ', rhs_loss.mean().item())

            loss.backward()
            optimizer.step()


            del loss,u,u_loss,var_u,var_u_loss,params#, t_params_loss

        _u_loss = torch.tensor(u_losses).mean().item()
        _var_u_loss = torch.tensor(var_u_losses).mean().item()

        mean_loss = torch.tensor(losses).mean().item()

        print_eq()
        #L.info(f'parameters\n{params}')
            #pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E}  xloss {x_loss:.3E} max eps {meps}\n')
        #print(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E}  xloss {_x_loss:.3E} vloss {_v_loss:.3E} max eps {meps}\n')
        L.info(f'run {run_id} epoch {epoch}, loss {mean_loss:.3E}  \
               uloss {_u_loss:.3E} var_u_loss {_var_u_loss:.3E}')
               
if __name__ == "__main__":
    train()

    print_eq()

# %%
