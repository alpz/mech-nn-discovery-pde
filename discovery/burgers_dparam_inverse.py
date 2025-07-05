#%%
from scipy.io import loadmat

from config import PDEConfig
import os
import numpy as np

import torch
torch.manual_seed(10)
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


log_dir, run_id = create_log_dir(root='logs_burgers2')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

DBL=True
dtype = torch.float64 if DBL else torch.float32
#STEP = 0.001
cuda=True
#T = 2000
#n_step_per_batch = T
#solver_dim=(10,256)
solver_dim=(32, 32)
#solver_dim=(50,64)
#solver_dim=(32,48)
n_grid=3
batch_size= 1
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1

noise =False

L.info(f'Burgers viscous ')
L.info(f'Solver dim {solver_dim} ')


loss_path = os.path.join(log_dir, 'losses.npy')

class BurgersDataset(Dataset):
    def __init__(self, solver_dim=(32,32)):
        #self.n_step_per_batch=n_step_per_batch
        #self.n_step=n_step

        self.down_sample = 1

        data=np.load(os.path.join(PDEConfig.burgers_dir, 'burgers_0.1_256.npy'))
        data = data[:32]
        print(data.shape)

        #print(data.keys())
        #t = torch.tensor(np.array(data['t'])).squeeze()
        #x = torch.tensor(np.array(data['x'])).squeeze()

        #self._t = t
        #self._x = x

        self.t_step = 0.025# t[1] - t[0]
        self.x_step = 20/data.shape[1]

        print('steps t,x ', self.t_step, self.x_step)

        #data = np.real(data['usol'])

        if noise:
            noise_f=20/100
            L.info(f'adding {noise_f*100}% noise')
            #rmse = mean_squared_error(u_data, np.zeros(u_data.shape), squared=False)
            rmse = np.sqrt(np.mean(data**2))
            # add 20% noise (note the impact on derivatives depends on step size...)
            data = data + np.random.normal(0, rmse *noise_f, data.shape) 

        #if noise:
        #    print('adding noise')
        #    rmse = mean_squared_error(data, np.zeros(data.shape), squared=False)
        #    # add 20% noise (note the impact on derivatives depends on step size...)
        #    data = data + np.random.normal(0, rmse / 5.0, data.shape) 

        self.data = torch.tensor(data, dtype=dtype)#.permute(1,0) 
        print('self ', self.data.shape)

        self.data_dim = self.data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1
        num_x_idx = self.data_dim[1] #- self.solver_dim[1] + 1


        self.num_t_idx = num_t_idx//solver_dim[0]  #+ 1
        self.num_x_idx = num_x_idx//solver_dim[1]  #+ 1
        #self.num_x_idx = num_x_idx-solver_dim[1]  #+ 1

        #if self.t_subsample < self.solver_dim[0]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[0]//self.t_subsample
        #if self.x_subsample < self.solver_dim[1]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[1]//self.x_subsample

        #self.length = self.num_t_idx*self.num_x_idx
        self.length = self.num_t_idx*self.num_x_idx

        #mask
        #mask = torch.rand_like(self.data)
        ##keep only 80% of data
        #mask = (mask>0.0).double()
        
        #self.data = self.data*mask
        #self.mask = mask
        


    def __len__(self):
        return self.length #self.x_train.shape[0]

    #def __getitem__(self, idx):
    #    #return self.data, self.t, self.x
    #    ##t_idx = idx//self.num_x_idx
    #    ##x_idx = idx - t_idx*self.num_x_idx
    #    (t_idx, x_idx) = np.unravel_index(idx, (self.num_t_idx, self.num_x_idx))

    #    t_idx = t_idx*solver_dim[0]
    #    x_idx = x_idx#*solver_dim[1]


    #    t_step = solver_dim[0]
    #    x_step = solver_dim[1]


    #    #t = self.t[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
    #    #x = self.x[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

    #    data = self.data[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
    #    #mask = self.mask[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

    #    return data[-1], data[0]#, t, x, mask
    #    #return data

    def __getitem__(self, idx):
        #return self.data, self.t, self.x
        ##t_idx = idx//self.num_x_idx
        ##x_idx = idx - t_idx*self.num_x_idx
        (t_idx, x_idx) = np.unravel_index(idx, (self.num_t_idx, self.num_x_idx))

        t_idx = t_idx*solver_dim[0]
        x_idx = x_idx*solver_dim[1]


        t_step = solver_dim[0]
        x_step = solver_dim[1]


        #t = self.t[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
        #x = self.x[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

        data = self.data[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
        #mask = self.mask[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
        ind = torch.linspace(x_idx/256, (x_idx+solver_dim[0])/256, solver_dim[0])

        return data[-1], data[0], ind#, t, x, mask
        #return data

#%%

ds = BurgersDataset(solver_dim=solver_dim)#.generate()

#
#%%
import matplotlib.pyplot as plt
u = ds.data.cpu().numpy().T
t = np.arange(32)
x = np.arange(256)
fig = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.pcolormesh(t, x, u)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title(r'$u(x, t)$', fontsize=16)
#plt.ion()
#plt.show(block=False)
fig.canvas.draw()
fig.canvas.flush_events()
plt.pause(1)
plt.cla()

def do_plot(x, y, z, epoch):
    n = x.shape[0]
    plt.plot(np.arange(n), x, label='learned')
    plt.plot(np.arange(n), y, label='var')
    plt.plot(np.arange(n), z, label='true')

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{log_dir}/fig_inv_{epoch:04d}.png", dpi=300)
    plt.close()


    #fig.canvas.draw()
    #fig.canvas.flush_events()
    #plt.pause(1)
    #plt.cla()

#%%
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False) 
test_loader =DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False) 



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
        self.iv_list = [lambda nx, ny: (0,0, [0,0],[0,ny-2]), 
                        lambda nx,ny: (1,0, [1,0], [nx-1, 0]), 
                        #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
                        #(0,0, [self.coord_dims[0]-1,1],[self.coord_dims[0]-1,self.coord_dims[1]-2]), 
                        #(1,2, [0,0], [self.coord_dims[0]-1, 0]),
                        #(1,3, [0,0], [self.coord_dims[0]-1, 0])
                        lambda nx,ny: (1,0, [0,ny-1], [nx-1, ny-1])
                        ]

        #self.iv_len = self.coord_dims[1]-1 + self.coord_dims[0]-1 + self.coord_dims[1]-2 + self.coord_dims[0]
        #self.iv_len = self.coord_dims[1]-1 + self.coord_dims[0]-1 + self.coord_dims[0]
        #print('iv len', self.iv_len)

        self.n_patches_t = 1 #ds.data.shape[0]//self.coord_dims[0]
        self.n_patches_x = 1 #ds.data.shape[1]//self.coord_dims[1]
        self.n_patches = 1 #self.n_patches_t*self.n_patches_x
        print('num patches ', self.n_patches)


        #self.pde = MultigridLayer(bs=bs, coord_dims=self.coord_dims, order=2, n_ind_dim=self.n_ind_dim, n_iv=1, 
        #                n_grid=n_grid,
        #                init_index_mi_list=self.iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)

        self.pde = PDEDenseLayer(bs=bs, coord_dims=self.coord_dims, order=2, 
                                n_ind_dim=self.n_dim, n_iv=1,
                                init_index_mi_list=self.iv_list,  n_iv_steps=1, 
                                double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders

        #self.iv_out = nn.Linear(13*32, self.iv_len)

        self.rnet1_2d = N.ResNet(out_channels=1, in_channels=1)
        self.rnet2_2d = N.ResNet(out_channels=1, in_channels=1)

        #self.rnet1_1d = N.ResNet1D(out_channels=32, in_channels=1)
        #self.rnet2_1d = N.ResNet1D(out_channels=32, in_channels=32)

        #self.rnet1_2d = N.ResNet2D(out_channels=1, in_channels=1)
        #self.rnet2_2d = N.ResNet2D(out_channels=1, in_channels=1)

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

        class IVNet(nn.Module):
            def __init__(self, out_len=32):
                super().__init__()
                #self.input = nn.Parameter(torch.randn(1,512)) #*(2/(4*512)))
                self.net = nn.Sequential(
                    nn.Linear(32, 1024),
                    #nn.ELU(),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),

                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, out_len),
                    #nn.Tanh()
                )
            def forward(self, x):
                y = self.net(x)
                return y

        #self.param_net= ParamNet()
        #self.param_net2= ParamNet()
        #self.param_net3= ParamNet()
        #self.param_net4= ParamNet()
        #self.param_net5= ParamNet()
        #self.param_net6= ParamNet()
        #self.param_net7= ParamNet()

        self.param_net_list = nn.ModuleList() 
        for i in range(3):
            self.param_net_list.append(ParamNet())

        self.iv_net_list = nn.ModuleList() 
        for out_len in [30,32,32]:
            self.iv_net_list.append(IVNet(out_len))

        #self.data_mlp1 = nn.Sequential(
        #    #nn.Linear(32*32, 1024),
        #    nn.Linear(self.pde.grid_size, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
        #    nn.ReLU(),
        #    #two polynomials, second order
        #    nn.Linear(1024,self.pde.grid_size)
        #)


        #self.data_mlp2 = nn.Sequential(
        #    #nn.Linear(32*32, 1024),
        #    nn.Linear(self.pde.grid_size, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
        #    nn.ReLU(),
        #    #two polynomials, second order
        #    nn.Linear(1024,self.pde.grid_size)
        #)

        self.param_in = nn.Parameter(torch.randn(1,512))
        self.param_net = nn.Sequential(
            nn.Linear(512, 1024),
            #nn.ELU(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 5),
            #nn.Tanh()
        )
        #self.param_net_out = nn.Linear(1024, 3)

        self.param_in2 = nn.Parameter(torch.randn(1,512))
        self.param_net2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 5),
            #nn.Tanh()
        )

        self.param_in3 = nn.Parameter(torch.randn(1,512))
        self.param_net3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 5),
            #nn.Tanh()
        )

        self.data_net = nn.Sequential(
            nn.Linear(32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 32*32),
            #nn.Tanh()
        )

        #self.in_iv = nn.Parameter(torch.randn(1,512))
        #self.iv_mlp = nn.Sequential(
        #    nn.Linear(self.n_patches*self.iv_len, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
        #    nn.ReLU(),
        #    #two polynomials, second order
        #    #nn.Linear(1024, 3*2),
        #    nn.Linear(1024, self.n_patches*self.iv_len),
        #    #nn.Tanh()
        #)


        #self.param_net2_out = nn.Linear(1024, 3)

        #self.param_net_out.weight.data.fill_(0.0)
        #self.param_net2_out.weight.data.fill_(0.0)

        #param_init = torch.randn(3)
        #self.param_net_out.bias.data.fill_(param_init)
        #self.param_net_out.bias = nn.Parameter(0.1*torch.randn(3))
        #self.param_net2_out.bias = nn.Parameter(0.1*torch.randn(3))

        self.t_step_size = ds.t_step #steps[0]
        self.x_step_size = ds.x_step #steps[1]
        print('steps ', steps)
        #self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,1,1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,1,1))

        #self.steps0 = nn.Parameter(self.steps0)
        #self.steps1 = nn.Parameter(self.steps1)


        #up_coeffs = torch.randn((1, 1, self.num_multiindex), dtype=dtype)
        #self.up_coeffs = nn.Parameter(up_coeffs)

        #self.stepsup0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.stepsup1 = torch.logit(self.x_step_size*torch.ones(2,self.coord_dims[1]-1))

    def get_params(self):
        #params_list = [(net()).squeeze() for net in self.param_net_list]
        params_list = [net() for net in self.param_net_list]
        params = torch.cat(params_list, dim=0)
        #params = params.transpose(1,0)
        #return u_params, v_params, w_params, x_params, y_params, z_params

        params[:,:] = 0.
        params[0, 1] = 1.
        params[1,0] = -0.1

        return params

    #def get_params(self):
    #    #params = self.param_net_out(self.param_net(self.param_in))
    #    #params2 =self.param_net2_out(self.param_net2(self.param_in2))

    #    params = (self.param_net(self.param_in))
    #    params2 =(self.param_net2(self.param_in2))
    #    params3 =(self.param_net3(self.param_in3))
    #    #params = params.reshape(-1,1,2, 3)
    #    #params = params.reshape(-1,1,2, 2)
    #    params = torch.stack([params, params2, params3], dim=-2)
    #    return params

    def get_iv(self, u):
        u1 = u[:,0, :self.coord_dims[1]-2+1]
        u2 = u[:, 1:self.coord_dims[0]-1+1:,0]
        #u3 = u[:, self.coord_dims[0]-1, 1:self.coord_dims[1]-2+1]
        u4 = u[:, 0:self.coord_dims[0]-1+1, self.coord_dims[1]-1]

        #u1 = self.iv_net_list[0](u)
        #u2 = self.iv_net_list[1](u)
        #u4 = self.iv_net_list[2](u)

        #ub = torch.cat([u1,u2,u3,u4], dim=-1)
        ub = torch.cat([u1,u2,u4], dim=-1)

        return ub

    def solve_chunks(self, u_patches, up_patches, up2_patches, params):
        bs = up_patches.shape[0]
        n_patches = up_patches.shape[1]
        u0_list = []
        eps_list = []

        steps0 = self.steps0.type_as(params).expand(self.bs, self.n_patches, self.coord_dims[0]-1)
        steps1 = self.steps1.type_as(params).expand(self.bs, self.n_patches, self.coord_dims[1]-1)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.5)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.5)
        steps_list = [steps0, steps1]

        #for i in range(n_patches):
        #for i in tqdm(range(n_patches)):
        u = u_patches#[:, i]
        up = up_patches#[:, i]
        up2 = up2_patches#[:, i]

        up = up.reshape(bs*n_patches, self.pde.grid_size)
        up2 = up2.reshape(bs*n_patches, self.pde.grid_size)
        # solve each chunk
        #can use either u or up for boundary conditions
        #upi = u.reshape(bs, *self.coord_dims)
        #upi = up.reshape(bs*n_patches, *self.coord_dims)
        upi = up.reshape(bs*n_patches, *self.coord_dims)
        #upi = upi + up2.reshape(bs, *self.coord_dims)
        #upi = upi/2
        iv_rhs = self.get_iv(upi)
        #iv_rhs = self.get_iv(u.reshape(bs, solver_dim[1]))
        #iv_rhs = upi + up2.reshape(bs, *self.coord_dims)
        #iv_rhs = iv_rhs.reshape(bs, n_patches*self.iv_len)
        #iv_rhs = self.iv_mlp(iv_rhs)


        basis = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up2), up2, up2**2], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up2), up2, up2**2], dim=-1)
        basis2 = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)
        #basis3 = torch.stack([torch.ones_like(up2), up2, up2**2, up2.pow(3), up2.pow(4)], dim=-1)
        basis3 = torch.stack([torch.ones_like(up), up, up**2, up.pow(3), up.pow(4)], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up), up, up**2], dim=-1)

        #p = (basis*params[...,0,:]).sum(dim=-1)
        #q = (basis2*params[...,1,:]).sum(dim=-1)
        #r = (basis3*params[...,2,:]).sum(dim=-1)

        #print('param', params.shape)

        p = (basis*params[...,0,:]).sum(dim=-1)
        q = (basis2*params[...,1,:]).sum(dim=-1)
        r = (basis3*params[...,2,:]).sum(dim=-1)


        coeffs = torch.zeros((bs*n_patches, self.pde.grid_size, self.pde.n_orders), device=u.device)
        #u, u_t, u_x, u_tt, u_xx
        #coeffs[..., 0] = r
        #u_t
        coeffs[..., 1] = 1.
        #u_x
        coeffs[..., 2] = p
        #u_xx
        coeffs[..., 4] = q

        #up = up.reshape(bs, *self.coord_dims)

        #rhs = torch.zeros(bs*n_patches, *self.coord_dims, device=u.device)
        rhs = r

        u0,_,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)
        #u0_list.append(u0)
        #eps_list.append(eps)

        #u0 = torch.stack(u0_list, dim=1)
        #eps = torch.stack(eps_list, dim=1).max()

        u0 = u0.reshape(up_patches.shape)

        return u0, eps
    
    def make_patches(self, x):
        x_patches = x.unfold(1, self.coord_dims[0], self.coord_dims[0]) 
        x_patches = x_patches.unfold(2, self.coord_dims[1], self.coord_dims[1]) 
        unfold_shape = x_patches.shape

        n_patch_t = x_patches.shape[1]
        n_patch_x = x_patches.shape[2]

        #x_patches = x_patches.reshape(-1, n_patch_t*n_patch_x, self.pde.grid_size)
        x_patches = x_patches.contiguous().view(-1, n_patch_t*n_patch_x, self.pde.grid_size)

        return x_patches, unfold_shape

    def join_patches(self, patches, unfold_shape):
        # Reshape back
        patches= patches.view(unfold_shape)
        n_t = unfold_shape[1] * unfold_shape[-2]
        n_x = unfold_shape[2] * unfold_shape[-1]
        merged = patches.permute(0,1,3,2,4).contiguous()
        merged = merged.view(-1, n_t, n_x)

        return merged

    def forward(self, u, u_idx):
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


        #cin = u.reshape(bs, 1, solver_dim[1])
        d = u.shape[1]


        #cin = u.unsqueeze(1).expand(-1, solver_dim[0], -1) 
        #cin = u.unsqueeze(1).expand(-1, solver_dim[0], -1) 
        cin = u.unsqueeze(1).expand(-1, solver_dim[0], -1) 

        ##cz = torch.zeros(cin.shape[0], cin.shape[1]+32, cin.shape[2]+32, device=u.device, dtype=u.dtype)
        ##cz[:, 16:-16, 16:-16] = cin
        cz = cin
        ##print(cin.shape)
        ##up = self.rnet1(cin.unsqueeze(1)).squeeze(1)
        ##up2 = self.rnet2(cin.unsqueeze(1)).squeeze(1)


        ##up = self.rnet1_2d(cin.unsqueeze(1)).squeeze(1)
        up = self.rnet1_2d(cz.unsqueeze(1)).squeeze(1)
        ##up = up[:, 16:-16, 16:-16]
        up2 = up #self.rnet2_2d(cin.unsqueeze(1)).squeeze(1)

        #up = self.data_conv2d(cin).squeeze(1)
        #up2 = self.data_conv2d2(cin).squeeze(1)

        #up = cin #self.rnet1_1d(cin)#.squeeze(1)
        #up2= cin #self.rnet2_1d(cin)#.squeeze(1)

        #up = self.rnet1_1d(u.unsqueeze(1))#.squeeze(1)
        #up2= up# self.rnet2_1d(cin)#.squeeze(1)

        #up  = up.double()
        #up2  = up2.double()
        #up =self.rnet1_1d(cin)#.squeeze(1)
        #up2 = self.rnet2_1d(cin)#.squeeze(1)

        up = up.reshape(bs, *solver_dim)
        up2 = up2.reshape(bs, *solver_dim)

        #iv = self.iv_conv2d(u)
        #iv = iv.reshape(-1, self.n_patches, 13*32)
        #iv = self.iv_out(iv)
        #iv = self.iv_mlp(self.in_iv)

        #up = self.data_net(cin)#.squeeze(1)
        #up2 = self.data_net2(cin)#.squeeze(1)

        #u = u.reshape(bs, *self.coord_dims)
        #up = up.reshape(bs, *self.coord_dims)
        #up2 = up2.reshape(bs, *self.coord_dims)

        #up = u + up
        #up2 = u + up2

        #chunk u, up, up2
        #u_patched, unfold_shape = self.make_patches(u)
        #up_patched, _ = self.make_patches(up)
        #up2_patched, _ = self.make_patches(up2)
        u_patched = u.unsqueeze(1)
        up_patched = up.unsqueeze(1)
        up2_patched= up2.unsqueeze(1)

        params = self.get_params()

        u0, eps = self.solve_chunks(u_patched, up_patched, up2_patched, params)

        #join chunks into solution
        #u0 = self.join_patches(u0_patches, unfold_shape)
        u0 = u0.squeeze(1)


        return u0, up,up2, eps, params
        #return u0, up,eps, params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step, ds.x_step), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum =0.9)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xi = model.get_params()
    params = xi.squeeze().detach().cpu().numpy()
    #print(params)
    return params
    #return code


def train():
    """Optimize and threshold cycle"""

    optimize()

@torch.no_grad()
def evaluate(epoch):
    x_list = []
    y_list = []
    xs = torch.zeros(256, device=device)
    counts = torch.zeros(256, device=device)
    ys = torch.zeros(256, device=device)
    vs = torch.zeros(256, device=device)

    for j, batch_in_d in enumerate(tqdm(test_loader)):
        #if i+32 == 256: #solver_dim[1]:
        #    break
        i = j*32
        counts[i:i+32] = counts[i:i+32]+1
        #optimizer.zero_grad()
        batch_in = batch_in_d[0].double().to(device)
        batch_eval = batch_in_d[1].double().to(device)
        batch_index = batch_in_d[2].double().to(device)

        out, var, var2, eps, params = model(batch_in, batch_index)

        bs = batch_in.shape[0]
        x0 = out[:, 0]
        xf = out[:, -1]
        vf = var[:, -1]
        v0 = var[:, 0]

        x0 = x0.reshape(bs, -1)
        xf = xf.reshape(bs, -1)
        batch_in = batch_in.reshape(bs, -1)
        batch_eval = batch_eval.reshape(bs, -1)

        xs[i:i+32] = xs[i:i+32] + x0[0]
        vs[i:i+32] = vs[i:i+32] + v0[0]

        if i % 32 == 0: #solver_dim[1]:
            ys[i:i+32] = batch_eval[0]
        #x_list.append(x0)
        x_list.append(v0)
        y_list.append(batch_eval)

    #x = torch.cat(x_list, dim=1)
    #y = torch.cat(y_list, dim=1)

    xs = xs/counts
    #do_plot(x[0].detach().cpu().numpy(), y[0].detach().cpu().numpy(), epoch)
    do_plot(xs.detach().cpu().numpy(), vs.detach().cpu().numpy(), ys.detach().cpu().numpy(), epoch)
    return


def optimize(nepoch=5000):
    #with tqdm(total=nepoch) as pbar:

    params=print_eq()
    L.info(f'parameters\n{params}')
    loss_list = []
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        x_losses = []
        var_losses = []
        eval_losses = []
        losses = []
        total_loss = 0
        for i, batch_in_d in enumerate(tqdm(train_loader)):
        #for i, batch_in in enumerate((train_loader)):
            optimizer.zero_grad()
            #batch_in,t,x,mask = batch_in[0], batch_in[1], batch_in[2], batch_in[3]
            #batch_in = batch_in[0], batch_in[1], batch_in[2], batch_in[3]
            batch_in = batch_in_d[0].double().to(device)
            batch_eval = batch_in_d[1].double().to(device)
            batch_index = batch_in_d[2].double().to(device)
            #mask = mask.double().to(device)

            out, var, var2, eps, params = model(batch_in, batch_index)

            bs = batch_in.shape[0]
            x0 = out[:, 0]
            xf = out[:, -1]
            vf = var[:, -1]
            #var = var[:, :-1]

            x0 = x0.reshape(bs, -1)
            xf = xf.reshape(bs, -1)
            batch_in = batch_in.reshape(bs, -1)
            batch_eval = batch_eval.reshape(bs, -1)
            #mask = mask.reshape(bs, -1)

            var =var.reshape(bs, -1)
            var2 =var2.reshape(bs, -1)
            #x_loss = (x0- batch_in).abs()#.pow(2)#.mean()
            #x_loss = (x0- batch_in).abs().mean(dim=-1)/batch_in.abs().mean(dim=-1)#.mean()
            #x_loss = (x0- batch_in).pow(2).mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)
            x_loss = (xf- batch_in).abs().mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)
            eval_loss = (x0- batch_eval).abs().mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)

            #var_loss = (var- batch_in).abs().mean(dim=-1)/batch_in.abs().mean(dim=-1)
            #var2_loss = (var2- x0).abs().mean(dim=-1)
            #var_loss = (var- out.reshape(bs,32,32)[:, :-1].reshape(bs, -1)).abs().mean(dim=-1)
            var_loss = (var- out.reshape(bs, -1)).abs().mean(dim=-1)
            #var_loss = var_loss + (vf- batch_in).abs().mean(dim=-1)

            #loss = x_loss + var_loss + time_loss
            param_loss = params.abs()
            #loss = x_loss.mean() + var_loss.mean() #+ 0.01*param_loss.mean()
            #loss = x_loss.mean() + var_loss.mean() + var2_loss.mean() + 0.0001*param_loss.mean()
            #loss = 2*x_loss.mean() + var_loss.mean() + var2_loss.mean() +  0.001*param_loss.mean()
            #loss = x_loss.mean() + var_loss.mean() +var2_loss.mean()  +  0.05*param_loss.mean()
            loss = x_loss.mean() + var_loss.mean()  #+  0.001*param_loss.mean()
            #loss = x_loss.mean() + var_loss.mean() + 0.001*param_loss.mean()
            #loss = x_loss.mean() #+ 0.01*param_loss.mean()
            #loss = var_loss.mean()
            #loss = x_loss +  (var- batch_in).abs().mean()
            #loss = x_loss +  (var- batch_in).pow(2).mean()
            x_losses.append(x_loss.detach().cpu())
            #var_losses.append(var_loss + var2_loss)
            var_losses.append(var_loss.detach().cpu())
            eval_losses.append(eval_loss.detach().cpu())
            #var_losses.append(var_loss )
            losses.append(loss.detach().cpu())
            #total_loss = total_loss + loss
            

            loss.backward()
            optimizer.step()

            del loss,var_loss, eval_loss, x_loss, var, var2, x0,xf,vf, params

        if (epoch+1) % 10 == 0:
            #print('plotting)')
            evaluate(epoch)

            #xi = xi.detach().cpu().numpy()
            #alpha = alpha.squeeze().item() #.detach().cpu().numpy()
            #beta = beta.squeeze().item()
        _x_loss = torch.cat(x_losses,dim=0).mean()
        _v_loss = torch.cat(var_losses,dim=0).mean()
        _eval_loss = torch.cat(eval_losses,dim=0).mean()

        #_x_loss = torch.tensor(x_losses).mean()
        #_v_loss = torch.tensor(var_losses).mean()

        #total_loss.backward()
        #optimizer.step()

        mean_loss = torch.tensor(losses).mean()

        loss_list.append(_x_loss.detach().cpu().numpy())
        np.save(loss_path, np.array(loss_list))

        meps = 1 #eps.max().item()
            #print(f'\nalpha, beta {xi}')
        params=print_eq()
        L.info(f'parameters\n{params}')
            #pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E}  xloss {x_loss:.3E} max eps {meps}\n')
        #print(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E}  xloss {_x_loss:.3E} vloss {_v_loss:.3E} max eps {meps}\n')
        L.info(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E} max eps {meps:.3E} xloss {_x_loss.item():.3E}  \
               vloss {_v_loss.item():.3E} evalloss {_eval_loss.item():.3E}')


if __name__ == "__main__":
    train()

    print_eq()
