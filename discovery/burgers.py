#%%
from scipy.io import loadmat

from config import PDEConfig
import os
import numpy as np

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

#import matplotlib.pyplot as plt
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
solver_dim=(32,32)
#solver_dim=(50,64)
#solver_dim=(32,48)
batch_size= 1
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.01
counter_threshold = 30

noise =False

L.info(f'Solver dim {solver_dim} ')


class BurgersDataset(Dataset):
    def __init__(self, solver_dim=(32,32)):
        #self.n_step_per_batch=n_step_per_batch
        #self.n_step=n_step

        self.down_sample = 1

        data=loadmat(os.path.join(PDEConfig.sindpy_data, 'burgers.mat'))

        print(data.keys())
        t = torch.tensor(np.array(data['t'])).squeeze()
        x = torch.tensor(np.array(data['x'])).squeeze()

        self._t = t
        self._x = x

        self.t_step = t[1] - t[0]
        self.x_step = x[1] - x[0]

        self.t = t.unsqueeze(1).expand(-1, x.shape[0])
        self.x = x.unsqueeze(0).expand(t.shape[0],-1)

        print('t x', self.t.shape, self.x.shape)

        #self.t_subsample = 10
        #self.x_subsample = 1

        #self.t_subsample = 32 #10
        #self.x_subsample = 32 #256


        #L.info(f'subsample {self.t_subsample}, {self.x_subsample} ')
        #self.t_subsample =50
        #self.x_subsample =64

        print(self.t.shape)
        print(self.x.shape)
        print(data['usol'].shape)

        data = np.real(data['usol'])

        if noise:
            print('adding noise')
            rmse = mean_squared_error(data, np.zeros(data.shape), squared=False)
            # add 20% noise (note the impact on derivatives depends on step size...)
            data = data + np.random.normal(0, rmse / 5.0, data.shape) 

        #permute time, x
        self.data = torch.tensor(data, dtype=dtype).permute(1,0) 
        print('self ', self.data.shape)

        self.data_dim = self.data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1
        num_x_idx = self.data_dim[1] #- self.solver_dim[1] + 1


        self.num_t_idx = num_t_idx//solver_dim[0]  #+ 1
        self.num_x_idx = num_x_idx//solver_dim[1]  #+ 1

        #if self.t_subsample < self.solver_dim[0]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[0]//self.t_subsample
        #if self.x_subsample < self.solver_dim[1]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[1]//self.x_subsample

        #self.length = self.num_t_idx*self.num_x_idx
        self.length = self.num_t_idx*self.num_x_idx


    def __len__(self):
        return self.length #self.x_train.shape[0]

    def __getitem__(self, idx):
        #return self.data, self.t, self.x
        ##t_idx = idx//self.num_x_idx
        ##x_idx = idx - t_idx*self.num_x_idx
        (t_idx, x_idx) = np.unravel_index(idx, (self.num_t_idx, self.num_x_idx))

        t_idx = t_idx*solver_dim[0]
        x_idx = x_idx*solver_dim[1]


        t_step = solver_dim[0]
        x_step = solver_dim[1]


        t = self.t[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
        x = self.x[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

        data = self.data[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

        return data, t, x

#%%

ds = BurgersDataset(solver_dim=solver_dim)#.generate()

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

        self.n_basis = 3
        self.n_poly = 2
        self.mask = torch.ones(1, self.n_poly, self.n_basis).to(device)
        self.counter = torch.zeros(1, self.n_poly, self.n_basis).to(device).int()

        self.param_in = nn.Parameter(torch.randn(1,64))

        self.coord_dims = solver_dim
        self.iv_list = [(0,0, [0,0],[0,self.coord_dims[1]-1]), 
                        (1,0, [1,0], [self.coord_dims[0]-1, 0]), 
                        #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
                        #(0,0, [self.coord_dims[0]-1,1],[self.coord_dims[0]-1,self.coord_dims[1]-2]), 
                        #(1,2, [0,0], [self.coord_dims[0]-1, 0]),
                        #(1,3, [0,0], [self.coord_dims[0]-1, 0])
                        (1,0, [1,self.coord_dims[1]-1], [self.coord_dims[0]-1, self.coord_dims[1]-1])
                        ]

        #self.iv_len = self.coord_dims[1]-1 + self.coord_dims[0]-1 + self.coord_dims[1]-2 + self.coord_dims[0]
        #self.iv_len = self.coord_dims[1]-1 + self.coord_dims[0]-1 + self.coord_dims[0]
        #print('iv len', self.iv_len)

        self.n_patches_t = 1 #ds.data.shape[0]//self.coord_dims[0]
        self.n_patches_x = 1 #ds.data.shape[1]//self.coord_dims[1]
        self.n_patches = 1 #self.n_patches_t*self.n_patches_x
        print('num patches ', self.n_patches)

        self.pde = PDEINDLayerEPS(bs=bs*self.n_patches, coord_dims=self.coord_dims, order=self.order, n_ind_dim=self.n_dim, 
                                  n_iv=self.n_iv, init_index_mi_list=self.iv_list,  
                                  n_iv_steps=1, double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders

        pm='circular'
        #TODO add time space dims
        #pm='zeros'
        self.data_net = nn.Sequential(
            nn.Conv1d(101, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,101, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )

        self.data_net2 = nn.Sequential(
            nn.Conv1d(101, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,101, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )


        self.data_conv2d = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            #nn.Conv2d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )

        self.data_conv2d2 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv2d(128,1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )


        #self.iv_conv1d = nn.Sequential(
        #    nn.Conv1d(101, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv1d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv1d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv1d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv1d(64,8, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    )

        #self.iv_conv2d = nn.Sequential(
        #    nn.Conv2d(1, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    #nn.ELU(),
        #    nn.Conv2d(128,128, kernel_size=5, padding=2, stride=2, padding_mode=pm),
        #    nn.ReLU(),
        #    #nn.ELU(),
        #    nn.Conv2d(128,128, kernel_size=5, padding=2, stride=2, padding_mode=pm),
        #    nn.ReLU(),
        #    #nn.ELU(),
        #    nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    #nn.ELU(),
        #    nn.Conv2d(128,128, kernel_size=5, padding=2, stride=2, padding_mode=pm),
        #    nn.ReLU(),
        #    #nn.ELU(),
        #    nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ReLU(),
        #    nn.Conv2d(128,self.n_patches, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    )

        #self.iv_out = nn.Linear(13*32, self.iv_len)

        self.rnet1 = N.ResNet(out_channels=1, in_channels=1)
        self.rnet2 = N.ResNet(out_channels=1, in_channels=1)
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
            nn.Linear(1024, 3),
            #nn.Tanh()
        )

        self.param_net[-1].weight.data.fill_(0.)
        self.param_net[-1].bias = nn.Parameter(0.05*torch.randn(3))

        self.param_net2[-1].weight.data.fill_(0.)
        self.param_net2[-1].bias = nn.Parameter(0.05*torch.randn(3))

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

        self.t_step_size = steps[0]
        self.x_step_size = steps[1]
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
        #params = self.param_net_out(self.param_net(self.param_in))
        #params2 =self.param_net2_out(self.param_net2(self.param_in2))

        params = (self.param_net(self.param_in))
        params2 =(self.param_net2(self.param_in2))
        #params = params.reshape(-1,1,2, 3)
        #params = params.reshape(-1,1,2, 2)
        params = torch.stack([params, params2], dim=-2)
        mask = self.mask.reshape(params.shape)
        return params*mask, mask

    def update_mask(self):
        #self.mask = self.mask*mask
        new_mask = self.mask
        new_mask[self.counter>counter_threshold] = 0.
        self.mask = self.mask*new_mask

    def reset_counter(self, params):
        #mask = self.counter>threshold_steps
        self.counter[params.abs()>=threshold] = 0.

    def inc_counter(self, params):
        #mask = self.counter>threshold_steps
        self.counter = self.counter + (params.abs() < threshold).int()
        

    def get_iv(self, u):
        #u1 = u[:,0, :self.coord_dims[1]-2+1]
        u1 = u[:,0, :self.coord_dims[1]]
        u2 = u[:, 1:self.coord_dims[0]-1+1:,0]
        #u3 = u[:, self.coord_dims[0]-1, 1:self.coord_dims[1]-2+1]
        #u4 = u[:, 0:self.coord_dims[0]-1+1, self.coord_dims[1]-1]
        u4 = u[:, 1:self.coord_dims[0]-1+1, self.coord_dims[1]-1]

        #ub = torch.cat([u1,u2,u3,u4], dim=-1)
        ub = torch.cat([u1,u2,u4], dim=-1)

        return ub

    def solve_chunks(self, u_patches, up_patches, up2_patches, params):
        bs = u_patches.shape[0]
        n_patches = u_patches.shape[1]
        u0_list = []
        eps_list = []

        steps0 = self.steps0.type_as(params).expand(-1, self.n_patches, self.coord_dims[0]-1)
        steps1 = self.steps1.type_as(params).expand(-1, self.n_patches, self.coord_dims[1]-1)
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
        #iv_rhs = upi + up2.reshape(bs, *self.coord_dims)
        #iv_rhs = iv_rhs.reshape(bs, n_patches*self.iv_len)
        #iv_rhs = self.iv_mlp(iv_rhs)


        basis = torch.stack([torch.ones_like(up), up, up**2], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up2), up2, up2**2], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up2), up2, up2**2], dim=-1)
        basis2 = torch.stack([torch.ones_like(up), up, up**2], dim=-1)
        #basis2 = torch.stack([torch.ones_like(up), up, up**2], dim=-1)

        p = (basis*params[...,0,:]).sum(dim=-1)
        q = (basis2*params[...,1,:]).sum(dim=-1)


        coeffs = torch.zeros((bs*n_patches, self.pde.grid_size, self.pde.n_orders), device=u.device)
        #u, u_t, u_x, u_tt, u_xx
        #u_t
        coeffs[..., 1] = 1.
        #u_x
        coeffs[..., 2] = p
        #u_xx
        coeffs[..., 4] = q

        #up = up.reshape(bs, *self.coord_dims)

        rhs = torch.zeros(bs*n_patches, *self.coord_dims, device=u.device)

        u0,_,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)
        #u0_list.append(u0)
        #eps_list.append(eps)

        #u0 = torch.stack(u0_list, dim=1)
        #eps = torch.stack(eps_list, dim=1).max()

        u0 = u0.reshape(u_patches.shape)

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

    def forward(self, u, t, x):
        bs = u.shape[0]
        #up = self.data_net(u)
        #up = up.reshape(bs, self.pde.grid_size)
        #cin = torch.stack([u,t,x], dim=1)
        cin = u.unsqueeze(1) #torch.stack([u,t,x], dim=1)
        #cin = u
        #print(cin.shape)

        #up = self.data_conv2d(cin).squeeze(1)
        #up2 = self.data_conv2d2(cin).squeeze(1)

        up = self.rnet1(cin).squeeze(1)
        up2 = up #self.rnet2(cin).squeeze(1)

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

        params, mask = self.get_params()

        u0, eps = self.solve_chunks(u_patched, up_patched, up2_patched, params)

        #join chunks into solution
        #u0 = self.join_patches(u0_patches, unfold_shape)
        u0 = u0.squeeze(1)

        return u0, up,up2, eps, params, mask
        #return u0, up,eps, params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step, ds.x_step), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum =0.9)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xi,mask = model.get_params()
    params = xi.squeeze().detach().cpu().numpy()
    mask = mask.squeeze().detach().cpu().numpy()
    #print(params)
    return params,mask
    #return code


def train():
    """Optimize and threshold cycle"""
    #model.reset_params()

    #max_iter = 10
    #for step in range(max_iter):
        #print(f'Optimizer iteration {step}/{max_iter}')
        ##threshold
        #xi = model.get_params()

        #L.info('Current params, mask')
        #L.info(xi)
        ##L.info(xi*model.mask)
        #L.info(model.mask)
        ##L.info(model.mask*mask)

        ##code = print_eq(stdout=True)
        ##simulate and plot

        ##x_sim = simulate(code)
        ##P.plot_lorenz(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

        ##set mask
        #if step > 0:
        #    mask = (xi.abs() > threshold).float()
        #    model.update_mask(mask)
        #    #model.reset_params()

    optimize(0)


def optimize(opt_step, nepoch=500):
    #with tqdm(total=nepoch) as pbar:

    #params=print_eq()
    #L.info(f'parameters\n{params}')
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        x_losses = []
        var_losses = []
        losses = []
        total_loss = 0
        for i, batch_in in enumerate(tqdm(train_loader)):
        #for i, batch_in in enumerate((train_loader)):
            optimizer.zero_grad()
            batch_in,t,x = batch_in[0], batch_in[1], batch_in[2]
            batch_in = batch_in.double().to(device)
            t = t.double().to(device)
            x = x.double().to(device)
            #time = time.to(device)
            #print(batch_in.shape)
            data_shape = batch_in.shape

            #optimizer.zero_grad()
            #x0, steps, eps, var,xi = model(index, batch_in)
            x0, var, var2, eps, params, mask = model(batch_in, t, x)

            #print(batch_in.shape, x0.shape, var.shape)
            t_end = x0.shape[1]
            x_end = x0.shape[2]

            batch_in = batch_in.reshape(*data_shape)[-1, :t_end, :x_end]
            var = var.reshape(*data_shape)[-1, :t_end, :x_end]
            #var2 = var2.reshape(*data_shape)[-1, :t_end, :x_end]


            #x_loss = (x0- batch_in).abs()#.pow(2)#.mean()
            x_loss = (x0- batch_in).pow(2)#.mean()
            #x_loss = (x0- batch_in).abs()#.mean()
            #x_loss = (x0- batch_in).pow(2).mean()
            #var_loss = (var- batch_in).abs()#.pow(2)#.mean()
            #var2_loss = (var2- batch_in).abs()#.pow(2)#.mean()

            var_loss = (var- batch_in).pow(2)#.mean()
            #var2_loss = (var2- batch_in).pow(2)#.mean()
            #var_loss = (var- batch_in).pow(2)#.mean()
            #var_loss = (var- batch_in).abs()#.mean()
            #var_loss = (var- batch_in).pow(2)#.mean()
            #time_loss = (time- var_time).pow(2).mean()
            #time_loss = (time- var_time).abs().mean()

            #loss = x_loss + var_loss + time_loss
            param_loss = params.abs()
            #loss = x_loss.mean() + var_loss.mean() #+ 0.01*param_loss.mean()
            #loss = x_loss.mean() + var_loss.mean() + var2_loss.mean() + 0.0001*param_loss.mean()
            #loss = 2*x_loss.mean() + var_loss.mean() + var2_loss.mean() +  0.001*param_loss.mean()
            loss = 2*x_loss.mean() + var_loss.mean()  +  0.001*param_loss.mean()
            #loss = x_loss.mean() + var_loss.mean() + 0.001*param_loss.mean()
            #loss = x_loss.mean() #+ 0.01*param_loss.mean()
            #loss = var_loss.mean()
            #loss = x_loss +  (var- batch_in).abs().mean()
            #loss = x_loss +  (var- batch_in).pow(2).mean()
            x_losses.append(x_loss)
            #var_losses.append(var_loss + var2_loss)
            var_losses.append(var_loss)
            #var_losses.append(var_loss )
            losses.append(loss)
            total_loss = total_loss + loss
            

            loss.backward()
            optimizer.step()

            
            #


            #xi = xi.detach().cpu().numpy()
            #alpha = alpha.squeeze().item() #.detach().cpu().numpy()
            #beta = beta.squeeze().item()

        model.inc_counter(params)
        model.reset_counter(params)
        model.update_mask()

        _x_loss = torch.cat(x_losses,dim=0).mean()
        _v_loss = torch.cat(var_losses,dim=0).mean()

        #total_loss.backward()
        #optimizer.step()

        mean_loss = torch.tensor(losses).mean()

        meps = eps.max().item()
            #print(f'\nalpha, beta {xi}')
        #params=print_eq()
        params = params.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        counter = model.counter.detach().cpu().numpy()
        L.info(f'parameters, mask, counter\n{params}\n{mask}\n{counter}')
            #pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E}  xloss {x_loss:.3E} max eps {meps}\n')
        #print(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E}  xloss {_x_loss:.3E} vloss {_v_loss:.3E} max eps {meps}\n')
        L.info(f'run {run_id} epoch {epoch}/{opt_step}, loss {mean_loss.item():.3E} max eps {meps:.3E} xloss {_x_loss.item():.3E} vloss {_v_loss.item():.3E}')


if __name__ == "__main__":
    train()

    print_eq()
