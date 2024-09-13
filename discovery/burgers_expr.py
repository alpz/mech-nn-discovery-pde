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
#solver_dim=(32,32)
solver_dim=(30,64)
batch_size= 1
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1

noise=True

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

        #num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1
        #num_x_idx = self.data_dim[1] #- self.solver_dim[1] + 1


        #self.num_t_idx = num_t_idx//self.t_subsample  #+ 1
        #self.num_x_idx = num_x_idx//self.x_subsample  #+ 1

        #if self.t_subsample < self.solver_dim[0]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[0]//self.t_subsample
        #if self.x_subsample < self.solver_dim[1]:
        #    self.num_t_idx = self.num_t_idx - self.solver_dim[1]//self.x_subsample

        #self.length = self.num_t_idx*self.num_x_idx
        self.length = 1 #self.num_t_idx*self.num_x_idx


    def __len__(self):
        return self.length #self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.data, self.t, self.x




#%%

#ds = BurgersDataset(n_step=T,n_step_per_batch=n_step_per_batch)#.generate()
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


        #Step size is fixed. Make this a parameter for learned step
        #self.step_size = (logit(0.01)*torch.ones(1,1,1))
        #_step_size = (logit(0.01)*torch.ones(1,1,1))
        #_step_size = (logit(0.001)*torch.ones(1,1,self.n_step_per_batch-1))
        #self.step_size = nn.Parameter(_step_size)
        self.param_in = nn.Parameter(torch.randn(1,64))
        #self.param_time = nn.Parameter(torch.randn(1,64))


        #form of equation u_t + 0*u_tt + p(x,t)u_x + q(x,t)u_xx = 0

        #init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2, dtype=dtype)
        #self.init_coeffs = nn.Parameter(init_coeffs)

        self.coord_dims = solver_dim
        self.iv_list = [(0,0, [0,0],[0,self.coord_dims[1]-2]), 
                        (1,0, [1,0], [self.coord_dims[0]-1, 0]), 
                        #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
                        (0,0, [self.coord_dims[0]-1,1],[self.coord_dims[0]-1,self.coord_dims[1]-2]), 
                        #(1,2, [0,0], [self.coord_dims[0]-1, 0]),
                        #(1,3, [0,0], [self.coord_dims[0]-1, 0])
                        (1,0, [0,self.coord_dims[1]-1], [self.coord_dims[0]-1, self.coord_dims[1]-1])
                        ]

        self.n_patches_t = ds.data.shape[0]//self.coord_dims[0]
        self.n_patches_x = ds.data.shape[1]//self.coord_dims[1]
        self.n_patches = self.n_patches_t*self.n_patches_x
        print('num patches ', self.n_patches)

        self.pde = PDEINDLayerEPS(bs=bs*self.n_patches, coord_dims=self.coord_dims, order=self.order, n_ind_dim=self.n_dim, 
                                  n_iv=self.n_iv, init_index_mi_list=self.iv_list,  
                                  n_iv_steps=1, double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders

        #pm='circular'
        #TODO add time space dims
        pm='zeros'
        self.data_net = nn.Sequential(
            nn.Conv1d(self.coord_dims[0], 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
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
            nn.Conv1d(64,self.coord_dims[0], kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )


        self.data_conv2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ELU(),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ELU(),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ELU(),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ELU(),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(64,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            #nn.ELU(),
            #nn.Conv2d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            #nn.ELU(),
            nn.Conv2d(64,1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )

        #self.data_conv2d2 = nn.Sequential(
        #    nn.Conv2d(1, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    #nn.ReLU(),
        #    nn.ELU(),
        #    nn.Conv2d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    #nn.ReLU(),
        #    nn.ELU(),
        #    nn.Conv2d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    #nn.ReLU(),
        #    nn.ELU(),
        #    nn.Conv2d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    nn.ELU(),
        #    nn.Conv2d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    #nn.ReLU(),
        #    nn.ELU(),
        #    nn.Conv2d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    #nn.ReLU(),
        #    #nn.ELU(),
        #    #nn.Conv2d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    #nn.ReLU(),
        #    nn.ELU(),
        #    nn.Conv2d(256,1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
        #    )


        #self.data_mlp = nn.Sequential(
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


        self.param_in = nn.Parameter(torch.randn(1,256))
        self.param_net = nn.Sequential(
            nn.Linear(256, 1024),
            #nn.ELU(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            #nn.ELU(),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ELU(),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 3),
            #nn.Tanh()
        )

        self.param_in2 = nn.Parameter(torch.randn(1,256))
        self.param_net2 = nn.Sequential(
            nn.Linear(256, 1024),
            #nn.ELU(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            #nn.ELU(),
            nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.ELU(),
            #nn.ReLU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 3),
            #nn.Tanh()
        )


        self.t_step_size = steps[0]
        self.x_step_size = steps[1]
        print('steps ', steps)
        #self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,1,1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,1,1))

        #self.steps0 = nn.Parameter(self.steps0)
        #self.steps1 = nn.Parameter(self.steps1)


        up_coeffs = torch.randn((1, 1, self.num_multiindex), dtype=dtype)
        self.up_coeffs = nn.Parameter(up_coeffs)

        #self.stepsup0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.stepsup1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))


    def get_params(self):
        params = self.param_net(self.param_in)
        params2 =self.param_net2(self.param_in2)
        #params = params.reshape(-1,1,2, 3)
        #params = params.reshape(-1,1,2, 2)
        params = torch.stack([params, params2], dim=-2)
        return params

    def get_iv(self, u):
        u1 = u[:,0, :self.coord_dims[1]-2+1]
        u2 = u[:, 1:self.coord_dims[0]-1+1:,0]
        u3 = u[:, self.coord_dims[0]-1, 1:self.coord_dims[1]-2+1]
        u4 = u[:, 0:self.coord_dims[0]-1+1, self.coord_dims[1]-1]

        ub = torch.cat([u1,u2,u3,u4], dim=-1)

        return ub

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

    def solve_chunks(self, rhs_chunks, u_chunks):
        bs = rhs_chunks.shape[0]
        #up_coeffs = self.up_coeffs.repeat(self.bs,self.pde.grid_size,1)
        coeffs = self.up_coeffs.expand(self.bs,self.n_patches, self.pde.grid_size,-1)
        steps0 = self.steps0.type_as(coeffs).expand(-1,self.n_patches, self.coord_dims[0]-1)
        steps1 = self.steps1.type_as(coeffs).expand(-1,self.n_patches, self.coord_dims[1]-1)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
        steps_list = [steps0, steps1]

        rhs = rhs_chunks.reshape(bs*self.n_patches, self.pde.grid_size)
        u_chunks = u_chunks.reshape(bs*self.n_patches, *self.coord_dims)

        iv_rhs = self.get_iv(u_chunks)

        u0,u,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)

        up = u0.reshape(bs, self.n_patches, *self.coord_dims)
        up_t = u[...,1].reshape(bs, self.n_patches, *self.coord_dims)
        up_x = u[...,2].reshape(bs, self.n_patches, *self.coord_dims)
        up_tt = u[...,3].reshape(bs, self.n_patches, *self.coord_dims)
        up_xx = u[...,4].reshape(bs, self.n_patches, *self.coord_dims)

        return {'up': up, 'up_t': up_t, 'up_x': up_x, 'up_tt': up_tt, 'up_xx': up_xx,
                'eps':eps.max()}

    def forward(self, u, t, x):
        bs = u.shape[0]
        #up = self.data_net(u)
        #up = up.reshape(bs, self.pde.grid_size)
        #cin = torch.stack([u,t,x], dim=1)
        cin = u.unsqueeze(1) #torch.stack([u,t,x], dim=1)
        #print(cin.shape)

        up_rhs = self.data_conv2d(cin).squeeze(1)
        #up_rhs = up

        u_chunked, unfold_shape = self.make_patches(u)
        up_rhs_chunked, _ = self.make_patches(up_rhs)

        up_dict = self.solve_chunks(up_rhs_chunked, u_chunked)
        eps = up_dict['eps']

        #join chunks into solution
        up = self.join_patches(up_dict['up'], unfold_shape)
        up_t = self.join_patches(up_dict['up_t'], unfold_shape)
        up_x = self.join_patches(up_dict['up_x'], unfold_shape)
        up_tt = self.join_patches(up_dict['up_tt'], unfold_shape)
        up_xx = self.join_patches(up_dict['up_xx'], unfold_shape)

        p = torch.stack([torch.ones_like(up), up, up**2], dim=-1)
        q = torch.stack([torch.ones_like(up), up, up**2], dim=-1)

        params = self.get_params()

        p = (p*params[...,0,:]).sum(dim=-1)
        q = (q*params[...,1,:]).sum(dim=-1)


        eq_loss = up_t - p*up_x - q*up_xx
        
        return up, eq_loss, eps, params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step, ds.x_step), device=device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum =0.9)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xi = model.get_params()
    print(xi.squeeze().detach().cpu().numpy())
    #return code


def train():
    """Optimize and threshold cycle"""

    optimize()


def optimize(nepoch=5000):
    #with tqdm(total=nepoch) as pbar:
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        x_losses = []
        eq_losses = []
        losses = []
        total_loss = 0
        for i, batch_in in enumerate(tqdm(train_loader)):
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
            x0, eq_loss, eps, params = model(batch_in, t, x)

            t_end = x0.shape[1]
            x_end = x0.shape[2]
            batch_in = batch_in.reshape(*data_shape)[-1, :t_end, :x_end]

            #var = var.reshape(*data_shape)[-1, :t_end, :x_end]
            #var2 = var2.reshape(*data_shape)[-1, :t_end, :x_end]

            #print(batch_in.shape, x0.shape, var.shape)
            batch_in = batch_in.reshape(x0.shape)
            eq_loss = eq_loss.reshape(x0.shape)

            #x_loss = (x0- batch_in).abs()#.pow(2)#.mean()
            #eq_loss = (eq_loss).abs()#.pow(2)#.pow(2)#.mean()

            #x_loss = (x0- batch_in).abs()#.pow(2)#.mean()
            #eq_loss = (eq_loss).abs()#.pow(2)#.pow(2)#.mean()

            x_loss = (x0- batch_in).abs()#.pow(2)#.mean()
            eq_loss = (eq_loss).abs()#.pow(2)#.pow(2)#.mean()

            param_loss = params.abs()
            #loss = x_loss.mean() + var_loss.mean() #+ 0.01*param_loss.mean()
            loss = x_loss.mean() + eq_loss.mean() +  0.01*param_loss.mean()
            #loss = x_loss.mean() #+ 0.01*param_loss.mean()
            #loss = var_loss.mean()
            #loss = x_loss +  (var- batch_in).abs().mean()
            #loss = x_loss +  (var- batch_in).pow(2).mean()
            x_losses.append(x_loss)
            eq_losses.append(eq_loss)
            losses.append(loss)
            #total_loss = total_loss + loss
            
            #total_loss.backward()
            loss.backward()
            optimizer.step()

        _x_loss = torch.cat(x_losses,dim=0).mean()
        _eq_loss = torch.cat(eq_losses,dim=0).mean()

        mean_loss = torch.tensor(losses).mean()

        meps = eps.max().item()
            #L.info(f'run {run_id} epoch {epoch}, loss {loss.item():.3E} max eps {meps:.3E} xloss {x_loss:.3E} time_loss {time_loss:.3E}')
            #print(f'\nalpha, beta {xi}')
            #L.info(f'\nparameters {xi}')
        print_eq()
            #pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E}  xloss {x_loss:.3E} max eps {meps}\n')
        print(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E}  xloss {_x_loss:.3E} eqloss {_eq_loss:.3E} max eps {meps}\n')


if __name__ == "__main__":
    train()

    print_eq()
