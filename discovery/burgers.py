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


log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=False)

DBL=True
dtype = torch.float64 if DBL else torch.float32
#STEP = 0.001
cuda=True
#T = 2000
#n_step_per_batch = T
solver_dim=(10,256)
batch_size= 1
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1




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

        self.t_subsample = 10
        self.x_subsample = 1

        print(self.t.shape)
        print(self.x.shape)
        print(data['usol'].shape)

        data = data['usol']
        #permute time, x
        self.data = torch.tensor(data, dtype=dtype).permute(1,0) 
        print('self ', self.data.shape)

        self.data_dim = self.data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] - self.solver_dim[0] + 1
        num_x_idx = self.data_dim[1] - self.solver_dim[1] + 1

        self.num_t_idx = num_t_idx//self.t_subsample 
        self.num_x_idx = num_x_idx//self.x_subsample 
        self.length = self.num_t_idx*self.num_x_idx


    def __len__(self):
        return self.length #self.x_train.shape[0]

    def __getitem__(self, idx):
        #t_idx = idx//self.num_x_idx
        #x_idx = idx - t_idx*self.num_x_idx
        (t_idx, x_idx) = np.unravel_index(idx, (self.num_t_idx, self.num_x_idx))

        t_idx = t_idx*self.t_subsample
        x_idx = x_idx*self.t_subsample

        t_step = self.solver_dim[0]
        x_step = self.solver_dim[1]

        t = self.t[t_idx:t_idx+t_step, x_idx:x_idx+x_step]
        x = self.x[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

        data = self.data[t_idx:t_idx+t_step, x_idx:x_idx+x_step]

        return data, t, x

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

        self.pde = PDEINDLayerEPS(bs=bs, coord_dims=self.coord_dims, order=self.order, n_ind_dim=self.n_dim, 
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
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            nn.ELU(),
            nn.Conv2d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            nn.ELU(),
            nn.Conv2d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            nn.ELU(),
            nn.Conv2d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            nn.ELU(),
            nn.Conv2d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            #nn.ELU(),
            #nn.Conv2d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.ReLU(),
            nn.ELU(),
            nn.Conv2d(128,1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )

        self.data_mlp = nn.Sequential(
            #nn.Linear(32*32, 1024),
            nn.Linear(self.pde.grid_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            #two polynomials, second order
            nn.Linear(1024,self.pde.grid_size)
        )


        self.param_in = nn.Parameter(torch.randn(1,64))
        self.param_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            #two polynomials, second order
            #nn.Linear(1024, 3*2),
            nn.Linear(1024, 2*2),
            #nn.Tanh()
        )


        self.t_step_size = steps[0]
        self.x_step_size = steps[1]
        print('steps ', steps)
        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

        self.steps0 = nn.Parameter(self.steps0)
        self.steps1 = nn.Parameter(self.steps1)

    def get_params(self):
        params = self.param_net(self.param_in)
        #params = params.reshape(-1,1,2, 3)
        params = params.reshape(-1,1,2, 2)
        return params

    def get_iv(self, u):
        u1 = u[:,0, :self.coord_dims[1]-2+1]
        u2 = u[:, 1:self.coord_dims[0]-1+1:,0]
        u3 = u[:, self.coord_dims[0]-1, 1:self.coord_dims[1]-2+1]
        u4 = u[:, 0:self.coord_dims[0]-1+1, self.coord_dims[1]-1]

        ub = torch.cat([u1,u2,u3,u4], dim=-1)

        return ub

    def forward(self, u, t, x):
        bs = u.shape[0]
        #up = self.data_net(u)
        #up = up.reshape(bs, self.pde.grid_size)
        #cin = torch.stack([u,t,x], dim=1)
        cin = u #torch.stack([u,t,x], dim=1)
        #print(cin.shape)

        #up = self.data_conv2d(cin)
        up = self.data_net(cin)
        #up = up.reshape(bs, self.coord_dims[0], self.coord_dims[1])

        #cin = u.reshape(1, self.pde.grid_size)
        #cin = u.reshape(-1,self.pde.grid_size) #+ t.reshape(-1,self.pde.grid_size) +x.reshape(-1,self.pde.grid_size)
        #up = self.data_mlp(u.reshape(-1,self.pde.grid_size))
        #up = self.data_mlp(cin)
        up = up.reshape(bs, self.pde.grid_size)
        u = u.reshape(bs, self.pde.grid_size)

        #p = torch.stack([torch.ones_like(up), up, up**2], dim=-1)
        #q = torch.stack([torch.ones_like(up), up, up**2], dim=-1)

        #p = torch.stack([torch.ones_like(up), up], dim=-1)
        #q = torch.stack([torch.ones_like(up), up], dim=-1)

        p = torch.stack([torch.ones_like(u), u], dim=-1)
        q = torch.stack([torch.ones_like(u), u], dim=-1)

        params = self.get_params()

        p = (p*params[...,0,:]).sum(dim=-1)
        q = (q*params[...,1,:]).sum(dim=-1)


        coeffs = torch.zeros((bs, self.pde.grid_size, self.pde.n_orders), device=u.device)
        #u, u_t, u_x, u_tt, u_xx
        #u_t
        coeffs[..., 1] = 1.
        #u_x
        coeffs[..., 2] = p
        #u_xx
        coeffs[..., 4] = q

        up = up.reshape(bs, *self.coord_dims)
        u = u.reshape(bs, *self.coord_dims)


        steps0 = self.steps0.type_as(coeffs)
        steps1 = self.steps1.type_as(coeffs)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
        steps_list = [steps0, steps1]

        rhs = torch.zeros(bs, *self.coord_dims, device=u.device)
        iv_rhs = self.get_iv(u)

        u0,u,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)

        #steps = self.step_size.repeat(self.bs, self.n_ind_dim, 1).type_as(u)

        #steps = torch.sigmoid(steps).clip(min=0.001).detach()
        #self.steps = self.steps.type_as(net_iv)

        #x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, steps)
        #x0 = x0.permute(0,2,1)
        #var = var.permute(0,2,1)
        #ts = ts.squeeze(1)

        #return x0, steps, eps, var,_xi
        return u0, up, eps, p.squeeze(),q.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, steps=(ds.t_step, ds.x_step), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xi = model.get_params()
    print(xi.squeeze().detach().cpu().numpy())
    #return code

#def simulate(gen_code):
#    #simulate learned equation
#    def f(state, t):
#        x0, x1, x2= state
#
#        dx0 = eval(gen_code[0])
#        dx1 = eval(gen_code[1])
#        dx2 = eval(gen_code[2])
#
#        return dx0, dx1, dx2
#        
#    state0 = [1.0, 1.0, 1.0]
#    time_steps = np.linspace(0, T*STEP, T)
#
#    x_sim = odeint(f, state0, time_steps)
#    return x_sim

def train():
    """Optimize and threshold cycle"""
    #model.reset_params()

    #max_iter = 1
    #for step in range(max_iter):
    #    print(f'Optimizer iteration {step}/{max_iter}')

    #    #threshold
    #    if step > 0:
    #        xi = model.get_xi()
    #        #mask = (xi.abs() > threshold).float()

    #        L.info(xi)
    #        #L.info(xi*model.mask)
    #        #L.info(model.mask)
    #        #L.info(model.mask*mask)

    #    #code = print_eq(stdout=True)
    #    #simulate and plot

    #    #x_sim = simulate(code)
    #    #P.plot_lorenz(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

    #    #set mask
    #    #if step > 0:
    #    #    model.update_mask(mask)
    #    #    model.reset_params()

    optimize()


def optimize(nepoch=5000):
    #with tqdm(total=nepoch) as pbar:
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        x_losses = []
        var_losses = []
        losses = []
        total_loss = 0
        optimizer.zero_grad()
        for i, batch_in in enumerate(tqdm(train_loader)):
            batch_in,t,x = batch_in[0], batch_in[1], batch_in[2]
            batch_in = batch_in.double().to(device)
            t = t.double().to(device)
            x = x.double().to(device)
            #time = time.to(device)
            #print(batch_in.shape)

            #optimizer.zero_grad()
            #x0, steps, eps, var,xi = model(index, batch_in)
            x0, var, eps, p,q = model(batch_in, t, x)

            #print(batch_in.shape, x0.shape, var.shape)
            batch_in = batch_in.reshape(x0.shape)
            var = var.reshape(x0.shape)


            x_loss = (x0- batch_in).pow(2)#.mean()
            #x_loss = (x0- batch_in).abs()#.mean()
            #x_loss = (x0- batch_in).pow(2).mean()
            var_loss = (var- batch_in).pow(2)#.mean()
            #var_loss = (var- batch_in).abs()#.mean()
            #var_loss = (var- batch_in).pow(2)#.mean()
            #time_loss = (time- var_time).pow(2).mean()
            #time_loss = (time- var_time).abs().mean()

            #loss = x_loss + var_loss + time_loss
            #param_loss = p.abs() + q.abs()
            #loss = x_loss.mean() + var_loss.mean() #+ 0.01*param_loss.mean()
            loss = x_loss.mean() #+ var_loss.mean() #+ 0.01*param_loss.mean()
            #loss = x_loss.mean() + 0.01*param_loss.mean()
            #loss = var_loss.mean()
            #loss = x_loss +  (var- batch_in).abs().mean()
            #loss = x_loss +  (var- batch_in).pow(2).mean()
            x_losses.append(x_loss)
            var_losses.append(var_loss)
            losses.append(loss)
            total_loss = total_loss + loss
            

            #loss.backward()
            #optimizer.step()


            #xi = xi.detach().cpu().numpy()
            #alpha = alpha.squeeze().item() #.detach().cpu().numpy()
            #beta = beta.squeeze().item()
        _x_loss = torch.cat(x_losses,dim=0).mean()
        _v_loss = torch.cat(var_losses,dim=0).mean()

        total_loss.backward()
        optimizer.step()

        mean_loss = torch.tensor(losses).mean()

        meps = eps.max().item()
            #L.info(f'run {run_id} epoch {epoch}, loss {loss.item():.3E} max eps {meps:.3E} xloss {x_loss:.3E} time_loss {time_loss:.3E}')
            #print(f'\nalpha, beta {xi}')
            #L.info(f'\nparameters {xi}')
        print_eq()
            #pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E}  xloss {x_loss:.3E} max eps {meps}\n')
        print(f'run {run_id} epoch {epoch}, loss {mean_loss.item():.3E}  xloss {_x_loss:.3E} vloss {_v_loss:.3E} max eps {meps}\n')


if __name__ == "__main__":
    train()

    print_eq()
