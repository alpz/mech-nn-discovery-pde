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
cuda=False
#T = 2000
#n_step_per_batch = T
batch_size= 1
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1




class BurgersDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000):
        #self.n_step_per_batch=n_step_per_batch
        #self.n_step=n_step

        self.down_sample = 1

        data=loadmat(os.path.join(PDEConfig.sindpy_data, 'burgers.mat'))

        print(data.keys())
        t = data['t']
        x = data['x']

        print(t.shape)
        print(x.shape)
        print(data['usol'].shape)

        self.x_train = torch.tensor(data, dtype=dtype) 

    def __len__(self):
        #return (self.n_step-self.n_step_per_batch)//self.down_sample
        return 1 #self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train


#ds = BurgersDataset(n_step=T,n_step_per_batch=n_step_per_batch)#.generate()
ds = BurgersDataset()#.generate()
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False) 



class Model(nn.Module):
    def __init__(self, bs, n_step,n_step_per_batch, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step #+ 1
        self.order = 2
        # state dimension
        self.bs = bs
        self.device = device
        self.n_iv=1
        self.n_ind_dim = 1
        self.n_step_per_batch = n_step_per_batch


        #Step size is fixed. Make this a parameter for learned step
        #self.step_size = (logit(0.01)*torch.ones(1,1,1))
        #_step_size = (logit(0.01)*torch.ones(1,1,1))
        _step_size = (logit(0.001)*torch.ones(1,1,self.n_step_per_batch-1))
        self.step_size = nn.Parameter(_step_size)
        self.param_in = nn.Parameter(torch.randn(1,64))
        self.param_time = nn.Parameter(torch.randn(1,64))

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders

        #form of equation u_t + 0*u_tt + p(x,t)u_x + q(x,t)u_xx = 0

        #init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2, dtype=dtype)
        #self.init_coeffs = nn.Parameter(init_coeffs)

        self.coord_dims = (32,32)
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

        pm='circular'
        self.data_net = nn.Sequential(
            nn.Conv1d(100,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
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
            nn.Conv1d(64,100, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )


        self.param_in = nn.Parameter(torch.randn(1,64))
        self.param_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            #two polynomials, second order
            nn.Linear(1024, 3*2)
        )


        self.t_step_size = 0.1
        self.x_step_size = 0.075
        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

    def get_params(self):
        params = self.param_net(self.param_in)
        params = params.reshape(-1,1,2, 3)
        return params

    def forward(self, u):
        bs = u.shape[0]
        up = self.data_net(u)
        up = up.reshape(bs, self.pde.grid_size)

        p = torch.stack([torch.ones_like(up), up, up**2], dim=-1)
        q = torch.stack([torch.ones_like(up), up, up**2], dim=-1)

        params = self.get_params()

        p = p*params[...,0,:].sum(dim=-1)
        q = q*params[...,1,:].sum(dim=-1)


        coeffs = torch.zeros((bs, self.pde.grid_size, self.pde.n_orders))
        #u, u_t, u_x, u_tt, u_xx
        #u_t
        coeffs[..., 1] = 1.
        coeffs[..., 2] = p
        coeffs[..., 4] = q

        up = up.reshape(bs, *self.coord_dims)


        steps0 = self.steps0.type_as(coeffs)
        steps1 = self.steps1.type_as(coeffs)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
        steps_list = [steps0, steps1]

        rhs = torch.zeros(bs, *self.coord_dims)

        u0,u,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)

        steps = self.step_size.repeat(self.bs, self.n_ind_dim, 1).type_as(net_iv)

        steps = torch.sigmoid(steps).clip(min=0.001).detach()
        #self.steps = self.steps.type_as(net_iv)

        x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, steps)
        x0 = x0.permute(0,2,1)
        var = var.permute(0,2,1)
        ts = ts.squeeze(1)

        #return x0, steps, eps, var,_xi
        return x0, steps, eps, var, ts, xi.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,n_step=T, n_step_per_batch=n_step_per_batch, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xi = model.get_params()
    print(xi.squeeze().cpu().numpy())
    #return code

def simulate(gen_code):
    #simulate learned equation
    def f(state, t):
        x0, x1, x2= state

        dx0 = eval(gen_code[0])
        dx1 = eval(gen_code[1])
        dx2 = eval(gen_code[2])

        return dx0, dx1, dx2
        
    state0 = [1.0, 1.0, 1.0]
    time_steps = np.linspace(0, T*STEP, T)

    x_sim = odeint(f, state0, time_steps)
    return x_sim

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
    with tqdm(total=nepoch) as pbar:
        for epoch in range(nepoch):
            pbar.update(1)
            #for i, (time, batch_in) in enumerate(train_loader):
            for i, batch_in in enumerate(train_loader):
                batch_in = batch_in.to(device)
                #time = time.to(device)

                optimizer.zero_grad()
                #x0, steps, eps, var,xi = model(index, batch_in)
                x0, steps, eps, var = model(batch_in)

                x_loss = (x0- batch_in).pow(2).mean()
                #x_loss = (x0- batch_in).abs().mean()
                #x_loss = (x0- batch_in).pow(2).mean()
                var_loss = (var- batch_in).pow(2).mean()
                #var_loss = (var- batch_in).abs().mean()
                #time_loss = (time- var_time).pow(2).mean()
                #time_loss = (time- var_time).abs().mean()

                #loss = x_loss + var_loss + time_loss
                loss = x_loss + var_loss
                #loss = x_loss +  (var- batch_in).abs().mean()
                #loss = x_loss +  (var- batch_in).pow(2).mean()
                

                loss.backward()
                optimizer.step()


            #xi = xi.detach().cpu().numpy()
            #alpha = alpha.squeeze().item() #.detach().cpu().numpy()
            #beta = beta.squeeze().item()
            meps = eps.max().item()
            #L.info(f'run {run_id} epoch {epoch}, loss {loss.item():.3E} max eps {meps:.3E} xloss {x_loss:.3E} time_loss {time_loss:.3E}')
            #print(f'\nalpha, beta {xi}')
            #L.info(f'\nparameters {xi}')
            print_eq()
            pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E} max eps {meps}\n xloss {x_loss:.3E} time_loss{time_loss:.3E}\n')


if __name__ == "__main__":
    train()

    print_eq()
