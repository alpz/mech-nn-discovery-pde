
#%%
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import gradcheck


from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

from solver.ode_layer import ODEINDLayer
import discovery.basis as B
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
STEP = 0.001
cuda=True
T = 1000
n_step_per_batch = T
batch_size= 1
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1


class RiccatiDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000):
        self.n_step_per_batch=n_step_per_batch
        self.n_step=n_step
        self.end= n_step*STEP
        x_train = self.generate()

        self.down_sample = 1

        self.x_train = torch.tensor(x_train, dtype=dtype) 
        self.x_train = self.x_train 

        #Create basis for some stats. Actual basis is in the model
        #basis,basis_vars =B.create_library(x_train, polynomial_order=2, use_trig=False, constant=True)
        #self.basis = torch.tensor(basis)
        #self.basis_vars = basis_vars
        #self.n_basis = self.basis.shape[1]

    def generate(self):
        def f(state, t):
            x = state
            #return 1.6 * x **(0.4)
            #return x**2 + t
            #return np.sqrt(np.abs(3*t**2-t))*x**2
            return np.power(np.abs(3*t**2-t),0.8)*x**2 + t
            #return (3*t**2+t)*x**2
            #return np.sqrt(t)*x**2

        state0 = [0.5]
        time_steps = np.linspace(0, self.end, self.n_step)
        self.time_steps = torch.tensor(time_steps, dtype=dtype)

        x_train = odeint(f, state0, time_steps)
        return x_train

    def __len__(self):
        #return (self.n_step-self.n_step_per_batch)//self.down_sample
        return 1 #self.x_train.shape[0]

    def __getitem__(self, idx):
        i = idx*self.down_sample
        d=  self.x_train[i:i+self.n_step_per_batch]
        t=  self.time_steps[i:i+self.n_step_per_batch]
        return t, d


ds = RiccatiDataset(n_step=T,n_step_per_batch=n_step_per_batch)#.generate()
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False) 

#plt.plot(ds.x_train)
#plt.pause(1)

#plot train data
#P.plot_lorenz(ds.x_train, os.path.join(log_dir, 'train.pdf'))

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.step_size = (logit(0.001)*torch.ones(1,1,1))
        self.param_in = nn.Parameter(torch.randn(1,64))
        self.param_time = nn.Parameter(torch.randn(1,64))

        #init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2, dtype=dtype)
        #self.init_coeffs = nn.Parameter(init_coeffs)
        
        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, solver_dbl=True, double_ret=True,
                                    n_iv=self.n_iv, n_iv_steps=1,  gamma=0.05, alpha=0, **kwargs)


        self.param_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3)
        )

        self.time_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch)
        )

        self.net = nn.Sequential(
            nn.Linear(self.n_step_per_batch*self.n_ind_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch*self.n_ind_dim)
        )
    
    #def reset_params(self):
    #    #reset basis weights to random values
    #    self.xi.data = torch.randn_like(self.init_xi)

    #def update_mask(self, mask):
    #    self.mask = self.mask*mask
    
    def get_xi(self):
        xi = self.param_net(self.param_in)
        xi = xi.reshape(1, 3)
        return xi

    def get_time(self):
        ts = self.time_net(self.param_time)
        ts = ts.reshape(1, self.n_step_per_batch)
        return ts

    def forward(self, net_iv):
        # apply mask
        xi = self.get_xi()
        ts = self.get_time()
        #xi = _xi

        #xi = self.mask*self.xi
        #xi = self.mask*xi
        xi = xi.repeat(self.bs, 1)
        ts = ts.repeat(self.bs, 1)

        alpha = xi[:, 0:1].unsqueeze(1)
        beta = xi[:, 1:2].unsqueeze(1)
        pow = xi[:, 2:3].unsqueeze(1)
        pow = torch.sigmoid(pow)
        ts = ts.unsqueeze(1)

        var = self.net(net_iv.reshape(self.bs,-1))
        #var = var.reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)
        var = var.reshape(self.bs, self.n_ind_dim, self.n_step_per_batch)
        #var = torch.relu(var)
        #var = var.abs()


        rhs = (alpha*ts + beta*ts**2).abs().pow(pow)*var**2 + ts
        #create basis
        #var_basis,_ = B.create_library_tensor_batched(var, polynomial_order=2, use_trig=False, constant=True)

        #rhs = var_basis@xi
        rhs = rhs.permute(0,2,1)

        z = torch.zeros(1, self.n_ind_dim, 1,1).type_as(net_iv)
        o = torch.ones(1, self.n_ind_dim, 1,1).type_as(net_iv)

        coeffs = torch.cat([z,o,z], dim=-1)
        coeffs = coeffs.repeat(self.bs,1,self.n_step_per_batch,1)

        init_iv = var[:,:,0]

        #steps = self.step_size*torch.ones(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(net_iv)
        steps = self.step_size.repeat(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(net_iv)

        steps = torch.sigmoid(steps)
        #self.steps = self.steps.type_as(net_iv)

        x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, steps)
        x0 = x0.permute(0,2,1)
        var = var.permute(0,2,1)
        ts = ts.squeeze(1)

        #return x0, steps, eps, var,_xi
        return x0, steps, eps, var, ts,alpha, beta, pow

model = Model(bs=batch_size,n_step=T, n_step_per_batch=n_step_per_batch, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if DBL:
    model = model.double()
model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    xi = model.get_xi()
    #repr_dict = B.basis_repr(model.xi*model.mask, ds.basis_vars)
    repr_dict = B.basis_repr(xi*model.mask, ds.basis_vars)
    code = []
    for k,v in repr_dict.items():
        L.info(f'{k} = {v}')
        if stdout:
            print(f'{k} = {v}')
        code.append(f'{v}')
    return code

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

    max_iter = 1
    for step in range(max_iter):
        print(f'Optimizer iteration {step}/{max_iter}')

        #threshold
        if step > 0:
            xi = model.get_xi()
            #mask = (xi.abs() > threshold).float()

            L.info(xi)
            #L.info(xi*model.mask)
            #L.info(model.mask)
            #L.info(model.mask*mask)

        #code = print_eq(stdout=True)
        #simulate and plot

        #x_sim = simulate(code)
        #P.plot_lorenz(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

        #set mask
        #if step > 0:
        #    model.update_mask(mask)
        #    model.reset_params()

        optimize()


def optimize(nepoch=2000):
    with tqdm(total=nepoch) as pbar:
        for epoch in range(nepoch):
            pbar.update(1)
            for i, (time, batch_in) in enumerate(train_loader):
                batch_in = batch_in.to(device)
                time = time.to(device)

                optimizer.zero_grad()
                #x0, steps, eps, var,xi = model(index, batch_in)
                x0, steps, eps, var, var_time ,alpha, beta, pow = model(batch_in)

                #x_loss = (x0- batch_in).abs().mean()
                #x_loss = (x0- batch_in).pow(2).mean()
                x_loss = (x0- batch_in).pow(2).mean()
                #x_loss = (x0- batch_in).abs().mean()
                var_loss = (var- batch_in).pow(2).mean()
                #var_loss = (var- batch_in).abs().mean()
                time_loss = (time- var_time).pow(2).mean()
                #time_loss = (time- var_time).abs().mean()
                loss = x_loss + var_loss + time_loss
                #loss = x_loss +  (var- batch_in).abs().mean()
                #loss = x_loss +  (var- batch_in).pow(2).mean()
                

                loss.backward()
                optimizer.step()


            #xi = xi.detach().cpu().numpy()
            alpha = alpha.squeeze().item() #.detach().cpu().numpy()
            beta = beta.squeeze().item()
            pow = pow.squeeze().item()
            meps = eps.max().item()
            L.info(f'run {run_id} epoch {epoch}, loss {loss.item():.3E} max eps {meps:.3E} xloss {x_loss:.3E} time_loss {time_loss:.3E}')
            print(f'\nalpha, beta, exp: {alpha}, {beta}, {pow}')
            L.info(f'\nalpha, beta, exp: {alpha}, {beta}, {pow}')
            pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item():.3E} max eps {meps}\n xloss {x_loss:.3E} time_loss{time_loss:.3E}\n')


if __name__ == "__main__":
    train()

    print_eq()
