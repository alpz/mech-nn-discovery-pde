#%%

import extras.source
log_dir, run_id = extras.source.create_log_dir(root='logs/gl')

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


#log_dir, run_id = create_log_dir(root='logs/gl')
extras.source.write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

DBL=True
dtype = torch.float64 if DBL else torch.float32
#STEP = 0.001
cuda=True
#T = 2000
#n_step_per_batch = T
#solver_dim=(10,256)
solver_dim=(8,32,32)
#solver_dim=(64,64,64)
#solver_dim=(50,64)
#solver_dim=(32,48)
n_grid=3
batch_size= 32
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1

noise =False
setmask =False
downsample=2

#We learn one equation at a time
first_equation = True

nn_transform = False

L.info(f'Ginzburg')
L.info(f'Solver dim {solver_dim} ')


class ReactDiffDataset(Dataset):
    def __init__(self, solver_dim=(32,32)):

        u_data=np.load(os.path.join(PDEConfig.ginzburg_dir, 'Ar_256_0_05.npy'))
        v_data=np.load(os.path.join(PDEConfig.ginzburg_dir, 'Ai_256_0_05.npy'))
        #uv_data=np.load(os.path.join(PDEConfig.ginzburg_dir, 'A2_256_0_05.npy'))
        #downsample time
        u_data = u_data[::downsample]
        v_data = v_data[::downsample]

        if not first_equation:
            #swap u and v if learning the second equation.
            L.info(f'Learing second equations')
            u_data, v_data = v_data, u_data

        #uv_data = uv_data[::downsample]
        L.info(f'data shape {u_data.shape}')
        L.info(f'downsample {downsample}')


        self.t_step_size = 0.05*downsample 
        self.x_step_size = 0.3906 #steps[1]
        self.y_step_size = 0.3906 #steps[2]


        if noise:
            L.info('adding 20% noise')
            #rmse = mean_squared_error(u_data, np.zeros(u_data.shape), squared=False)
            rmse = np.sqrt(np.mean(u_data**2))
            # add 20% noise (note the impact on derivatives depends on step size...)
            u_data = u_data + np.random.normal(0, rmse / 5.0, u_data.shape) 
            v_data = v_data + np.random.normal(0, rmse / 5.0, v_data.shape) 
            #uv_data = uv_data + np.random.normal(0, rmse / 5.0, uv_data.shape) 

        u_data = torch.tensor(u_data, dtype=dtype)#.permute(1,0,2,3) 
        v_data = torch.tensor(v_data, dtype=dtype)#.permute(1,0,2,3) 
        #uv_data = torch.tensor(uv_data, dtype=dtype)#.permute(1,0,2,3) 


        data_shape = tuple(u_data.shape)
        self.t = torch.linspace(0,1,data_shape[0]).reshape(-1,1,1).expand(-1, data_shape[1], data_shape[2])
        self.x = torch.linspace(0,1,data_shape[1]).reshape(1,-1,1).expand(data_shape[0], -1, data_shape[2])        
        self.y = torch.linspace(0,1,data_shape[2]).reshape(1,1,-1).expand(data_shape[0], data_shape[1], -1)        
        

        # learn over a smaller subset
        self.u_data = u_data[:128+128, :128, :128]
        self.v_data = v_data[:128+128, :128, :128]
        #self.uv_data = uv_data[128:128+128, :128, :128]

        print('u,v ', self.u_data.shape, self.v_data.shape)

        self.data_dim = self.u_data.shape
        self.solver_dim = solver_dim

        num_t_idx = self.data_dim[0] #- self.solver_dim[0] + 1
        num_x_idx = self.data_dim[1] #- self.solver_dim[1] + 1
        num_y_idx = self.data_dim[2] #- self.solver_dim[1] + 1


        self.num_t_idx = num_t_idx//solver_dim[0]  #+ 1
        self.num_x_idx = num_x_idx//solver_dim[1]  #+ 1
        self.num_y_idx = num_y_idx//solver_dim[2]  #+ 1

        self.length = self.num_t_idx*self.num_x_idx*self.num_y_idx

        #if setmask:
        #    ##mask
        #    mask = torch.rand_like(self.u_data)
        #    ##keep only 80% of data
        #    mask = (mask>0.2).double()
        #    L.info(f'20% mask')
        #else:
        #    mask = torch.ones_like(self.u_data)

        
        #self.u_data = self.u_data
        #self.v_data = self.v_data
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

        #uv_data = self.uv_data[t_idx:t_idx+t_step, 
        #                     x_idx:x_idx+x_step,
        #                     y_idx:y_idx+y_step]
        #mask = self.mask[t_idx:t_idx+t_step, 
        #                     x_idx:x_idx+x_step,
        #                     y_idx:y_idx+y_step]


        t = self.t[t_idx:t_idx+t_step, x_idx:x_idx+x_step,y_idx:y_idx+y_step]
        x = self.x[ t_idx:t_idx+t_step, x_idx:x_idx+x_step,
                             y_idx:y_idx+y_step]#.unsqueeze(0)

        y = self.y[t_idx:t_idx+t_step, x_idx:x_idx+x_step,
                             y_idx:y_idx+y_step]#.unsqueeze(0)

        return u_data, v_data,t, x, y

#%%

ds = ReactDiffDataset(solver_dim=solver_dim)#.generate()

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
    def __init__(self, bs, solver_dim, device=None, **kwargs):
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
                        #lambda nt, nx, ny: (0,0, [nt-1,1,1],[nt-1, nx-2, ny-2]), 
                        ]



        self.pde = MultigridLayer(bs=bs, coord_dims=self.coord_dims, order=2, n_ind_dim=self.n_ind_dim, n_iv=1, 
                        n_grid=n_grid, evolution=False, downsample_first=False,
                        init_index_mi_list=self.iv_list,  n_iv_steps=1, double_ret=True, solver_dbl=True)

        # u, u_t, u_tt, u_x, u_xx
        self.num_multiindex = self.pde.n_orders


        #self.rnet2d_1 = N.ResNet2D(out_channels=1, in_channels=1)
        #self.rnet2d_2 = N.ResNet2D(out_channels=1, in_channels=1)

        self.rnet2d_1 = N.ResNet(out_channels=1, in_channels=1)
        self.rnet2d_2 = N.ResNet(out_channels=1, in_channels=1)

        #self.params_u = nn.Parameter(0.5*torch.randn(1,4))
        class ParamNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Parameter(torch.randn(1,512)) #*(2/(4*512)))
                self.net = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 10),
                )
            def forward(self):
                y = self.net(self.input)
                return y


        self.param_net_list = nn.ModuleList() 
        for i in range(4):
            self.param_net_list.append(ParamNet())




        self.t_step_size = ds.t_step_size #0.05*downsample #steps[0]
        self.x_step_size = ds.x_step_size #0.3906 #steps[1]
        self.y_step_size = ds.y_step_size #0.3906 #steps[2]
        #print('steps ', steps)
        ##self.steps0 = torch.logit(self.t_step_size*torch.ones(1,self.coord_dims[0]-1))
        ##self.steps1 = torch.logit(self.x_step_size*torch.ones(1,self.coord_dims[1]-1))

        self.steps0 = torch.logit(self.t_step_size*torch.ones(1,1))
        self.steps1 = torch.logit(self.x_step_size*torch.ones(1,1))
        self.steps2 = torch.logit(self.y_step_size*torch.ones(1,1))


    def get_params(self):
        params_list = [(net()).squeeze() for net in self.param_net_list]
        #return u_params, v_params, w_params, x_params, y_params, z_params
        return params_list

    def get_iv(self, u):
        bs = u.shape[0]

        #u = self.rnet3d_2(u.unsqueeze(1)).squeeze(1)
        u1 = u[:,0, :self.coord_dims[1], :self.coord_dims[2]].reshape(bs, -1)
        u2 = u[:,1:self.coord_dims[0], 0, :self.coord_dims[2]].reshape(bs, -1)
        u3 = u[:,1:self.coord_dims[0], 1:self.coord_dims[1], 0].reshape(bs, -1)

        u4 = u[:,1:self.coord_dims[0], -1, 1:self.coord_dims[2]].reshape(bs, -1)
        u5 = u[:,1:self.coord_dims[0], 1:self.coord_dims[1]-1, -1].reshape(bs, -1)
        u6 = u[:,-1, 1:self.coord_dims[1]-1, 1:self.coord_dims[2]-1].reshape(bs, -1)

        #ub = torch.cat([u1,u2,u3,u4,u5,u6], dim=-1)
        ub = torch.cat([u1,u2,u3,u4,u5], dim=-1)

        return ub


    def get_steps(self, u, t, x, y):
        x = x.squeeze()
        y = y.squeeze()
        x = x[:, :, 0]
        y = y[:, 0,:]

        steps0 = self.steps0.type_as(u).expand(self.bs, self.coord_dims[0]-1)
        steps1 = self.steps1.type_as(u).expand(self.bs, self.coord_dims[1]-1)
        steps2 = self.steps2.type_as(u).expand(self.bs, self.coord_dims[2]-1)
        steps0 = torch.sigmoid(steps0)#.clip(min=0.01, max=0.8)
        steps1 = torch.sigmoid(steps1)#.clip(min=0.01, max=0.55)
        steps2 = torch.sigmoid(steps2)#.clip(min=0.01, max=0.55)


        steps_list = [steps0, steps1, steps2]

        return steps_list

    def solve(self, u, v, up, vp, params_list, steps_list):
        bs = u.shape[0]

        ui = u.reshape(bs, *self.coord_dims)
        #vi = v.reshape(bs, *self.coord_dims)
        #vpi = vp.reshape(bs, *self.coord_dims)
        upi = up[:,0].reshape(bs, *self.coord_dims)
        #upi = upi/2
        iv_rhs_u = self.get_iv(upi)
        #iv_rhs_u = self.get_iv(ui)
        #iv_rhs_v = self.get_iv(vi)


        up = up.reshape(bs, 1, self.pde.grid_size)
        u = u.reshape(bs, self.pde.grid_size)
        v = v.reshape(bs, self.pde.grid_size)
        vp = vp.reshape(bs, 1, self.pde.grid_size)
        
        up0 = up[:,0]
        #up1 = up[:,1]
        #up2 = up[:,2]

        vp0 = vp[:,0]
        #vp1 = vp[:,1]
        #vp2 = vp[:,2]

        #basis1 = torch.stack([torch.ones_like(up0), up0, up0.pow(2), up0.pow(3), vp0, vp0.pow(2), vp0.pow(3), up0*vp0, up0.pow(2)*vp0, up0*vp0.pow(2)], dim=-1)
        basis0 = torch.stack([torch.ones_like(up0), up0, up0.pow(2), vp0, vp0.pow(2), up0*vp0], dim=-1)
        basis2= torch.stack([torch.ones_like(up0), up0, up0.pow(2)], dim=-1)
        basis3 = torch.stack([vp0, vp0.pow(2), vp0.pow(3)], dim=-1)


        p0 = (basis0*params_list[0][:6]).sum(dim=-1)
        p1 = (basis2*params_list[1][:3]).sum(dim=-1)
        p2 = (basis2*params_list[2][:3]).sum(dim=-1)
        p3 = (basis3*params_list[3][:3]).sum(dim=-1)


        coeffs_u = torch.zeros((bs, self.pde.grid_size, self.pde.n_orders), device=u.device)
        coeffs_u[..., 0] = p0 #(params_list[0][0]+ params_list[0][1]*A2[:,0]) #+ params_u[2]*A2.pow(2))
        #u_t
        coeffs_u[..., 1] =  1. #params_list[3][0]
        #u_tt
        #u_xx
        coeffs_u[..., 5] = p1 #params_list[2][0]#+ params_list[3][1]*A2
        #u_yy
        coeffs_u[..., 6] = p2 #params_list[2][1]#+ params_list[5][1]*A2
        rhs_u = p3 

        rhs_loss = None #(rhs_u - rhs_u_true).abs().mean()


        u0,_,eps = self.pde(coeffs_u, rhs_u, iv_rhs_u, steps_list)
        u = u0
        v = 0*u0 #v0 #None#u0[1]

        return u, v, rhs_loss
    
    def forward(self, u, v, t, x,y):
        bs = u.shape[0]
        ts = solver_dim[0]

        # u batch, time, x, y

        u_in = u.unsqueeze(1) #torch.stack([u,t, x,y], dim=1) 
        v_in = v.unsqueeze(1) #torch.stack([v,t, x,y], dim=1) 
        #uv_in = uv.unsqueeze(1) #torch.stack([v,t, x,y], dim=1) 

        u_in = u_in.reshape(bs*ts, 1, solver_dim[1], solver_dim[2])
        v_in = v_in.reshape(bs*ts, 1, solver_dim[1], solver_dim[2])



        if nn_transform:
            up = self.rnet2d_1(u_in)
            vp = self.rnet2d_2(v_in)
        else:
            up = u_in
            vp = v_in

        up = up.reshape(bs,1, *solver_dim)
        vp = vp.reshape(bs,1, *solver_dim)

        params = self.get_params()
        steps_list = self.get_steps(u, t,x,y)

        u0, v0, rhs_loss = self.solve(u,v, up, vp, params, steps_list)

        return u0, v0,up,vp, params
        #return u0, up,eps, params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(bs=batch_size,solver_dim=solver_dim, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

if DBL:
    model = model.double()

model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    params = model.get_params()
    params = [p.squeeze().detach().cpu().numpy() for p in params]

    for i,p in enumerate(params):
        L.info(f'param {i}\n{p}')
    #print(params)
    #return code


def train(nepoch=500):

    print_eq()
    for epoch in range(nepoch):
        #pbar.update(1)
        #for i, (time, batch_in) in enumerate(train_loader):
        u_losses = []
        v_losses = []
        var_u_losses = []
        var_v_losses = []
        var_uv_losses = []
        t_losses = []
        losses = []
        total_loss = 0
        for i, batch_in in enumerate(tqdm(train_loader)):
        #for i, batch_in in enumerate((train_loader)):
            optimizer.zero_grad()
            batch_u, batch_v = batch_in[0], batch_in[1]
            t,x,y = batch_in[2], batch_in[3], batch_in[4]
            batch_u = batch_u.double().to(device)
            batch_v = batch_v.double().to(device)
            #mask = mask.double().to(device)
            #batch_uv = batch_uv.double().to(device)

            t = t.double().to(device)
            x = x.double().to(device)
            y = y.double().to(device)

            data_shape = batch_u.shape

            #optimizer.zero_grad()
            #x0, steps, eps, var,xi = model(index, batch_in)
            u, v, var_u, var_v, params = model(batch_u, batch_v, t, x,y)


            bs = batch_u.shape[0]
            u = u.reshape(bs, -1)
            #v = v.reshape(bs, -1)
            batch_u = batch_u.reshape(bs, -1)
            batch_v = batch_v.reshape(bs, -1)
            #batch_uv = batch_uv.reshape(bs, -1)
            var_u =var_u.reshape(bs,1, -1)
            var_v =var_v.reshape(bs,1, -1)

            u_loss = (u- batch_u).abs().mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)
            v_loss = 0*u_loss #(v- batch_v).abs().mean(dim=-1) #+ (x0**2- batch_in**2).pow(2).mean(dim=-1)

            var_u_loss = (var_u- batch_u.unsqueeze(1)).abs().mean(dim=-1)
            var_v_loss = (var_v- batch_v.unsqueeze(1)).abs().mean(dim=-1)
            var_uv_loss = 0*var_u_loss #(var_uv- batch_uv.unsqueeze(1)).pow(2).mean(dim=-1)

            param_loss = torch.stack(params).abs().sum()

            #t_params_loss = (t_params-1).abs().mean()
            t_params_loss = 0*(params[1][1]-1).abs().mean()
            #jloss = u_loss.mean() +  v_loss.mean() + var_u_loss.mean() + var_v_loss.mean()
            #loss = u_loss.mean() + v_loss.mean() +  var_u_loss.mean() + var_v_loss.mean() + t_params_loss
            loss = u_loss.mean() +  var_u_loss.mean() + var_v_loss.mean() #+ var_uv_loss.mean()#+ t_params_loss
            #loss = u_loss.mean() +  var_u_loss.mean() 
            #loss = loss + rhs_loss.mean()
            loss = loss +  0.0001*param_loss.mean()

            u_losses.append(u_loss.mean().item())
            v_losses.append(v_loss.mean().item())
            #var_losses.append(var_loss + var2_loss)
            var_u_losses.append(var_u_loss.mean().item())
            var_v_losses.append(var_v_loss.mean().item())
            var_uv_losses.append(var_uv_loss.mean().item())
            #var_losses.append(var_loss )
            losses.append(loss.detach().item())
            t_losses.append(t_params_loss.detach().item())
            #total_loss = total_loss + loss
            
            #print('rhs loss ', rhs_loss.mean().item())

            loss.backward()
            optimizer.step()


            del loss,u,u_loss,v,v_loss,var_u,var_v,var_u_loss,var_v_loss,var_uv_loss, params#, t_params_loss

        _u_loss = torch.tensor(u_losses).mean().item()
        _v_loss = torch.tensor(v_losses).mean().item()
        _var_u_loss = torch.tensor(var_u_losses).mean().item()
        _var_v_loss = torch.tensor(var_v_losses).mean().item()
        _var_uv_loss = torch.tensor(var_uv_losses).mean().item()
        t_loss = torch.tensor(t_losses).mean().item()

        mean_loss = torch.tensor(losses).mean().item()

        print_eq()
        L.info(f'run {run_id} epoch {epoch}, loss {mean_loss:.3E}  \
               uloss {_u_loss:.3E} vloss {_v_loss:.3E} \
               var_u_loss {_var_u_loss:.3E} var_v_loss {_var_v_loss:.3E}  \
               var_uv_loss {_var_uv_loss:.3E}  \ t_loss {t_loss:.3E}')
        print('steps ', torch.sigmoid(model.steps0).squeeze().item(), 
                        torch.sigmoid(model.steps1).squeeze().item(), 
                        torch.sigmoid(model.steps2).squeeze().item())

if __name__ == "__main__":
    train()

    print_eq()

# %%
