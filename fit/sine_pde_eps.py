
import torch.nn as nn
import torch

#from solver.ode_layer import ODEINDLayer
from solver.pde_layer import PDEINDLayerEPS
from torch.nn.parameter import Parameter
import numpy as np

import torch
import ipdb
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import Dataset

#Fit a noisy exponentially damped sine wave with a second order ODE

class SineDataset(Dataset):
    def __init__(self, end=1, coord_dims=(50,32)):
        _y = np.linspace(0, end, coord_dims[0])
        _x = np.linspace(0, end, coord_dims[1])
        y0 = np.sin(3*_y) #+ 0.5*np.random.randn(*_y.shape)
        y1 = np.cos(3*_y) #+ 0.5*np.random.randn(*_y.shape)
        y0 = torch.tensor(y0)

        _xx = np.linspace(0, end, coord_dims[0])[:,np.newaxis]
        _yy= np.linspace(0, end, coord_dims[1])[np.newaxis, :]

        #self.damp = (np.exp(-0.05*((_xx-end/2)**2 + (_yy-end/2)**2))).reshape(coord_dims)
        self.damp = (np.exp(-0.1*(_xx) + (_yy-end/2)**2)).reshape(coord_dims)
        #self.damp = np.exp(-0.1*(_yy)).reshape(1,coord_dims[1])
        #self.damp = np.exp(-0.1*(_yy+_xx)).reshape(coord_dims[0],coord_dims[1])
        #self.y = y0*damp 
        self.y = y0.unsqueeze(-1).repeat(1,coord_dims[1])#.transpose(1,0)
        #self.y = np.sin((2*_yy+_xx))#.transpose(1,0)
        #self.y = np.sin((2*_yy+_xx))#.transpose(1,0)
        #self.y = np.sin((7*_yy+3*_xx))*np.cos(4*_xx)#.transpose(1,0)
        self.y = self.y*self.damp
        
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        y = self.y #+ 0.5*torch.randn_like(self.y)
        return y
    
class SineDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=True)
        return train_loader 
        
        

class Method(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.model = Sine(device=self.device)
        self.model = self.model.double()
        
        self.func_list = []
        self.y_list = []
        self.funcp_list = []
        self.funcpp_list = []
        self.steps_list = []
        self.coeffs_list = []
        self.rhs_list = []

    def forward(self, init_list):
        return self.model(init_list,check=False)
    

    def training_step(self, batch, batch_idx):
        y = batch

        #y = y.reshape((32,32))
        #y = y.reshape((32,32))
        y = y.reshape((48,48))
        t0 = y[0, 0:-1].reshape(-1)
        tn = y[-1, 1:-1].reshape(-1)
        x0 = y[1:, 0].reshape(-1)
        xn = y[:, -1].reshape(-1)

        #print(y.shape)
        eps, u0, cf = self([t0,x0, tn, xn])
        

        loss = (u0.reshape(-1)-y.reshape(-1)).pow(2).mean()
        #loss = (u0.reshape(-1)-y.reshape(-1)).abs().mean()
        #loss = loss + 0.1*cf.pow(2).mean()
        #loss = loss + eps.pow(2).mean()
        #loss = loss + eps.pow(2).sum()
        
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('eps', eps, prog_bar=True, logger=True)
        
        self.func_list.append(u0.detach().cpu().numpy())
        self.coeffs_list.append(self.model.coeffs.squeeze().detach().cpu().numpy())
        self.rhs_list.append(self.model.rhs.detach().cpu().numpy())
        #self.funcp_list.append(u1.detach().cpu().numpy())
        #self.funcpp_list.append(u2.detach().cpu().numpy())
        #self.steps_list.append(steps.detach().cpu().numpy())
        
        self.y_list.append(y.detach().cpu().numpy())
        
        #return {"loss": loss, 'y':y, 'u0':u0, 'u1':u1,'u2': u2}
        return {"loss": loss, 'y':y, 'u0':u0}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class Sine(nn.Module):
    def __init__(self, bs=1, device=None):
        super().__init__()

        self.step_size = 0.1
        #self.end = 500* self.step_size
        self.end =1
        #kself.n_step = int(self.end /self.step_size)
        self.order = 2
        #state dimension
        self.n_dim = 1
        self.bs = bs
        #kself.n_coeff = self.n_step * (self.order + 1)
        self.device  = device
        dtype = torch.float64
        #self.coord_dims = (64,32)
        #self.coord_dims = (10,15)
        #self.coord_dims = (32,32)
        self.coord_dims = (48,48)
        #self.coord_dims = (8,8)
        #self.coord_dims = (10,10)
        #self.coord_dims = (128,128)
        #self.coord_dims = (64,32)
        self.n_iv = 1
        #coord, mi_index, begin, end
        self.iv_list = [(0,0, [0,0],[0,self.coord_dims[1]-2]), 
                        (1,0, [1,0], [self.coord_dims[0]-1, 0]), 
                        #(0,1, [0,0],[0,self.coord_dims[1]-1]), 
                        (0,0, [self.coord_dims[0]-1,1],[self.coord_dims[0]-1,self.coord_dims[1]-2]), 
                        #(1,2, [0,0], [self.coord_dims[0]-1, 0]),
                        #(1,3, [0,0], [self.coord_dims[0]-1, 0])
                        (1,0, [0,self.coord_dims[1]-1], [self.coord_dims[0]-1, self.coord_dims[1]-1])
                        ]
        #self.iv_list = [(1,0), (0,1)]
        #self.iv_list = [(0,0), (1,0)]
        #self.iv_list = [(0,0), (0,1),(1,0)]

        self.pde = PDEINDLayerEPS(bs=bs, coord_dims=self.coord_dims, order=self.order, n_ind_dim=self.n_dim, n_iv=self.n_iv, init_index_mi_list=self.iv_list,  
                                    n_iv_steps=1, gamma=0.5, alpha=0., double_ret=True, solver_dbl=True)

        #_coeffs = torch.tensor(np.random.random((self.n_dim, self.pde.grid_size, self.pde.n_orders)), dtype=dtype)
        #_coeffs = torch.tensor(np.random.random((self.n_dim, self.pde.grid_size, self.pde.n_orders)), dtype=dtype)
        #_coeffs = torch.randn((self.n_dim, self.pde.grid_size, self.pde.n_orders), dtype=dtype)
        #_coeffs = torch.rand((self.n_dim, 1, self.pde.n_orders), dtype=dtype)
        #_coeffs = torch.rand((self.n_dim, 1, self.pde.n_orders), dtype=dtype)
        _coeffs = torch.randn((self.n_dim, 1, 1024), dtype=dtype)
        #_coeffs = torch.randn((self.n_dim, self.pde.grid_size, self.pde.n_orders), dtype=dtype)
        #_coeffs = torch.randn((self.n_dim, self.pde.grid_size, 256), dtype=dtype)
        self.coeffs = Parameter(_coeffs)

        #initial values grad and up
        if self.n_iv > 0:
            #iv_rhs00 = torch.randn(1,self.coord_dims[0], dtype=dtype)
            #iv_rhs10 = torch.randn(1,self.coord_dims[1], dtype=dtype)
            #iv_rhs01 = torch.randn(1,self.coord_dims[0], dtype=dtype)
            #iv_rhs11 = torch.randn(1,self.coord_dims[1], dtype=dtype)
            
            #iv_rhs = np.array([0]*self.n_dim).reshape(self.n_dim,self.n_iv)
            #iv_rhs = torch.zeros(1,self.coord_dims[1] + 2*self.coord_dims[0], dtype=dtype)
            #iv_rhs = torch.randn(1,2*self.coord_dims[1] + 2*self.coord_dims[0], dtype=dtype)
            iv_rhs = torch.randn(1,2*self.coord_dims[0]-1 + 2*self.coord_dims[1]-1, dtype=dtype)
            #iv_rhs = torch.randn(1,self.coord_dims[0], dtype=dtype)
            #iv_rhs = torch.randn(1,self.coord_dims[1] + self.coord_dims[0], dtype=dtype)
            self.iv_rhs = Parameter(iv_rhs)
            #self.iv_rhs00 = Parameter(iv_rhs00)
            #self.iv_rhs01 = Parameter(iv_rhs01)
            #self.iv_rhs10 = Parameter(iv_rhs10)
            #self.iv_rhs11 = Parameter(iv_rhs11)
            #self.iv_rhs = (iv_rhs)
        else:
            #self.iv_rhs = None
            self.iv_rhs = torch.tensor([])

        _rhs = np.array([0] * self.pde.grid_size)
        #_rhs = np.array([0])
        #_rhs = torch.tensor(_rhs, dtype=dtype, device=self.device).reshape(1,1).repeat(self.bs, self.pde.grid_size)
        _rhs = torch.tensor(_rhs, dtype=dtype, device=self.device)#.reshape(1,1).repeat(self.bs, self.pde.grid_size)
        #self.rhs = _rhs #nn.Parameter(_rhs)
        self.rhs = nn.Parameter(_rhs)

        #self.steps0 = torch.logit(self.step_size*torch.ones(1,*self.pde.step_grid_shape[0]))
        #self.steps0 = torch.logit(self.step_size*torch.ones(1,*self.pde.step_grid_shape[0]))
        #self.steps0 = torch.logit(self.step_size*torch.ones(1,self.pde.step_grid_shape[0][0],1))
        self.steps0 = torch.logit(self.step_size*torch.ones(1,self.coord_dims[0]-1))
        #self.steps0 = torch.logit(self.step_size*torch.ones(1,1,1))
        #self.steps0 = nn.Parameter(self.steps0)

        self.steps1 = torch.logit(self.step_size*torch.ones(1,self.coord_dims[1]-1))
        #self.steps1 = torch.logit(self.step_size*torch.ones(1,*self.pde.step_grid_shape[1]))
        #self.steps1 = torch.logit(self.step_size*torch.ones(1,1,1))
        #self.steps1 = nn.Parameter(self.steps1)

        self._dfnn = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(1024,1024),
            #nn.ReLU(),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            #nn.Linear(1024,self.pde.n_orders),
            #nn.Tanh()
        )

        self.rhs_nn = nn.Sequential(
            #nn.Linear(1024,1024),

            #nn.ReLU(),
            #nn.Linear(1024,1),
            nn.Linear(1024,self.pde.grid_size),
            #nn.Tanh()
        )

        self.cf_nn = nn.Sequential(
            nn.Linear(1024,self.pde.n_orders),
            #nn.Tanh()
        )

        self.iv_nn = nn.Sequential(
            nn.Linear(1024,2*self.coord_dims[0]-1 + 2*self.coord_dims[1]-3),
            #nn.Tanh()
        )


        
    def forward(self, init_list, check=False):
        #print('cin ', self.coeffs)
        _res = self._dfnn(self.coeffs)
        #_coeffs = 3*self.cf_nn(self.coeffs)
        #_coeffs = 3*self.cf_nn(_res)
        _coeffs = self.cf_nn(_res)
        #_coeffs = self.cf_nn(_res)
        #_coeffs[..., -2] = 1.
        rhs = self.rhs_nn(_res)
        #iv_rhs = self.iv_nn(_res)
        #_coeffs = self.cf_nn(self.coeffs).clip(min=-5, max=5)
        #cnorm = _coeffs.pow(2).sum(dim=-1,keepdim=True).clip(min=1e-5)

        #_coeffs = self.coeffs#.clone()
        #_coeffs = 5*torch.tanh(_coeffs)
        #coeffs = coeffs.clone()
        #coeffs = _coeffs#/cnorm
        #coeffs[:,:,0] = 0.
        #coeffs = coeffs
        #print(coeffs, coeffs.shape)
        #coeffs = self.coeffs.unsqueeze(0).repeat(self.bs,1,self.pde.grid_size,1)
        coeffs = _coeffs.unsqueeze(0).repeat(self.bs,1,self.pde.grid_size,1)
        #coeffs = self.coeffs.unsqueeze(0).repeat(self.bs,1,self.pde.grid_size,1)
        #coeffs = coeffs.unsqueeze(0).repeat(self.bs,1,self.pde.grid_size,1)
        #coeffs = self.coeffs.unsqueeze(0).repeat(self.bs,1,1,1)
        #coeffs[:,:,:,0] = 0.
        #coeffs[:,:,:,2] = 0.
        #coeffs[:,:,:,3] = 0.
        #coeffs[:,:,:,4] = 0.
        #coeffs[:,:,:,5] = 0.

        #iv_rhs = self.iv_rhs
        iv_rhs = torch.cat(init_list, dim=0)


        #iv_rhs10[0] = iv_rhs00[0]

        #iv_rhs = torch.cat([iv_rhs00, iv_rhs10,iv_rhs01,iv_rhs11], dim=1)
        #iv_rhs = torch.cat([iv_rhs00, iv_rhs10], dim=1)
        #iv_rhs = torch.cat([iv_rhs00, iv_rhs11], dim=1)

        #if self.n_iv > 0:
        #    #iv_rhs = iv_rhs.unsqueeze(0).repeat(self.bs,1, 1).type_as(coeffs)
        #    iv_rhs = iv_rhs.unsqueeze(0).repeat(self.bs,1).type_as(coeffs)


        #steps0 = self.step_size*torch.ones(1,*self.pde.step_grid_shape[0]).type_as(coeffs)
        #steps1 = self.step_size*torch.ones(1,*self.pde.step_grid_shape[1]).type_as(coeffs)

        #steps0 = steps0.repeat(1,*self.pde.step_grid_shape[0]).type_as(coeffs)
        #steps1 = steps1.repeat(1,*self.pde.step_grid_shape[1]).type_as(coeffs)

        steps0 = self.steps0.type_as(coeffs)
        steps1 = self.steps1.type_as(coeffs)
        #steps0 = self.steps0
        #steps1 = self.steps1
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
        #print(steps0)
        #print(steps1)
        #steps = steps.repeat(self.bs,1, 1)#.double()
        steps_list = [steps0, steps1]

        #rhs = self.rhs#.reshape(1,1).repeat(self.bs, self.pde.grid_size)
        #rhs = rhs.reshape(1,1).repeat(self.bs, self.pde.grid_size)
        #rhs = rhs.type_as(coeffs)

        u0,u,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)

        #print(eps, eps.abs().min(), eps.abs().mean())


        return eps.max(), u0,_coeffs#, u1,u2,steps

method = Method()
dataset = SineDataset(end=method.model.end, coord_dims=method.model.coord_dims)

def train():
    datamodule = SineDataModule(dataset=dataset)

    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #accelerator="cpu",
        devices=1,
        callbacks=[
            #pl.callbacks.ModelCheckpoint(mode="max", monitor="val_accuracy"),
            #pl.callbacks.RichProgressBar(),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(method, datamodule=datamodule)


if __name__ == "__main__":
    train()