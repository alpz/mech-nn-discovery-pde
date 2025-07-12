
import torch.nn as nn
import torch

from torch.nn.parameter import Parameter
import numpy as np

import torch
import ipdb
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import Dataset
from solver.pde_layer_dense import PDEDenseLayer

#Fit a noisy exponentially damped sine wave with a second order ODE

class SineDataset(Dataset):
    def __init__(self, end=1, coord_dims=(50,32)):
        _y = np.linspace(0, end, coord_dims[0])
        y0 = np.sin(3*_y) 
        y0 = torch.tensor(y0)

        _xx = np.linspace(0, end, coord_dims[0])[:,np.newaxis]
        _yy= np.linspace(0, end, coord_dims[1])[np.newaxis, :]

        self.damp = (np.exp(-0.1*(_xx) + (_yy-end/2)**2)).reshape(coord_dims)
        self.y = y0.unsqueeze(-1).repeat(1,coord_dims[1])#.transpose(1,0)
        self.y = self.y*self.damp
        
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        y = self.y 
        return y
    
class SineDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=True)
        return train_loader 
        
        

class Method(pl.LightningModule):
    def __init__(self, time_varying_source=True):
        super().__init__()
        self.learning_rate = 0.0001
        self.model = Sine(time_varying_source=time_varying_source, device=self.device)
        self.model = self.model.double()
        
        self.func_list = []
        self.y_list = []
        self.coeffs_list = []
        self.rhs_list = []

    def forward(self, init_list):
        return self.model(init_list,check=False)
    

    def training_step(self, batch, batch_idx):
        y = batch

        y = y.reshape((32,32))

        t0 = y[0, 0:-1].reshape(-1)
        tn = y[-1, 1:-1].reshape(-1)
        x0 = y[1:, 0].reshape(-1)
        xn = y[:, -1].reshape(-1)

        u0, cf,u = self([t0,x0, tn, xn])
        

        loss = (u0.reshape(-1)-y.reshape(-1)).pow(2).mean()
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        self.func_list.append(u0.detach().cpu().numpy())
        self.coeffs_list.append(self.model.coeffs.squeeze().detach().cpu().numpy())
        self.rhs_list.append(self.model.rhs.detach().cpu().numpy())
        
        self.y_list.append(y.detach().cpu().numpy())
        
        return {"loss": loss, 'y':y, 'u0':u0}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class Sine(nn.Module):
    def __init__(self, bs=1, time_varying_source=True, device=None):
        super().__init__()

        self.step_size = 0.05
        self.end =1
        self.order = 2
        #state dimension
        self.n_dim = 1
        self.bs = bs
        self.device  = device
        dtype = torch.float64
        self.coord_dims = (32,32)
        self.time_varying_source = time_varying_source

        #self.n_iv = 1
        #coord, mi_index, begin, end
        self.iv_list = [lambda nx,ny: (0,0, [0,0],[0,ny-2]), 
                        lambda nx,ny:(1,0, [1,0], [nx-1, 0]), 
                        lambda nx,ny:(0,0, [nx-1,1],[nx-1,ny-2]), 
                        lambda nx,ny:(1,0, [0,ny-1], [nx-1, ny-1])
                        ]
        self.pde = PDEDenseLayer(bs=bs, coord_dims=self.coord_dims, order=2, 
                                n_ind_dim=self.n_dim, n_iv=1,
                                init_index_mi_list=self.iv_list,  n_iv_steps=1, 
                                double_ret=True, solver_dbl=True)

        _coeffs = torch.randn((self.n_dim, 1, 1024), dtype=dtype)
        self.coeffs = Parameter(_coeffs)

        #_rhs = np.array([0] * self.pde.grid_size)
        #_rhs = torch.tensor(_rhs, dtype=dtype, device=self.device)#.reshape(1,1).repeat(self.bs, self.pde.grid_size)
        #self.rhs = _rhs #nn.Parameter(_rhs)

        self.steps0 = torch.logit(self.step_size*torch.ones(1,self.coord_dims[0]-1))
        self.steps1 = torch.logit(self.step_size*torch.ones(1,self.coord_dims[1]-1))

        self._dfnn = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
        )

        self.rhs_nn = nn.Sequential(
            nn.Linear(1024,self.pde.grid_size),
        )

        self.cf_nn = nn.Sequential(
            nn.Linear(1024,self.pde.n_orders),
        )

        self.iv_nn = nn.Sequential(
            nn.Linear(1024,2*self.coord_dims[0]-1 + 2*self.coord_dims[1]-3),
        )


        
    def forward(self, init_list, check=False):
        _res = self._dfnn(self.coeffs)
        _coeffs = self.cf_nn(_res)

        if self.time_varying_source:
            rhs = self.rhs_nn(_res)
        else:
            #learn without source term
            rhs = torch.zeros(self.bs, self.pde.grid_size, device=_coeffs.device)

        coeffs = _coeffs.unsqueeze(0).repeat(self.bs,1,self.pde.grid_size,1)

        iv_rhs = torch.cat(init_list, dim=0)
        #iv_rhs = self.iv_nn(_res)


        steps0 = self.steps0.type_as(coeffs)
        steps1 = self.steps1.type_as(coeffs)
        steps0 = torch.sigmoid(steps0).clip(min=0.005, max=0.1)
        steps1 = torch.sigmoid(steps1).clip(min=0.005, max=0.1)
        steps_list = [steps0, steps1]

        u0,u,eps = self.pde(coeffs, rhs, iv_rhs, steps_list)

        return u0,_coeffs,u

def create_model(time_varying_source=True):
    method = Method(time_varying_source=time_varying_source)
    dataset = SineDataset(end=method.model.end, coord_dims=method.model.coord_dims)
    return method, dataset

def train(method, dataset, epochs=100):

    datamodule = SineDataModule(dataset=dataset)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        #callbacks=[
        #],
        log_every_n_steps=1,
    )
    trainer.fit(method, datamodule=datamodule)


if __name__ == "__main__":
    train()