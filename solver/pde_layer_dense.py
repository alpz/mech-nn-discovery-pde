
import numpy as np
import torch

import cupy as cp
import cupyx.scipy.sparse as CSP
import cupyx.scipy.sparse.linalg as CSPLA

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#seed_everything(10)

import math
import scipy.sparse as sp
import torch.nn as nn
import ipdb
import scipy.sparse as SP
from typing import List

import torch.nn.functional as F
from solver.lp_pde_central_diff import PDESYSLP as PDESYSLPEPS #as ODELP_sys

#add multigrid
#from solver.qp_dual_indirect_sparse_pde import QPFunction as QPFunctionSys
#from solver.qp_dual_sparse_multigrid_normal import QPFunction as QPFunctionSys
import solver.qp_dual_dense_normal_kkt as MGS #import QPFunction as QPFunctionSys
#import solver.qp_dual_sparse_multigrid_normal2 as MGS #import QPFunction as QPFunctionSys
#import solver.qp_dual_sparse_multigrid_normal_dense as MGS #import QPFunction as QPFunctionSys
from config import PDEConfig as config
#import solver.cg as CG

#from torch.autograd import gradcheck
# set of KKT matrices
#torch.autograd.detect_anomaly()


class PDEDenseLayer(nn.Module):
    """ Dense PDE Layer layer """
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
                    evolution=False,
                    gamma=0.5, alpha=0.1, double_ret=False,device=None):
        super().__init__()
        # placeholder step size
        self.step_size = 0.01
        #self.end = n_step * self.step_size
        #self.n_step = n_step #int(self.end / self.step_size)
        self.coord_dims = coord_dims
        self.n_coord = len(coord_dims)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1 
        self.n_equations =1 # n_equations
        self.n_iv = n_iv
        self.n_iv_steps = 1 #n_iv_steps
        self.bs = bs
        #self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.solver_dbl = solver_dbl
        self.evolution=evolution
        #dtype = torch.float64 if DBL else torch.float32

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        #self.mg_solver = MultigridSolver(bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, 
        #                            n_iv_steps, solver_dbl=True, n_grid=n_grid,
        #                            evolution=self.evolution,
        #                            gamma=0.5, alpha=0.1, double_ret=False, 
        #                            device=None)
        self.pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, coord_dims=self.coord_dims, 
                               step_size=self.step_size, order=self.order, evolution=False,
                                n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, 
                                n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        #self.pde = self.mg_solver.pde_list[0]
        self.n_orders = len(self.pde.var_set.mi_list)
        self.grid_size = self.pde.var_set.grid_size
        #self.step_grid_size = self.pde.step_grid_size
        self.step_grid_shape = self.pde.step_grid_shape
        #self.iv_grid_size = self.pde.t0_grid_size

        #self.qpf = MGS.QPFunction(self.pde, self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)
        self.qpf = MGS.QPFunction(self.pde, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps_list):
        #interpolate and fill grids: coeffs, rhs, iv_rhs, steps
        #
        #self.mg_solver.device = rhs.device
        
        #build finest grid data
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.grid_size, self.n_orders)
        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.grid_size)

        if iv_rhs is not None:
            #iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv*self.iv_grid_size)
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)

        for i in range(self.n_coord):
            #steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])
            steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,self.coord_dims[i]-1)


        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps_list = [steps.double() for steps in steps_list]

        derivative_constraints = self.pde.build_derivative_tensor(steps_list)
        eq_constraints = self.pde.build_equation_tensor(coeffs)

        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints, coeffs, steps_list)
        eps = None #x[:, self.pde.var_set.num_vars:].abs()#.max(dim=1)[0]

        #shape: batch, grid, order
        u = self.pde.get_solution_reshaped(x)

        u = u.reshape(self.bs, self.n_ind_dim, *u.shape[1:])
        #shape: batch, step, vars, order
        #u = u.permute(0,2,1,3)

        u0 = u[:,:,:,0]
        #u1 = u[:,:,:,1]
        #u2 = u[:,:,:,2]
        
        #return u0, u1, u2, eps#, steps
        return u0, u, eps#,out
