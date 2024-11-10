
import numpy as np
import torch
import math
import scipy.sparse as sp
import torch.nn as nn
import ipdb
import scipy.sparse as SP
from typing import List

import torch.nn.functional as F
from solver.lp_pde_central_diff import PDESYSLP as PDESYSLPEPS #as ODELP_sys

#add multigrid
from solver.qp_dual_indirect_sparse_pde import QPFunction as QPFunctionSys

# set of KKT matrices

#set of coarse grids
class MultigridSolver():
    #def __init__(self, coord_dims):
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
                    gamma=0.5, alpha=0.1, double_ret=False, device=None):
        super().__init__()
        dtype = torch.float64 if self.solver_dbl else torch.float32

        # placeholder step size
        self.step_size = 0.01
        self.coord_dims = coord_dims
        self.n_coord = len(coord_dims)
        self.order = order
        self.n_ind_dim = n_ind_dim
        self.n_dim = 1 
        self.n_equations =1 
        self.n_iv = n_iv
        self.n_iv_steps = 1 
        self.bs = bs
        self.device = device
        self.solver_dbl = solver_dbl
        self.init_index_mi_list = init_index_mi_list

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")


        self.dim_list = []
        self.size_list = []
        #for 32 only
        dims = coord_dims
        self.n_grids = 3
        for i in range(self.n_grid):
            dims = np.array(dims)
            size = np.prod(dims)
            assert(np.max(dims)>=8)
            self.dim_list.append(tuple(dims))
            self.size_list.append(size)
            dims = dims//2

        self.pde_list: List[PDESYSLPEPS] = []
        for dim in self.dim_list:
            pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, 
                        coord_dims=dim, step_size=self.step_size, order=self.order,
                        n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, 
                        dtype=dtype, device=self.device)
            self.pde_list.append(pde)

    def fill_coarse_grid(self, pde: PDESYSLPEPS, new_shape, coeffs, rhs, iv_rhs, steps_list):
        n_orders = len(pde.var_set.mi_list)

        #downsample coeffs, rhs, step, iv_rhs
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, *self.coord_dims, n_orders)
        rhs = rhs.reshape(self.bs*self.n_ind_dim, *self.coord_dims)

        for i in range(self.n_coord):
            steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,self.coord_dims[i]-1)

        iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)
        iv_list = []
        offset = 0
        for ivf in self.init_index_mi_list:
            pair = ivf(*self.coord_dims)
            range_begin = np.array(pair[2])
            range_end = np.array(pair[3])
            iv_shape_dims = range_end+1 - range_begin
            iv_size = np.prod(iv_shape_dims)

            iv = iv_rhs[:, offset:offset+iv_size]
            offset = offset+iv_size

            iv_list.append(iv)



        pass

    def downsample_coeffs(self, coeffs, old_shape,  new_shape, n_orders):
        grid_size = np.prod(np.array(old_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, grid_size, n_orders)
        coeffs = coeffs.permute(0,2,1)
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, *old_shape)

        if len(old_shape) == 2:
            mode='bilinear'
        elif len(old_shape) == 3:
            mode='trilinear'
        else:
            raise ValueError('incorrect num coordinates')

        coeffs = F.interpolate(coeffs, size=new_shape, mode=mode, align_corners=True)

        grid_size = np.prod(np.array(new_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, grid_size)
        coeffs = coeffs.permute(0,2,1)

        return coeffs

    def downsample_rhs(self, rhs, old_shape,  new_shape):
        grid_size = np.prod(np.array(old_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, 1, *old_shape)

        if len(old_shape) == 2:
            mode='bilinear'
        elif len(old_shape) == 3:
            mode='trilinear'
        else:
            raise ValueError('incorrect num coordinates')

        rhs = F.interpolate(rhs, size=new_shape, mode=mode, align_corners=True)

        grid_size = np.prod(np.array(new_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, grid_size)

        return rhs

    def downsample_steps(self, steps_list, old_shape,  new_shape):
        new_steps_list = []
        for i in range(self.n_coord):
            #steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])
            steps = steps_list[i]
            steps = steps.reshape(self.bs*self.n_ind_dim,old_shape[i]-1)

            steps = F.interpolate(steps, size=new_shape, mode='linear', align_corners=True)
            steps = steps.reshape(self.bs*self.n_ind_dim,new_shape[i]-1)
            new_steps_list.append(steps)

        return new_steps_list


    def fill_grid(self, pde: PDESYSLPEPS, coeffs, rhs, iv_rhs, steps_list):
        grid_size = pde.var_set.grid_size
        n_orders = len(pde.var_set.mi_list)
        step_grid_shape = pde.step_grid_shape

        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, grid_size, n_orders)
        rhs = rhs.reshape(self.bs*self.n_ind_dim, grid_size)

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

        derivative_constraints = pde.build_derivative_tensor(steps_list)
        eq_constraints = pde.build_equation_tensor(coeffs)

        return derivative_constraints, eq_constraints, steps_list, iv_rhs

    def restriction(self):
        pass

    def prolongation(self):
        pass

    def smooth_jacobi(self, nsteps):
        pass

class MultigridLayer(nn.Module):
    """ Multigrid layer """
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
                    gamma=0.5, alpha=0.1, double_ret=False, device=None):
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
        #dtype = torch.float64 if DBL else torch.float32

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        self.mg_solver = MultigridSolver(bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, 
                                    n_iv_steps, solver_dbl=True,
                                    gamma=0.5, alpha=0.1, double_ret=False, device=None)
        #self.pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, coord_dims=self.coord_dims, step_size=self.step_size, order=self.order,
        #                 n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        self.pde = self.mg_solver.pde_list[0]
        self.n_orders = len(self.pde.var_set.mi_list)
        self.grid_size = self.pde.var_set.grid_size
        #self.step_grid_size = self.pde.step_grid_size
        self.step_grid_shape = self.pde.step_grid_shape
        #self.iv_grid_size = self.pde.t0_grid_size

        self.qpf = QPFunctionSys(self.pde, self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps_list):
        #interpolate and fill grids: coeffs, rhs, iv_rhs, steps
        #
        
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


        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x,lam = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)
        #eps = x[:,0]

        #print(x)
        eps = x[:, self.pde.var_set.num_vars:].abs()#.max(dim=1)[0]

        #shape: batch, grid, order
        u = self.pde.get_solution_reshaped(x)

        u = u.reshape(self.bs, self.n_ind_dim, *u.shape[1:])
        #shape: batch, step, vars, order
        #u = u.permute(0,2,1,3)

        u0 = u[:,:,:,0]
        #u1 = u[:,:,:,1]
        #u2 = u[:,:,:,2]
        
        #return u0, u1, u2, eps#, steps
        return u0, u, eps
