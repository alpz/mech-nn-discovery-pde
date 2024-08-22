import sys
#sys.path.append("../../")

import torch.nn as nn
import torch
#from lp_dyn_cent_sys_dim import ODESYSLP as ODELP
#from lp_dyn_cent_sys_sparse import ODESYSLP as ODELP_sys
#from solver.lp_sparse_forward_diff import ODESYSLP #as ODELP_sys
import solver.lp_pde
from solver.lp_pde import PDESYSLP #as ODELP_sys
from solver.lp_pde_central_diff import PDESYSLP as PDESYSLPEPS #as ODELP_sys
from torch.nn.parameter import Parameter
import numpy as np

from torch.autograd import gradcheck

from config import PDEConfig as config, SolverType
#from qp_primal_direct_batched_sys import QPFunction
if config.pde_linear_solver == SolverType.DENSE_CHOLESKY:
    from solver.qp_primal_direct_pde import QPFunction as QPFunctionSys
elif config.pde_linear_solver == SolverType.DENSE_CHOLESKY_DUAL:
    from solver.qp_dual_direct_pde import QPFunction as QPFunctionSys
elif config.pde_linear_solver == SolverType.SPARSE_INDIRECT_CG:
    from solver.qp_primal_indirect_batched_sparse_pde import QPFunction as QPFunctionSys
elif config.pde_linear_solver == SolverType.SPARSE_INDIRECT_BLOCK_CG:
    from solver.qp_primal_indirect_batched_block_sparse_pde import QPFunction as QPFunctionSys
else:
    raise ValueError("unknown solver type")


class PDEINDLayer(nn.Module):
    """ class for PDE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, gamma=0.5, alpha=0.1, double_ret=False, device=None):
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

        self.pde = PDESYSLP(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, coord_dims=self.coord_dims, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        self.n_orders = len(self.pde.var_set.mi_list)
        self.grid_size = self.pde.var_set.grid_size
        #self.step_grid_size = self.pde.step_grid_size
        self.step_grid_shape = self.pde.step_grid_shape
        #self.iv_grid_size = self.pde.t0_grid_size

        self.qpf = QPFunctionSys(self.pde, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps_list):
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.grid_size, self.n_orders)
        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.grid_size)

        if iv_rhs is not None:
            #iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv*self.iv_grid_size)
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)

        for i in range(self.n_coord):
            steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])


        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps_list = [steps.double() for steps in steps_list]

        derivative_constraints = self.pde.build_derivative_tensor(steps_list)
        eq_constraints = self.pde.build_equation_tensor(coeffs)

        #_values = eq_constraints._values()
        #print(_values.shape, rhs.shape)
        #repr = self.pde.repr_eq(values =_values,rhs=rhs[0], type=solver.lp_pde.ConstraintType.Equation)
        #print(repr)

        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)
        eps = x[:,0]

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

class PDEINDLayerEPS(nn.Module):
    """ class for PDE with dimensions modeled independently"""
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

        self.pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, coord_dims=self.coord_dims, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        self.n_orders = len(self.pde.var_set.mi_list)
        self.grid_size = self.pde.var_set.grid_size
        #self.step_grid_size = self.pde.step_grid_size
        self.step_grid_shape = self.pde.step_grid_shape
        #self.iv_grid_size = self.pde.t0_grid_size

        self.qpf = QPFunctionSys(self.pde, self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps_list):
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

        check=False
        if check:
            import sys
            #test = gradcheck(self.qpf, iv_rhs, eps=1e-4, atol=1e-3, rtol=0.001, check_undefined_grad=False, check_batched_grad=True)
            #test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-5, rtol=0.001, check_undefined_grad=True, check_batched_grad=True)
            try: 
                torch.set_printoptions(precision=4, threshold=1000000, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

                test = gradcheck(self.qpf, (eq_constraints, rhs, iv_rhs, derivative_constraints), 
                                eps=1e-10, atol=1e-4, rtol=0.001, fast_mode=False)
            except Exception as e:
                string = e.args[0].split('tensor')
                numerical = string[1].split('analytical')[0]
                analytical = 'torch.tensor' + string[2]
                numerical = 'torch.tensor' + numerical

                #print(e)
                print(string[0])
                print('numerical', numerical)
                print('--------')
                print('analytical', analytical)
                d = eval(numerical)
                a = eval(analytical)
                print('diff')
                diff = (d-a).abs()
                print(diff)
                print(diff> 0.01)
                print(diff.max())
                print(d.shape)
            sys.exit(0)

        #_values = eq_constraints._values()
        #print(_values.shape, rhs.shape)
        #repr = self.pde.repr_eq(values =_values,rhs=rhs[0], type=solver.lp_pde.ConstraintType.Equation)
        #print(repr)

        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x,lam = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)
        #eps = x[:,0]

        #print(x)
        eps = x[:, self.pde.var_set.num_vars:].abs().max(dim=1)[0]

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
