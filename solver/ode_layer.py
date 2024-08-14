import sys

import torch.nn as nn
import torch
from solver.lp_sparse_forward_diff import ODESYSLP #as ODELP_sys
#from solver.lp_sparse_central_diff import ODESYSLP as ODELP_central_sys
from solver.lp_sparse_central_diff_eps import ODESYSLP as ODESYSLPEPS
from torch.nn.parameter import Parameter
import numpy as np

import torch

from config import ODEConfig, SolverType

if ODEConfig.linear_solver == SolverType.DENSE_CHOLESKY:
    from solver.qp_primal_direct_batched_sparse_sys import QPFunction as QPFunctionSys
if ODEConfig.linear_solver == SolverType.SPARSE_INDIRECT_BLOCK_CG:
    from solver.qp_primal_indirect_batched_block_sparse_sys import QPFunction as QPFunctionSys

from solver.qp_dual_indirect_sparse import QPFunction as QPFunctionSysEPS
from torch.autograd import gradcheck


#DBL=False

class ODEINDLayer(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, step_size=0.1, solver_dbl=True, gamma=0.5, alpha=0.1, central_diff=True, double_ret=False, device=None):
        super().__init__()
        # placeholder step size
        self.step_size = step_size
        #self.end = n_step * self.step_size
        self.n_step = n_step #int(self.end / self.step_size)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1 #n_dim
        self.n_equations =1 # n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.solver_dbl = solver_dbl
        #dtype = torch.float64 if DBL else torch.float32

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        self.ode = ODESYSLP(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=self.n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)


        self.qpf = QPFunctionSys(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_step,self.n_dim, self.order + 1)


        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_step)
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv_steps*self.n_iv)

        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,1)


        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_constraints = self.ode.build_equation_tensor(coeffs)

        check=False
        if check:
            import sys
            #test = gradcheck(self.qpf, iv_rhs, eps=1e-4, atol=1e-3, rtol=0.001, check_undefined_grad=False, check_batched_grad=True)
            #test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-5, rtol=0.001, check_undefined_grad=True, check_batched_grad=True)
            test = gradcheck(self.qpf, (eq_constraints, rhs, iv_rhs, derivative_constraints), eps=1e-4, atol=1e-3, rtol=0.001)
            sys.exit(0)

        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)

        eps = x[:,0]

        #shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        u = u.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)
        #shape: batch, step, vars, order
        #u = u.permute(0,2,1,3)

        u0 = u[:,:,:,0]
        u1 = u[:,:,:,1]
        u2 = u[:,:,:,2] if self.order >=2 else None
        
        return u0, u1, u2, eps, steps

class ODESYSLayer(nn.Module):
    def __init__(self, bs, order, n_ind_dim, n_dim, n_equations, n_iv, n_iv_steps, n_step, periodic_boundary=False, solver_dbl=True, gamma=0.5, alpha=0.1, double_ret=True, device=None):
        super().__init__()

        # placeholder step size
        self.step_size = 0.1
        self.n_step = n_step 
        self.order = order

        self.n_dim = n_dim
        self.n_ind_dim = n_ind_dim
        self.n_equations = n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps

        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.solver_dbl = solver_dbl

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if solver_dbl else torch.float32

        self.ode = ODESYSLP(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                        periodic_boundary=periodic_boundary, n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)


        self.qpf = QPFunctionSys(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_equations, self.n_step,self.n_dim, self.order + 1)

        #n_equation, n_step
        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_equations*self.n_step)

        #iv_steps, n_dim, n_iv
        if iv_rhs is not None:
            #iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv_steps*self.n_dim*self.n_iv)
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)

        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,self.n_dim)


        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_constraints = self.ode.build_equation_tensor(coeffs)

        check=False
        if check:
            import sys
            #test = gradcheck(self.qpf, iv_rhs, eps=1e-4, atol=1e-3, rtol=0.001, check_undefined_grad=False, check_batched_grad=True)
            #test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-5, rtol=0.001, check_undefined_grad=True, check_batched_grad=True)
            try: 
                test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-3, rtol=0.001)
            except Exception as e:
                print(e)

            sys.exit(0)

        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)

        eps = x[:,0]

        #shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        u = u.reshape(self.bs,self.n_ind_dim, self.n_step, self.n_dim, self.order+1)
        #shape: batch, step, vars, order
        #u = u.permute(0,2,1,3)

        u0 = u[:,:,:,:,0]
        u1 = u[:,:,:,:,1]
        u2 = u[:,:,:,:,2]
        
        return u0, u1, u2,eps, u
        #return u0, None, u2,eps, u

class ODEINDLayerTest(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, step_size=0.1, solver_dbl=True, gamma=0.5, alpha=0.1, central_diff=True, double_ret=False, device=None):
        super().__init__()
        # placeholder step size
        self.step_size = step_size
        #self.end = n_step * self.step_size
        self.n_step = n_step #int(self.end / self.step_size)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1 #n_dim
        self.n_equations =1 # n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.solver_dbl = solver_dbl
        #dtype = torch.float64 if DBL else torch.float32

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        self.ode = ODESYSLPEPS(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=self.n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, 
                         add_eps_constraint=False, device=self.device)


        self.qpf =QPFunctionSysEPS(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

        self.num_var = self.ode.num_vars
        self.num_eps = self.ode.num_added_eps_vars

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_step,self.n_dim, self.order + 1)


        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_step)
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv_steps*self.n_iv)

        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,1)


        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        #derivative_A = self.ode.build_derivative_tensor_test(steps)
        derivative_A = self.ode.build_derivative_tensor(steps)
        #derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_A = self.ode.build_equation_tensor(coeffs)


        At, ub = self.ode.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        #x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)

        #eps = x[:,0]

        ##shape: batch, step, vars (== 1), order
        #u = self.ode.get_solution_reshaped(x)

        #u = u.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)
        ##shape: batch, step, vars, order
        ##u = u.permute(0,2,1,3)

        #u0 = u[:,:,:,0]
        #u1 = u[:,:,:,1]
        #u2 = u[:,:,:,2]
        
        #return u0, u1, u2, eps, steps, eq_constraints, self.ode.initial_A,  derivative_constraints, self.ode.eps_A 
        #return None, None, None, None, None, eq_constraints, self.ode.initial_A,  derivative_constraints, self.ode.eps_A 
        return None, None, None, None, None,  At, ub



class ODEINDLayerTestEPS(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, step_size=0.1, solver_dbl=True, gamma=0.5, alpha=0.1, central_diff=True, double_ret=False, device=None):
        super().__init__()
        # placeholder step size
        self.step_size = step_size
        #self.end = n_step * self.step_size
        self.n_step = n_step #int(self.end / self.step_size)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1 #n_dim
        self.n_equations =1 # n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.solver_dbl = solver_dbl
        #dtype = torch.float64 if DBL else torch.float32

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        self.ode = ODESYSLPEPS(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=self.n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, 
                         add_eps_constraint=False, device=self.device)

        self.num_var = self.ode.num_vars
        self.num_eps = self.ode.num_added_eps_vars
        self.num_constraints = self.ode.num_added_constraints

        self.qpf = QPFunctionSysEPS(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps, lam_init):
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_step,self.n_dim, self.order + 1)


        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_step)
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv_steps*self.n_iv)

        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,1)


        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_A = self.ode.build_derivative_tensor(steps)
        #derivative_A = self.ode.build_derivative_tensor_test(steps)
        #derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_A = self.ode.build_equation_tensor(coeffs)

        #eq_A = eq_A.detach()
        #eq_A.requires_grad=False
        #rhs.requires_grad=True

        check=False
        if check:
            import sys
            #test = gradcheck(self.qpf, iv_rhs, eps=1e-4, atol=1e-3, rtol=0.001, check_undefined_grad=False, check_batched_grad=True)
            #test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-5, rtol=0.001, check_undefined_grad=True, check_batched_grad=True)
            try: 
                torch.set_printoptions(precision=4, threshold=1000000, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

                test = gradcheck(self.qpf, (eq_A, rhs, iv_rhs, derivative_A), eps=1e-6, atol=1e-4, rtol=0.001)
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

        #At, ub = self.ode.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x,lam = self.qpf(eq_A, rhs, iv_rhs, derivative_A, lam_init)
        #x = self.solve(eq_A, rhs, iv_rhs, derivative_A)

        #eps = x[:,0]

        ##shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        #eps = x[:, self.ode.num_vars:].abs().max(dim=1)[0]
        eps = x[:, self.ode.num_vars:].abs()#.max(dim=1)[0]
        #eps = x[0]

        #u = u.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)
        ##shape: batch, step, vars, order
        ##u = u.permute(0,2,1,3)

        u0 = u[:,:,:,0]
        u1 = u[:,:,:,1]
        u2 = u[:,:,:,2] if self.order >=2 else None
        
        #return u0, u1, u2, eps, steps, eq_constraints, self.ode.initial_A,  derivative_constraints, self.ode.eps_A 
        return u0, u1, u2, eps, steps, lam#, eq_constraints, self.ode.initial_A,  derivative_constraints, self.ode.eps_A 
        #return None, None, None, None, None, eq_constraints, self.ode.initial_A,  derivative_constraints, self.ode.eps_A 
        #return None, None, None, None, None,  At, ub

    def solve(self, eq_A, rhs, iv_rhs, derivative_A):
    #def forward(ctx, coeffs, rhs, iv_rhs):
        #bs = coeffs.shape[0]
        bs = rhs.shape[0]
        #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
        #At, ub = ode.fill_block_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
        #A, A_rhs = self.ode.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
        A, A_rhs = self.ode.fill_constraints_dense_torch(eq_A, rhs, iv_rhs, derivative_A)
        #A = eq_A.to_dense()
        #A_rhs = rhs #= self.ode.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
        #At = A.transpose(1,2)

        A = A.to_dense()
        At = A.transpose(1,2)

        num_eps = self.ode.num_added_eps_vars
        num_var = self.ode.num_vars
        #u = l
        P_diag = torch.ones(num_eps).type_as(rhs)*1e1
        P_zeros = torch.zeros(num_var).type_as(rhs) +1e-2
        P_diag = torch.cat([P_zeros, P_diag])
        P_diag_inv = 1/P_diag
        P_diag_inv = P_diag_inv.unsqueeze(0)

        #c = torch.zeros(num_var+num_eps, device=A.device).type_as(rhs)
        c = torch.ones(num_var+num_eps, device=A.device).type_as(rhs)
        rhs = c
        rhs = P_diag_inv*rhs
        rhs = torch.bmm(A, rhs.unsqueeze(2))
        #TODO rhs is zero upto here. remove the above
        rhs = rhs.squeeze(2) + A_rhs
        #rhs =  A_rhs

        #lam, info = cg_matvec([A, P_diag_inv, At], rhs, maxiter=2000)


        ######### dense
        #A = A.to_dense()
        #At = At.to_dense()
        PAt = P_diag_inv.unsqueeze(2)*At
        APAt = torch.bmm(A, PAt)
        L,info = torch.linalg.cholesky_ex(APAt,upper=False)
        lam = torch.cholesky_solve(rhs.unsqueeze(2), L)
        lam = lam.squeeze(2)
        #########

        #print('torch cg info ', info)
        #lam,info = SPSLG.lgmres(pdmat, pd_rhs)
        #xl = -Pinv_s@(A_s.T@lam -q)

        #xl = -Pinv_s@(A_s.T@lam -c)
        x = lam.unsqueeze(2)
        x = torch.bmm(At, x)
        x = -P_diag_inv*(x.squeeze(2) - c)
        
        #ctx.save_for_backward(A, P_diag_inv, x, lam, L)
        
        #if not double_ret:
        #    x = x.float()
        return x