
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

import solver.qp_dual_sparse_multigrid_normal_kkt as MGS #import QPFunction as QPFunctionSys

from config import PDEConfig as config
import solver.cg as CG

from torch.autograd import gradcheck
# set of KKT matrices
#torch.autograd.detect_anomaly()

#set of coarse grids
class MultigridSolver():
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
                    evolution=False, downsample_first=True,
                    gamma=0.5, alpha=0.1, double_ret=False, n_grid=2, device=None):
        super().__init__()
        dtype = torch.float64 if solver_dbl else torch.float32

        print("Multigrid: num grid ", n_grid)
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
        self.evolution=evolution
        #whether to downsample the time dimension.
        self.downsample_first = downsample_first

        
        interp_modes={1:'linear', 2:'bilinear', 3:'trilinear'}
        #print('nearest')
        #interp_modes={1:'nearest', 2:'nearest', 3:'nearest'}
        self.align_corners = True
        #self.align_corners = None
        self.interp_mode = interp_modes[self.n_coord]

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        if self.evolution:
            print('building evolution equation')


        self.dim_list = []
        self.size_list = []
        dims = coord_dims
        self.n_grid = n_grid
        for i in range(self.n_grid):
            dims = np.array(dims)
            size = np.prod(dims)
            assert(np.min(dims)>=8)
            self.dim_list.append(tuple(dims))
            self.size_list.append(size)

            if downsample_first:
                dims = dims//2
            else:
                dims[1:] = dims[1:]//2

        print('multigrid dimension list', self.dim_list)
        self.pde_list: List[PDESYSLPEPS] = []
        for dim in self.dim_list:
            pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, 
                        evolution=self.evolution,
                        coord_dims=dim, step_size=self.step_size, order=self.order,
                        n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, 
                        dtype=dtype, device=self.device)
            self.pde_list.append(pde)

    #@torch.no_grad()
    def fill_coarse_grids(self, coeffs, rhs, iv_rhs, steps_list):

        A_list = []
        A_rhs_list = []
        #deriv_weights_list = []


        #steps are incrementally downsampled by adding pairs. The rest are done directly
        new_steps_list = steps_list
        new_coeffs = coeffs
        new_rhs = rhs
        new_iv_rhs = iv_rhs
        for k in range(1, self.n_grid):
            coarsest = True if k==self.n_grid-1 else False


            pde = self.pde_list[k]
            new_shape = self.dim_list[k]
            old_shape = self.dim_list[k-1]

            n_orders = len(pde.var_set.mi_list)

            new_coeffs = self.downsample_coeffs(new_coeffs, old_shape,  new_shape, n_orders)
            new_rhs = self.downsample_rhs(new_rhs, old_shape,  new_shape)
            new_steps_list = self.downsample_steps(new_steps_list, old_shape)
            new_iv_rhs = self.downsample_iv(new_iv_rhs, old_shape,  new_shape)

            if self.solver_dbl:
                new_coeffs = new_coeffs.double()
                new_rhs = new_rhs.double()
                new_iv_rhs = new_iv_rhs.double() if iv_rhs is not None else None
                new_steps_list = [steps.double() for steps in new_steps_list]

            derivative_constraints = pde.build_derivative_tensor(new_steps_list)
            eq_constraints = pde.build_equation_tensor(new_coeffs)

            if coarsest:
                A, A_rhs = pde.fill_constraints_torch(eq_constraints, new_rhs, new_iv_rhs, derivative_constraints)
            else:
                A, A_rhs = pde.fill_block_constraints_torch(eq_constraints, new_rhs, new_iv_rhs, derivative_constraints)

            num_eps = pde.var_set.num_added_eps_vars
            num_var = pde.var_set.num_vars


            A_list.append(A)
            A_rhs_list.append(A_rhs)

        return A_list, A_rhs_list


    def make_coarse_AtA_matrices(self, A_list, A_rhs_list):
        AtA_list = []
        D_list = []
        rhs_list = []
        L_list = []
        U_list = []
        for i in range(1,self.n_grid):
            coarsest = True if i==self.n_grid-1 else False
            AtA,D,rhs,L,U = self.make_AtA(self.pde_list[i], A_list[i-1], A_rhs_list[i-1], coarsest=coarsest)
            AtA_list.append(AtA)
            D_list.append(D)
            L_list.append(L)
            U_list.append(U)

            rhs_list.append(rhs)

        return AtA_list, rhs_list, D_list, L_list,U_list

        
    def get_tril(self, M):
        """ TODO: get block diag M"""
        #M = M[0]
        indices = M._indices()
        values = M._values()
        rows = indices[0]
        cols = indices[1]

        mask = (cols <= rows)

        new_indices = indices[:, mask]
        new_values = values[mask]

        L = torch.sparse_coo_tensor(new_indices, new_values,
                                       size=M.size(), dtype=M.dtype)

        mask = (cols > rows)
        new_indices = indices[:, mask]
        new_values = values[mask]

        U = torch.sparse_coo_tensor(new_indices, new_values,
                                       size=M.size(), dtype=M.dtype)
        #U = U.to_sparse_csr()
        return L,U

    def make_AtA(self, pde: PDESYSLPEPS, A, A_rhs, coarsest=False, ds=1e2, save=False):
        num_eq = pde.num_added_equation_constraints + pde.num_added_initial_constraints
        num_ineq = pde.num_added_derivative_constraints
        bs = A_rhs.shape[0]

        if coarsest:
            A = A.to_dense()
            At = A.transpose(1,2)
            AtA = torch.bmm(At, A)#.unsqueeze(0)
            D = None #torch.sum(A*A, dim=1)
            AtPrhs =torch.bmm(At, A_rhs.unsqueeze(2)).squeeze(2)#.to_dense()
            L,U=None,None 
        else:
            #A = A.to_dense()#[:, :, :num_var]
            At = A.transpose(0,1)

            AtA = torch.sparse.mm(At, A)#.unsqueeze(0)

            D =None # D.to_dense()

            AtPrhs =torch.mm(At, A_rhs.unsqueeze(1)).squeeze(1)#.to_dense()


            cval = cp.asarray(AtA._values())
            crow = cp.asarray(AtA._indices()[0])
            ccol = cp.asarray(AtA._indices()[1])

            cAtA = CSP.coo_matrix((cval, (crow, ccol)), shape=AtA.shape)
            L = CSP.tril(cAtA, k=0, format='csr')
            U = CSP.triu(cAtA, k=1, format='csr')
        return AtA, D, AtPrhs,L,U#, AtA_act, P_diag


    def downsample_coeffs(self, coeffs, old_shape,  new_shape, n_orders):
        grid_size = np.prod(np.array(old_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, grid_size, n_orders)
        _coeffs = coeffs
        coeffs = coeffs.permute(0,2,1)
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, *old_shape)

        coeffs = F.interpolate(coeffs, size=new_shape, mode=self.interp_mode, align_corners=self.align_corners)

        new_grid_size = np.prod(np.array(new_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, new_grid_size)
        coeffs = coeffs.permute(0,2,1)

        return coeffs

    def downsample_rhs(self, rhs, old_shape,  new_shape):
        grid_size = np.prod(np.array(old_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, 1, *old_shape)
        _rhs = rhs

        rhs = F.interpolate(rhs, size=new_shape, mode=self.interp_mode, align_corners=self.align_corners)

        new_grid_size = np.prod(np.array(new_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, new_grid_size)

        return rhs


    def downsample_steps(self, steps_list, old_shape):
        new_steps_list = []
        for i in range(self.n_coord):
            steps = steps_list[i]
            steps = steps.reshape(self.bs*self.n_ind_dim,old_shape[i]-1)
            if i==0:
                if self.downsample_first:
                    steps = steps[:, :-1].reshape(-1, old_shape[i]//2-1, 2).sum(dim=-1)
            else:
                steps = steps[:, :-1].reshape(-1, old_shape[i]//2-1, 2).sum(dim=-1)

            new_steps_list.append(steps)


        return new_steps_list

    def downsample_iv(self, iv_rhs, old_shape,  new_shape):

        iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)
        _iv_rhs = iv_rhs.clone()
        #print(iv_rhs.shape, old_shape, new_shape)
        #ipdb.set_trace()
        iv_list = []
        offset = 0
        for ivf in self.init_index_mi_list:
            pair = ivf(*old_shape)
            range_begin = np.array(pair[2])
            range_end = np.array(pair[3])
            #iv_old_shape = np.squeeze(range_end+1 - range_begin)
            iv_old_shape = (range_end+1 - range_begin)

            new_pair = ivf(*new_shape)
            new_range_begin = np.array(new_pair[2])
            new_range_end = np.array(new_pair[3])
            #iv_new_shape = np.squeeze(new_range_end+1 - new_range_begin)
            iv_new_shape = (new_range_end+1 - new_range_begin)
            #iv_new_shape = np.array([i for i in iv_new_shape if i!= 1])
            #iv_new_shape = np.squeeze(iv_new_shape)

            iv_old_size = np.prod(iv_old_shape)

            iv = _iv_rhs[:, offset:offset+iv_old_size]
            offset = offset+iv_old_size

            _iv = iv.reshape(self.bs*self.n_ind_dim, *iv_old_shape)

            #print(iv.shape, old_shape, tuple(iv_new_shape))

            #iv = _iv.reshape(-1, 8, 2)[:,:, 1]
            iv = F.interpolate(_iv.unsqueeze(1), size=tuple(iv_new_shape), 
                               mode=self.interp_mode, 
                               #mode='bilinear', 
                               align_corners=self.align_corners)
                               #align_corners=True)
                               #align_corners=None)
            #print('interp iv ', iv.shape)
            iv = iv.reshape(self.bs*self.n_ind_dim, -1)
            #ipdb.set_trace()

            iv_list.append(iv)

        if len(iv_list) == 0:
            iv_rhs = torch.empty(self.bs, 0, device=iv_rhs.device)
        else:
            iv_rhs = torch.cat(iv_list, dim=-1)

        return iv_rhs


    def restrict(self, idx, x, back=False):
        pde = self.pde_list[idx]
        #pro_pde = self.pde_list[idx-1]

        x = torch.tensor(x, device=self.device)
        x = x.reshape(self.bs, -1)

        #bs, grid, num_mi
        x = pde.get_solution_reshaped(x)
        x = x.permute(0,2,1)

        x = x.reshape(*x.shape[0:2], *self.dim_list[idx])

        x = F.interpolate(x, size=self.dim_list[idx+1], 
                            mode=self.interp_mode, 
                            align_corners=self.align_corners)
        x = x.reshape(*x.shape[0:2], self.size_list[idx+1])

        x = x.permute(0,2,1).reshape(x.shape[0], -1)

        x = x.reshape(-1)
        x = cp.asarray(x)

        return x


    def prolong(self, idx, x, back=False):
        pde = self.pde_list[idx]
        #pro_pde = self.pde_list[idx-1]

        x = torch.as_tensor(x, device=self.device)
        x = x.reshape(self.bs, -1)

        #bs, grid, num_mi
        x = pde.get_solution_reshaped(x)
        x = x.permute(0,2,1)

        x = x.reshape(*x.shape[0:2], *self.dim_list[idx])

        x = F.interpolate(x, size=self.dim_list[idx-1], 
                          #mode='bilinear',
                          mode=self.interp_mode, 
                          #align_corners=self.align_corners)
                          align_corners=self.align_corners)
        x = x.reshape(*x.shape[0:2], self.size_list[idx-1])

        x = x.permute(0,2,1).reshape(x.shape[0], -1)

        x = x.reshape(-1)
        x = cp.asarray(x)

        return x

    def mult_AtA(self, A, x):
        x = torch.as_tensor(x, device=self.device)
        x = torch.mm(A, x.unsqueeze(1)).squeeze(1)
        x = cp.asarray(x)
        return x#.to_dense()

    def smooth_gs(self, L, U,  b, x, nsteps=20):
        """GS iteration"""

        for i in range(nsteps):
            x = -U@x + b
            x = CSPLA.spsolve_triangular(L,x,lower=True)
        return x

    def smooth_jacobi(self, As, b, x, D, nsteps=200, w=0.55, back=False):
        """Weighted Jacobi iteration"""
        Dinv = 1/D
        if back:
            w=0.4 #config.jacobi_w
        else:
            w=0.45
        for i in range(nsteps):
            x = x - w*Dinv*self.mult_AtA(As, x) + w*Dinv*b
        return x

    def smooth_cg(self, As, b, x, nsteps=100):
        Alist = [As[0].transpose(1,2), 1/As[1], As[0]]
        x, _  =CG.cg_matvec(Alist, b, x, maxiter=nsteps)

        return x

    def get_residual_norm(self, A, x, b):
        x = torch.as_tensor(x, device=self.device)
        b = torch.as_tensor(b, device=self.device)
        r = b - torch.mm(A, x.unsqueeze(1)).squeeze(1) #torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        r = r.reshape(self.bs, -1)
        b = b.reshape(self.bs, -1)

        d = b.pow(2).sum(dim=-1).sqrt()

        rnorm = r.pow(2).sum(dim=-1).sqrt()
        rrnorm = rnorm/d

        return rnorm, rrnorm

    def factor_coarsest(self, A):
        L,info = torch.linalg.cholesky_ex(A,upper=False, check_errors=True)
        return L

    def solve_coarsest(self, L, b):
        #L,info = torch.linalg.cholesky_ex(A,upper=False, check_errors=True)
        b = torch.tensor(b, device=L.device)
        b = b.reshape(self.bs, -1)
        lam = torch.cholesky_solve(b.unsqueeze(2), L)
        lam = lam.squeeze(2)
        lam = lam.reshape(-1)
        lam = cp.asarray(lam)
        return lam

    @torch.no_grad()
    def v_cycle_gs(self, idx, A_list, AL_list, AU_list, b, x, L, back=False):
        #A,L are torch tensors. Rest are cupy arrays
        A = A_list[idx]
        AL = AL_list[idx]
        AU = AU_list[idx]
        #b = b
        nstep = config.mg_gauss_seidel_steps_pre  #10 # 5 if back and idx == 0 else 5

        x = self.smooth_gs(AL, AU, b, x, nsteps=nstep)
        r = b-self.mult_AtA(A, x) #torch.bmm(A, x.unsqueeze(2)).squeeze(2)

        rH = self.restrict(idx, r, back=back)

        if idx ==self.n_grid-2:
            #deltaH = self.solve_coarsest(A_list[self.n_grid-1], rH)
            if not back:
                #deltaH = self.smooth_jacobi(As_list[idx+1], rH, torch.zeros_like(rH), D_list[idx+1], nsteps=100)
                deltaH = self.solve_coarsest(L, rH)
            else:
                deltaH = self.solve_coarsest(L, rH)
        else:
            xH0 = cp.zeros_like(rH)
            #deltaH,_ = self.v_cycle_jacobi(idx+1, As_list, rH,xH0, D_list,L, back=back)
            deltaH = self.v_cycle_gs(idx+1, A_list, AL_list, AU_list, rH,xH0, L, back=back)

        delta = self.prolong(idx+1, deltaH, back=back)
        x = x+delta

        nstep= config.mg_gauss_seidel_steps_post #10
        x = self.smooth_gs(AL, AU, b, x, nsteps=nstep)

        out = None


        return x

    @torch.no_grad()
    def v_cycle_gs_start(self, A_list, b, AL_list, AU_list,L, n_step=1, back=False):

        b=cp.asarray(b)
        x = cp.zeros_like(b)
        n_step =config.mg_steps_backward if back else config.mg_steps_forward
        for step in range(n_step):
            x = self.v_cycle_gs(0, A_list, AL_list, AU_list, b, x, L, back=back)
        x= torch.as_tensor(x, device=self.device)
        return x#, out#.to_dense()

    #@torch.no_grad()
    #def v_cycle_jacobi_start(self, A_list, b_list, D_list,L, n_step=1, back=False):
    #    x = torch.zeros_like(b_list[0])
    #    #x = torch.randn_like(b_list[0])
    #    #x = torch.rand_like(b_list[0])
    #    n_step =2 # 200 if back else 200
    #    #n_step =1 if back else 1
    #    #n_step=1000
    #    for step in range(n_step):
    #        x, out = self.v_cycle_jacobi(0, A_list, b_list[0], x, D_list,L, back=back)
    #        #if back:
    #        #r,rr = self.get_residual_norm(A_list[0], x, b_list[0] )
    #        #print(f'vcycle end norm: ',step, r,rr.item(),back,'\n')
    #    #x = x.to_dense()
    #    #return x, out#.to_dense()
    #    return x#, out#.to_dense()

    #def full_multigrid_jacobi_start(self, A_list, b_list, D_list,L, back=False):
    #    u = self.solve_coarsest(L, b_list[-1])
    #    for idx in reversed(range(self.n_grid-1)):
    #        print('fmg idx', idx)
    #        u = self.prolong(idx+1, u)
    #        for k in range(1):
    #            u,_ = self.v_cycle_jacobi(idx, A_list, b_list[idx], u, D_list,L, back=back)

    #            r,rr = self.get_residual_norm(A_list[idx], u, b_list[idx] )
    #            print(f'fmg step norm: ', r,rr)

    #    #print(self.AtA_act.shape, b_list[0].shape, u.shape)
    #    #u,_ = CG.gmres(self.AtA_act, b_list[0], u, restart=20, maxiter=800)
    #    #u,_ = CG.cg(self.AtA_act, b_list[0], u, maxiter=100)

    #    #r,rr = self.get_residual_norm(A_list[0], u, b_list[0] )
    #    #print(f'gmres step norm: ', r,rr)
    #    return u

class MultigridLayer(nn.Module):
    """ Multigrid layer """
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
                    evolution=False, downsample_first=True,
                    gamma=0.5, alpha=0.1, double_ret=False,n_grid=2, device=None):
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

        self.mg_solver = MultigridSolver(bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, 
                                    n_iv_steps, solver_dbl=True, n_grid=n_grid,
                                    evolution=self.evolution, downsample_first=downsample_first,
                                    gamma=0.5, alpha=0.1, double_ret=False, 
                                    device=None)
        self.pde = self.mg_solver.pde_list[0]
        self.n_orders = len(self.pde.var_set.mi_list)
        self.grid_size = self.pde.var_set.grid_size
        #self.step_grid_size = self.pde.step_grid_size
        self.step_grid_shape = self.pde.step_grid_shape
        #self.iv_grid_size = self.pde.t0_grid_size

        self.qpf = MGS.QPFunction(self.pde, self.mg_solver, self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)

    def forward(self, coeffs, rhs, iv_rhs, steps_list):
        #interpolate and fill grids: coeffs, rhs, iv_rhs, steps
        #
        self.mg_solver.device = rhs.device
        
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
        eps = None 

        #shape: batch, grid, order
        u = self.pde.get_solution_reshaped(x)

        u = u.reshape(self.bs, self.n_ind_dim, *u.shape[1:])

        u0 = u[:,:,:,0]
        #u1 = u[:,:,:,1]
        #u2 = u[:,:,:,2]
        
        return u0, u, eps
