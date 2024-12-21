
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
import solver.qp_dual_sparse_multigrid_normal_kkt as MGS #import QPFunction as QPFunctionSys
#import solver.qp_dual_sparse_multigrid_normal2 as MGS #import QPFunction as QPFunctionSys
#import solver.qp_dual_sparse_multigrid_normal_dense as MGS #import QPFunction as QPFunctionSys
from config import PDEConfig as config
import solver.cg as CG

from torch.autograd import gradcheck
# set of KKT matrices
#torch.autograd.detect_anomaly()

#set of coarse grids
class MultigridSolver():
    #def __init__(self, coord_dims):
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
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
            dims = dims//2

        self.pde_list: List[PDESYSLPEPS] = []
        for dim in self.dim_list:
            pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, 
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
            #print('creating grid ', self.dim_list[k])

            n_orders = len(pde.var_set.mi_list)

            #new_coeffs = self.downsample_coeffs(coeffs, self.coord_dims,  new_shape, n_orders)
            #new_rhs = self.downsample_rhs(rhs, self.coord_dims,  new_shape)
            #new_steps_list = self.downsample_steps(new_steps_list, old_shape)
            #new_iv_rhs = self.downsample_iv(iv_rhs, self.coord_dims,  new_shape)

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

            #A, A_rhs = pde.fill_constraints_torch2(eq_constraints.coalesce(), new_rhs, new_iv_rhs, derivative_constraints.coalesce())
            if coarsest:
                A, A_rhs = pde.fill_constraints_torch(eq_constraints, new_rhs, new_iv_rhs, derivative_constraints)
            else:
                A, A_rhs = pde.fill_block_constraints_torch(eq_constraints, new_rhs, new_iv_rhs, derivative_constraints)
            #A, A_rhs = pde.fill_constraints_torch_dense(eq_constraints.to_dense(), new_rhs, new_iv_rhs, derivative_constraints.to_dense())

            num_eps = pde.var_set.num_added_eps_vars
            num_var = pde.var_set.num_vars

            #A = A.to_dense()#[:, :, :num_var]

            A_list.append(A)
            A_rhs_list.append(A_rhs)
            #deriv_weights_list.append(derivative_weights)

        return A_list, A_rhs_list#, deriv_weights_list


    def make_coarse_AtA_matrices(self, A_list, A_rhs_list):
        AtA_list = []
        D_list = []
        rhs_list = []
        L_list = []
        U_list = []
        #ds_list = [1e6]*self.n_grid
        for i in range(1,self.n_grid):
            coarsest = True if i==self.n_grid-1 else False
            AtA,D,rhs,L,U = self.make_AtA(self.pde_list[i], A_list[i-1], A_rhs_list[i-1], coarsest=coarsest)
            AtA_list.append(AtA)
            D_list.append(D)
            L_list.append(L)
            U_list.append(U)

            #At = A.transpose(1,2)
            #rhs = P_diag_inv*A_rhs_list[i]
            #rhs = torch.bmm(At, rhs.unsqueeze(2)).squeeze(2)
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
        #L = L.to_sparse_csr()

        mask = (cols > rows)
        new_indices = indices[:, mask]
        new_values = values[mask]

        U = torch.sparse_coo_tensor(new_indices, new_values,
                                       size=M.size(), dtype=M.dtype)
        #U = U.to_sparse_csr()
        return L,U

    def make_AtA(self, pde: PDESYSLPEPS, A, A_rhs, coarsest=False, ds=1e2, save=False):
    #def make_AAt(self, pde: PDESYSLPEPS, A, us=1e1, ds=1e-2):
        #AGinvAt
        #P_diag = torch.ones(num_eps).type_as(rhs)*1e3
        #P_zeros = torch.zeros(num_var).type_as(rhs) +1e-5
        num_eq = pde.num_added_equation_constraints + pde.num_added_initial_constraints
        num_ineq = pde.num_added_derivative_constraints
        bs = A_rhs.shape[0]

        #num_eps = pde.var_set.num_added_eps_vars
        #num_var = pde.var_set.num_vars

        #_P_diag = torch.ones(num_ineq, dtype=A.dtype, device='cpu')*config.ds#*us
        #_P_ones = torch.ones(num_eq, dtype=A.dtype, device='cpu')#/ds#/config.ds# +ds
        #ipdb.set_trace()

        #bs = deriv_weights.shape[0]
        #print(deriv_weights.shape, num_ineq)
        #_P_diag = torch.ones(bs, num_ineq, dtype=A.dtype, device=A.device) #ds #config.ds#*us
        ##_P_diag = 1/deriv_weights
        ##_P_diag = deriv_weights
        #_P_ones = torch.ones(bs, num_eq, dtype=A.dtype, device=A.device)#*0.01#/ds#/config.ds# +ds
        #P_diag = torch.cat([_P_ones, _P_diag], dim=-1)#.to(A.device)
        #P_diag_inv = 1/P_diag
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
            #if save:
            #    At.register_hook(lambda grad: print("At grad"))
            #PinvA = P_diag_inv.unsqueeze(1)*A
            #PinvA = P_diag_inv.unsqueeze(2)*A

            #with torch.no_grad():
            AtA = torch.sparse.mm(At, A)#.unsqueeze(0)

            #AtA = [A, P_diag]

            #D = torch.sparse.sum(PinvA*A, dim=1)
            D = torch.sparse.sum(A*A, dim=1)
            D = D.to_dense()

            #P_rhs = P_diag_inv*A_rhs

            #AtPrhs =torch.bmm(At, P_rhs.unsqueeze(2)).squeeze(2)#.to_dense()
            #AtPrhs =torch.bmm(At, A_rhs.unsqueeze(1)).squeeze(1)#.to_dense()
            AtPrhs =torch.mm(At, A_rhs.unsqueeze(1)).squeeze(1)#.to_dense()
            #AtPrhs.register_hook(lambda grad: print('aptrhs grad'))


            cval = cp.asarray(AtA._values())
            crow = cp.asarray(AtA._indices()[0])
            ccol = cp.asarray(AtA._indices()[1])

            cAtA = CSP.coo_matrix((cval, (crow, ccol)), shape=AtA.shape)
            L = CSP.tril(cAtA, k=0, format='csr')
            U = CSP.triu(cAtA, k=1, format='csr')
        return AtA, D, AtPrhs,L,U#, AtA_act, P_diag

    #@torch.no_grad()
    #def get_AtA_dense(self, As):
    #    A = As[0]
    #    P_diag = As[1]
    #    P_diag_inv = 1/P_diag
    #    A = A.to_dense()
    #    At = A.transpose(1,2)
    #    PinvA = P_diag_inv.unsqueeze(2)*A
    #    #PinvA = A

    #    #TODO fix mm
    #    AtA = torch.bmm(At, PinvA)#.unsqueeze(0)

    #    return AtA


    #def downsample_grads(self, coeffs, old_shape,  new_shape, n_orders):
    #    grid_size = np.prod(np.array(old_shape))
    #    coeffs = coeffs.reshape(self.bs*self.n_ind_dim, grid_size, n_orders)
    #    coeffs = coeffs.permute(0,2,1)
    #    coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, *old_shape)

    #    m = old_shape[0]
    #    #print('ols s ', m)
    #    coeffs = coeffs.reshape(self.bs*self.n_ind_dim,n_orders,m//2,2,m//2,2)
    #    coeffs = coeffs[:, :, :, 0, :, 0]

    #    new_grid_size = np.prod(np.array(new_shape))
    #    coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, new_grid_size)
    #    coeffs = coeffs.permute(0,2,1)

    #    return coeffs

    def downsample_coeffs(self, coeffs, old_shape,  new_shape, n_orders):
        grid_size = np.prod(np.array(old_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, grid_size, n_orders)
        _coeffs = coeffs
        coeffs = coeffs.permute(0,2,1)
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, *old_shape)

        #if len(old_shape) == 2:
        #    mode='bilinear'
        #elif len(old_shape) == 3:
        #    mode='trilinear'
        #else:
        #    raise ValueError('incorrect num coordinates')

        #print('ds coeffs ', coeffs.shape, new_shape)
        coeffs = F.interpolate(coeffs, size=new_shape, mode=self.interp_mode, align_corners=self.align_corners)
        #m = old_shape[0]

        #coeffs = coeffs.reshape(self.bs*self.n_ind_dim,n_orders,m//2,2,m//2,2)
        #coeffs = coeffs[:, :, :, 0, :, 0]
        #print('ds coeffs ', coeffs.shape)

        new_grid_size = np.prod(np.array(new_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, new_grid_size)
        coeffs = coeffs.permute(0,2,1)

        return coeffs

    def downsample_rhs(self, rhs, old_shape,  new_shape):
        grid_size = np.prod(np.array(old_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, 1, *old_shape)
        _rhs = rhs

        #if len(old_shape) == 2:
        #    mode='bilinear'
        #elif len(old_shape) == 3:
        #    mode='trilinear'
        #else:
        #    raise ValueError('incorrect num coordinates')

        #print('ds rhs ', rhs.shape)
        rhs = F.interpolate(rhs, size=new_shape, mode=self.interp_mode, align_corners=self.align_corners)
        #print('ds rhs ', rhs.shape)

        new_grid_size = np.prod(np.array(new_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, new_grid_size)

        return rhs

    ###@torch.no_grad()
    #def downsample_grad(self, gradient):
    #    bs = gradient.shape[0]
    #    grad_list = []
    #    new_grad = gradient
    #    old_shape = self.coord_dims
    #    for k in range(1, self.n_grid):
    #        #pde = self.pde_list[k]
    #        new_shape = self.dim_list[k]
    #        pde = self.pde_list[k]

    #        n_orders = len(pde.var_set.mi_list)
    #        #new_grad = self.downsample_coeffs(new_grad, old_shape,  new_shape, n_orders)
    #        #new_grad = self.downsample_grads(gradient, self.coord_dims,  new_shape, n_orders)
    #        new_grad = self.downsample_grads(new_grad.clone(), old_shape,  new_shape, n_orders)
    #        old_shape = new_shape
    #        new_grad = new_grad.reshape(bs,-1)
    #        grad_list.append(new_grad.clone())

    #    return grad_list
    

    def downsample_steps(self, steps_list, old_shape):
        new_steps_list = []
        for i in range(self.n_coord):
            #steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])
            steps = steps_list[i]
            steps = steps.reshape(self.bs*self.n_ind_dim,old_shape[i]-1)
            #print('steps', old_shape)
            #steps = steps[:, :-1].reshape(-1, old_shape[i]//2-1, 2).sum(dim=-1)
            steps = steps[:, :-1].reshape(-1, old_shape[i]//2-1, 2).sum(dim=-1)
            #steps = steps[:, :-1].reshape(-1, old_shape[i]//2-1, 2)[:,:, 0]

            new_steps_list.append(steps)


        return new_steps_list

    def downsample_iv(self, iv_rhs, old_shape,  new_shape):
        #if len(old_shape) == 2:
        #    mode='bilinear'
        #elif len(old_shape) == 3:
        #    mode='trilinear'
        #else:
        #    raise ValueError('incorrect num coordinates')
        
        #bs = iv_rhs.shape[0]

        #x_steps = torch.linspace(0, 2*np.pi, 8, device=iv_rhs.device)
        #y_steps = torch.linspace(0, 2*np.pi, 8, device=iv_rhs.device)

        #x_bc = torch.sin(x_steps) 
        #y_bc = torch.sin(y_steps) 

        ##x_bc = x_bc.reshape(8,2)[:, 1]
        ##y_bc = y_bc.reshape(8,2)[:, 1]
        
        #iv0 = x_bc[:-1]
        #iv1 = y_bc[1:]
        #iv2 = x_bc[1:-1]
        #iv3 = y_bc[:]

        ##print('iv0', iv0.shape)
        #iv_rhs = torch.cat([iv0,iv1, iv2, iv3], dim =-1).to(iv_rhs.device)
        #iv_rhs = torch.stack([iv_rhs]*bs,dim=0)

        #return iv_rhs


        iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)
        _iv_rhs = iv_rhs.clone()
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

        back=False
        if back:
            #x = F.interpolate(x, size=self.dim_list[idx+1], 
            #              mode='bilinear',
            #              align_corners=True)

            #print('rback')
            s = self.dim_list[idx][0]
            x = x.reshape(1, -1, s//2,2,s//2,2)
            #x = torch.repeat_interleave(x, 2, dim=2)
            #x = torch.repeat_interleave(x, 2, dim=3)
            #x = x.sum(dim=-1).sum(dim=3)
            x = x[:,:,:,1,:,1]
        else:
            x = F.interpolate(x, size=self.dim_list[idx+1], 
                            mode=self.interp_mode, 
                            align_corners=self.align_corners)
        x = x.reshape(*x.shape[0:2], self.size_list[idx+1])

        x = x.permute(0,2,1).reshape(x.shape[0], -1)
        #eq, f_list,b_list, init_list = pde.lambda_flat_to_grid_set(x)
        #x_rst = pde.lambda_grids_to_flat(eq, f_list, b_list, init_list)
        #return x_rst

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

        back=False
        if back:
            s = self.dim_list[idx][0]
            _x = x.reshape(1, -1, s,s)
            _x = self.interp1(_x).reshape(1,-1,s,2*s)
            _x = _x.permute(0,1,3,2)
            _x = self.interp1(_x).reshape(1,-1,2*s,2*s)
            x = _x.permute(0,1,3,2)
            #x = x.reshape(1, -1, s,1,s,1)
            #y = torch.repeat_interleave(x, 2, dim=3)
            #y = torch.repeat_interleave(x, 2, dim=5)

            #_x1 = _x[]

        else:
            x = F.interpolate(x, size=self.dim_list[idx-1], 
                          #mode='bilinear',
                          mode=self.interp_mode, 
                          #align_corners=self.align_corners)
                          align_corners=self.align_corners)
            #s = self.dim_list[idx][0]
            #x = x.reshape(1, -1, s,s)
            #x = torch.repeat_interleave(x, 2, dim=2)
            #x = torch.repeat_interleave(x, 2, dim=3)
        #else:
        #    x = F.interpolate(x, size=self.dim_list[idx-1], 
        #                  mode=self.interp_mode, 
        #                  align_corners=self.align_corners)
        x = x.reshape(*x.shape[0:2], self.size_list[idx-1])

        x = x.permute(0,2,1).reshape(x.shape[0], -1)
        #eq, f_list,b_list, init_list = pde.lambda_flat_to_grid_set(x)
        #x_rst = pde.lambda_grids_to_flat(eq, f_list, b_list, init_list)
        #return x_rst

        x = x.reshape(-1)
        x = cp.asarray(x)

        return x

    def mult_AtA(self, A, x):
        #x = torch.tensor(x, device=A.device)
        #print(x.device)
        x = torch.as_tensor(x, device=self.device)
        x = torch.mm(A, x.unsqueeze(1)).squeeze(1)
        x = cp.asarray(x)
        return x#.to_dense()

    def smooth_gs(self, L, U,  b, x, nsteps=20):
    #def smooth_gs(self, A, b, x, nsteps=20):
        """GS iteration"""

        #A = A[0]
        #At = A.transpose(1,2)
        #AtA_act = torch.sparse.mm(At[0], A[0])#.unsqueeze(0)
        #ALU = AtA_act.unsqueeze(0).to_dense()
        #L = torch.tril(ALU, diagonal=0)
        #U = torch.triu(ALU, diagonal=1)

        #L = L.to_dense()
        #U = U.to_dense()

        #Ld = torch.tril(L, diagonal=-1)
        #D = L-Ld
        #w = 0.9
        #L = torch.tril(A.to_dense())[0]
        #A = A.to_dense()[0]
        #U = A-L
        #print('ll',L.shape)
        #nsteps =100

        ##SOR
        #for i in range(nsteps):
        #    x1 = w*b - torch.bmm((w*U+(w-1)*D), x.unsqueeze(2)).squeeze(2)
        #    x = torch.linalg.solve_triangular(D+w*Ld,x1.unsqueeze(2),upper=False, 
        #                                      left=True).squeeze(2)
        #return x
        
        
        for i in range(nsteps):
            #x = -torch.bmm(U, x.unsqueeze(2)).squeeze(2) + b
            #x = -torch.mm(U, x.unsqueeze(1)).squeeze(1) + b
            x = -U@x + b
            #x = torch.sparse.spsolve(L, x)
            #x = torch.linalg.solve_triangular(L,x.unsqueeze(2),upper=False, 
            #                                  left=True).squeeze(2)
            #x = CSP.linalg.spsolve_triangular(L,x,upper=False)
            x = CSPLA.spsolve_triangular(L,x,lower=True)
        return x

    def smooth_jacobi(self, As, b, x, D, nsteps=200, w=0.55, back=False):
        """Weighted Jacobi iteration"""
        Dinv = 1/D
        if back:
            w=0.4 #config.jacobi_w
        else:
            w=0.45
        #A = As[0]

        #I = torch.sparse.spdiags(torch.ones(A.shape[2]), torch.tensor([0]), (A.shape[2], A.shape[2]), 
        #                        layout=torch.sparse_coo)
        #I = I.to(A.device).unsqueeze(0)
        #I = I.to_dense()

        #I = torch.eye(A.shape[1], device=A.device)

        #print('diff',(I-I2).pow(2).sum())

        #Jx = x - w*Dinv.uns(2)*Ax
        #J = I - w*Dinv.unsqueeze(2)*A

        for i in range(nsteps):
            #x = torch.bmm(J, x.unsqueeze(2)).squeeze(2) + w*Dinv*b
            #print('jacobi', x.shape)
            #x = x - w*Dinv.unsqueeze(2)*self.mult_AtA(As, x) + w*Dinv*b
            x = x - w*Dinv*self.mult_AtA(As, x) + w*Dinv*b
        return x

    def smooth_cg(self, As, b, x, nsteps=100):
        Alist = [As[0].transpose(1,2), 1/As[1], As[0]]
        x, _  =CG.cg_matvec(Alist, b, x, maxiter=nsteps)

        return x

    def get_residual_norm(self, A, x, b):
        x = torch.as_tensor(x, device=self.device)
        b = torch.as_tensor(b, device=self.device)
        #r = b - torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        #r = b - self.mult_AtA(As, x) #torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        r = b - torch.mm(A, x.unsqueeze(1)).squeeze(1) #torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        r = r.reshape(self.bs, -1)
        b = b.reshape(self.bs, -1)

        d = b.pow(2).sum(dim=-1).sqrt()

        rnorm = r.pow(2).sum(dim=-1).sqrt()
        rrnorm = rnorm/d

        return rnorm, rrnorm

    def factor_coarsest(self, A):
        #dense cholesky factor
        #A = As[0]
        #A = A.to_dense()
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

    #def solve_direct(self, A, b):
    #    #At = A.transpose(1,2)#.to_dense()
    #    #PAt = P_diag_inv.unsqueeze(2)*At
    #    #APAt = torch.bmm(A, PAt)
    #    A = A.to_dense()
    #    #TODO: move factorization outside loop
    #    L,info = torch.linalg.cholesky_ex(A,upper=False, check_errors=True)
    #    lam = torch.cholesky_solve(b.unsqueeze(2), L)
    #    lam = lam.squeeze(2)
    #    return lam

    @torch.no_grad()
    def v_cycle_gs(self, idx, A_list, AL_list, AU_list, b, x, L, back=False):
        #A,L are torch tensors. Rest are cupy arrays
        A = A_list[idx]
        AL = AL_list[idx]
        AU = AU_list[idx]
        #b = b
        #D = D_list[idx]

        #print(As[0].shape, b.shape)
        ##if back:
        #dr, drn = self.get_residual_norm(As, x, b)
        #print('resid before smooth',idx, dr, drn)
        ##pre-smooth
        nstep =10 # 5 if back and idx == 0 else 5
        #nstep =50 if back and idx == 0 else 10
        #x = self.smooth_jacobi(As, b, x, D, nsteps=nstep, back=back)


        x = self.smooth_gs(AL, AU, b, x, nsteps=nstep)
        #x = self.smooth_cg(As, b, x, nsteps=200)


        #dr, drn = self.get_residual_norm(As, x, b)
        #print('resid after smooth',idx, dr, drn)

        #ipdb.set_trace()
        #print(A.shape, x.shape, b.shape, D.shape)
        #r = b-torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        r = b-self.mult_AtA(A, x) #torch.bmm(A, x.unsqueeze(2)).squeeze(2)

        cx = r.reshape(2, -1)
        dd = cx[0] - cx[1]
        #dd = dd.pow(2).sum()
        dd = (dd**2).sum()
        #print(dd, 'resid')
        #if back:
        #dr, drn = self.get_residual_norm(As, x, b)
        #print('resid init',idx, dr, drn)


        rH = self.restrict(idx, r, back=back)

        cx = rH.reshape(2, -1)
        dd = cx[0] - cx[1]
        #dd = dd.pow(2).sum()
        dd = (dd**2).sum()
        #print(dd, 'restrict')

        #print(idx, self.n_grid, len(As_list))
        if idx ==self.n_grid-2:
            #deltaH = self.solve_coarsest(A_list[self.n_grid-1], rH)
            if not back:
                #deltaH = self.smooth_jacobi(As_list[idx+1], rH, torch.zeros_like(rH), D_list[idx+1], nsteps=100)
                deltaH = self.solve_coarsest(L, rH)
            else:
                deltaH = self.solve_coarsest(L, rH)
        else:
            #xH0 = torch.zeros_like(rH)
            xH0 = cp.zeros_like(rH)
            #print('els')
            #deltaH,_ = self.v_cycle_jacobi(idx+1, As_list, rH,xH0, D_list,L, back=back)
            deltaH = self.v_cycle_gs(idx+1, A_list, AL_list, AU_list, rH,xH0, L, back=back)
            #print('one')
            #deltaH,_ = self.v_cycle_jacobi(idx+1, As_list, rH,deltaH, D_list,L, back=back)
            #deltaH = self.v_cycle_jacobi(idx+1, As_list, rH,deltaH, D_list,L, back=back)
            #print('two')
            #deltaH = self.v_cycle_jacobi(idx+1, As_list, rH,deltaH, D_list,L, back=back)
            #deltaH = self.v_cycle_jacobi(idx+1, As_list, rH,deltaH, D_list,L, back=back)

        delta = self.prolong(idx+1, deltaH, back=back)
        #delta = deltaup
        #if back:
        #ipdb.set_trace()
        #dr, drn = self.get_residual_norm(As, x, b)
        #print('resid delta',idx, dr, drn, delta.shape)
        #print('af idx', x.shape, delta.shape, idx)
        #correct
        #if back:
        #    x = x+delta
        #else:
        x = x+delta
        #if back:
        #dr, drn = self.get_residual_norm(As, x, b)
        #print('resid plus delta',idx, dr, drn)

        #smooth
        #x = self.smooth_jacobi(As, b, x, D, nsteps=nstep, back=back)
        nstep=10
        x = self.smooth_gs(AL, AU, b, x, nsteps=nstep)
        #x = self.smooth_cg(As, b, x, nsteps=200)


        #if back:
        #dr, drn = self.get_residual_norm(As, x, b)
        #print('resid smooth delta',idx, dr, drn)

        #out = {'deltaH': deltaH, 'deltaup': deltaup, 'delta': delta.reshape(16,16,5)}
        out = None


        return x

    @torch.no_grad()
    def v_cycle_gs_start(self, A_list, b, AL_list, AU_list,L, n_step=1, back=False):

        b=cp.asarray(b)
        x = cp.zeros_like(b)
        #x = torch.zeros_like(b_list[0])
        #x = torch.randn_like(b_list[0])
        #x = torch.rand_like(b_list[0])
        #n_step =1000 if back else 200
        n_step =2 if back else 1
        #n_step=1000
        for step in range(n_step):
            x = self.v_cycle_gs(0, A_list, AL_list, AU_list, b, x, L, back=back)
            #if back:
            #r,rr = self.get_residual_norm(A_list[0], x, b)
            #print(f'vcycle end norm: ',step, r,rr.item(),back,'\n')
        #x = x.to_dense()
        #return x, out#.to_dense()
        x= torch.as_tensor(x, device=self.device)
        return x#, out#.to_dense()

    @torch.no_grad()
    def v_cycle_jacobi_start(self, A_list, b_list, D_list,L, n_step=1, back=False):
        x = torch.zeros_like(b_list[0])
        #x = torch.randn_like(b_list[0])
        #x = torch.rand_like(b_list[0])
        n_step =2 # 200 if back else 200
        #n_step =1 if back else 1
        #n_step=1000
        for step in range(n_step):
            x, out = self.v_cycle_jacobi(0, A_list, b_list[0], x, D_list,L, back=back)
            #if back:
            #r,rr = self.get_residual_norm(A_list[0], x, b_list[0] )
            #print(f'vcycle end norm: ',step, r,rr.item(),back,'\n')
        #x = x.to_dense()
        #return x, out#.to_dense()
        return x#, out#.to_dense()

    def full_multigrid_jacobi_start(self, A_list, b_list, D_list,L, back=False):
        u = self.solve_coarsest(L, b_list[-1])
        for idx in reversed(range(self.n_grid-1)):
            print('fmg idx', idx)
            u = self.prolong(idx+1, u)
            for k in range(1):
                u,_ = self.v_cycle_jacobi(idx, A_list, b_list[idx], u, D_list,L, back=back)

                r,rr = self.get_residual_norm(A_list[idx], u, b_list[idx] )
                print(f'fmg step norm: ', r,rr)

        #print(self.AtA_act.shape, b_list[0].shape, u.shape)
        #u,_ = CG.gmres(self.AtA_act, b_list[0], u, restart=20, maxiter=800)
        #u,_ = CG.cg(self.AtA_act, b_list[0], u, maxiter=100)

        #r,rr = self.get_residual_norm(A_list[0], u, b_list[0] )
        #print(f'gmres step norm: ', r,rr)
        return u

class MultigridLayer(nn.Module):
    """ Multigrid layer """
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
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
        #dtype = torch.float64 if DBL else torch.float32

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        self.mg_solver = MultigridSolver(bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, 
                                    n_iv_steps, solver_dbl=True, n_grid=n_grid,
                                    gamma=0.5, alpha=0.1, double_ret=False, 
                                    device=None)
        #self.pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, coord_dims=self.coord_dims, step_size=self.step_size, order=self.order,
        #                 n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

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

        #build coarse grids
        #with torch.no_grad():
        #coarse_A_list, coarse_rhs_list = self.mg_solver.fill_coarse_grids(coeffs, 
        #                                                        rhs, iv_rhs, steps_list)

        #A, A_rhs = self.pde.fill_constraints_torch2(eq_constraints.coalesce(), rhs, iv_rhs, 
        #                                            derivative_constraints.coalesce())
        #A, A_rhs = self.pde.fill_constraints_torch2(eq_constraints, rhs, iv_rhs, 
        #                                            derivative_constraints)

        #A0, A_rhs0 = self.pde.fill_constraints_torch(eq_constraints, rhs, iv_rhs, 
        #                                            derivative_constraints)

        #Adiff = A0.to_dense() - A.to_dense()
        #Adiff = Adiff.pow(2).sum()
        #rdiff = (A_rhs - A_rhs0).pow(2).sum()
        #print('dif A r', Adiff, rdiff)
        rhs_in = rhs[:,:20]

        #with torch.no_grad():
        #AtA,D, AtPrhs,A_L, A_U,AtA_act,_ = self.mg_solver.make_AtA(self.pde, A, A_rhs, save=True)
        #AtA_act.register_hook(lambda grad: print('ataact'))
        ##AtA.register_hook(lambda grad: print('at', grad))
        #AtPrhs.register_hook(lambda grad: print('atprhs'))
        #def run1(coeffs, rhs, iv_rhs):
        def run1(rhs_in2):
            rhs2 = torch.cat([rhs_in2, rhs[:,20:]], dim=-1)
            #derivative_constraints = self.pde.build_derivative_tensor(steps_list)
            #eq_constraints = self.pde.build_equation_tensor(coeffs)
            x = self.qpf(eq_constraints, rhs2, iv_rhs, derivative_constraints, coeffs, steps_list)
            x= x.reshape(16,16,5)[:,:,0]
            return x


        check=False
        if check:
            import sys
            #test = gradcheck(self.qpf, iv_rhs, eps=1e-4, atol=1e-3, rtol=0.001, check_undefined_grad=False, check_batched_grad=True)
            #test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-5, rtol=0.001, check_undefined_grad=True, check_batched_grad=True)
            try: 
                torch.set_printoptions(precision=4, threshold=1000000, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

                #test = gradcheck(self.qpf, (eq_constraints, rhs, iv_rhs, derivative_constraints, coeffs, steps_list ), 
                #                eps=1e-10, atol=1e-4, rtol=0.001, fast_mode=False)

                #test = gradcheck(run1, (coeffs, rhs, iv_rhs), 
                test = gradcheck(run1, (rhs_in), 
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

        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        #x,lam = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints, 
        #                 coarse_A_list, coarse_rhs_list)

        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints, coeffs, steps_list)
        out={}
        #x = self.qpf(AtA_act, AtPrhs, AtA, D, None, None, coarse_A_list, coarse_rhs_list)
        #x = MGS.solve_mg(self.pde, self.mg_solver, AtA, AtPrhs, D, coarse_A_list, coarse_rhs_list)
        #x = MGS.solve_mg_gmres(self.pde, self.mg_solver, AtA, AtPrhs, D, coarse_A_list, coarse_rhs_list)
        #x = MGS.solve_mg_gs(self.pde, self.mg_solver, AtA, AtPrhs, D, A_L,A_U, coarse_A_list, coarse_rhs_list)


        #x,lam = MGS.run(self.pde, eq_constraints, rhs, iv_rhs, derivative_constraints, 
        #                 coarse_A_list, coarse_rhs_list)
        #eps = x[:,0]

        #print(x)
        #self.pde = self.mg_solver.pde_list[-1]
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

#def solve_direct(A, b):
#    #At = A.transpose(1,2)#.to_dense()
#    #PAt = P_diag_inv.unsqueeze(2)*At
#    #APAt = torch.bmm(A, PAt)
#    A = A.to_dense()
#    #TODO: move factorization outside loop
#    L,info = torch.linalg.cholesky_ex(A,upper=False, check_errors=True)
#    lam = torch.cholesky_solve(b.unsqueeze(2), L)
#    lam = lam.squeeze(2)
#    return lam

#class MultigridLayer2(nn.Module):
#    """ Multigrid layer """
#    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
#                    gamma=0.5, alpha=0.1, double_ret=False,n_grid=2, device=None):
#        super().__init__()
#        # placeholder step size
#        self.step_size = 0.01
#        #self.end = n_step * self.step_size
#        #self.n_step = n_step #int(self.end / self.step_size)
#        self.coord_dims = coord_dims
#        self.n_coord = len(coord_dims)
#        self.order = order
#
#        self.n_ind_dim = n_ind_dim
#        self.n_dim = 1 
#        self.n_equations =1 # n_equations
#        self.n_iv = n_iv
#        self.n_iv_steps = 1 #n_iv_steps
#        self.bs = bs
#        #self.n_coeff = self.n_step * (self.order + 1)
#        self.device = device
#        self.solver_dbl = solver_dbl
#        #dtype = torch.float64 if DBL else torch.float32
#
#        if solver_dbl:
#            print("Using double precision solver")
#        else:
#            print("Using single precision solver")
#
#        dtype = torch.float64 if self.solver_dbl else torch.float32
#
#        #self.mg_solver = MultigridSolver(bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, 
#        #                            n_iv_steps, solver_dbl=True, n_grid=n_grid,
#        #                            gamma=0.5, alpha=0.1, double_ret=False, 
#        #                            device=None)
#        self.pde = PDESYSLPEPS(bs=bs*self.n_ind_dim, n_equations=self.n_equations, n_auxiliary=0, coord_dims=self.coord_dims, step_size=self.step_size, order=self.order,
#                         n_iv=self.n_iv, init_index_mi_list=init_index_mi_list, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)
#
#        #self.pde = self.mg_solver.pde_list[0]
#        self.n_orders = len(self.pde.var_set.mi_list)
#        self.grid_size = self.pde.var_set.grid_size
#        #self.step_grid_size = self.pde.step_grid_size
#        self.step_grid_shape = self.pde.step_grid_shape
#        #self.iv_grid_size = self.pde.t0_grid_size
#
#        self.qpf = MGS.QPFunction(self.pde, self.n_iv, gamma=gamma, alpha=alpha, double_ret=double_ret)
#
#    def make_AtA(self, pde: PDESYSLPEPS, A, A_rhs, ds=1e3):
#    #def make_AAt(self, pde: PDESYSLPEPS, A, us=1e1, ds=1e-2):
#        #AGinvAt
#        #P_diag = torch.ones(num_eps).type_as(rhs)*1e3
#        #P_zeros = torch.zeros(num_var).type_as(rhs) +1e-5
#        num_eq = pde.num_added_equation_constraints + pde.num_added_initial_constraints
#        num_ineq = pde.num_added_derivative_constraints
#
#        num_eps = pde.var_set.num_added_eps_vars
#        num_var = pde.var_set.num_vars
#
#        _P_diag = torch.ones(num_ineq, dtype=A.dtype, device='cpu')*ds#*us
#        _P_ones = torch.ones(num_eq, dtype=A.dtype, device='cpu')#/ds#/config.ds# +ds
#        P_diag = torch.cat([_P_ones, _P_diag]).to(A.device)
#        P_diag_inv = 1/P_diag
#
#        #A = A.to_dense()#[:, :, :num_var]
#        At = A.transpose(1,2)
#        PinvA = P_diag_inv.unsqueeze(1)*A
#        AtA = torch.mm(At[0], PinvA[0]).unsqueeze(0)
#
#        # diagonal of AtG-1A
#        D = (PinvA*A).sum(dim=1).to_dense()
#
#        #P_rhs = P_diag_inv.sqrt()*A_rhs
#        P_rhs = P_diag_inv*A_rhs
#        #P_rhs = A_rhs
#        #AtPrhs = -torch.bmm(At, P_rhs.unsqueeze(2)).squeeze(2)
#        Atrhs = torch.bmm(At, A_rhs.unsqueeze(2)).squeeze(2)
#        AtPrhs = -torch.bmm(At, P_rhs.unsqueeze(2)).squeeze(2)
#        #AtPrhs = torch.bmm(At, P_rhs.unsqueeze(2)).squeeze(2)
#
#        return AtA, AtPrhs, Atrhs
#
#    def forward(self, coeffs, rhs, iv_rhs, steps_list):
#        #interpolate and fill grids: coeffs, rhs, iv_rhs, steps
#        #
#        
#        #build finest grid data
#        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.grid_size, self.n_orders)
#        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.grid_size)
#
#        if iv_rhs is not None:
#            #iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv*self.iv_grid_size)
#            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)
#
#        for i in range(self.n_coord):
#            #steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])
#            steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,self.coord_dims[i]-1)
#
#
#        if self.solver_dbl:
#            coeffs = coeffs.double()
#            rhs = rhs.double()
#            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
#            steps_list = [steps.double() for steps in steps_list]
#
#        derivative_constraints = self.pde.build_derivative_tensor(steps_list)
#        eq_constraints = self.pde.build_equation_tensor(coeffs)
#
#        #build coarse grids
#        #coarse_A_list, coarse_rhs_list = self.mg_solver.fill_coarse_grids(coeffs, 
#        #                                                        rhs, iv_rhs, steps_list)
#        A, A_rhs = self.pde.fill_constraints_torch_dense(eq_constraints.to_dense(), rhs, iv_rhs, derivative_constraints.to_dense())
#        #A, A_rhs = self.pde.fill_constraints_torch2(eq_constraints, rhs, iv_rhs, derivative_constraints)
#        #B3 = A2.to_dense()
#        #C2 = A.to_dense()
#
#        #b2 = A_rhs.to_dense()
#        #c2 = A_rhs2.to_dense()
#
#        #diff = (B2-C2).pow(2).sum()
#        #diff2 = (b2-c2).pow(2).sum()
#        #print('diff ', diff, diff2)
#        AtA,AtPrhs, At_rhs,A_L,A_U = self.make_AtA(self.pde, A, A_rhs)
#        #AtA = AtA.to_dense()
#        #At_rhs = At_rhs.to_dense()
#
#        check=False
#        if check:
#            import sys
#            #test = gradcheck(self.qpf, iv_rhs, eps=1e-4, atol=1e-3, rtol=0.001, check_undefined_grad=False, check_batched_grad=True)
#            #test = gradcheck(self.qpf, (coeffs,rhs,iv_rhs), eps=1e-6, atol=1e-5, rtol=0.001, check_undefined_grad=True, check_batched_grad=True)
#            #try: 
#            torch.set_printoptions(precision=4, threshold=1000000, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
#
#            test = gradcheck(self.qpf, (AtA, At_rhs), 
#                            eps=1e-12, atol=1e-4, rtol=0.001, fast_mode=False)
#            #except Exception as e:
#            #    string = e.args[0].split('tensor')
#            #    numerical = string[1].split('analytical')[0]
#            #    analytical = 'torch.tensor' + string[2]
#            #    numerical = 'torch.tensor' + numerical
#
#            #    #print(e)
#            #    print(string[0])
#            #    print('numerical', numerical)
#            #    print('--------')
#            #    print('analytical', analytical)
#            #    d = eval(numerical)
#            #    a = eval(analytical)
#            #    print('diff')
#            #    diff = (d-a).abs()
#            #    print(diff)
#            #    print(diff> 0.01)
#            #    print(diff.max())
#            #    print(d.shape)
#            sys.exit(0)
#
#
#        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
#        #x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints, 
#        #                 None, None)
#
#        #x= solve_direct(AtA, AtPrhs)
#        x = self.qpf(AtA, AtPrhs)
#        #x = self.qpf(AtA, At_rhs)
#
#        #x,lam = MGS.run(self.pde, eq_constraints, rhs, iv_rhs, derivative_constraints, 
#        #                 None, None)
#        #eps = x[:,0]
#
#        #print(x)
#        #self.pde = self.mg_solver.pde_list[-1]
#        eps =  None #x[:, self.pde.var_set.num_vars:].abs()#.max(dim=1)[0]
#
#        #shape: batch, grid, order
#        u = self.pde.get_solution_reshaped(x)
#
#        u = u.reshape(self.bs, self.n_ind_dim, *u.shape[1:])
#        #shape: batch, step, vars, order
#        #u = u.permute(0,2,1,3)
#
#        u0 = u[:,:,:,0]
#        #u1 = u[:,:,:,1]
#        #u2 = u[:,:,:,2]
#        
#        #return u0, u1, u2, eps#, steps
#        return u0, u, eps
#