
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
#from solver.qp_dual_indirect_sparse_pde import QPFunction as QPFunctionSys
#from solver.qp_dual_sparse_multigrid_normal import QPFunction as QPFunctionSys
import solver.qp_dual_sparse_multigrid_normal as MGS #import QPFunction as QPFunctionSys
from config import PDEConfig as config

# set of KKT matrices

#set of coarse grids
class MultigridSolver():
    #def __init__(self, coord_dims):
    def __init__(self, bs, order, n_ind_dim, n_iv, init_index_mi_list, coord_dims, n_iv_steps, solver_dbl=True, 
                    gamma=0.5, alpha=0.1, double_ret=False, n_grid=2, device=None):
        super().__init__()
        dtype = torch.float64 if solver_dbl else torch.float32

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
        self.n_grid = n_grid
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

    @torch.no_grad()
    def fill_coarse_grids(self, coeffs, rhs, iv_rhs, steps_list):

        A_list = []
        A_rhs_list = []

        #steps are incrementally downsampled by adding pairs. The rest are done directly
        new_steps_list = steps_list
        for k in range(1, self.n_grid):
            pde = self.pde_list[k]
            new_shape = self.dim_list[k]
            old_shape = self.dim_list[k-1]

            n_orders = len(pde.var_set.mi_list)

            new_coeffs = self.downsample_coeffs(coeffs, self.coord_dims,  new_shape, n_orders)
            new_rhs = self.downsample_rhs(rhs, self.coord_dims,  new_shape)
            new_steps_list = self.downsample_steps(new_steps_list, old_shape)
            new_iv_rhs = self.downsample_iv(iv_rhs, self.coord_dims,  new_shape)

            if self.solver_dbl:
                new_coeffs = new_coeffs.double()
                new_rhs = new_rhs.double()
                new_iv_rhs = new_iv_rhs.double() if iv_rhs is not None else None
                new_steps_list = [steps.double() for steps in new_steps_list]

            derivative_constraints = pde.build_derivative_tensor(new_steps_list)
            eq_constraints = pde.build_equation_tensor(new_coeffs)

            A, A_rhs = pde.fill_constraints_torch(eq_constraints, new_rhs, new_iv_rhs, derivative_constraints)
            A_list.append(A)
            A_rhs_list.append(A_rhs)

        return A_list, A_rhs_list

    def make_AAt_matrices(self, A_list, A_rhs_list):
        AAt_list = []
        D_list = []
        #rhs_list = []
        for i in range(self.n_grid):
            AAt,D = self.make_AAt(self.pde_list[i], A_list[i])
            AAt_list.append(AAt)
            D_list.append(D)

        return AAt_list, A_rhs_list, D_list


    def make_AAt(self, pde: PDESYSLPEPS, A, us=1e5, ds=1e-5):
    #def make_AAt(self, pde: PDESYSLPEPS, A, us=1e1, ds=1e-2):
        #AGinvAt
        #P_diag = torch.ones(num_eps).type_as(rhs)*1e3
        #P_zeros = torch.zeros(num_var).type_as(rhs) +1e-5
        num_eps = pde.var_set.num_added_eps_vars
        num_var = pde.var_set.num_vars

        _P_diag = torch.ones(num_eps, dtype=A.dtype, device='cpu')*us
        _P_zeros = torch.zeros(num_var, dtype=A.dtype, device='cpu') +ds
        P_diag = torch.cat([_P_zeros, _P_diag]).to(A.device)
        P_diag_inv = 1/P_diag

        At = A.transpose(1,2)
        PinvAt = P_diag_inv.unsqueeze(1)*At
        AAt = torch.mm(A[0], PinvAt[0]).unsqueeze(0)

        # diagonal of AG-1At
        D = (PinvAt*At).sum(dim=1).to_dense()


        return AAt, D


    def make_kkt_matrices(self, A_list, A_rhs_list):
        KKT_list = []
        KKT_diag_list = []
        for i in range(self.n_grids):
            KKT, KKT_diag = self.make_kkt(self.pde_list[i], A_list[i], A_rhs_list[i])
            KKT_list.append(KKT)
            KKT_diag_list.append(KKT_diag)

        return KKT_list, KKT_diag_list


    def make_kkt(pde: PDESYSLPEPS, A, A_rhs, us=1e2, ds=1e-2):
        #P_diag = torch.ones(num_eps).type_as(rhs)*1e3
        #P_zeros = torch.zeros(num_var).type_as(rhs) +1e-5
        num_eps = pde.var_set.num_added_eps_vars
        num_var = pde.var_set.num_vars

        _P_diag = torch.ones(num_eps, dtype=A_rhs.dtype, device='cpu')*us
        _P_zeros = torch.zeros(num_var, dtype=A_rhs.dtype, device='cpu') +ds
        P_diag = torch.cat([_P_zeros, _P_diag])

        #torch bug: can't make diagonal tensor on gpu
        G = torch.sparse.spdiags(P_diag, torch.tensor([0]), (P_diag.shape[0], P_diag.shape[0]), 
                                layout=torch.sparse_coo)

        G = G.to(A.device)
        G = G.unsqueeze(0)
        G = torch.cat([G]*A_rhs.shape[0], dim=0)
        GA = torch.cat([G, A], dim=1)

        Z = torch.sparse_coo_tensor(torch.empty([2,0]), [], size=(A.shape[1], A.shape[1]), dtype=A.dtype, device=A_rhs.device)
        Z = Z.unsqueeze(0)#.to(rhs.device)
        Z = torch.cat([Z]*A_rhs.shape[0], dim=0)

        AtZ = torch.cat([A.transpose(1,2), Z], dim =1)
        KKT = torch.cat([GA, AtZ], dim =2)

        return KKT, P_diag

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

        #print('ds coeffs ', coeffs.shape, new_shape)
        coeffs = F.interpolate(coeffs, size=new_shape, mode=mode, align_corners=False)
        #print('ds coeffs ', coeffs.shape)

        new_grid_size = np.prod(np.array(new_shape))
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, n_orders, new_grid_size)
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

        #print('ds rhs ', rhs.shape)
        rhs = F.interpolate(rhs, size=new_shape, mode=mode, align_corners=False)
        #print('ds rhs ', rhs.shape)

        new_grid_size = np.prod(np.array(new_shape))
        rhs = rhs.reshape(self.bs*self.n_ind_dim, new_grid_size)

        return rhs


    def downsample_steps(self, steps_list, old_shape):
        new_steps_list = []
        for i in range(self.n_coord):
            #steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])
            steps = steps_list[i]
            steps = steps.reshape(self.bs*self.n_ind_dim,old_shape[i]-1)
            #print('steps', old_shape)
            steps = steps[:, :-1].reshape(-1, old_shape[i]//2-1, 2).sum(dim=-1)

            print(steps)
            new_steps_list.append(steps)


        return new_steps_list

    def downsample_iv(self, iv_rhs, old_shape,  new_shape):
        if len(old_shape) == 2:
            mode='bilinear'
        elif len(old_shape) == 3:
            mode='trilinear'
        else:
            raise ValueError('incorrect num coordinates')

        iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)
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

            iv = iv_rhs[:, offset:offset+iv_old_size]
            offset = offset+iv_old_size

            iv = iv.reshape(self.bs*self.n_ind_dim, *iv_old_shape)

            #print(iv.shape, old_shape, tuple(iv_new_shape))

            iv = F.interpolate(iv.unsqueeze(1), size=tuple(iv_new_shape), mode='bilinear', align_corners=True)
            #print('interp iv ', iv.shape)
            iv = iv.reshape(self.bs*self.n_ind_dim, -1)

            iv_list.append(iv)

        iv_rhs = torch.cat(iv_list, dim=-1)

        return iv_rhs


    #def fill_grid(self, pde: PDESYSLPEPS, coeffs, rhs, iv_rhs, steps_list):
    #    grid_size = pde.var_set.grid_size
    #    n_orders = len(pde.var_set.mi_list)
    #    #step_grid_shape = pde.step_grid_shape

    #    coeffs = coeffs.reshape(self.bs*self.n_ind_dim, grid_size, n_orders)
    #    rhs = rhs.reshape(self.bs*self.n_ind_dim, grid_size)

    #    if iv_rhs is not None:
    #        #iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv*self.iv_grid_size)
    #        iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)

    #    for i in range(self.n_coord):
    #        #steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,*self.step_grid_shape[i])
    #        steps_list[i] = steps_list[i].reshape(self.bs*self.n_ind_dim,self.coord_dims[i]-1)


    #    if self.solver_dbl:
    #        coeffs = coeffs.double()
    #        rhs = rhs.double()
    #        iv_rhs = iv_rhs.double() if iv_rhs is not None else None
    #        steps_list = [steps.double() for steps in steps_list]

    #    derivative_constraints = pde.build_derivative_tensor(steps_list)
    #    eq_constraints = pde.build_equation_tensor(coeffs)

    #    return derivative_constraints, eq_constraints, steps_list, iv_rhs

    def restrict(self, idx, x):
        pde = self.pde_list[idx]
        rst_pde = self.pde_list[idx+1]

        eq, init_list, c_list, f_list,b_list = pde.lambda_flat_to_grid_set(x)

        rst_grid_shape = self.dim_list[idx+1]


        csh = rst_pde.central_grid_shapes
        fsh = rst_pde.forward_grid_shapes
        bsh = rst_pde.backward_grid_shapes
        ish = rst_pde.initial_grid_shapes

        rst_eq = F.interpolate(eq.unsqueeze(1), size=rst_grid_shape, mode='bilinear', align_corners=True)

        rst_central = []
        for i,c in enumerate(c_list):
            #print('forward ', f.shape)
            #rst_c = F.interpolate(c.unsqueeze(1), size=csh[i], mode='bilinear', align_corners=True)
            #rst_c = rst_c.squeeze(1)
            #rst_central.append(rst_c)

            cin = c.permute(0,3,1,2)
            #print('central permed ', cin.shape, csh[i])
            rst_c = F.interpolate(cin, size=(8,8), mode='bilinear', align_corners=False)
            #print('df ', rst_c.shape)

            #rst_c = rst_c.permute(ran[0],*ran[2:], ran[1])
            rst_c = rst_c.permute(0,2,3,1)

            rst_central.append(rst_c)

        rst_forward = []
        for i,f in enumerate(f_list):
            #print('forward ', f.shape)
            rst_f = F.interpolate(f.unsqueeze(1), size=fsh[i], mode='bilinear', align_corners=False)
            rst_f = rst_f.squeeze(1)
            rst_forward.append(rst_f)

        rst_backward = []
        for i,b in enumerate(b_list):
            rst_b = F.interpolate(b.unsqueeze(1), size=bsh[i], mode='bilinear', align_corners=False)
            rst_b = rst_b.squeeze(1)
            rst_backward.append(rst_b)

        rst_init = []
        for i,init in enumerate(init_list):

            #print('init ', init.shape, ish[i])
            rst_i = F.interpolate(init.unsqueeze(1), size=tuple(ish[i]), mode='bilinear', align_corners=False)
            rst_i = rst_i.squeeze(1)
            #print('interp init ', init.shape)
            #rst_i = F.interpolate(init, size=ish[i][-1], mode='linear', align_corners=True)
            rst_init.append(rst_i)

        x_rst = rst_pde.lambda_grids_to_flat(rst_eq, rst_init, rst_central, rst_forward, rst_backward)

        return x_rst

    #def prolong(self, idx, x):
    #    pde = self.pde_list[idx]
    #    pro_pde = self.pde_list[idx-1]

    #    eq, f_list,b_list, init_list = pde.lambda_flat_to_grid_set(x)
    #    #x_rst = pde.lambda_grids_to_flat(eq, f_list, b_list, init_list)
    #    #return x_rst

    #    rst_grid_shape = self.dim_list[idx-1]

    #    fsh = pro_pde.forward_grid_shapes
    #    bsh = pro_pde.backward_grid_shapes
    #    ish = pro_pde.initial_grid_shapes

    #    rst_eq = F.interpolate(eq, size=rst_grid_shape, mode='bilinear', align_corners=True)
    #    rst_forward = []
    #    for i,f in enumerate(f_list):
    #        rst_f = F.interpolate(f.unsqueeze(1), size=fsh[i], mode='bilinear', align_corners=True)
    #        rst_f = rst_f.squeeze(1)
    #        rst_forward.append(rst_f)

    #    rst_backward = []
    #    for i,b in enumerate(b_list):
    #        rst_b = F.interpolate(b.unsqueeze(1), size=bsh[i], mode='bilinear', align_corners=True)
    #        rst_b = rst_b.squeeze(1)
    #        rst_backward.append(rst_b)

    #    rst_init = []
    #    for i,init in enumerate(init_list):
    #        rst_i = F.interpolate(init.unsqueeze(1), size=tuple(ish[i]), mode='bilinear', align_corners=True)
    #        rst_i = rst_i.squeeze(1)
    #        rst_init.append(rst_i)

    #    x_rst = pro_pde.lambda_grids_to_flat(rst_eq, rst_forward, rst_backward, rst_init)

    #    return x_rst

    def prolong2(self, idx, x, x_target):
    #def prolong(self, idx, x):
        pde = self.pde_list[idx]
        pro_pde = self.pde_list[idx-1]

        eq, init_list, c_list, f_list,b_list = pde.lambda_flat_to_grid_set(x)
        eq_t, init_list_t, c_list_t, f_list_t,b_list_t = pro_pde.lambda_flat_to_grid_set(x_target)
        #x_rst = pde.lambda_grids_to_flat(eq, f_list, b_list, init_list)
        #return x_rst

        rst_grid_shape = self.dim_list[idx-1]

        csh = pro_pde.central_grid_shapes
        fsh = pro_pde.forward_grid_shapes
        bsh = pro_pde.backward_grid_shapes
        ish = pro_pde.initial_grid_shapes

        print('interp ', eq.shape, rst_grid_shape)
        rst_eq = F.interpolate(eq.unsqueeze(1), size=rst_grid_shape, mode='bilinear', align_corners=True)

        rst_central = []
        #for i,c in enumerate(c_list):
        #    ran = tuple(range(len(c.shape)))
        #    print('shape ', ran)
        #    print('central ', c.shape, csh[i])
        #    c = c.permute(ran[0],ran[-1], *ran[1:-1])
        #    print('central permed ', c.shape, csh[i])
        #    rst_c = F.interpolate(c, size=csh[i][:-1], mode='bilinear', align_corners=True)
        #    print('df ', rst_c.shape)

        #    rst_c = rst_c.permute(ran[0],*ran[2:], ran[1])
        #    print('dfaf permed', rst_c.shape)

        #    #print(rst_c[i][...,0],c_list_t[i][...,0])

        #    #rst_c = rst_c.squeeze(1)
        #    rst_central.append(rst_c)

        for i,c in enumerate(c_list):
            #ran = tuple(range(len(c.shape)))
            #print('shape ', ran)
            #coord 0
            cin = c.permute(0,3,1,2).reshape(1,2,8,8).permute(0,1,3,2).reshape(1,2*8,8)
            cin = F.interpolate(cin, size=16, mode='linear', align_corners=None)
            #cin = torch.repeat_interleave(cin, repeats=2, dim=-1)
            print('aft ',cin.shape)
            cin = cin.reshape(1,2,8,16).permute(0,1,3,2)

            cin = cin.reshape(1,2*16,8)
            cin = F.interpolate(cin, size=16, mode='nearest', align_corners=None)
            #cin = torch.repeat_interleave(cin, repeats=2, dim=-1)
            cin = cin.reshape(1,2,16,16)

            #rst_c = rst_c.permute(ran[0],*ran[2:], ran[1])
            rst_c = cin.permute(0,2,3,1)
            print('dfaf permed', rst_c.shape)

            #print(rst_c[i][...,0],c_list_t[i][...,0])

            #rst_c = rst_c.squeeze(1)
            rst_central.append(rst_c)

        rst_forward = []
        for i,f in enumerate(f_list):
            #rst_f = F.interpolate(f.unsqueeze(1), size=fsh[i], mode='bilinear', align_corners=False)
            #rst_f = F.interpolate(f.unsqueeze(1), size=fsh[i], mode='bilinear', align_corners=None)
            rst_f = F.interpolate(f.unsqueeze(1), size=fsh[i], mode='nearest', align_corners=None)
            rst_f = rst_f.squeeze(1)
            rst_forward.append(rst_f)

        rst_backward = []
        for i,b in enumerate(b_list):
            #rst_b = F.interpolate(b.unsqueeze(1), size=bsh[i], mode='bilinear', align_corners=None)
            rst_b = F.interpolate(b.unsqueeze(1), size=bsh[i], mode='nearest', align_corners=None)
            rst_b = rst_b.squeeze(1)
            rst_backward.append(rst_b)

        rst_init = []
        for i,init in enumerate(init_list):
            print('init ', init.shape, ish[i])
            #rst_i = F.interpolate(init.unsqueeze(1), size=tuple(ish[i]), mode='bilinear', align_corners=None)
            rst_i = F.interpolate(init.unsqueeze(1), size=tuple(ish[i]), mode='nearest', align_corners=None)
            rst_i = rst_i.squeeze(1)
            rst_init.append(rst_i)

        new_c = [rst_central[0]] + c_list_t[1:]
        #x_rst = pro_pde.lambda_grids_to_flat(rst_eq,init_list_t, rst_central, f_list_t, b_list_t)
        #x_rst = pro_pde.lambda_grids_to_flat(rst_eq,rst_init, rst_central, f_list_t, b_list_t)
        #x_rst = pro_pde.lambda_grids_to_flat(rst_eq,init_list_t, rst_central, rst_forward, rst_backward)
        x_rst = pro_pde.lambda_grids_to_flat(eq_t,init_list_t, new_c, f_list_t, b_list_t)

        return x_rst

    def smooth_jacobi(self, A, b, x, D, nsteps=5, w=2/3):
        Dinv = 1/D

        #w = 2/3
        w = 0.1
        I = torch.sparse.spdiags(torch.ones(A.shape[1]), torch.tensor([0]), (A.shape[1], A.shape[2]), 
                                layout=torch.sparse_coo)
        I = I.to(A.device).unsqueeze(0)

        J = I - w*Dinv.unsqueeze(2)*A

        for i in range(nsteps):
            x = torch.bmm(J, x.unsqueeze(2)).squeeze(2) + w*Dinv*b

        return x

    def smooth_qmres(self, A, b,   nsteps):
        pass

    def smooth_uzawa(self, A, b,   nsteps):
        pass

    def get_residual_norm(self, A, x, b):
        r = b - torch.bmm(A, x.unsqueeze(2)).squeeze(2)
        d = b.pow(2).sum(dim=-1).sqrt()

        rnorm = r.pow(2).sum(dim=-1).sqrt()
        rrnorm = rnorm/d

        return rnorm, rrnorm

    def solve_coarsest(self, A, b):
        #At = A.transpose(1,2)#.to_dense()
        #PAt = P_diag_inv.unsqueeze(2)*At
        #APAt = torch.bmm(A, PAt)
        A = A.to_dense()
        L,info = torch.linalg.cholesky_ex(A,upper=False)
        lam = torch.cholesky_solve(b.unsqueeze(2), L)
        lam = lam.squeeze(2)
        return lam

    def v_cycle_jacobi(self, idx, A_list, b_list, x, D_list):
        A = A_list[idx]
        b = b_list[idx]
        D = D_list[idx]

        print('idx ', x.shape, idx)
        #pre-smooth
        rn, rrn = self.get_residual_norm(A, x, b)
        print('before smooth ', rn, rrn)
        x = self.smooth_jacobi(A, b, x, D)
        rn, rrn = self.get_residual_norm(A, x, b)
        print('frst smooth ', rn, rrn)
        r = b-torch.bmm(A, x.unsqueeze(2)).squeeze(2)

        #ipdb.set_trace()
        rH = self.restrict(idx, r)

        if idx ==self.n_grid-2:
            deltaH = self.solve_coarsest(A_list[self.n_grid-1], rH)
            dr, drn = self.get_residual_norm(A_list[self.n_grid-1], deltaH, rH)
            print('resid ', dr, drn)
        else:
            xH0 = torch.zeros_like(b_list[idx+1])
            deltaH = self.v_cycle(self, idx+1, A_list, b_list,xH0, D_list)

        delta = self.prolong(idx+1, deltaH)
        print('af idx', x.shape, delta.shape, idx)
        #correct
        x = x+delta
        dr, drn = self.get_residual_norm(A_list[0], x, b)
        print('resid delta', dr, drn)

        #smooth
        x = self.smooth_jacobi(A, b, x, D)

        dr, drn = self.get_residual_norm(A_list[0], x, b)
        print('resid delta', dr, drn)


        return x

    def v_cycle_jacobi_start(self, A_list, b_list, D_list, n_step=10):
        x = torch.zeros_like(b_list[0])
        for step in range(n_step):
            x = self.v_cycle_jacobi(0, A_list, b_list, x, D_list)
            r,rr = self.get_residual_norm(A_list[0], x, b_list[0] )
            print('vcycle end norm ', r,rr)
        return x

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
                                    gamma=0.5, alpha=0.1, double_ret=False, 
                                    n_grid=config.n_grid, device=None)
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
        coarse_A_list, coarse_rhs_list = self.mg_solver.fill_coarse_grids(coeffs, 
                                                                rhs, iv_rhs, steps_list)
        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x,lam = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints, 
                         coarse_A_list, coarse_rhs_list)
        #eps = x[:,0]

        #print(x)
        self.pde = self.mg_solver.pde_list[-1]
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
