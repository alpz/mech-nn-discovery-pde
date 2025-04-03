import torch
from torch.autograd import Function
import numpy as np

#from sksparse.cholmod import cholesky, cholesky_AAt, analyze, analyze_AAt

import scipy.sparse.linalg as spla
import scipy.linalg as spl

import scipy.sparse as SP
import torch.linalg as TLA


#from solver.cg import cg_matvec


import solver.cg as cg

from config import PDEConfig as config
import ipdb
import scipy.sparse as SPS
#import scipy.sparse as SPS
#from solver.lsmr import lsmr
#import solver.lsmr as LSMR
import solver.minres_torch as MINRES
#import solver.minres_torch_chol as MINRES
#import solver.multigrid as MG
import torch.nn.functional as F


def to_torch_coo(KKTs):
    row = torch.tensor(KKTs.row)
    col = torch.tensor(KKTs.col)
    values =torch.tensor(KKTs.data)

    shape = list(KKTs.shape)
    indices = torch.stack([row, col], dim=0)

    KKT = torch.sparse_coo_tensor(indices=indices, values=values, size=shape)

    return KKT

def to_scipy_coo(KKTs):
    print(KKTs._indices().shape, KKTs._values().shape)
    indices = KKTs._indices().cpu().numpy()
    values = KKTs._values().cpu().numpy()
    shape = list(KKTs.shape)
    KKTs = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)
    return KKTs

def apply_sparse_row_perm(M, permutation):
    #M = M[0].to_dense()[perm.unsqueeze(1), perm].unsqueeze(0)
    #return M
    #M = M.coalesce()
    indices=M.indices().clone()
    indices2=M.indices().clone()
    values=M.values().clone()
    rows = indices[1]
    #cols = indices[2]
    permuted_rows = permutation[rows]
    #permuted_cols = permutation[cols]
    indices2[1] = permuted_rows
    #indices2[2] = permuted_cols

    #print('rows ', rows)
    #print('perm ', permutation)
    #print('permed ', permuted_rows)

    D = torch.sparse_coo_tensor(indices=indices2, 
                            values=values, size=M.shape)
    return D 

def apply_sparse_perm(M, permutation):
    #M = M[0].to_dense()[perm.unsqueeze(1), perm].unsqueeze(0)
    #return M
    M = M.coalesce()
    indices=M.indices().clone()
    indices2=M.indices().clone()
    values=M.values().clone()
    rows = indices[1]
    cols = indices[2]
    permuted_rows = permutation[rows]
    permuted_cols = permutation[cols]
    indices2[1] = permuted_rows
    indices2[2] = permuted_cols

    print('rows ', rows)
    print('perm ', permutation)
    print('permed ', permuted_rows)

    D = torch.sparse_coo_tensor(indices=indices2, 
                            values=values, size=M.shape)
    return D 

def get_AAT_diagonal(A, G):
    """AG^-1At diagonal for Jaboci Iteration"""

    Ginv = 1/G

    AGinv = A*Ginv.unsqueeze(1)
    diagonal = (AGinv*A).sum(dim=1)

    return diagonal


def solve_direct(A, b):
    #At = A.transpose(1,2)#.to_dense()
    #PAt = P_diag_inv.unsqueeze(2)*At
    #APAt = torch.bmm(A, PAt)
    A = A.to_dense()
    #TODO: move factorization outside loop
    L,info = torch.linalg.cholesky_ex(A,upper=False, check_errors=True)
    lam = torch.cholesky_solve(b.unsqueeze(2), L)
    lam = lam.squeeze(2)
    return lam

def QPFunction(pde, double_ret=True):
    class QPFunctionFn(Function):
        #csr_rows = None
        #csr_cols = None
        #nnz = None
        #QRSolver = None
        #dim = None
        #perm = None

        @staticmethod
        def forward(ctx, eq_constraints, rhs, iv_rhs, derivative_constraints, coeffs, steps_list):
            #print('input nnz', eq_constraints._nnz(), derivative_constraints._nnz())
            #print('input shape', eq_constraints.shape, derivative_constraints.shape)

            bs = pde.bs
            #coarse_A_list, coarse_rhs_list = mg.fill_coarse_grids(coeffs, rhs, iv_rhs, steps_list)

            #A, A_rhs = pde.fill_constraints_torch2(eq_constraints.coalesce(), rhs, iv_rhs, 
            #                                            derivative_constraints.coalesce())

            #AUNB, Aub_rhs = pde.fill_constraints_torch(eq_constraints, rhs, iv_rhs, 
            #                                            derivative_constraints)

            #A, A_rhs = pde.fill_block_constraints_torch(eq_constraints, rhs, iv_rhs, 
            #                                            derivative_constraints)

            A, A_rhs = pde.fill_constraints_torch(eq_constraints, rhs, iv_rhs, 
                                                        derivative_constraints)
            #ipdb.set_trace()
            #AtA,D, AtPrhs,A_L, A_U,AtA_act,G = mg.make_AtA(pde, A, A_rhs, derivative_weights, save=True)
            #AtA,D, AtPrhs,A_L, A_U = mg.make_AtA(pde, A, A_rhs)

            A = A.to_dense()
            At = A.transpose(1,2)
            AtA = torch.bmm(At, A)
            AtPrhs = torch.bmm(At, A_rhs.unsqueeze(2)).squeeze(2)

            #ipdb.set_trace()

            #G=identity
            #G = G.squeeze(2)
            #A_kkt = mg.make_kkt(G[0], A)
            

            #AtA_act.register_hook(lambda grad: print('ataact'))
            ##AtA.register_hook(lambda grad: print('at', grad))
            #AtPrhs.register_hook(lambda grad: print('atprhs'))

            #AtA_list, rhs_list, D_list,L_list,U_list = mg.make_coarse_AtA_matrices(coarse_A_list, 
            #                                                         coarse_rhs_list)

            AtA_list = AtA #+ AtA_list
            rhs_list = AtPrhs #+ rhs_list
            #D_list = [D] + D_list
            #AL_list = [A_L] + L_list
            #AU_list = [A_U] + U_list

            #negate
            #rhs_list  = [rhs for rhs in rhs_list]


            #AtA_coarsest = mg.get_AtA_dense(AtA_list[-1])
            #AtA_coarsest = AtA_list[-1].to_dense()
            #L= mg.factor_coarsest(AtA_list[-1])
            #L= mg.factor_coarsest(AtA_coarsest)

            #L= mg.factor_coarsest(AtA_list[-1])

            #x,out = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)
            #x = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)
            #x = mg.v_cycle_gs_start(AtA_list, rhs_list[0], AL_list, AU_list, L)

            #x = solve_direct(AtA_list[0].unsqueeze(0), rhs_list[0].reshape(1, -1))
            ##x = mg.full_multigrid_jacobi_start(AtA_list, rhs_list, D_list, L)
            #mg_args = [AtA_list, AL_list, AU_list, L]

            L,info = torch.linalg.cholesky_ex(AtA,upper=False, check_errors=True)
            x = torch.cholesky_solve(AtPrhs.unsqueeze(2), L).squeeze(2)

            #x,_ = cg.fgmres_matvec(AtA, 
            #                rhs_list[0].reshape(-1),
            #                x0=torch.zeros_like(rhs_list[0]).reshape(-1),
            #                MG=mg, MG_args=mg_args, restart=40, maxiter=80)
            ##x,_ = cg.fgmres(AtA_list[0].unsqueeze(0), 
            ##                rhs_list[0].unsqueeze(0),
            ##                x0=torch.zeros_like(rhs_list[0]).unsqueeze(0), 
            ##                MG=mg, MG_args=mg_args, restart=40, maxiter=80)

            #x = x.reshape(-1)
            #r,rr = mg.get_residual_norm(AtA_list[0], x, rhs_list[0])
            #print(f'gmres step norm: ', r,rr)
                                                                            
            #G (-lam) = Ax -b
            #r = torch.bmm(A, x.unsqueeze(2)).squeeze(2) - A_rhs
            #r = torch.mm(A, x.unsqueeze(1)).squeeze(1) - A_rhs
            r = torch.bmm(A, x.unsqueeze(2)).squeeze(2) - A_rhs
            #lam = -(1/G)*r
            lam = -r

            #x = -x
            #lam = -lam

            ctx.A = A
            ##ctx.AUNB = AUNB
            ##ctx.A_kkt = A_kkt
            ##ctx.G = G
            #ctx.AtA_list = AtA_list
            #ctx.AL_list = AL_list
            #ctx.AU_list = AU_list
            #ctx.AtA_act = AtA_act
            #ctx.D_list = D_list
            #ctx.rhs_list = rhs_list
            ctx.L = L
            ctx.eqA = eq_constraints
            ctx.DA = derivative_constraints
            
            ctx.save_for_backward(x, lam)
            
            x =x.reshape(bs, -1)
            #print('qpf', x.shape)
            return x#,out
        
        @staticmethod
        def backward(ctx, dl_dzhat):
            _x,_lam = ctx.saved_tensors[0:2]
            A = ctx.A
            eqA = ctx.eqA
            DA = ctx.DA
            #AUNB = ctx.AUNB
            #A_kkt = ctx.A_kkt
            #G = ctx.G
            #AtA_list = ctx.AtA_list
            #AtA_act = ctx.AtA_act
            #D_list = ctx.D_list
            #rhs_list = ctx.rhs_list
            L = ctx.L

            #AL_list = ctx.AL_list
            #AU_list = ctx.AU_list

            #print(dl_dzhat.reshape(32,32,5)[:,:,0])
            ##shape: batch, grid, order

            #coarse_grads = mg.downsample_grad(dl_dzhat.clone())
            #grad_list = [dl_dzhat] #+ coarse_grads
            #grad_list =  [g.reshape(-1) for g in grad_list]
            
            #dnu = mg.v_cycle_jacobi_start(AtA_list, grad_list, D_list, L)
            #dz,_ = mg.v_cycle_jacobi_start(AtA_list, grad_list, D_list, L, back=True)
            #dz = mg.v_cycle_gs_start(AtA_list, grad_list[0], AL_list, AU_list, L, back=True)

            #dz = mg.full_multigrid_jacobi_start(AtA_list, grad_list, D_list, L, back=True)

            #AtA0 = mg.get_AtA_dense(AtA_list[0])
            #dz = solve_direct(AtA_list[0].unsqueeze(0), grad_list[0].reshape(1,-1))

            ##mg_args = [AtA_list, D_list, L, (A.shape[1], A.shape[2])]

            dz = torch.cholesky_solve(dl_dzhat.unsqueeze(2), L).squeeze(2)
            #mg_args = [AtA_list, AL_list, AU_list, L]
            #dz,_ = cg.fgmres_matvec(AtA_list[0], 
            #                 grad_list[0].reshape(-1),
            #                 x0=torch.zeros_like(grad_list[0]).reshape(-1),
            #               MG=mg, MG_args=mg_args, restart=40, maxiter=80, back=True)

            #dz,_ = cg.fgmres(AtA_list[0].unsqueeze(0), 
            #                 grad_list[0].unsqueeze(0),
            #                 x0=torch.zeros_like(grad_list[0]).unsqueeze(0), 
            #               MG=mg, MG_args=mg_args, restart=40, maxiter=80, back=True)
            #dz = dz.reshape(-1)

            #nr, nrr = mg.get_residual_norm(AtA_list[0],dz, grad_list[0].reshape(-1))
            #print('backward', nr, nrr)

            ## dnu + G^(-1)*Adnu = 0
            #dnu = torch.bmm(A, dz.unsqueeze(2)).squeeze(2)
            #dnu = torch.bmm(A, dz.unsqueeze(1)).squeeze(1)
            #dnu = torch.mm(A, dz.unsqueeze(1)).squeeze(1)
            dnu = torch.bmm(A, dz.unsqueeze(2)).squeeze(2)
            #dnu = -(1/G)*dnu
            dnu = -(1)*dnu

            #dnu = solve_direct(AtA_list[0], grad_list[0])

            #dx = torch.bmm(A, dnu.unsqueeze(2)).squeeze(2)
            #dx = P_diag_inv*(dx)

            #_dx, _dnu = -dx,-dnu
            dz = dz.reshape(pde.bs, -1)
            dnu = dnu.reshape(pde.bs, -1)
            _x = _x.reshape(pde.bs, -1)
            _lam = _lam.reshape(pde.bs, -1)

            db = -dnu
            #dz = dz

            drhs = db[:, :pde.num_added_equation_constraints] #torch.tensor(-dnu.squeeze())
            div_rhs = db[:, pde.num_added_equation_constraints:pde.num_added_equation_constraints + pde.num_added_initial_constraints]#.squeeze(2)

            drhs = pde.add_pad(drhs).reshape(drhs.shape[0], -1)
            

            mask = torch.sparse_coo_tensor(eqA._indices(), torch.ones_like(eqA._values()), 
                                           size=eqA.size(), device=eqA.device)
            # eq grad
            dA1 = pde.sparse_grad_eq_constraint(dz,_lam, mask)
            dA2 = pde.sparse_grad_eq_constraint(_x,dnu, mask)


            #ipdb.set_trace()

            Dmask = torch.sparse_coo_tensor(DA._indices(), torch.ones_like(DA._values()), 
                                           size=DA.size(), device=DA.device)
            # step gradient
            #dD1 = pde.sparse_grad_derivative_constraint(dz,_lam, Dmask)
            #dD2 = pde.sparse_grad_derivative_constraint(_x,dnu, Dmask)
            dD1 = pde.sparse_grad_derivative_constraint(dz,_lam, Dmask)
            dD2 = pde.sparse_grad_derivative_constraint(_x,dnu, Dmask)

            #ipdb.set_trace()

            #Workaround: adding sparse matrices directly doubles the nnz. 
            dA_values = (dA1._values() + dA2._values())
            dA = torch.sparse_coo_tensor(dA1._indices(), dA_values, size=dA1.size(),
                                       dtype=dA1.dtype, device=dA1.device)
            #dA = dA1 + dA2

            dD_values = (dD1._values() + dD2._values())
            dD = torch.sparse_coo_tensor(dD1._indices(), dD_values, 
                                       dtype=dD1.dtype, device=dD1.device)

            dA = dA.to_dense()                           
            dD = dD.to_dense()

            return dA, drhs,div_rhs,dD, None, None, None

    return QPFunctionFn.apply
