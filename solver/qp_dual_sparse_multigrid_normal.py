import torch
from torch.autograd import Function
import numpy as np

#from sksparse.cholmod import cholesky, cholesky_AAt, analyze, analyze_AAt

import scipy.sparse.linalg as spla
import scipy.linalg as spl

import scipy.sparse as SP
import torch.linalg as TLA


from solver.cg import cg_matvec


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

def do_minres(Aperm, KKT, R, perm, perminv, block_L=None, schur_diag=None, KKT_diag=None, num_qvar=None):

    print('do_minres')

    num_var = KKT_diag.shape[0]
    num_constraint = Aperm.shape[1]

    KKT_diag[:num_qvar] = 1
    KKT_diag = KKT_diag.unsqueeze(0)
    # AG^{-1/2}
    AG = Aperm*(1/KKT_diag).sqrt()       

    #I = torch.sparse.spdiags(1e-4*torch.ones(KKTperm_torch.shape[1]), torch.tensor([0]), 
    #                         (KKT.shape[1], KKT.shape[1]), 
    #                                    layout=torch.sparse_coo)

    #I = I.unsqueeze(0).to(KKT.device)

    #D = (KKTperm_torch*KKTperm_torch).sum(dim=1).to_dense().sqrt()
    #Dinv = 1/D
    #Dinv = torch.ones_like(Dinv)

    #print('building blocks')
    #block_L = LSMR.get_blocks(KKT, block_size=300, stride=150)
    #block_L = SYMMLQ.get_blocks(KKTperm_torch, block_size=100, stride=100)
    #block_L = MINRES.get_blocks(KKTperm_torch, block_size=200, stride=200)
    #print('end blocks, shape ', KKT.shape)

    print('starting')
    #sol, info ,iter = LSMR.lsmr(KKTperm_torch, Rperm_torch , Dinv, x0=x0_torch, #M=None, 
    #sol, info ,iter = LSMR.lsmr_bdp(KKT, R , block_L, block_size=300,stride=150, x0=x0_torch, #M=None, 
    #sol, info ,iter = SYMMLQ.symmlq(KKT, R , M=block_L, block_size=300,stride=150, x0=x0_torch, #M=None, 

    if block_L is None:
        print('building blocks')
        block_L, info = MINRES.get_blocks(AG, block_size=config.block_size, stride=config.block_size//4)
        #block_L,info = MINRES.get_blocks(AGperm_torch, block_size=config.block_size, stride=config.block_size//2)
        print('end blocks, shape ', KKT.shape)

    #sol, info ,iter = spla.minres(KKTperm, Rpermnp ,x0=x0_torch[0].cpu().numpy(), #M=None, 
    #                  maxiter=10000, show=True, tol=1e-6)[:3]
    

    #sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch , M=None, block_size=10,
    #sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch , M=block_L, block_size=100,
    #                                stride=100, x0=x0_torch, #M=None, 
    #                  maxiter=10000, rtol=1e-5)
    x0_torch = torch.zeros_like(R)

    #_max = MINRES._get_tensor_max(R)
    sol, info ,iter = MINRES.minres(Aperm, KKT, R, M1=block_L, M2=KKT_diag,  
                                    block_size=config.block_size, stride=config.block_size//4,
                                    #perm=perm, perminv=perminv, 
    #sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch , M=block_L, block_size=100,
                                    mlens=(num_var, num_constraint),
                                    x0=x0_torch, maxiter=10000, rtol=1e-5)
    #sol = Dinv*sol
    #sol = LSMR.apply_block_jacobi_M(block_L, sol, upper=False, block_size=50, stride=50)

    #sol, info = cg.gmres(KKTperm_torch, R, x0=sol, M=None, 
    #                    #maxiter=config.pde_gmres_max_iter, 
    #                    maxiter=100, #config.pde_gmres_max_iter, 
    #                    #restart=config.pde_gmres_repeat)
    #                    restart=50)


    #sol = sol[0].cpu().numpy()
    #resid = Rpermnp - KKTperm@sol
    #resid = np.sqrt((resid** 2).sum())
    print('minres torch', info, iter)
                      

    #if perm is not None:
    #    sol = sol[:,perminv]

    #sol = torch.tensor(sol).unsqueeze(0).to(R.device)

    residual = R -torch.bmm(KKT,sol.unsqueeze(2)).squeeze(2)
    residual = residual.pow(2).sum(dim=-1)[0]
    d = R.pow(2).sum(dim=-1)[0]
    #residual = Rperm_torch -torch.bmm(KKTperm_torch,sol.unsqueeze(2)).squeeze(2)
    print('residual  sumsq', sol.shape, residual)
    print('relative norm',  residual.sqrt()/d.sqrt())

    return sol,info,block_L


def QPFunction(pde, mg, n_iv, gamma=1, alpha=1, double_ret=True):
    class QPFunctionFn(Function):
        #csr_rows = None
        #csr_cols = None
        #nnz = None
        #QRSolver = None
        #dim = None
        #perm = None

        @staticmethod
        def forward(ctx, eq_A, rhs, iv_rhs, derivative_A, coarse_A_list, coarse_rhs_list ):
        #def forward(ctx, coeffs, rhs, iv_rhs):
            #bs = coeffs.shape[0]
            bs = rhs.shape[0]
            n_coarse_grid = len(coarse_A_list)
            #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
            #At, ub = ode.fill_block_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            A, A_rhs = pde.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            At = A.transpose(1,2)

            A_list = [A] + coarse_A_list
            rhs_list = [A_rhs] + coarse_rhs_list

            AAt_list, _, D_list = mg.make_AAt_matrices(A_list, rhs_list)
            #lam = mg.v_cycle_jacobi_start(AAt_list, rhs_list, D_list)
            
            #num_eps = mg.pde_list[-1].var_set.num_added_eps_vars
            #num_var = mg.pde_list[-1].var_set.num_vars

            num_eps = pde.var_set.num_added_eps_vars
            num_var = pde.var_set.num_vars

            #u = l
            P_diag = torch.ones(num_eps).type_as(rhs)*1e5
            P_zeros = torch.zeros(num_var).type_as(rhs) +1e-5
            P_diag = torch.cat([P_zeros, P_diag])
            P_diag_inv = 1/P_diag
            P_diag_inv = P_diag_inv.unsqueeze(0)

            lam_orig = mg.solve_coarsest(AAt_list[0], rhs_list[0])
            lam = mg.solve_coarsest(AAt_list[1], rhs_list[1])
            lam = mg.prolong2(1, lam, lam_orig)

            #At = coarse_A_list[-1].transpose(1,2)
            #c = torch.zeros(num_var+num_eps, device=A.device).type_as(rhs)
            #rhs = c
            #rhs = P_diag_inv*rhs
            #rhs = torch.bmm(A, rhs.unsqueeze(2))
            ##TODO rhs is zero upto here. remove the above
            #rhs = rhs.squeeze(2) + A_rhs
            ##rhs =  A_rhs

            ##lam, info = cg_matvec([A, P_diag_inv, At], rhs, x0=None, tol=1e-3, maxiter=16000)
            ##L=None
            ##print(info[1], info[2], lam.shape, A.shape)
            


            ############ dense
            #A = A.to_dense()
            #At = A.transpose(1,2)#.to_dense()
            #PAt = P_diag_inv.unsqueeze(2)*At
            #APAt = torch.bmm(A, PAt)
            #L,info = torch.linalg.cholesky_ex(APAt,upper=False)
            #lam = torch.cholesky_solve(rhs.unsqueeze(2), L)
            #lam = lam.squeeze(2)
            ############

            #print('torch cg info ', info)
            #lam,info = SPSLG.lgmres(pdmat, pd_rhs)
            #xl = -Pinv_s@(A_s.T@lam -q)

            #xl = -Pinv_s@(A_s.T@lam -c)
            x = lam.unsqueeze(2)
            x = torch.bmm(At, x)
            #x = P_diag_inv*(x.squeeze(2) - c)
            #c==0
            x = P_diag_inv*(x.squeeze(2))


            ####### check
            #G = torch.diag(1/P_diag_inv[0]).type_as(A)
            #Z = torch.zeros((A.shape[1], A.shape[1])).type_as(A)

            #K = torch.cat ([G, At[0]], dim=1)
            #K2 = torch.cat([A[0], Z], dim=1)
            #K = torch.cat([K,K2], dim=0)
            #sol = torch.cat([-x[0], lam[0]], dim=0)
            #msol = torch.mm(K, sol.unsqueeze(1)).squeeze()

            #z = torch.zeros(A.shape[2]).type_as(A)
            #t = torch.cat([c, -A_rhs[0]])

            #diff = (msol - t).pow(2).sum()
            #print('ff ', diff)
            ###########
            
            ctx.save_for_backward(A, P_diag_inv, x, lam, None)
            
            #if not double_ret:
            #    x = x.float()
            #print(lam)
            return x, lam
        
        @staticmethod
        def backward(ctx, dl_dzhat, dl_dlam):
            A,P_diag_inv, _x, _y, L = ctx.saved_tensors
            At = A.transpose(1,2)
            #n = A.shape[1]
            #m = A.shape[2]
            #At = A.transpose(1,2)
            
            bs = dl_dzhat.shape[0]
            m = pde.num_constraints

            dl_dzhat = -dl_dzhat

            #z = torch.zeros(bs, m, device=dl_dzhat.device).type_as(_x)
            rhs = dl_dzhat #torch.cat([-dl_dzhat, z], dim=-1)

            #rhs = -dl_dzhat
            rhs = P_diag_inv*rhs
            rhs = torch.bmm(A, rhs.unsqueeze(2))
            #TODO rhs is zero upto here. remove the above
            rhs = rhs.squeeze(2) 

            #dnu, info = cg_matvec([A, P_diag_inv, At], rhs, maxiter=16000)

            #print('back', info[1], info[2], dnu.shape)
            ###### dense
            dnu = torch.cholesky_solve(rhs.unsqueeze(2), L)
            dnu = dnu.squeeze(2)
            ######

            dx = dnu.unsqueeze(2)
            dx = torch.bmm(At, dx)
            dx = P_diag_inv*(dx.squeeze(2)- dl_dzhat ) 


            ####### check
            #G = torch.diag(1/P_diag_inv[0]).type_as(A)
            #Z = torch.zeros((A.shape[1], A.shape[1])).type_as(A)

            #K = torch.cat ([G, At[0]], dim=1)
            #K2 = torch.cat([A[0], Z], dim=1)
            #K = torch.cat([K,K2], dim=0)
            #sol = torch.cat([dx[0], dnu[0]], dim=0)
            #msol = torch.mm(K, sol.unsqueeze(1)).squeeze()

            #z = torch.zeros(A.shape[1]).type_as(A)
            #t = torch.cat([dl_dzhat[0], z])

            #diff = (msol - t).pow(2).sum()
            #print('grad ', diff)
            ###############
            
            _dx = dx
            _dnu = dnu

            db = _dnu[:, :pde.num_added_equation_constraints] #torch.tensor(-dnu.squeeze())
            db = -db.squeeze(-1) #.reshape(bs,n_step*ode.n_equations)
            #div_rhs = torch.tensor(div_rhs)
            
            #dA = dA.reshape(bs,n_step,t_vars, order+1)

            if pde.n_iv == 0:
                div_rhs = None
            else:
                #div_rhs = _dx[:, n_step*ode.n_equations:(n_step+n_iv)*ode.n_equations].squeeze(2)
                #div_rhs = _dx[:, ode.num_added_equation_constraints:ode.num_added_equation_constraints + ode.num_added_initial_constraints]#.squeeze(2)
                div_rhs = _dnu[:, pde.num_added_equation_constraints:pde.num_added_equation_constraints + pde.num_added_initial_constraints]#.squeeze(2)
                div_rhs = -div_rhs#.reshape(bs,n_iv*ode.n_equations)
            

            # step gradient
            dD = pde.sparse_grad_derivative_constraint(_dx,_y)
            dD = dD + pde.sparse_grad_derivative_constraint(_x,_dnu)

            # eq grad
            dA = pde.sparse_grad_eq_constraint(_dx,_y)
            dA = dA + pde.sparse_grad_eq_constraint(_x,_dnu)

            #if not double_ret:
            #    dA = dA.float()
            #    db =db.float()
            #    div_rhs = div_rhs.float() if div_rhs is not None else None
            #    dD = dD.float()
            
            #print(dA.abs().mean(), dA.abs().max())
            return dA, db,div_rhs, dD, None, None

    return QPFunctionFn.apply
