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

def solve_direct_AtA(As, b):
    #At = A.transpose(1,2)#.to_dense()
    #PAt = P_diag_inv.unsqueeze(2)*At
    #APAt = torch.bmm(A, PAt)
    A = As[0]
    G = 1/As[1]
    A = A.to_dense()
    At= A.transpose(1,2)

    AtA = torch.bmm(At, G.unsqueeze(2)*A)
    #TODO: move factorization outside loop
    L,info = torch.linalg.cholesky_ex(AtA,upper=False, check_errors=True)
    lam = torch.cholesky_solve(b.unsqueeze(2), L)
    lam = lam.squeeze(2)
    return lam


def solve_mg_gs(pde, mg, AtA, At_rhs, D, A_L, A_U, coarse_A_list, coarse_rhs_list ):
    AtA_list, rhs_list, D_list, L_list, U_list  = mg.make_coarse_AtA_matrices(coarse_A_list, 
                                                                coarse_rhs_list)
    AtA_list = [AtA] + AtA_list
    rhs_list = [At_rhs] + rhs_list
    D_list = [D] + D_list
    L_list = [A_L] + L_list
    U_list = [A_U] + U_list

    #negate
    rhs_list  = [-rhs for rhs in rhs_list]

    #make coarsest dense. TODO: use torch.spsolve
    #AtA_list[-1]= AtA_list[-1].to_dense()

    AtA_coarsest = mg.get_AtA_dense(AtA_list[-1])
    #L= mg.factor_coarsest(AtA_list[-1])
    L= mg.factor_coarsest(AtA_coarsest)

    x = mg.v_cycle_gs_start(AtA_list, rhs_list, L_list, U_list, L)
    #x = mg.full_multigrid_jacobi_start(AtA_list, rhs_list, D_list, L)
    return x


def solve_mg(pde, mg, AtA, At_rhs, D, coarse_A_list, coarse_rhs_list ):
    AtA_list, rhs_list, D_list,_,_ = mg.make_coarse_AtA_matrices(coarse_A_list, 
                                                                coarse_rhs_list)
    AtA_list = [AtA] + AtA_list
    rhs_list = [At_rhs] + rhs_list
    D_list = [D] + D_list

    #negate
    #rhs_list  = [-rhs for rhs in rhs_list]
    rhs_list  = [-rhs for rhs in rhs_list]

    #make coarsest dense. TODO: use torch.spsolve
    #AtA_list[-1]= AtA_list[-1].to_dense()

    AtA_coarsest = mg.get_AtA_dense(AtA_list[-1])
    #L= mg.factor_coarsest(AtA_list[-1])
    L= mg.factor_coarsest(AtA_coarsest)

    #L= mg.factor_coarsest(AtA_list[-1])

    #x = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)
    #x = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)

    #print('solving direct ata')
    x = solve_direct_AtA(AtA_list[0], rhs_list[0])
    #x = mg.full_multigrid_jacobi_start(AtA_list, rhs_list, D_list, L)

    return x

def solve_mg_gmres(pde, mg, AtA, At_rhs, D, coarse_A_list, coarse_rhs_list ):
    AtA_list, rhs_list, D_list,_,_ = mg.make_coarse_AtA_matrices(coarse_A_list, 
                                                                coarse_rhs_list)
    AtA_list = [AtA] + AtA_list
    rhs_list = [At_rhs] + rhs_list
    D_list = [D] + D_list

    #negate
    rhs_list  = [-rhs for rhs in rhs_list]

    #make coarsest dense. TODO: use torch.spsolve
    #AtA_list[-1]= AtA_list[-1].to_dense()

    AtA_coarsest = mg.get_AtA_dense(AtA_list[-1])
    #L= mg.factor_coarsest(AtA_list[-1])
    L= mg.factor_coarsest(AtA_coarsest)

    #L= mg.factor_coarsest(AtA_list[-1])

    #x = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)
    #x = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)

    #print('solving direct ata')
    #x = solve_direct_AtA(AtA_list[0], rhs_list[0])
    #x = mg.full_multigrid_jacobi_start(AtA_list, rhs_list, D_list, L)
    mg_args = [AtA_list, D_list, L]

    x,_ = cg.gmres(mg.AtA_act, rhs_list[0],x0=torch.zeros_like(rhs_list[0]), MG=mg, MG_args=mg_args, restart=20, maxiter=100)

    r,rr = mg.get_residual_norm(AtA_list[0], x, rhs_list[0])
    print(f'gmres step norm: ', r,rr)

    return x


def QPFunction(pde, mg, n_iv, gamma=1, alpha=1, double_ret=True):
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
            coarse_A_list, coarse_rhs_list = mg.fill_coarse_grids(coeffs, rhs, iv_rhs, steps_list)

            #A, A_rhs = pde.fill_constraints_torch2(eq_constraints.coalesce(), rhs, iv_rhs, 
            #                                            derivative_constraints.coalesce())

            #AUNB, Aub_rhs = pde.fill_constraints_torch(eq_constraints, rhs, iv_rhs, 
            #                                            derivative_constraints)

            A, A_rhs = pde.fill_block_constraints_torch(eq_constraints, rhs, iv_rhs, 
                                                        derivative_constraints)
            #ipdb.set_trace()
            #AtA,D, AtPrhs,A_L, A_U,AtA_act,G = mg.make_AtA(pde, A, A_rhs, derivative_weights, save=True)
            AtA,D, AtPrhs,A_L, A_U = mg.make_AtA(pde, A, A_rhs)

            #ipdb.set_trace()

            #G=identity
            #G = G.squeeze(2)
            #A_kkt = mg.make_kkt(G[0], A)
            

            #AtA_act.register_hook(lambda grad: print('ataact'))
            ##AtA.register_hook(lambda grad: print('at', grad))
            #AtPrhs.register_hook(lambda grad: print('atprhs'))

            AtA_list, rhs_list, D_list,L_list,U_list = mg.make_coarse_AtA_matrices(coarse_A_list, 
                                                                     coarse_rhs_list)

            AtA_list = [AtA] + AtA_list
            rhs_list = [AtPrhs] + rhs_list
            #D_list = [D] + D_list
            AL_list = [A_L] + L_list
            AU_list = [A_U] + U_list

            #negate
            #rhs_list  = [rhs for rhs in rhs_list]


            #AtA_coarsest = mg.get_AtA_dense(AtA_list[-1])
            AtA_coarsest = AtA_list[-1].to_dense()
            #L= mg.factor_coarsest(AtA_list[-1])
            L= mg.factor_coarsest(AtA_coarsest)

            #L= mg.factor_coarsest(AtA_list[-1])

            #x,out = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)
            #x = mg.v_cycle_jacobi_start(AtA_list, rhs_list, D_list, L)
            #x = mg.v_cycle_gs_start(AtA_list, rhs_list[0], AL_list, AU_list, L)

            x = solve_direct(AtA_list[0].unsqueeze(0), rhs_list[0].reshape(1, -1))
            ##x = mg.full_multigrid_jacobi_start(AtA_list, rhs_list, D_list, L)
            #mg_args = [AtA_list, AL_list, AU_list, L]

            #x,_ = cg.fgmres_matvec(AtA, 
            #                rhs_list[0].reshape(-1),
            #                x0=torch.zeros_like(rhs_list[0]).reshape(-1),
            #                MG=mg, MG_args=mg_args, restart=40, maxiter=80)
            #x,_ = cg.fgmres(AtA_list[0].unsqueeze(0), 
            #                rhs_list[0].unsqueeze(0),
            #                x0=torch.zeros_like(rhs_list[0]).unsqueeze(0), 
            #                MG=mg, MG_args=mg_args, restart=40, maxiter=80)

            x = x.reshape(-1)
            r,rr = mg.get_residual_norm(AtA_list[0], x, rhs_list[0])
            print(f'gmres step norm: ', r,rr)
                                                                            
            #G (-lam) = Ax -b
            #r = torch.bmm(A, x.unsqueeze(2)).squeeze(2) - A_rhs
            r = torch.mm(A, x.unsqueeze(1)).squeeze(1) - A_rhs
            #lam = -(1/G)*r
            lam = -r

            #x = -x
            #lam = -lam

            ctx.A = A
            #ctx.AUNB = AUNB
            #ctx.A_kkt = A_kkt
            #ctx.G = G
            ctx.AtA_list = AtA_list
            ctx.AL_list = AL_list
            ctx.AU_list = AU_list
            #ctx.AtA_act = AtA_act
            #ctx.D_list = D_list
            #ctx.rhs_list = rhs_list
            ctx.L = L
            ctx.eqA = eq_constraints
            ctx.DA = derivative_constraints
            
            ctx.save_for_backward(x, lam)
            
            x =x.reshape(bs, -1)
            print('qpf', x.shape)
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
            AtA_list = ctx.AtA_list
            #AtA_act = ctx.AtA_act
            #D_list = ctx.D_list
            #rhs_list = ctx.rhs_list
            L = ctx.L

            AL_list = ctx.AL_list
            AU_list = ctx.AU_list

            #print(dl_dzhat.reshape(32,32,5)[:,:,0])
            ##shape: batch, grid, order

            #coarse_grads = mg.downsample_grad(dl_dzhat.clone())
            grad_list = [dl_dzhat] #+ coarse_grads
            #grad_list =  [g.reshape(-1) for g in grad_list]
            
            #dnu = mg.v_cycle_jacobi_start(AtA_list, grad_list, D_list, L)
            #dz,_ = mg.v_cycle_jacobi_start(AtA_list, grad_list, D_list, L, back=True)
            #dz = mg.v_cycle_gs_start(AtA_list, grad_list[0], AL_list, AU_list, L, back=True)

            #dz = mg.full_multigrid_jacobi_start(AtA_list, grad_list, D_list, L, back=True)

            #AtA0 = mg.get_AtA_dense(AtA_list[0])
            dz = solve_direct(AtA_list[0].unsqueeze(0), grad_list[0].reshape(1,-1))

            ##mg_args = [AtA_list, D_list, L, (A.shape[1], A.shape[2])]
            #mg_args = [AtA_list, AL_list, AU_list, L]
            #dz,_ = cg.fgmres_matvec(AtA_list[0], 
            #                 grad_list[0].reshape(-1),
            #                 x0=torch.zeros_like(grad_list[0]).reshape(-1),
            #               MG=mg, MG_args=mg_args, restart=40, maxiter=80, back=True)

            #dz,_ = cg.fgmres(AtA_list[0].unsqueeze(0), 
            #                 grad_list[0].unsqueeze(0),
            #                 x0=torch.zeros_like(grad_list[0]).unsqueeze(0), 
            #               MG=mg, MG_args=mg_args, restart=40, maxiter=80, back=True)
            dz = dz.reshape(-1)

            nr, nrr = mg.get_residual_norm(AtA_list[0],dz, grad_list[0].reshape(-1))
            print('backward', nr, nrr)

            ## dnu + G^(-1)*Adnu = 0
            #dnu = torch.bmm(A, dz.unsqueeze(2)).squeeze(2)
            #dnu = torch.bmm(A, dz.unsqueeze(1)).squeeze(1)
            dnu = torch.mm(A, dz.unsqueeze(1)).squeeze(1)
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

                                       

            #print('grad nnz', dA._nnz(), dD._nnz())
            #print('grad shape', dA.shape, dD.shape)

            #dQ1 = pde.sparse_AtA_grad(dnu, _x)
            ##print('nnz1 ', dQ1._nnz(), AtA_act._nnz())
            #dQ2 = pde.sparse_AtA_grad(_x, dnu)
            ##print('nnz2 ', dQ2._nnz(), AtA_act._nnz())


            ##adding sparse matrices directly doubles the nnz.
            #dQ_values = (dQ1._values() + dQ2._values())/2
            ##dQ = dQ.coalesce()

            #dQ = torch.sparse_coo_tensor(AtA_act.indices(), dQ_values, 
            #                           #size=(self.num_added_derivative_constraints, self.num_vars), 
            #                           dtype=AtA_act.dtype, device=AtA_act.device)
            ##print('nnz ', dQ._nnz(), AtA_act._nnz())

            ##dQ = dnu.unsqueeze(1)*_x.unsqueeze(2)
            ##dQ = dQ + dnu.unsqueeze(2)*_x.unsqueeze(1)
            ##dQ = dQ/2

            #dq = dnu
            #ipdb.set_trace()

            #return dQ, dq,None, None, None,None,None
            #print(dQ, dq)
            return dA, drhs,div_rhs,dD, None, None, None

    return QPFunctionFn.apply
