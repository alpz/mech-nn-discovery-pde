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
#import solver.symmlq_torch as SYMMLQ
import solver.minres_torch as MINRES

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

def do_minres(KKT, R, perm, perminv):

    print('do_symlq')

    if perm is not None:
        KKTsp = to_scipy_coo(KKT[0])
        Rsp = R[0].detach().cpu().numpy()
        KKTcsr = KKTsp.tocsr()
        permnp = perm.cpu().numpy()
        KKTperm = KKTcsr[permnp[:,np.newaxis], permnp]
        Rpermnp = Rsp[permnp]

        x0_torch = torch.zeros_like(R)

        KKTperm_torch = to_torch_coo(KKTperm.tocoo()).unsqueeze(0).to(R.device)
        Rperm_torch = torch.tensor(Rpermnp).unsqueeze(0).to(R.device)
    else:
        KKTperm_torch = KKT
        Rperm_torch = R
        x0_torch = torch.zeros_like(R)

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


    #sol, info ,iter = spla.minres(KKTperm, Rpermnp ,x0=x0_torch[0].cpu().numpy(), #M=None, 
    #                  maxiter=10000, show=True, tol=1e-6)[:3]
    

    #sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch , M=None, block_size=10,
    #sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch , M=block_L, block_size=100,
    #                                stride=100, x0=x0_torch, #M=None, 
    #                  maxiter=10000, rtol=1e-5)

    _max = MINRES._get_tensor_max(Rperm_torch)
    sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch,
    #sol, info ,iter = MINRES.minres(KKTperm_torch, Rperm_torch , M=block_L, block_size=100,
                                    x0=x0_torch, #M=None, 
                      maxiter=10000, rtol=1e-5, _max=_max)
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
                      

    if perm is not None:
        sol = sol[:,perminv]

    #sol = torch.tensor(sol).unsqueeze(0).to(R.device)

    residual = R -torch.bmm(KKT,sol.unsqueeze(2)).squeeze(2)
    residual = residual.pow(2).sum()
    d = R.pow(2).sum()
    #residual = Rperm_torch -torch.bmm(KKTperm_torch,sol.unsqueeze(2)).squeeze(2)
    print('residual  sumsq', sol.shape, residual)
    print('relative norm',  residual.sqrt()/d.sqrt())

    return sol,info 


def QPFunction(pde, n_iv, n_step=10, gamma=1, alpha=1, double_ret=True):
    class QPFunctionFn(Function):
        #csr_rows = None
        #csr_cols = None
        #nnz = None
        #QRSolver = None
        #dim = None
        perm = None
        perminv = None

        first=True


        @staticmethod
        def forward(ctx, eq_A, rhs, iv_rhs, derivative_A):
        #def forward(ctx, coeffs, rhs, iv_rhs):
            #bs = coeffs.shape[0]
            bs = rhs.shape[0]
            #print(eq_A.shape, rhs.shape, iv_rhs.shape, derivative_A.shape )
            #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
            #At, ub = ode.fill_block_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            A, A_rhs = pde.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            #At = A.transpose(1,2)


            num_eps = pde.var_set.num_added_eps_vars
            num_var = pde.var_set.num_vars
            #u = l

            #torch bug: can't make diagonal tensor on gpu
            def make_kkt(us=1e3, ds=1e-5):
                #P_diag = torch.ones(num_eps).type_as(rhs)*1e3
                #P_zeros = torch.zeros(num_var).type_as(rhs) +1e-5

                P_diag = torch.ones(num_eps, dtype=rhs.dtype)*1e2
                P_zeros = torch.zeros(num_var, dtype=rhs.dtype) #+1e-5

                #P_diag = torch.ones(num_eps)*1e3
                #P_zeros = torch.zeros(num_var) +1e-3
                P_diag = torch.cat([P_zeros, P_diag])
                G = torch.sparse.spdiags(P_diag, torch.tensor([0]), (P_diag.shape[0], P_diag.shape[0]), 
                                        layout=torch.sparse_coo)

                G = G.to(rhs.device)
                G = G.unsqueeze(0)
                G = torch.cat([G]*rhs.shape[0], dim=0)
                GA = torch.cat([G, A], dim=1)
                Z = torch.sparse_coo_tensor(torch.empty([2,0]), [], size=(A.shape[1], A.shape[1]), dtype=A.dtype)
                Z = Z.unsqueeze(0).to(rhs.device)
                Z = torch.cat([Z]*rhs.shape[0], dim=0)

                AtZ = torch.cat([A.transpose(1,2), Z], dim =1)
                KKT = torch.cat([GA, AtZ], dim =2)
                return G, KKT
            G,KKT = make_kkt(us=1e3, ds=1e-4)
            R = torch.cat([torch.zeros(rhs.shape[0],G.shape[1]).type_as(rhs), -A_rhs], dim=1)

            if config.permute and QPFunctionFn.perm is None:
                KKTsp = to_scipy_coo(KKT[0])
                Rsp = R[0].detach().cpu().numpy()
                x0 = np.zeros_like(Rsp)

                print('do_csr')
                KKTcsr = KKTsp.tocsr()
                print('start perm')
                perm = SPS.csgraph.reverse_cuthill_mckee(KKTcsr,symmetric_mode=True)
                print('done computing perm')
                perminv = np.empty_like(perm)
                perminv[perm] = np.arange(perm.size)

                QPFunctionFn.perm = torch.tensor(perm.copy()).to(R.device)
                QPFunctionFn.perminv = torch.tensor(perminv).to(R.device)



            sol, info = do_minres(KKT, R,perm=QPFunctionFn.perm, perminv=QPFunctionFn.perminv)
            print('minres ', info)
            

            x = -sol[:, :num_var+num_eps]
            lam = sol[:, num_var+num_eps:]
            

            ctx.save_for_backward(A, None, x, lam, KKT)
            
            #if not double_ret:
            #    x = x.float()
            #print(lam)
            return x, lam
        
        @staticmethod
        def backward(ctx, dl_dzhat, dl_dlam):
            A,P_diag_inv, _x, _y, KKT = ctx.saved_tensors
            #M_list = ctx.M_list
            #At = A.transpose(1,2)
            #n = A.shape[1]
            #m = A.shape[2]
            #At = A.transpose(1,2)
            
            bs = dl_dzhat.shape[0]
            m = pde.num_constraints
            dl_dzhat = -dl_dzhat

            z = torch.zeros(bs, m, device=dl_dzhat.device).type_as(_x)
            R = torch.cat([dl_dzhat, z], dim=-1)
            

            sol, info = do_minres(KKT, R,perm=QPFunctionFn.perm, perminv=QPFunctionFn.perminv)
            print('minres grad ', info)


            dx = sol[:, :pde.var_set.num_vars+pde.var_set.num_added_eps_vars]
            dnu = sol[:, pde.var_set.num_vars+pde.var_set.num_added_eps_vars:]

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

            return dA, db,div_rhs, dD,None

    return QPFunctionFn.apply
