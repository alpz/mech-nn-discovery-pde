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
from config import ODEConfig as config
import ipdb



def block_mv(A, x):
    """shape x: (b, d), A sparse block"""
    b = x.shape[0]
    x = x.reshape(-1)

    y = torch.mv(A, x)
    y = y.reshape(b, -1)
    return y

def solve_kkt_indirect_cg(A, AAt, g, h, gamma):
    """
        Solve min x'Gx + d'x
            Ax = b

            g := d
            h := -b
            p := x*
            G := gamma*I
    """
    At = A.t() #transpose(1,2)
    rhs1 = block_mv(A,g) - gamma*h
    #y = torch.cholesky_solve(rhs1, L)
    #y = y

    #rhs = cupy.asarray(rhs1.squeeze())#.cpu().numpy()
    #print('starting ')
    #rhs = rhs1.squeeze(-1)
    y,info = cg.cg_block(AAt, rhs1, maxiter=config.cg_max_iter)
    #res = rhs1 - block_mv(AAt,y)
    #print('done ', res.mean().item() )
    
    p = block_mv(At,y) - g
    p = p/gamma
    
    return p,y


def QPFunction(ode, n_iv, order, n_step=10, gamma=1, alpha=1, double_ret=True):


    class QPFunctionFn(Function):
        #csr_rows = None
        #csr_cols = None
        #nnz = None
        #QRSolver = None
        #dim = None
        #perm = None

        @staticmethod
        def forward(ctx, eq_A, rhs, iv_rhs, derivative_A, lam_init):
        #def forward(ctx, coeffs, rhs, iv_rhs):
            #bs = coeffs.shape[0]
            bs = rhs.shape[0]
            #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
            #At, ub = ode.fill_block_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            A, A_rhs = ode.fill_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            At = A.transpose(1,2)


            num_eps = ode.num_added_eps_vars
            num_var = ode.num_vars
            #u = l
            P_diag = torch.ones(num_eps).type_as(rhs)*1e7
            P_zeros = torch.zeros(num_var).type_as(rhs) +1e-7
            P_diag = torch.cat([P_zeros, P_diag])
            P_diag_inv = 1/P_diag
            P_diag_inv = P_diag_inv.unsqueeze(0)

            c = torch.zeros(num_var+num_eps, device=A.device).type_as(rhs)
            rhs = c
            rhs = P_diag_inv*rhs
            rhs = torch.bmm(A, rhs.unsqueeze(2))
            #TODO rhs is zero upto here. remove the above
            rhs = rhs.squeeze(2) + A_rhs
            #rhs =  A_rhs

            lam, info = cg_matvec([A, P_diag_inv, At], rhs, maxiter=8000)
            L=None
            print(info[1], info[2], lam.shape)


            ######### dense
            #A = A.to_dense()
            #At = At.to_dense()
            #PAt = P_diag_inv.unsqueeze(2)*At
            #APAt = torch.bmm(A, PAt)
            #L,info = torch.linalg.cholesky_ex(APAt,upper=False)
            #lam = torch.cholesky_solve(rhs.unsqueeze(2), L)
            #lam = lam.squeeze(2)
            #########

            #print('torch cg info ', info)
            #lam,info = SPSLG.lgmres(pdmat, pd_rhs)
            #xl = -Pinv_s@(A_s.T@lam -q)

            #xl = -Pinv_s@(A_s.T@lam -c)
            x = lam.unsqueeze(2)
            x = torch.bmm(At, x)
            x = P_diag_inv*(x.squeeze(2) - c)


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
            
            ctx.save_for_backward(A, P_diag_inv, x, lam, L)
            
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
            m = ode.num_constraints

            dl_dzhat = -dl_dzhat

            #z = torch.zeros(bs, m, device=dl_dzhat.device).type_as(_x)
            rhs = dl_dzhat #torch.cat([-dl_dzhat, z], dim=-1)

            #rhs = -dl_dzhat
            rhs = P_diag_inv*rhs
            rhs = torch.bmm(A, rhs.unsqueeze(2))
            #TODO rhs is zero upto here. remove the above
            rhs = rhs.squeeze(2) 

            dnu, info = cg_matvec([A, P_diag_inv, At], rhs, maxiter=8000)

            print('back', info[1], info[2], dnu.shape)
            ##### dense
            #dnu = torch.cholesky_solve(rhs.unsqueeze(2), L)
            #dnu = dnu.squeeze(2)
            #####

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

            db = _dnu[:, :ode.num_added_equation_constraints] #torch.tensor(-dnu.squeeze())
            db = -db.squeeze(-1) #.reshape(bs,n_step*ode.n_equations)
            #div_rhs = torch.tensor(div_rhs)
            
            #dA = dA.reshape(bs,n_step,t_vars, order+1)

            if ode.n_iv == 0:
                div_rhs = None
            else:
                #div_rhs = _dx[:, n_step*ode.n_equations:(n_step+n_iv)*ode.n_equations].squeeze(2)
                #div_rhs = _dx[:, ode.num_added_equation_constraints:ode.num_added_equation_constraints + ode.num_added_initial_constraints]#.squeeze(2)
                div_rhs = _dnu[:, ode.num_added_equation_constraints:ode.num_added_equation_constraints + ode.num_added_initial_constraints]#.squeeze(2)
                div_rhs = -div_rhs#.reshape(bs,n_iv*ode.n_equations)
            

            # step gradient
            dD = ode.sparse_grad_derivative_constraint(_dx,_y)
            dD = dD + ode.sparse_grad_derivative_constraint(_x,_dnu)

            # eq grad
            dA = ode.sparse_grad_eq_constraint(_dx,_y)
            dA = dA + ode.sparse_grad_eq_constraint(_x,_dnu)

            #if not double_ret:
            #    dA = dA.float()
            #    db =db.float()
            #    div_rhs = div_rhs.float() if div_rhs is not None else None
            #    dD = dD.float()
            
            #print(dA.abs().mean(), dA.abs().max())
            return dA, db,div_rhs, dD,None

    return QPFunctionFn.apply
