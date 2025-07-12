import torch
from torch.autograd import Function
import numpy as np

#import solver.cg as cg

from config import PDEConfig as config
import ipdb
#import scipy.sparse as SPS
import torch.nn.functional as F
import extras.logger as logger

import extras.source

log_dir,_ = extras.source.create_log_dir()
LOG = logger.setup(log_dir, 'solver', 'solver.txt', stdout=True)


def QPFunction(pde, double_ret=True):
    class QPFunctionFn(Function):

        @staticmethod
        def forward(ctx, eq_constraints, rhs, iv_rhs, derivative_constraints, coeffs, steps_list):

            bs = pde.bs

            A, A_rhs = pde.fill_constraints_torch(eq_constraints, rhs, iv_rhs, 
                                                        derivative_constraints)

            A = A.to_dense()
            At = A.transpose(1,2)
            AtA = torch.bmm(At, A)
            AtPrhs = torch.bmm(At, A_rhs.unsqueeze(2)).squeeze(2)


            AtA_list = AtA #+ AtA_list
            rhs_list = AtPrhs #+ rhs_list

            L,info = torch.linalg.cholesky_ex(AtA,upper=False, check_errors=True)
            x = torch.cholesky_solve(AtPrhs.unsqueeze(2), L).squeeze(2)

            r = torch.bmm(A, x.unsqueeze(2)).squeeze(2) - A_rhs
            lam = -r


            ctx.A = A
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
            L = ctx.L

            dz = torch.cholesky_solve(dl_dzhat.unsqueeze(2), L).squeeze(2)
            dnu = torch.bmm(A, dz.unsqueeze(2)).squeeze(2)
            #dnu = -(1/G)*dnu
            dnu = -(1)*dnu


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
