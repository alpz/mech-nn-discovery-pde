import torch
from torch.autograd import Function
import numpy as np


import solver.fgmres as fgmres 

from config import PDEConfig as config
import ipdb
import torch.nn.functional as F

import extras.logger as logger

import extras.source

log_dir,_ = extras.source.create_log_dir()
LOG = logger.setup(log_dir, 'solver', 'solver.txt', stdout=False)



def QPFunction(pde, mg, n_iv, gamma=1, alpha=1, double_ret=True):
    class QPFunctionFn(Function):

        @staticmethod
        def forward(ctx, eq_constraints, rhs, iv_rhs, derivative_constraints, coeffs, steps_list):

            bs = pde.bs
            coarse_A_list, coarse_rhs_list = mg.fill_coarse_grids(coeffs, rhs, iv_rhs, steps_list)


            A, A_rhs = pde.fill_block_constraints_torch(eq_constraints, rhs, iv_rhs, 
                                                        derivative_constraints)
            AtA,D, AtPrhs,A_L, A_U = mg.make_AtA(pde, A, A_rhs)

            AtA_list, rhs_list, D_list,L_list,U_list = mg.make_coarse_AtA_matrices(coarse_A_list, 
                                                                     coarse_rhs_list)

            AtA_list = [AtA] + AtA_list
            rhs_list = [AtPrhs] + rhs_list
            #D_list = [D] + D_list
            AL_list = [A_L] + L_list
            AU_list = [A_U] + U_list



            AtA_coarsest = AtA_list[-1].to_dense()
            L= mg.factor_coarsest(AtA_coarsest)

            mg_args = [AtA_list, AL_list, AU_list, L]

            x,_ = fgmres.fgmres_matvec(AtA, 
                            rhs_list[0].reshape(-1),
                            x0=torch.zeros_like(rhs_list[0]).reshape(-1),
                            MG=mg, MG_args=mg_args, 
                            restart=config.mg_fgmres_restarts_forward, 
                            maxiter=config.mg_fgmres_max_iter_forward)

            x = x.reshape(-1)
            r,rr = mg.get_residual_norm(AtA_list[0], x, rhs_list[0])
            LOG.info(f'gmres step norm: {r},{rr}')
                                                                            
            r = torch.mm(A, x.unsqueeze(1)).squeeze(1) - A_rhs
            lam = -r


            ctx.A = A
            ctx.AtA_list = AtA_list
            ctx.AL_list = AL_list
            ctx.AU_list = AU_list
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
            AtA_list = ctx.AtA_list
            L = ctx.L

            AL_list = ctx.AL_list
            AU_list = ctx.AU_list


            grad_list = [dl_dzhat] #+ coarse_grads
            mg_args = [AtA_list, AL_list, AU_list, L]
            dz,_ = fgmres.fgmres_matvec(AtA_list[0], 
                             grad_list[0].reshape(-1),
                             x0=torch.zeros_like(grad_list[0]).reshape(-1),
                           MG=mg, MG_args=mg_args, 
                           restart=config.mg_fgmres_restarts_backward, 
                           maxiter=config.mg_fgmres_max_iter_backward, 
                           back=True)

            dz = dz.reshape(-1)

            nr, nrr = mg.get_residual_norm(AtA_list[0],dz, grad_list[0].reshape(-1))
            #print('backward', nr, nrr)
            LOG.info(f'backward norms: {nr}, {nrr}')

            ## dnu + G^(-1)*Adnu = 0
            #dnu = torch.bmm(A, dz.unsqueeze(2)).squeeze(2)
            #dnu = torch.bmm(A, dz.unsqueeze(1)).squeeze(1)
            dnu = torch.mm(A, dz.unsqueeze(1)).squeeze(1)
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

                                       
            return dA, drhs,div_rhs,dD, None, None, None

    return QPFunctionFn.apply
