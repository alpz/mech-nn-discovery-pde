
from enum import Enum
import os

class SolverType(Enum):
    DENSE_CHOLESKY = 1
    DENSE_CHOLESKY_DUAL = 5
    SPARSE_QR = 10
    SPARSE_INDIRECT_CG = 100
    SPARSE_INDIRECT_CG_DUAL = 200
    SPARSE_INDIRECT_BLOCK_CG = 1000


class ODEConfig:
    linear_solver = SolverType.DENSE_CHOLESKY_DUAL
    #linear_solver = SolverType.SPARSE_INDIRECT_CG

class PDEConfig:
    #pde_linear_solver = SolverType.DENSE_CHOLESKY_DUAL
    #pde_linear_solver = SolverType.DENSE_CHOLESKY
    pde_linear_solver = SolverType.SPARSE_INDIRECT_CG_DUAL
    pde_gmres_max_iter = 100 
    pde_gmres_repeat =  40

    #pde_gmres_max_iter = 1000 
    #pde_gmres_repeat = 500
    permute=False
    block_size= 512
    ilu_preconditioner=False
    ilu_fill_factor= 60.0
    sindpy_data = os.path.join(os.path.expanduser('~'),'data/pysindy/')
    #relax
    ds = 1e4

    #uncomment to choose linear solver sparse conjuate gradient
    #linear_solver = SolverType.SPARSE_INDIRECT_BLOCK_CG
    #cg_max_iter = 200


