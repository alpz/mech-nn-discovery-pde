
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

    data_root = os.path.expanduser('~')
    sindpy_data = os.path.join(data_root ,'data/pysindy/')
    brusselator_dir = os.path.join(data_root,'data', 'brusselator')
    ginzburg_dir = os.path.join(data_root,'data', 'ginzburg')
    euler_dir = os.path.join(data_root,'data', 'euler')
    porous_dir = os.path.join(data_root,'data', 'porous')
    rheology_dir = os.path.join(data_root,'data', 'rheology')
    burgers_dir = os.path.join(data_root,'data', 'burgers')

    #relax
    ds = 1e2


    # multigrid options
    mg_gauss_seidel_steps_pre = 5
    mg_gauss_seidel_steps_post = 5

    mg_steps_forward  = 1
    mg_steps_backward = 1

    mg_fgmres_max_iter_forward = 40
    mg_fgmres_restarts_forward = 10

    mg_fgmres_max_iter_backward = 40
    mg_fgmres_restarts_backward = 10


    jacobi_w = 0.4


