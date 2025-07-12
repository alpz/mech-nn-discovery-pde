
from enum import Enum
import os

class PDEConfig:
    #data_root = os.path.expanduser('~')
    data_root = 'data'
    ginzburg_dir = os.path.join(data_root,'ginzburg')
    rheology_dir = os.path.join(data_root,'kamani')
    burgers_dir = os.path.join(data_root,'burgers')


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


