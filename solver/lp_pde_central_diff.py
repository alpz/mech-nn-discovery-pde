import numpy as np
import torch
import math
import scipy.sparse as sp
import scipy.optimize as spopt
import torch.nn as nn
from enum import Enum, IntEnum
import ipdb
import scipy.sparse as SP

from itertools import combinations
from dataclasses import dataclass

            
from functools import cmp_to_key            

class VarType(Enum):
    EPS = 1
    Mesh = 10

class ConstraintType(Enum):
    Equation = 1
    Initial = 10
    Derivative = 20

class Const(IntEnum):
    #placeholder
    PH = -100 




class QPVariableSet():
    def __init__(self, coord_dims, order):
        self.coord_dims = coord_dims
        self.n_coord = len(coord_dims)

        self.grid_size = np.prod(self.coord_dims)
        self.grid_size_excl_edge = np.prod(np.array(self.coord_dims)-2)

        #make grid. reshape (grid_size, n_coord)
        self.grid_indices = np.indices(self.coord_dims).reshape((-1,self.grid_size)).transpose((1,0))
        #make grid ready only
        self.grid_indices.setflags(write=False)
        #total order (1 or 2)
        self.order = order

        #allowed order multi indices
        self.get_order_mi()
        self.mi_to_index = dict()
        for i in self.mi_indices:
            self.mi_to_index[tuple(self.mi_list[i])] = i


        #number of variables for each grid point.
        self.n_vars_per_step = len(self.mi_list)

        #self.num_vars = self.n_vars_per_step*self.grid_size + 1
        self.num_vars = self.n_vars_per_step*self.grid_size
        self.num_pde_vars = self.num_vars

        self.multi_index_shape = (self.grid_size, self.n_vars_per_step)

        self.num_added_eps_vars = 0

    #def get_eps(self):
    #    return 0 
    def make_grid(self, coord_dims, write=False):
        grid_size = np.prod(coord_dims)

        #make grid. reshape (grid_size, n_coord)
        grid_indices = np.indices(coord_dims).reshape((-1,grid_size)).transpose((1,0))

        #make grid ready only
        grid_indices.setflags(write=write)

        return grid_indices, grid_size
    

    def next_eps_var(self):
        #add new eps var after the equation variables
        offset = self.num_vars
        index = offset + self.num_added_eps_vars
        self.num_added_eps_vars = self.num_added_eps_vars+ 1

        return index

    def get_variable_from_mi_index(self, index):
        """index = (grid_index, mi_index). Returns qp variable index"""
        # 0 is epsilon, step, grad_index
        offset = 0 #self.get_eps() + 1

        grid_pointer = np.ravel_multi_index(index[0], self.coord_dims, order='C')
        index = (grid_pointer, index[1])
        #index = self.get_coefficient_index(mesh_index, grad_index)
        out_index = np.ravel_multi_index(index, self.multi_index_shape, order='C')
        out_index = offset + out_index

        return out_index

    def get_variable_from_mi(self, grid_index, mi):
        mi_index = self.mi_to_index[mi]
        return self.get_variable_from_mi_index((grid_index,mi_index))

    def zeroth_order_index(self, coord, mi):
        """return multi index where coord order is 0 """
        coord_order = mi[coord]
        order = np.sum(mi)
        
        if coord_order == 0 :
            raise ValueError("Tried to access zeroth order from itself")

        prev_mi = list(mi)
        prev_mi[coord] = 0 #mi[coord]-1
        prev_mi = tuple(prev_mi)

        mi_index = self.mi_to_index[prev_mi]
        return prev_mi, mi_index

    def next_order_index(self, coord, mi):
        """return next higher order index """
        coord_order = mi[coord]
        order = np.sum(mi)
        
        #TODO replace with max order of coordinate in mi
        if order >= self.order:
            assert(order==self.order)
            return None

        next_mi = list(mi)
        next_mi[coord] = mi[coord]+1
        next_mi = tuple(next_mi)

        mi_index = self.mi_to_index[next_mi]
        return mi_index

    def prev_order_index(self, coord, mi):
        """return prev order index """
        coord_order = mi[coord]
        order = np.sum(mi)
        
        if coord_order == 0 :
            raise ValueError("Tried to access order lower than zero")

        prev_mi = list(mi)
        prev_mi[coord] = mi[coord]-1
        prev_mi = tuple(prev_mi)

        mi_index = self.mi_to_index[prev_mi]
        return prev_mi, mi_index

    def next_adjacent_grid_index(self, grid_index, coord):
        """return adjacent grid index of same order"""
        coord_offset = grid_index[coord]
        
        coord_dim = self.coord_dims[coord]
        if coord_offset >= coord_dim-1:
            #return None
            raise ValueError("Tried to access next adjacent to edge")

        next_index = np.copy(grid_index)
        next_index[coord] = grid_index[coord]+1

        return next_index

    def prev_adjacent_grid_index(self, grid_index, coord):
        """return adjacent grid index of same order"""
        coord_offset = grid_index[coord]
        
        if coord_offset <= 0:
            #return None
            raise ValueError("Tried to access previous adjacent to edge")

        prev_index = np.copy(grid_index)
        prev_index[coord] = grid_index[coord]-1

        return prev_index

    def next_orders_all(self, coord, mi_index):
        mi_index_list = []
        #mi_index_list.append(mi_index)

        while mi_index is not None:
            mi_index_list.append(mi_index)
            mi = self.mi_list[mi_index]
            mi_index = self.next_order_index(coord, mi)

        return mi_index_list

    def is_right_edge(self, grid_index, coord):
        return grid_index[coord] == self.coord_dims[coord]-1

    def is_right_edge_or_adjacent(self, grid_index, coord):
        return grid_index[coord] == self.coord_dims[coord]-1 or \
            grid_index[coord] == self.coord_dims[coord]-2

    def is_left_edge(self, grid_index, coord):
        return grid_index[coord] == 0
        
    def is_left_edge_or_adjacent(self, grid_index, coord):
        return grid_index[coord] == 0 or grid_index[coord] == 1

    def is_right_edge_any(self, grid_index):
        for coord in range(self.n_coord):
            if self.is_right_edge(grid_index, coord):
                return True
        return False

    def is_left_edge_any(self, grid_index):
        for coord in range(self.n_coord):
            if self.is_left_edge(grid_index, coord):
                return True
        return False

    def is_edge_any(self, grid_index):
        if self.is_left_edge_any(grid_index) or self.is_right_edge_any(grid_index):
                return True
        return False


    def get_order_counts(self, coord):
        """Create counts"""
        #for mi_index in self.var_set.central_mi_indices:
        #    mi = self.var_set.mi_list[mi_index]
        order_count = {}
        for mi in self.mi_list:
            order = mi[coord]
            order_count[order] = order_count.get(order,0) + 1 
        return order_count

    def get_higher_order_sorted_mi_indices(self, coord):
        """get list of higher order mi indices for coord, sorted by order"""
        #for mi_index in self.var_set.central_mi_indices:
        #    mi = self.var_set.mi_list[mi_index]

        mi_list = [] #[self.mi_list[mi_index] for mi_index in self.central_mi_indices]
        #for mi_index in self.central_mi_indices
        #for mi_index in self.mi_indices
        #    mi = self.mi_list[mi_index]
        for mi in self.mi_list:
            order = mi[coord]

            if order == 0:
                continue
            mi_list.append(mi)

        def compare(x, y):
            if x[coord] < y[coord]:
                return -1
            elif x[coord] > y[coord]:
                return 1
            else:
                return 0

        sorted_list = sorted(mi_list, key=cmp_to_key(compare))
        return sorted_list

    def get_order_mi(self):
        """mi's for total order 1 and 2"""

        n = self.n_coord
        order = self.order

        l0 = tuple((tuple(0 for i in range(n)),))
        l1 = tuple(tuple(1 if i in comb else 0 for i in range(n))
                for comb in combinations(np.arange(n), 1))
        l12 = tuple(tuple(2 if i in comb else 0 for i in range(n))
                for comb in combinations(np.arange(n), 1))
        l2 = tuple(tuple(1 if i in comb else 0 for i in range(n))
                for comb in combinations(np.arange(n), 2))

        #order representations u, u_x0, u_x1 u_x0x0, u_x0x1
        r0 = ['u']
        _r1 = [['x'+str(i) if i in comb else '' for i in range(n)]
                for comb in combinations(np.arange(n), 1)]
        r1 = ['u_'+''.join(l) for l in _r1]

        _r12 = [['x'+str(i)+'x'+str(i) if i in comb else '' for i in range(n)]
                for comb in combinations(np.arange(n), 1)]
        r12 = ['u_'+''.join(l) for l in _r12]

        _r2 = [['x'+str(i) if i in comb else '' for i in range(n)]
                for comb in combinations(np.arange(n), 2)]

        r2 = ['u_'+''.join(l) for l in _r2]
        

        if order ==2:
            l =  l0 + l1 + l12 #+ l2
            repr = r0+r1+r12#+r2
            #return (),l)
            self.mi_list = l
            self.mi_list_repr = repr
            #mi without max total order
            #self.taylor_mi_list = l0+l1

            self.mi_indices = list(range(len(self.mi_list)))
            #self.taylor_mi_indices = list(range(len(l0+l1)))
            self.taylor_mi_indices = list(range(len(l0)))
            #self.central_mi_indices = [] #list(range(len(l0+l1)))
            #self.maximal_mi_indices = list(range(len(l0+l1), len(l)))
        elif order==1:
            l =  l0 + l1
            repr = r0+r1
            #return (),l)
            self.mi_list = l
            self.mi_list_repr = repr
            #mi without max total order
            #mi without max total order
            #self.taylor_mi_list = l0

            self.mi_indices = list(range(len(self.mi_list)))
            self.taylor_mi_indices = list(range(len(l0)))
            #self.central_mi_indices = [] #list(range(len(l0+l1)))
            #self.maximal_mi_indices = list(range(len(l0), len(l)))

        else:
            raise ValueError('unsupported total order')
        print('mi list', self.mi_list)

        #indices sorted by order for each coord and counts for each order
        self.sorted_central_mi_indices = {}
        self.central_mi_index_order_count = {}
        self.order_count = {}
        for coord in range(self.n_coord):
            self.sorted_central_mi_indices[coord] = self.get_higher_order_sorted_mi_indices(coord)
            self.order_count[coord] = self.get_order_counts(coord)


class PDESYSLP(nn.Module):
    def __init__(self, bs=1, coord_dims=(5,6), n_iv=1, init_index_mi_list=[], n_auxiliary=0, n_equations=1, step_size=0.25, order=2, dtype=torch.float32, n_iv_steps=1, step_list = None, build=True, device=None):
        super().__init__()
        
        #dimension of each coordinate
        self.coord_dims = coord_dims
        #n_coord, number of  pde coordinates t,x,y,z ..
        self.n_coord = len(coord_dims)

        self.step_grid_size= {} 
        self.step_grid_shape= {} 
        self.step_grid_expand_shape = {}
        self.step_grid_unsqueeze_shape = {}
        step_coords = np.array(coord_dims)
        for i in range(self.n_coord):
            one_hot = np.array([1 if k==i else 0 for k in range(self.n_coord)])
            #unsqueeze_shape = np.array([1 if k==i else 0 for k in range(self.n_coord)])
            unsqueeze_shape = np.array([-1 if k==i else 1 for k in range(self.n_coord)])
            expand_shape = np.array([-1 if k==i else step_coords[k] for k in range(self.n_coord)])
            #print(one_hot)
            self.step_grid_size[i] = np.prod(step_coords - one_hot)

            self.step_grid_shape[i] = tuple(step_coords - one_hot)
            self.step_grid_expand_shape[i] = tuple(expand_shape)
            self.step_grid_unsqueeze_shape[i] = tuple(unsqueeze_shape)

            #self.step_grid_expand_shape[i] = tuple(expand_shape)

        self.forward_backward_shape = lambda coord, coord_dims: [d if i!=coord else d-1 for i,d in enumerate(coord_dims)]

        #placeholder step
        self.step_size = step_size
        #if step_list is None:
        #    step_list = step_size*np.ones(n_step-1)
        #self.step_list = step_list

        #initial constraint steps starting from step 0
        #self.n_iv_steps = n_iv_steps
        self.n_iv = n_iv
        #list of pairs of (coord_index, mi_index) for inital vals at coord_index = 0 (fixed to 0 for now)
        self.iv_list = []
        self.init_index_mi_list = init_index_mi_list

        self.num_constraints = 0

        #tracks number of added constraints
        self.num_added_constraints = 0
        self.num_added_equation_constraints = 0
        self.num_added_initial_constraints = 0
        self.num_added_derivative_constraints = 0


        self.bs = bs
        self.dtype = dtype

        #total order
        self.order = order

        self.var_set = QPVariableSet(self.coord_dims, self.order)


        #grid constraint list for row permutation. assumes order
        self.grid_constraint_list = [list() for i in range(self.var_set.grid_size)]

        #### sparse constraint arrays
        # constraint coefficients
        self.value_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # constraint indices
        self.row_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # variable indices
        self.col_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # rhs values
        self.rhs_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}

        #representation for debugging  
        self.repr_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}


        #list for taylor constraint exponetial and denominators
        self.grad_step_exp_dict = {i: [] for i in range(self.n_coord)}
        self.grad_step_denom_dict = {i: [] for i in range(self.n_coord)}
        #self.grad_step_exp_list = []
        #self.grad_step_denom_list = []

        self.back_grad_step_exp_dict = {i: [] for i in range(self.n_coord)}
        self.back_grad_step_denom_dict = {i: [] for i in range(self.n_coord)}

        #self.init_interpolation_grids()

        #self.back_grad_step_exp_list = []
        #self.back_grad_step_denom_list = []
        ##exponents for step sizes for central diff grad for similar error order.
        #self.central_exp_list = []
        self.tc_count = None
        self.act_central_mi_index_count = None


        if build:
            self.build_constraints()

        ##self.compute_constraint_grid_sizes_and_shapes()
        #print('added constraints ', self.num_added_constraints)
        #self.compute_constraint_grid_sizes_and_shapes()
        #z = torch.rand(1, self.num_added_constraints)
        #self.test_grid_transfer(z)

    #def init_interpolation_grids(self):
    #    """interpolation grids for multigrid on multipliers"""
    #    # constraints indexes for linear interpolation
    #    self.equation_constraint_indices = -1*np.ones(self.var_set.grid_size)

    #    # add grids for storing smoothness constraint indexes per coord per mi
    #    #central
    #    self.smoothness_constraint_indices = {} #{i: [] for i in range(self.n_coord)}
    #    for coord in range(self.n_coord):
    #        self.smoothness_constraint_indices[coord] = {}

    #        for mi in self.var_set.sorted_central_mi_indices[coord]:
    #            mi_index = self.var_set.mi_to_index[mi]
    #            self.smoothness_constraint_indices[coord][mi_index] = -1*np.ones(self.var_set.grid_size)

    #    #forward backward
    #    self.zeroth_smoothness_constraint_indices = {} #{i: [] for i in range(self.n_coord)}
    #    for coord in range(self.n_coord):
    #        self.zeroth_smoothness_constraint_indices[coord] = {}
    #        for direction in ['forward', 'backward']:
    #            self.zeroth_smoothness_constraint_indices[coord][direction] = []

    #    self.initial_constraint_indices = {i: [] for i in range(self.init_index_mi_list)}

    #    self.constraint_grid_sizes = []
    #    self.constraint_grid_shapes = []

    def get_solution_reshaped(self, x):
        """remove eps and reshape solution"""
        #x = x[:, 1:]
        x = x[:, :self.var_set.num_vars]
        x = x.reshape(-1, *self.var_set.multi_index_shape)

        #x = x.reshape(x.shape[0], -1)
        #x = x.reshape(-1, *self.multi_index_shape)
        return x

    def add_constraint(self, var_list, values, rhs, constraint_type, grid_num):
        """ var_list: list of multindex tuples or eps enum """

        if constraint_type == ConstraintType.Equation:
            constraint_index = self.num_added_equation_constraints 
        elif constraint_type == ConstraintType.Initial:
            constraint_index = self.num_added_initial_constraints
        elif constraint_type == ConstraintType.Derivative:
            constraint_index = self.num_added_derivative_constraints
        else:
            raise ValueError("Unknown or missing constraint type")

        for i,v in enumerate(var_list):
            if v == VarType.EPS:
                continue
                var_index = self.var_set.next_eps_var()
                #var_index = self.var_set.get_eps()
                var_repr = 'EPS'
            else:
                var_index = self.var_set.get_variable_from_mi_index(v)
                var_repr = self.var_set.mi_list_repr[v[1]]
                var_repr = var_repr + str(tuple(v[0])) 

            self.col_dict[constraint_type].append(var_index)
            self.value_dict[constraint_type].append(values[i])
            #self.row_dict[constraint_type].append(self.num_added_constraints)
            self.row_dict[constraint_type].append(constraint_index)

            self.repr_dict[constraint_type].append(var_repr)
        
        self.rhs_dict[constraint_type].append(rhs)

        ###########store list of constraints per grid num##
        #print(self.var_set.grid_indices[grid_num], var_index)
        self.grid_constraint_list[grid_num].append(self.num_added_constraints)
        ###########

        self.num_added_constraints = self.num_added_constraints+1
        if constraint_type == ConstraintType.Equation:
            self.num_added_equation_constraints += 1
        elif constraint_type == ConstraintType.Initial:
            self.num_added_initial_constraints += 1
        elif constraint_type == ConstraintType.Derivative:
            self.num_added_derivative_constraints += 1

    def compute_constraint_grid_sizes_and_shapes(self):
        #keep order
        #equation
        grid_size = np.prod(self.coord_dims)
        #coord_dims_up = np.array(self.coord_dims)*2
        #coord_dims_down = np.array(self.coord_dims)//2

        #count central constraint grids over mi over coord
        #central
        self.central_grid_shapes = []
        self.central_grid_sizes = []
        for coord in range(self.n_coord):
            nmi = len(self.var_set.sorted_central_mi_indices[coord])
            size = self.var_set.grid_size*nmi
            shape = list(self.coord_dims) + [nmi]

            self.central_grid_shapes.append(shape)
            self.central_grid_sizes.append(size)

        #ccshape = (self.n_coord, )
        #ccount = len([1 for coord in range(self.n_coord) for mi in self.var_set.sorted_central_mi_indices[coord]])
        #self.num_grids_equation_central = ccount+1

        self.forward_grid_shapes = []
        self.forward_grid_sizes = []
        #forward
        for coord in range(self.n_coord):
            coord_dims = list(self.coord_dims)
            coord_dims[coord] -= 1
            grid_size = np.prod(coord_dims)

            self.forward_grid_shapes.append(coord_dims)
            self.forward_grid_sizes.append(grid_size)


        self.backward_grid_shapes = []
        self.backward_grid_sizes = []
        #backward
        for coord in range(self.n_coord):
            coord_dims = list(self.coord_dims)
            coord_dims[coord] -= 1
            grid_size = np.prod(coord_dims)

            self.backward_grid_shapes.append(coord_dims)
            self.backward_grid_sizes.append(grid_size)

        #initial/boundary shapes and sizes
        self.initial_grid_shapes = []
        self.initial_grid_sizes = []
        #initial
        for init_num, f in enumerate(self.init_index_mi_list):
            pair = f(*self.coord_dims)
            range_begin = np.array(pair[2])
            range_end = np.array(pair[3])
            coord_dims = range_end+1 - range_begin
            grid_size = np.prod(coord_dims)

            self.initial_grid_shapes.append(coord_dims)
            self.initial_grid_sizes.append(grid_size)

    def lambda_flat_to_grid_set(self, z):
        """converts a flat Lagrange multiplier vector to a set of grids for interpolation/downsampling"""
        bs = z.shape[0]
        len0 = 0

        len0 = len0 + self.var_set.grid_size
        offset = len0

        eq_grid = z[:, :len0].reshape(bs, *self.coord_dims)


        #forward backward grid shapes. TODO Assumes same
        #coord_dims = list(self.coord_dims)
        #coord_dims[0] -= 1
        #fb_grid_size = np.prod(coord_dims)

        #len1 = len0 + self.num_grids_forward_backward*self.var_set.grid_size
        #fb_grids = z[:, :len1].reshape(bs, self.num_grids_forward_backward, *self.coord_dims)
        initial_grids = []
        for grid_shape, grid_size in zip(self.initial_grid_shapes,self.initial_grid_sizes):
            len0 = offset + grid_size
            grid = z[:, offset:len0].reshape(bs, *grid_shape)
            offset = len0
            initial_grids.append(grid)

        central_grids = []
        for grid_shape, grid_size in zip(self.central_grid_shapes,self.central_grid_sizes):
            len0 = offset + grid_size
            print(offset, len0, grid_shape )
            grid = z[:, offset:len0].reshape(bs, *tuple(grid_shape))
            offset = len0
            central_grids.append(grid)

        forward_grids = []
        for grid_shape, grid_size in zip(self.forward_grid_shapes,self.forward_grid_sizes):
            len0 = len0 + grid_size
            grid = z[:, offset:len0].reshape(bs, *grid_shape)
            offset = len0
            forward_grids.append(grid)

        backward_grids = []
        for grid_shape, grid_size in zip(self.backward_grid_shapes,self.backward_grid_sizes):
            len0 = len0 + grid_size
            grid = z[:, offset:len0].reshape(bs, *grid_shape)
            offset = len0
            backward_grids.append(grid)


        return eq_grid, initial_grids, central_grids, forward_grids, backward_grids

    def lambda_grids_to_flat(self, eq_grid,initial_grids, central_grids, forward_grids, backward_grids):
        bs = eq_grid.shape[0]
        #eq_central_grids, forward_grids, backward_grids, initial_grids = self.lambda_flat_to_grid_set(z)

        eq_grid = eq_grid.reshape(bs, -1)

        initial_grids = [grid.reshape(bs, -1) for grid in initial_grids]
        central_grids = [grid.reshape(bs, -1) for grid in central_grids]
        forward_grids = [grid.reshape(bs, -1) for grid in forward_grids]
        backward_grids = [grid.reshape(bs, -1) for grid in backward_grids]

        zp = torch.cat([eq_grid] +initial_grids+central_grids+forward_grids+backward_grids, dim=1)

        return zp

    def test_grid_transfer(self, z):
        bs = z.shape[0]

        eq_grid, initial_grids, central_grids, forward_grids, backward_grids = self.lambda_flat_to_grid_set(z)

        eq_grid = eq_grid.reshape(bs, -1)

        central_grids = [grid.reshape(bs, -1) for grid in central_grids]
        forward_grids = [grid.reshape(bs, -1) for grid in forward_grids]
        backward_grids = [grid.reshape(bs, -1) for grid in backward_grids]
        initial_grids = [grid.reshape(bs, -1) for grid in initial_grids]

        zp = torch.cat([eq_grid]+initial_grids+central_grids+forward_grids+backward_grids, dim=1)

        diff = (z-zp).pow(2).sum()
        print('constraint diff check', diff)

        assert(diff < 1e-4)

        return
        #fb_grids = z[:, :len1].reshape(bs, self.num_grids_forward_backward, *self.coord_dims)
        

    #build constraint string representations for check
    def repr_eq(self, rows=None, cols=None, rhs=None, values=None, type=ConstraintType.Equation):
        if cols is None:
            cols =self.col_dict[type] 
        if rows is None:
            rows =self.row_dict[type] 
        if rhs is None:
            rhs =self.rhs_dict[type] 
        if values is None:
            values =self.value_dict[type] 
        var_repr =self.repr_dict[type] 

        urows,counts = np.unique(rows, return_counts=True)

        cidx = 0
        repr_list = []
        for i,r in enumerate(urows):
            count = counts[i]
            constraint_repr = []
            for coffset in range(count):
                repr = f'{str(values[cidx])}*{var_repr[cidx]}'
                constraint_repr.append(repr)
                cidx = cidx + 1

            repr_list.append('+ '.join(constraint_repr) + f' ={rhs[i]}')

        return '\n'.join(repr_list)

    def repr_taylor(self, rows=None, cols=None, values=None, rhs=None, print_row=False):
        if cols is None:
            cols =self.col_dict[ConstraintType.Derivative] 
        if rows is None:
            rows =self.row_dict[ConstraintType.Derivative] 
        if rhs is None:
            rhs =self.rhs_dict[ConstraintType.Derivative] 
        if values is None:
            values =self.value_dict[ConstraintType.Derivative] 
        var_repr =self.repr_dict[ConstraintType.Derivative] 

        urows,counts = np.unique(rows, return_counts=True)

        cidx = 0
        repr_list = []
        for i,r in enumerate(urows):
            if print_row:
                repr_list.append(str(i) + ': ')
            count = counts[i]
            constraint_repr = []
            for coffset in range(count):
                repr = f'{values[cidx]}*{var_repr[cidx]}'
                constraint_repr.append(repr)
                cidx = cidx + 1
            
            repr_list.append('+ '.join(constraint_repr) + f' ={rhs[i]}')

        return '\n'.join(repr_list)

    def build_equation_constraints(self):
        #for grid_index in self.var_set.grid_indices:
        for grid_num,grid_index in enumerate(self.var_set.grid_indices):
            #skip constraints on edges. Assumes initial conditions on box
            if self.var_set.is_edge_any(grid_index):
                continue
            var_list = []
            val_list = []
            for mi_index in self.var_set.mi_indices:
                var_list.append((grid_index, mi_index))
                val_list.append(Const.PH)
            self.add_constraint(var_list = var_list, values=val_list, rhs=Const.PH, constraint_type=ConstraintType.Equation,
                                grid_num=grid_num)

            #self.store_equation_constraint_indices()

    def tc_list_append(self, coord, exp, denom, forward=True):
        """taylor constraint lists"""
        if forward:
            self.grad_step_exp_dict[coord].append(exp)
            self.grad_step_denom_dict[coord].append(denom)
        else:
            self.back_grad_step_exp_dict[coord].append(exp)
            self.back_grad_step_denom_dict[coord].append(denom)

    def tc_tensor(self):
        self.grad_step_exp_dict = {i: torch.tensor(self.grad_step_exp_dict[i]) for i in range(self.n_coord)}
        self.grad_step_denom_dict = {i: torch.tensor(self.grad_step_denom_dict[i]) for i in range(self.n_coord)}

        self.back_grad_step_exp_dict = {i: torch.tensor(self.back_grad_step_exp_dict[i]) for i in range(self.n_coord)}
        self.back_grad_step_denom_dict = {i: torch.tensor(self.back_grad_step_denom_dict[i]) for i in range(self.n_coord)}


    def _add_forward_backward_constraint(self, coord, grid_index, grid_num, forward=True):
        #Keep track of how many times a step has to repeat for each allowed (coord,grid_index). 
        tc_count = 0
        # add one constraint for each non-maximal multiindex
        for mi_index in self.var_set.taylor_mi_indices:
            mi = self.var_set.mi_list[mi_index]

            order = np.sum(mi)
            #TODO replace with maximal coordinate order
            if order == self.order:
                #maximal order, shouldn't happen
                raise ValueError("Tried adding Taylor constraint for maximal order ")
            var_list = []
            val_list = []

            #epsilon
            #var_list.append(VarType.EPS)
            #val_list.append(1)

            #self.tc_list_append(coord, 0, 1, forward=forward)
            #tc_count = tc_count+1

            #i = order
            i = mi[coord]

            #all indices upto maximal including current
            mi_index_list = self.var_set.next_orders_all(coord=coord, mi_index=mi_index)
            assert(len(mi_index_list)>=2)

            #TODO check order scale
            #diff between maximum number of taylor terms (order+1) and current terms
            order_diff =0 # self.order+2- len(mi_index_list)
            for _j,ts_mi_index in enumerate(mi_index_list):
                j = _j +order_diff
                #h = self.step_size**(j)
                if forward:
                    h = self.step_size**(j)
                else:
                    h = (-self.step_size)**(j)
                #d = math.factorial(j-i)
                d = math.factorial(_j)
                h = h/d

                var_list.append((grid_index, ts_mi_index))
                val_list.append(h)

                self.tc_list_append(coord, j, d, forward=forward)
                tc_count = tc_count+1


            if forward:
                next_grid = self.var_set.next_adjacent_grid_index(grid_index=grid_index, coord=coord)
                var_list.append((next_grid, mi_index))
            else:
                prev_grid = self.var_set.prev_adjacent_grid_index(grid_index=grid_index, coord=coord)
                var_list.append((prev_grid, mi_index))
            
            if forward:
                h_pow_i = self.step_size**(order_diff)
                #h_pow_i = self.step_size**(4)
                val_list.append(-h_pow_i)

                self.tc_list_append(coord, order_diff, -1, forward=forward)
                tc_count = tc_count+1
            else:
                #neg sign is used with all steps in the backward approximation when filling in during training
                #denom corrects the sign
                h_pow_i = (-self.step_size)**(order_diff)
                #h_pow_i = (-self.step_size)**(4)
                d = 1 #(-1)**(order_diff)
                #d = (-1)**(order_diff)
                d = -d

                val_list.append(d*h_pow_i)

                self.tc_list_append(coord, order_diff, d, forward=forward)
                tc_count = tc_count+1

            self.add_constraint(var_list=var_list, values=val_list, rhs=0, constraint_type=ConstraintType.Derivative, grid_num=grid_num)
            #self.store_zeroth_smoothness_constraint_indices(coord, forward)

        #
        self.tc_count = tc_count if self.tc_count is None else self.tc_count

    def forward_constraints(self):
        for coord in range(self.n_coord):
            for grid_num,grid_index in enumerate(self.var_set.grid_indices):
                #skip right edge
                if self.var_set.is_right_edge(grid_index=grid_index, coord=coord):
                    continue
                self._add_forward_backward_constraint(coord, grid_index, grid_num, forward=True)

    def backward_constraints(self):
        for coord in range(self.n_coord):
            for grid_num,grid_index in enumerate(self.var_set.grid_indices):
                #skip left edge
                if self.var_set.is_left_edge(grid_index=grid_index, coord=coord):
                    continue
                #for mi_index in self.var_set.mi_indices:
                self._add_forward_backward_constraint(coord, grid_index, grid_num, forward=False)

    def _add_central_constraint_edge(self, coord, grid_index, grid_num, backward=False):
        #forward/backward differences for left edge. 4th order accuracy
        for mi in self.var_set.sorted_central_mi_indices[coord]:
            mi_index = self.var_set.mi_to_index[mi]
            #mi = self.var_set.mi_list[mi_index]

            #if mi[coord] == 0:
            assert(mi[coord]!= 0)
                #continue
            #act_mi_index_count += 1

            #_, prev_order_index = self.var_set.prev_order_index(coord, mi)
            _, zeroth_order_index = self.var_set.zeroth_order_index(coord, mi)

            if backward: 
                #prev_grid = self.var_set.next_adjacent_grid_index(grid_index=grid_index, coord=coord)
                next_grid = self.var_set.prev_adjacent_grid_index(grid_index=grid_index, coord=coord)
                next_2_grid = self.var_set.prev_adjacent_grid_index(grid_index=next_grid, coord=coord)
                next_3_grid = self.var_set.prev_adjacent_grid_index(grid_index=next_2_grid, coord=coord)
                next_4_grid = self.var_set.prev_adjacent_grid_index(grid_index=next_3_grid, coord=coord)
            else:
                #prev_grid = self.var_set.prev_adjacent_grid_index(grid_index=grid_index, coord=coord)
                next_grid = self.var_set.next_adjacent_grid_index(grid_index=grid_index, coord=coord)
                next_2_grid = self.var_set.next_adjacent_grid_index(grid_index=next_grid, coord=coord)
                next_3_grid = self.var_set.next_adjacent_grid_index(grid_index=next_2_grid, coord=coord)
                next_4_grid = self.var_set.next_adjacent_grid_index(grid_index=next_3_grid, coord=coord)

            #var_list = [ VarType.EPS]
            var_list = []
            #var_list.append((prev_grid, zeroth_order_index))
            var_list.append((grid_index, zeroth_order_index))
            var_list.append((next_grid, zeroth_order_index))
            var_list.append((next_2_grid, zeroth_order_index))
            var_list.append((next_3_grid, zeroth_order_index))
            var_list.append((next_4_grid, zeroth_order_index))
            var_list.append((grid_index, mi_index))

            h= self.step_size
            m = (2*h)**(self.order-1)

            if mi[coord]==1:
                #values= [ -1, 1/12, -2/3, 0, 2/3, -1/12, -1*h]
                #values= [-1, -25/12, 4, -3, 4/3,-1/4, -1*h]
                values= [-25/12, 4, -3, 4/3,-1/4, -1*h]
                if backward:
                    #values= [-1, 25/12, -4, 3,-4/3,1/4, -1*h]
                    values= [25/12, -4, 3,-4/3,1/4, -1*h]
            elif mi[coord] ==2:
                #continue
                #values= [ -1, -1/12, 4/3, -5/2, 4/3, -1/12, -1*h**2]
                #values= [ -1, 35/12, -104/12, 114/12, -56/12, 11/12, -1*h**2]
                values= [35/12, -104/12, 114/12, -56/12, 11/12, -1*h**2]
            else:
                raise ValueError('Central diff not implemented for ' + str(mi[coord]))

            self.add_constraint(var_list=var_list, values= values, rhs=0, constraint_type=ConstraintType.Derivative, grid_num=grid_num)
            #self.store_central_smoothness_constraint_indices_on_grid(coord, mi_index, grid_num)

    def _add_central_constraint(self, coord, grid_index, grid_num):
        act_mi_index_count = 0
        #get maximal order indices
        #for mi_index in self.var_set.maximal_mi_indices:
        #for mi_index in self.var_set.central_mi_indices:
        #for mi_index in sorted_mi_indices:
        #print('sorted mi', self.var_set.sorted_central_mi_indices)
        for mi in self.var_set.sorted_central_mi_indices[coord]:
            mi_index = self.var_set.mi_to_index[mi]
            #mi = self.var_set.mi_list[mi_index]

            #if mi[coord] == 0:
            assert(mi[coord]!= 0)
                #continue
            act_mi_index_count += 1

            #_, prev_order_index = self.var_set.prev_order_index(coord, mi)
            _, zeroth_order_index = self.var_set.zeroth_order_index(coord, mi)
            next_grid = self.var_set.next_adjacent_grid_index(grid_index=grid_index, coord=coord)
            next_next_grid = self.var_set.next_adjacent_grid_index(grid_index=next_grid, coord=coord)
            prev_grid = self.var_set.prev_adjacent_grid_index(grid_index=grid_index, coord=coord)
            prev_prev_grid = self.var_set.prev_adjacent_grid_index(grid_index=prev_grid, coord=coord)

            #var_list = [ VarType.EPS]
            var_list = []
            var_list.append((prev_prev_grid, zeroth_order_index))
            var_list.append((prev_grid, zeroth_order_index))
            var_list.append((grid_index, zeroth_order_index))
            var_list.append((next_grid, zeroth_order_index))
            var_list.append((next_next_grid, zeroth_order_index))
            var_list.append((grid_index, mi_index))

            h= self.step_size
            m = (2*h)**(self.order-1)

            if mi[coord]==1:
                #values= [ -1, 1/12, -2/3, 0, 2/3, -1/12, -1*h]
                values= [ 1/12, -2/3, 0, 2/3, -1/12, -1*h]
            elif mi[coord] ==2:
                #values= [ -1, -1/12, 4/3, -5/2, 4/3, -1/12, -1*h**2]
                values= [ -1/12, 4/3, -5/2, 4/3, -1/12, -1*h**2]
            else:
                raise ValueError('Central diff not implemented for ' + str(mi[coord]))

            self.add_constraint(var_list=var_list, values= values, rhs=0, constraint_type=ConstraintType.Derivative, grid_num=grid_num)
            #self.store_central_smoothness_constraint_indices_on_grid(coord, mi_index, grid_num)

        self.act_central_mi_index_count = act_mi_index_count if self.act_central_mi_index_count is None else self.act_central_mi_index_count

    def central_constraints(self):
        #5 point central diff estimate
        for coord in range(self.n_coord):
            #sort_mi_indices = self.var_set.get_order_sorted_mi_indices(coord)
            for grid_num,grid_index in enumerate(self.var_set.grid_indices):
                #skip left and right edge
                #if self.var_set.is_right_edge(grid_index=grid_index, coord=coord) or self.var_set.is_left_edge(grid_index=grid_index, coord=coord):
                if self.var_set.is_left_edge_or_adjacent(grid_index=grid_index, coord=coord):
                    self._add_central_constraint_edge(coord, grid_index, grid_num, backward=False)
                elif self.var_set.is_right_edge_or_adjacent(grid_index=grid_index, coord=coord):
                    self._add_central_constraint_edge(coord, grid_index, grid_num, backward=True)
                else:
                #for mi_index in self.var_set.mi_indices:
                    self._add_central_constraint(coord, grid_index, grid_num)

    def build_initial_constraints(self):
        #if(self.n_iv > 1):
        #    raise ValueError("Not implemented n_iv>1")
        #TODO range

        for init_num, f in enumerate(self.init_index_mi_list):
            pair = f(*self.coord_dims)
            coord_index = pair[0]
            mi_index = pair[1]
            range_begin = np.array(pair[2])
            range_end = np.array(pair[3])

            #for grid_index in t0_grid:
            for grid_num,grid_index in enumerate(self.var_set.grid_indices):
                if (grid_index < range_begin).any() or (grid_index > range_end).any():
                    continue
                for iv in range(self.n_iv):
                    var_list = []
                    val_list = []
                    #add initial function values
                    #mi_index = 0
                    var_list.append((grid_index, mi_index))
                    val_list.append(1)
                    self.add_constraint(var_list = var_list, values=val_list, rhs=Const.PH, constraint_type=ConstraintType.Initial, grid_num=grid_num)
                    #self.initial_constraint_indices(init_num)
        return

        #for pair in self.init_index_mi_list:
        #    coord_index = pair[0]
        #    mi_index = pair[1]

        #    #t0_dims = (1,) + self.coord_dims[1:]
        #    t0_dims = self.coord_dims[:coord_index] + (1,) + self.coord_dims[coord_index+1:]
        #    t0_size  = np.prod(t0_dims)
        #    t0_grid = np.indices(t0_dims).reshape(self.n_coord, t0_size).transpose(1,0)
        #    self.t0_grid_size = t0_size

        #    for grid_index in t0_grid:
        #        for iv in range(self.n_iv):
        #            var_list = []
        #            val_list = []
        #            #add initial function values
        #            #mi_index = 0
        #            var_list.append((grid_index, mi_index))
        #            val_list.append(1)
        #            self.add_constraint(var_list = var_list, values=val_list, rhs=Const.PH, constraint_type=ConstraintType.Initial)
        #return


    def build_derivative_constraints(self):
        self.central_constraints()
        self.forward_constraints()
        self.backward_constraints()


    def build_constraints(self):
        
        self.build_equation_constraints()
        self.build_derivative_constraints()
        self.build_initial_constraints()


        total_vars = self.var_set.num_vars + self.var_set.num_added_eps_vars
        eq_A = torch.sparse_coo_tensor([self.row_dict[ConstraintType.Equation],self.col_dict[ConstraintType.Equation]],
                                       self.value_dict[ConstraintType.Equation], 
                                       size=(self.num_added_equation_constraints, total_vars), 
                                       #size=(self.num_added_constraints, self.num_vars), 
                                       #dtype=self.dtype, device=self.device)
                                       dtype=self.dtype)
        
        #if self.n_iv > 0:
        initial_A = torch.sparse_coo_tensor([self.row_dict[ConstraintType.Initial],self.col_dict[ConstraintType.Initial]],
                                       self.value_dict[ConstraintType.Initial], 
                                       size=(self.num_added_initial_constraints, total_vars), 
                                       #dtype=self.dtype, device=self.device)
                                       dtype=self.dtype)
        #else:
        #    initial_A =None


        derivative_A = torch.sparse_coo_tensor([self.row_dict[ConstraintType.Derivative],self.col_dict[ConstraintType.Derivative]],
                                       self.value_dict[ConstraintType.Derivative], 
                                       size=(self.num_added_derivative_constraints, total_vars), 
                                       #dtype=self.dtype, device=self.device)
                                       dtype=self.dtype)

        full_A = torch.cat([eq_A, initial_A, derivative_A], dim=0)#.coalesce()
        #full_AtA = torch.mm(full_A.transpose(0,1),full_A).coalesce()
        #print('fullatannz', full_AtA._nnz())

        print(f'Constraints Shape eq {eq_A.shape}, init {initial_A.shape}, deriv {derivative_A.shape}')
        #print('first ', eq_A.shape, initial_A.shape, derivative_A.shape)
        self.num_constraints = full_A.shape[0]
        self.build_block_diag(full_A)

        derivative_rhs = self.rhs_dict[ConstraintType.Derivative]
        #derivative_rhs = torch.tensor(derivative_rhs, dtype=self.dtype, device=self.device)
        derivative_rhs = torch.tensor(derivative_rhs, dtype=self.dtype)

        self.set_row_col_sorted_indices()


        #Add batch dim
        #(b, r1, c)
        eq_A = eq_A.unsqueeze(0)
        eq_A = torch.cat([eq_A]*self.bs, dim=0)
        

        batch_A = torch.cat([full_A.unsqueeze(0)]*self.bs, dim=0)
        batch_A = batch_A.coalesce()
        self.batch_A_indices = batch_A.indices()
        self.batch_A_size = batch_A.shape

        #(b, r2, c)
        if initial_A is not None:
            initial_A = initial_A.unsqueeze(0)
            initial_A = torch.cat([initial_A]*self.bs, dim=0)
            print ('Initial/boundary Constraints shape ', initial_A.shape)

        #(b, r3, c)
        derivative_A = derivative_A.unsqueeze(0)
        derivative_A = torch.cat([derivative_A]*self.bs, dim=0)

        derivative_rhs = derivative_rhs.unsqueeze(0).repeat(self.bs,1)

        self.tc_tensor()

        self.derivative_rhs = derivative_rhs
        self.eq_A = eq_A
        #self.register_buffer("mask_A", mask_A)
        self.initial_A =  initial_A
        self.derivative_A =  derivative_A

        #build permutation
        perm_list = [torch.tensor(row) for row in self.grid_constraint_list]
        self.row_perm_inv = torch.cat(perm_list)

        self.row_perm = torch.empty_like(self.row_perm_inv)
        self.row_perm[self.row_perm_inv] = torch.arange(self.row_perm_inv.shape[0])


    #def apply_sparse_row_perm(self, M, permutation):
    #    #M = M[0].to_dense()[perm.unsqueeze(1), perm].unsqueeze(0)
    #    #return M
    #    #M = M.coalesce()
    #    indices=M._indices().clone()
    #    indices2=M._indices().clone()
    #    values=M._values().clone()
    #    rows = indices[1]
    #    #cols = indices[2]
    #    permuted_rows = permutation[rows]
    #    #permuted_cols = permutation[cols]
    #    indices2[1] = permuted_rows
    #    #indices2[2] = permuted_cols

    #    #print('rows ', rows)
    #    #print('perm ', permutation)
    #    #print('permed ', permuted_rows)

    #    D = torch.sparse_coo_tensor(indices=indices2, 
    #                            values=values, size=M.shape)
    #    return D 

    def build_block_diag(self, A):
        print('Building block diagonal A')
        #AtA = A.t()@A
        A_mat = SP.coo_matrix((A._values(), (A._indices()[0], A._indices()[1])), shape=A.shape)
        #AtA_mat = SP.coo_matrix((AtA._values(), (AtA.indices[0], AtA.indices[1])), shape=AtA.shape)

        A_list = [A_mat]*self.bs
        #AtA_list = [AtA_mat]*self.bs

        A_block = SP.block_diag(A_list)
        #AtA_block = SP.block_diag(AtA_list)

        A_block_indices = np.stack([A_block.row,A_block.col],axis=0)
        #AtA_block_indices = np.stack([AtA_block.row,AtA_block.col],axis=0)

        self.A_block_indices =torch.tensor(A_block_indices)
        #self.AtA_block_indices =torch.tensor(AtA_block_indices)

        self.A_block_shape = A_block.shape
        #self.AtA_block_shape = AtA_block.shape

        #bsmat = torch.sparse_coo_tensor(indices = bindices, values =bvalues, size=bmat.shape)
    
    def get_row_col_sorted_indices(self, row, col, num_constraints):
        """ Compute indices sorted by row and column and repeats. Useful for sparse outer product when computing constraint derivatives"""
        indices = np.stack([row, col], axis=0)

        row_sorted = indices[:, indices[0,:].argsort()]
        column_sorted = indices[:, indices[1,:].argsort()]

        #_, row_counts = np.unique(row_sorted[0], return_counts=True)
        #_, column_counts = np.unique(column_sorted[1], return_counts=True)


        row_counts = np.bincount(row_sorted[0], minlength=num_constraints)
        total_vars = self.var_set.num_vars + self.var_set.num_added_eps_vars
        #row_counts = np.bincount(row_sorted[0], minlength=total_vars)
        column_counts = np.bincount(column_sorted[1], minlength=total_vars)
        #column_counts = np.bincount(column_sorted[1], minlength=num_constraints)

        row_count = row.shape[0]
        #add batch dimension
        batch_dim = torch.arange(self.bs).repeat_interleave(repeats=row_count).unsqueeze(0)

        row_sorted = torch.tensor(row_sorted)
        column_sorted = torch.tensor(column_sorted)

        row_sorted = row_sorted.repeat(1, self.bs)
        column_sorted = column_sorted.repeat(1, self.bs)

        row_sorted = torch.cat([batch_dim, row_sorted], dim=0)
        column_sorted = torch.cat([batch_dim, column_sorted], dim=0)

        #ipdb.set_trace()
        
        return row_sorted, column_sorted, row_counts, column_counts

    def set_row_col_sorted_indices(self):
        ############### AtA indices
        #full_AtA_indices = full_AtA._indices()
        #full_AtA_size = full_AtA.shape
        #AtA_rows = full_AtA_indices[0]
        #AtA_columns = full_AtA_indices[1]

        #row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(AtA_rows, 
        #                                    AtA_columns, 
        #                                    full_AtA.shape[0])
        #print('rc sum', row_counts.sum(), column_counts.sum())

        #self.AtA_row_sorted = torch.tensor(row_sorted)
        #self.AtA_column_sorted = torch.tensor(column_sorted)
        #self.AtA_row_counts = torch.tensor(row_counts)
        #self.AtA_column_counts = torch.tensor(column_counts)

        ############## derivative indices sorted and counted
        derivative_rows = np.array(self.row_dict[ConstraintType.Derivative])
        derivative_columns = np.array(self.col_dict[ConstraintType.Derivative])
        #row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(derivative_rows, derivative_columns)
        row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(derivative_rows, derivative_columns, self.num_added_derivative_constraints)
        

        self.derivative_row_sorted = torch.tensor(row_sorted)
        self.derivative_column_sorted = torch.tensor(column_sorted)
        self.derivative_row_counts = torch.tensor(row_counts)
        self.derivative_column_counts = torch.tensor(column_counts)
        ##############


        ###############equation indices sorted and counted
        eq_rows = np.array(self.row_dict[ConstraintType.Equation])
        eq_columns = np.array(self.col_dict[ConstraintType.Equation])
        #row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(eq_rows, eq_columns)
        row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(eq_rows, eq_columns, self.num_added_equation_constraints)

        self.eq_row_sorted = torch.tensor(row_sorted)
        self.eq_column_sorted = torch.tensor(column_sorted)
        self.eq_row_counts = torch.tensor(row_counts)
        self.eq_column_counts = torch.tensor(column_counts)
        #################

    def expand_steps(self, coord, steps):
        expand_shape_step = self.step_grid_expand_shape[coord]
        new_shape_step = self.step_grid_unsqueeze_shape[coord]

        c_shape = steps.shape
        new_shape = c_shape[:1] + new_shape_step + c_shape[2:]
        #expand_shape = (-1,)*len(c_shape[:1]) + expand_shape_step + (-1,)*len(c_shape[2:])
        expand_shape = c_shape[:1] + expand_shape_step + c_shape[2:]

        steps = steps.reshape(new_shape)
        #expand over grid
        steps = steps.expand(expand_shape)
        return steps

    def expand_steps_edge(self, coord, steps):
        expand_shape_step = self.step_grid_expand_edge_shape[coord]
        new_shape_step = self.step_grid_unsqueeze_shape[coord]

        c_shape = steps.shape
        new_shape = c_shape[:1] + new_shape_step + c_shape[2:]
        #expand_shape = (-1,)*len(c_shape[:1]) + expand_shape_step + (-1,)*len(c_shape[2:])
        expand_shape = c_shape[:1] + expand_shape_step + c_shape[2:]

        steps = steps.reshape(new_shape)
        #expand over grid
        steps = steps.expand(expand_shape)
        return steps


    def solve_5pt_stencil_edge(self, coord, steps, backward=False):
        #steps shape b,  n_step-1
        # 5 point stencil starting at 0 

        if backward:
            end = torch.zeros_like(steps[:, -2:])
            stepn1 = steps[:, -3:-1]
            stepn2 = steps[:, -4:-2]
            stepn3 = steps[:, -5:-3]
            stepn4 = steps[:, -6:-4]

            left1 = -stepn1
            left2 = left1-stepn2
            left3 = left2-stepn3
            left4 = left3-stepn4

            #b, step, var, 5
            #matrix = torch.stack([left4, left3, left2, left1, end], dim=-1)
            matrix = torch.stack([end, left1, left2, left3, left4], dim=-1)
        else:
            begin = torch.zeros_like(steps[:, 0:2])
            stepn1 = steps[:, 1:3]
            stepn2 = steps[:, 2:4]
            stepn3 = steps[:, 3:5]
            stepn4 = steps[:, 4:6]

            right1 = stepn1
            right2 = right1+stepn2
            right3 = right2+stepn3
            right4 = right3+stepn4

            #b, step, var, 5
            matrix = torch.stack([begin, right1, right2, right3, right4], dim=-1)

        ones = torch.ones_like(matrix)
        mp2 = matrix.pow(2)
        matrix = torch.stack([ones, matrix, mp2 , matrix*mp2, mp2*mp2], dim=-2)

        #shape 5,2
        b = torch.tensor([[0,1,0,0,0], [0,0,2,0,0]]).type_as(matrix).T

        coeffs = torch.linalg.solve(matrix, b)

        ones = torch.ones_like(steps[:,0:2]).unsqueeze(-1)
        #values_list = []
        #coeffs1 = torch.cat([-ones, coeffs[...,0]*stepn1.unsqueeze(-1)**2, -ones*stepn1.unsqueeze(-1)**2], dim=-1)
        #coeffs1 = torch.cat([-ones, coeffs[...,0]*stepn1.unsqueeze(-1)**2, -ones*stepn1.unsqueeze(-1)**2], dim=-1)
        #coeffs2 = torch.cat([-ones, coeffs[...,1]*stepn1.unsqueeze(-1)**2, -ones*stepn1.unsqueeze(-1)**2], dim=-1)

        coeffs1 = torch.cat([ coeffs[...,0]*stepn1.unsqueeze(-1)**1, -ones*stepn1.unsqueeze(-1)**1], dim=-1)
        coeffs2 = torch.cat([ coeffs[...,1]*stepn1.unsqueeze(-1)**2, -ones*stepn1.unsqueeze(-1)**2], dim=-1)

        #coeffs1 = torch.cat([ coeffs[...,0]*stepn1.unsqueeze(-1), -ones*stepn1.unsqueeze(-1)], dim=-1)
        #coeffs2 = torch.cat([ coeffs[...,1]*stepn1.unsqueeze(-1), -ones*stepn1.unsqueeze(-1)], dim=-1)

        #coeffs1 = torch.cat([ coeffs[...,0], -ones], dim=-1)
        #coeffs2 = torch.cat([ coeffs[...,1], -ones], dim=-1)


        coeffs_list = []
        n_order1 = self.var_set.order_count[coord].get(1,0)
        if n_order1 > 0:
            ex_shape = coeffs1.shape
            ex_shape = ex_shape[:2] + (n_order1,) + ex_shape[2:]
            #print(coeffs1.shape, ex_shape, self.var_set.order_count, n_order1, coord)
            #coeffs1 = coeffs1.unsqueeze(2).repeat(self.var_set.order_count[coord].get(1,0),dim=2)
            coeffs1 = coeffs1.unsqueeze(2).expand(ex_shape)
            coeffs_list.append(coeffs1)

        #coeffs order 2 shape b, steps, num_values
        #repeat num order 2 indices b, steps,num_index, num_values

        n_order2 = self.var_set.order_count[coord].get(2,0)
        if self.var_set.order_count[coord].get(2,0) > 0:
            ex_shape = coeffs2.shape
            ex_shape = ex_shape[:2] + (n_order2,) + ex_shape[2:]
            #coeffs2 = coeffs2.unsqueeze(2).repeat(self.var_set.order_count[coord].get(2,0),dim=2)
            coeffs2 = coeffs2.unsqueeze(2).expand(ex_shape)
            coeffs_list.append(coeffs2)

        #concat along mi indices
        coeffs = torch.cat(coeffs_list,dim=2)
        #print('insi ', coeffs.shape)

        #expand over grid
        expand_shape_step = self.step_grid_expand_shape[coord]
        new_shape_step = self.step_grid_unsqueeze_shape[coord]

        c_shape = coeffs.shape
        new_shape = c_shape[:1] + new_shape_step + c_shape[2:]
        expand_shape = c_shape[:1] + expand_shape_step + c_shape[2:]

        coeffs = coeffs.reshape(new_shape)
        coeffs = coeffs.expand(expand_shape)

        #coeffs = coeffs.reshape(steps.shape[0],-1)
        return coeffs#, coeffs1, coeffs2
    
    def solve_5pt_central_stencil(self, coord, steps):
        #steps shape b,  n_step-1
        # 5 point stencil centered at 0

        center = torch.zeros_like(steps[:, 2:-1])
        stepn1 = steps[:, 2:-1]
        stepn2 = steps[:, 3:]
        stepp1 = steps[:, 1:-2]
        stepp2 = steps[:, :-3]

        left1 = -stepp1
        left2 = left1 - stepp2
        right1 = stepn1
        right2 = stepn1+stepn2

        #b, step, var, 5
        matrix = torch.stack([left2, left1, center, right1, right2], dim=-1)
        ones = torch.ones_like(matrix)
        mp2 = matrix.pow(2)
        matrix = torch.stack([ones, matrix, mp2 , matrix*mp2, mp2*mp2], dim=-2)

        #shape 5,2
        b = torch.tensor([[0,1,0,0,0], [0,0,2,0,0]]).type_as(matrix).T

        coeffs = torch.linalg.solve(matrix, b)
        #print(coeffs.shape)
        ones = torch.ones_like(center).unsqueeze(-1)
        values_list = []
        #coeffs1 = torch.cat([-ones, coeffs[...,0], -ones], dim=-1)
        #coeffs1 = torch.cat([-ones, coeffs[...,0]*stepn1.unsqueeze(-1), -ones*stepn1.unsqueeze(-1)], dim=-1)
        #coeffs1 = torch.cat([-ones, coeffs[...,0]*stepn1.unsqueeze(-1)**2, -ones*stepn1.unsqueeze(-1)**2], dim=-1)
        coeffs1 = torch.cat([coeffs[...,0]*stepn1.unsqueeze(-1)**1, -ones*stepn1.unsqueeze(-1)**1], dim=-1)
        coeffs2 = torch.cat([ coeffs[...,1]*stepn1.unsqueeze(-1)**2, -ones*stepn1.unsqueeze(-1)**2], dim=-1)

        #coeffs1 = torch.cat([coeffs[...,0]*stepn1.unsqueeze(-1), -ones*stepn1.unsqueeze(-1)], dim=-1)
        #coeffs2 = torch.cat([ coeffs[...,1]*stepn1.unsqueeze(-1), -ones*stepn1.unsqueeze(-1)], dim=-1)

        #coeffs1 = torch.cat([coeffs[...,0], -ones], dim=-1)
        #coeffs2 = torch.cat([ coeffs[...,1], -ones], dim=-1)


        #coeffs2 = torch.cat([-ones, coeffs[...,1], -ones], dim=-1)
            #coeffs2 = torch.cat([-ones, coeffs[...,1], -ones], dim=-1)
        #values_list.append(coeffs2)

        #coeffs order 1 shape b, steps, num_values
        #repeat num order 1 indices b, steps,num_index, num_values
        coeffs_list = []
        n_order1 = self.var_set.order_count[coord].get(1,0)
        if n_order1 > 0:
            ex_shape = coeffs1.shape
            ex_shape = ex_shape[:2] + (n_order1,) + ex_shape[2:]
            #print(coeffs1.shape, ex_shape, self.var_set.order_count, n_order1, coord)
            #coeffs1 = coeffs1.unsqueeze(2).repeat(self.var_set.order_count[coord].get(1,0),dim=2)
            coeffs1 = coeffs1.unsqueeze(2).expand(ex_shape)
            coeffs_list.append(coeffs1)

        #coeffs order 2 shape b, steps, num_values
        #repeat num order 2 indices b, steps,num_index, num_values
        n_order2 = self.var_set.order_count[coord].get(2,0)
        if self.var_set.order_count[coord].get(2,0) > 0:
            ex_shape = coeffs2.shape
            ex_shape = ex_shape[:2] + (n_order2,) + ex_shape[2:]
            #coeffs2 = coeffs2.unsqueeze(2).repeat(self.var_set.order_count[coord].get(2,0),dim=2)
            coeffs2 = coeffs2.unsqueeze(2).expand(ex_shape)
            coeffs_list.append(coeffs2)

        #concat along mi indices
        coeffs = torch.cat(coeffs_list,dim=2)
        #print('insi ', coeffs.shape)

        #expand over grid
        expand_shape_step = self.step_grid_expand_shape[coord]
        new_shape_step = self.step_grid_unsqueeze_shape[coord]

        c_shape = coeffs.shape
        new_shape = c_shape[:1] + new_shape_step + c_shape[2:]
        #expand_shape = (-1,)*len(c_shape[:1]) + expand_shape_step + (-1,)*len(c_shape[2:])
        expand_shape = c_shape[:1] + expand_shape_step + c_shape[2:]
        #print(new_shape, expand_shape)

        coeffs = coeffs.reshape(new_shape)
        #expand over grid
        #print('reshape ', coeffs.shape)
        coeffs = coeffs.expand(expand_shape)
        #print('expand ', coeffs.shape)

        #concat along mi index axis
        #shape b, steps, num_values
        #values = torch.cat(values_list,dim=-1)

        #repeat

        #coeffs = coeffs.reshape(steps.shape[0],-1)
        return coeffs#, coeffs1, coeffs2

    def get_central_weights(self, steps, coord):
        b = steps.shape[0]
        #steps shape b, dimn-1

        cstep = steps[:, 2:-1]
        rstep = steps[:, -3:-1]
        lstep = steps[:, 1:3]

        steps = torch.cat([lstep, cstep, rstep], dim=-1)

        n_order1 = self.var_set.order_count[coord].get(1,0)
        n_order2 = self.var_set.order_count[coord].get(2,0)
        n_mi = n_order1+n_order2

        #expand b, dim1, dim2 .. dimcoord-1, dim...

        expand_shape_step = self.step_grid_expand_shape[coord]
        new_shape_step = self.step_grid_unsqueeze_shape[coord]

        c_shape = steps.shape
        new_shape = c_shape[:1] + new_shape_step + (1,)
        expand_shape = c_shape[:1] + expand_shape_step + (n_mi,)
        

        steps = steps.reshape(new_shape)
        steps = steps.expand(expand_shape)

        steps = steps.reshape(b,-1)
        weights = steps.pow(2)

        return weights

    def build_central_values(self, steps_list):

        values_list = []
        #weights_list = []
        for coord in range(self.n_coord):
            #coeffs shape b, step_grid, num_indices, num_values
            coeffs = self.solve_5pt_central_stencil(coord, steps_list[coord])
            left_coeffs = self.solve_5pt_stencil_edge(coord, steps_list[coord], backward=False)
            right_coeffs = self.solve_5pt_stencil_edge(coord, steps_list[coord], backward=True)

            #weights_list.append(self.get_central_weights(steps_list[coord], coord))

            coeffs = torch.cat([left_coeffs, coeffs, right_coeffs], dim=1+coord)
            coeffs = coeffs.reshape(steps_list[coord].shape[0],-1)
            values_list.append(coeffs) 

        #stack along coord: b, num_coord, step_grid, num_indices, num_values 
        values = torch.cat(values_list, dim=1)
        #values = values.reshape(steps_list[0].shape[0], -1)
        #return values
        #weights = torch.cat(weights_list, dim=-1)
        return values


    def build_forward_values(self, steps_list):

        values_list = []
        #weights_list = []
        for coord, steps in enumerate(steps_list):
            b = steps.shape[0]
            #TODO expand steps
            steps = self.expand_steps(coord, steps)
            #weights_list.append(torch.ones_like(steps.reshape(b, -1)))
            #weights_list.append((steps.reshape(b, -1)).pow(2))
            #weights_list.append((steps.reshape(b, -1)))

            #steps shape b, step_grid
            repeats = (1,)*len(steps.shape)
            steps = steps.unsqueeze(dim=-1).repeat(*repeats, self.tc_count)
            steps = steps.reshape(b, self.step_grid_size[coord], self.tc_count)

            exp_tensor = self.grad_step_exp_dict[coord].type_as(steps)
            denom_tensor = self.grad_step_denom_dict[coord].type_as(steps)

            exp_tensor = exp_tensor.reshape(1,-1)
            denom_tensor = denom_tensor.reshape(1,-1)

            steps = steps.reshape(b, -1)
            steps = steps**exp_tensor
            steps = steps/denom_tensor

            values_list.append(steps)

        values = torch.cat(values_list, dim=-1)
        #weights = torch.cat(weights_list, dim=-1)
        return values

    def build_backward_values(self, steps_list):
        values_list = []
        #weights_list = []
        for coord, steps in enumerate(steps_list):
            b = steps.shape[0]
            #TODO expand steps
            steps = self.expand_steps(coord, steps)
            #eights_list.append(torch.ones_like(steps.reshape(b, -1)))
            #weights_list.append((steps.reshape(b, -1)).pow(2))
            #weights_list.append((steps.reshape(b, -1)))

            steps = -steps
            b = steps.shape[0]
            #steps shape b, step_grid
            repeats = (1,)*len(steps.shape)
            steps = steps.unsqueeze(dim=-1).repeat(*repeats, self.tc_count)
            steps = steps.reshape(b, self.step_grid_size[coord], self.tc_count)

            exp_tensor = self.back_grad_step_exp_dict[coord].type_as(steps)
            denom_tensor = self.back_grad_step_denom_dict[coord].type_as(steps)

            exp_tensor = exp_tensor.reshape(1,-1)
            denom_tensor = denom_tensor.reshape(1,-1)

            steps = steps.reshape(b, -1)
            steps = steps**exp_tensor
            steps = steps/denom_tensor

            values_list.append(steps)

        values = torch.cat(values_list, dim=-1)
        #eights = torch.cat(weights_list, dim=-1)
        return values


    def build_derivative_values(self, steps_list):
        #list of n_coord tensors
        #tensor i has dimension i with length coord_dim[i] -1, otherwise coord_dim[i]
        fv = self.build_forward_values(steps_list)
        cv = self.build_central_values(steps_list)
        bv = self.build_backward_values(steps_list)

        built_values = torch.cat([cv,fv,bv], dim=-1)
        #built_weights = torch.cat([cw,fw,bw], dim=-1)
        #built_values = torch.cat([cv], dim=-1)
        #built_values = torch.cat([cv,fv], dim=-1)

        return built_values

    def add_pad(self, eq_values):
        bs = eq_values.shape[0]
        new_tensor = torch.zeros(self.bs, *self.var_set.coord_dims, device=eq_values.device)
        new_shape = np.array(self.var_set.coord_dims)-2
        new_shape = tuple(new_shape)
        if self.n_coord==3:
            eq_values = eq_values.reshape(bs, *new_shape)
            new_tensor[:, 1:-1, 1:-1, 1:-1] = eq_values
        elif self.n_coord==2:
            eq_values = eq_values.reshape(bs, *new_shape)
            new_tensor[:, 1:-1, 1:-1] = eq_values
        else: 
            eq_values = eq_values.reshape(bs, *new_shape)
            new_tensor[:, 1:-1] = eq_values
        return new_tensor

    def remove_pad(self, eq_values, coeffs=True):
        if coeffs:
            eq_values = eq_values.reshape(self.bs, *self.var_set.coord_dims, len(self.var_set.mi_list))
            if self.n_coord==3:
                eq_values = eq_values[:, 1:-1, 1:-1, 1:-1, :]
            elif self.n_coord==2:
                eq_values = eq_values[:, 1:-1, 1:-1, :]
            else: 
                #self.n_coord==1:
                eq_values = eq_values[:, 1:-1, :]
        else:
            eq_values = eq_values.reshape(self.bs, *self.var_set.coord_dims)
            if self.n_coord==3:
                eq_values = eq_values[:, 1:-1, 1:-1, 1:-1]
            elif self.n_coord==2:
                eq_values = eq_values[:, 1:-1, 1:-1]
            else: 
                #self.n_coord==1:
                eq_values = eq_values[:, 1:-1]
        return eq_values

    def build_equation_tensor(self, eq_values):
        #eq_values = self.build_equation_values(steps).reshape(-1)
        #shape batch, n_eq, n_step, n_vars, order+1

        eq_values = self.remove_pad(eq_values, coeffs=True)

        eq_values = eq_values.reshape(-1)

        eq_indices = self.eq_A._indices()
        #G = torch.sparse_coo_tensor(eq_indices, eq_values, dtype=self.dtype, device=eq_values.device)
        G = torch.sparse_coo_tensor(eq_indices, eq_values, self.eq_A.shape, dtype=self.dtype, device=eq_values.device, )

        return G
    
    def build_derivative_tensor(self, steps_list):
        #self.derivative_A = self.derivative_A.to(steps_list[0].device)
        derivative_A = self.derivative_A.to(steps_list[0].device)
        derivative_values = self.build_derivative_values(steps_list)#.reshape(-1)
        derivative_values = derivative_values.reshape(-1)

        #print('built', len(derivative_values))
        derivative_indices = derivative_A._indices()
        G = torch.sparse_coo_tensor(derivative_indices, derivative_values, size=self.derivative_A.shape, dtype=self.dtype)

        return G


    def fill_block_constraints_torch(self, eq_A, eq_rhs, iv_rhs, derivative_A):
        #self.initial_A = self.initial_A.type_as(derivative_A)
        initial_A = self.initial_A.type_as(derivative_A)

        eq_rhs = self.remove_pad(eq_rhs, coeffs=False).reshape(self.bs, -1)

        eq_values = eq_A._values().reshape(self.bs,-1)
        #init_values = self.initial_A._values().reshape(self.bs, -1)
        init_values = initial_A._values().reshape(self.bs, -1)
        deriv_values = derivative_A._values().reshape(self.bs, -1)


        values = torch.cat([eq_values, init_values, deriv_values], dim=1)
        #values = torch.cat([eq_values, deriv_values], dim=1)
        values = values.reshape(-1)

        #self.A_block_indices = self.A_block_indices.to(values.device)
        A_block_indices = self.A_block_indices.to(values.device)

        #A_block = torch.sparse_coo_tensor(indices=self.A_block_indices, values=values, size=self.A_block_shape)
        A_block = torch.sparse_coo_tensor(indices=A_block_indices, values=values, size=self.A_block_shape)

        #self.derivative_rhs = self.derivative_rhs.type_as(eq_rhs)
        derivative_rhs = self.derivative_rhs.type_as(eq_rhs)
        #rhs = torch.cat([eq_rhs, iv_rhs, self.derivative_rhs], axis=1)
        rhs = torch.cat([eq_rhs, iv_rhs, derivative_rhs], axis=1)
        #rhs = torch.cat([eq_rhs, derivative_rhs], axis=1)
        rhs = rhs.reshape(-1)

        return A_block, rhs


    def fill_constraints_torch(self, eq_A, eq_rhs, iv_rhs, derivative_A):
        bs = eq_rhs.shape[0]

        eq_rhs = self.remove_pad(eq_rhs, coeffs=False).reshape(self.bs, -1)

        initial_A = self.initial_A.type_as(eq_A)
        AG = torch.cat([eq_A, initial_A, derivative_A], dim=1)
        #AG = torch.cat([eq_A, derivative_A], dim=1)
        #AG = torch.cat([eq_A, initial_A], dim=1)
        #if self.n_iv > 0:
        rhs = torch.cat([eq_rhs, iv_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)
        #rhs = torch.cat([eq_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)
        #else:
        #    rhs = torch.cat([eq_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)

        return AG, rhs


    def fill_constraints_torch2(self, eq_A, eq_rhs, iv_rhs, derivative_A):
        """fill without using sparse cat"""
        bs = eq_rhs.shape[0]

        eq_rhs = self.remove_pad(eq_rhs, coeffs=False).reshape(self.bs, -1)

        self.initial_A = self.initial_A.to(eq_A.device).coalesce()
        self.batch_A_indices = self.batch_A_indices.to(eq_A.device)

        initial_values = self.initial_A.values().reshape(bs, -1)
        eq_values = eq_A.values().reshape(bs, -1)
        derivative_values = derivative_A.values().reshape(bs,-1)

        #print('shapes',initial_values.shape, eq_values.shape, eq_A.shape)
        values = torch.cat([eq_values, initial_values, derivative_values], dim=-1)
        #values = torch.cat([eq_values, derivative_values], dim=-1)
        values = values.reshape(-1)

        AG = torch.sparse_coo_tensor(self.batch_A_indices, values, size=self.batch_A_size, 
                                     device=eq_A.device, dtype=self.dtype)

        #AG = torch.cat([eq_A, initial_A, derivative_A], dim=1)
        #AG = torch.cat([eq_A, initial_A], dim=1)
        #if self.n_iv > 0:
        rhs = torch.cat([eq_rhs, iv_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)
        #rhs = torch.cat([eq_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)
        #else:
        #    rhs = torch.cat([eq_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)

        return AG, rhs

    def fill_constraints_torch_dense(self, eq_A, eq_rhs, iv_rhs, derivative_A):
        bs = eq_rhs.shape[0]

        eq_rhs = self.remove_pad(eq_rhs, coeffs=False).reshape(self.bs, -1)

        initial_A = self.initial_A.to_dense().type_as(eq_A)
        AG = torch.cat([eq_A, initial_A, derivative_A], dim=1)
        #AG = torch.cat([eq_A, derivative_A], dim=1)
        #AG = torch.cat([eq_A, initial_A], dim=1)
        if self.n_iv > 0:
            rhs = torch.cat([eq_rhs, iv_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)
            #rhs = torch.cat([eq_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)
        else:
            rhs = torch.cat([eq_rhs, self.derivative_rhs.type_as(eq_rhs)], axis=1)

        return AG, rhs

    ##def fill_constraints_torch(self, eq_values, eq_rhs, iv_rhs, derivative_A):
    #def fill_constraints_torch_old(self, eq_A, eq_rhs, iv_rhs, derivative_A):
    #    bs = eq_rhs.shape[0]

    #    # (b, *)
    #    self.constraint_rhs = eq_rhs
    #    self.initial_rhs = iv_rhs

    #    self.derivative_rhs = self.derivative_rhs.type_as(eq_rhs)

    #    #ipdb.set_trace()


    #    if derivative_A is None:
    #        G = self.derivative_A
    #    else:
    #        G = derivative_A
    #        #print(G.to_dense())
    #    #G = G.type_as(constraint_A)

    #    if self.initial_A is not None:
    #        initial_A = self.initial_A.type_as(G)

    #    #print(eq_A.shape, initial_A.shape, G.shape)
    #    #print(self.constraint_A.shape, initial_A.shape, G.shape, flush=True)
    #    if self.initial_A is not None:
    #        self.AG = torch.cat([eq_A, initial_A, G], dim=1)
    #    else:
    #        self.AG = torch.cat([eq_A, G], dim=1)
    #    #self.AG = torch.cat([constraint_A, G], dim=1)
    #    #print('AG ', self.AG.shape, flush=True)

    #    self.num_constraints = self.AG.shape[1]
    #    #self.ub = torch.cat([self.constraint_rhs, self.boundary_rhs, self.derivative_ub], axis=1)

    #    if self.initial_A is not None:
    #        self.ub = torch.cat([self.constraint_rhs, self.initial_rhs, self.derivative_rhs], axis=1)
    #    else:
    #        self.ub = torch.cat([self.constraint_rhs, self.derivative_rhs], axis=1)
    #    #print('ub ', self.ub.shape, flush=True)


    def build_pde(self, coeffs, rhs, iv_rhs, derivative_A):
        self.fill_constraints_torch(coeffs, rhs, iv_rhs, derivative_A)
        #self.fill_constraints_torch_test(coeffs, rhs, iv_rhs, derivative_A)

    #def sparse_AtA_grad(self, _x, _y):
    #    """ sparse x y' for AtA"""
    #    #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
    #    #dA = dx*nu.reshape(bs, 1,num_coeffs)
    #    #correct x, y shapes

    #    b = _x.shape[0]
    #    #copy x across columns. copy y across rows
    #    #y = _y[:, self.num_added_equation_constraints+self.num_added_initial_constraints: self.num_added_equation_constraints+self.num_added_initial_constraints+self.num_added_derivative_constraints]
    #    #x = _x[:, :self.var_set.num_vars+self.var_set.num_added_eps_vars]


    #    #ipdb.set_trace()
    #    ####### dense
    #    #x = x.reshape(b, 1, -1)
    #    #y = y.reshape(b, -1, 1)

    #    #dA = y*x#.reshape(b, -1, 1)
    #    #return dA
    #    ##########

    #    x = _x.reshape(b,-1)
    #    y = _y.reshape(b,-1)

    #    self.AtA_row_counts = self.AtA_row_counts.to(x.device)
    #    self.AtA_column_counts = self.AtA_column_counts.to(x.device)

    #    y_repeat = torch.repeat_interleave(y, self.AtA_row_counts, dim=-1)
    #    x_repeat = torch.repeat_interleave(x, self.AtA_column_counts, dim=-1)

    #    x_repeat = x_repeat.reshape(-1)
    #    y_repeat = y_repeat.reshape(-1)


    #    X = torch.sparse_coo_tensor(self.AtA_row_sorted, x_repeat, 
    #                                   #size=(self.num_added_derivative_constraints, self.num_vars), 
    #                                   dtype=self.dtype, device=x.device)



    #    Y = torch.sparse_coo_tensor(self.AtA_column_sorted, y_repeat, 
    #                                   #size=(self.num_added_derivative_constraints, self.num_vars), 
    #                                   dtype=self.dtype, device=x.device)

    #    #print('outer', X.shape, Y.shape)
    #    #ipdb.set_trace()

    #    dD = X*Y

    #    return dD

    def sparse_grad_derivative_constraint(self, _x, _y):
        """ sparse x y' for derivative constraint"""
        #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
        #dA = dx*nu.reshape(bs, 1,num_coeffs)
        #correct x, y shapes

        b = _x.shape[0]
        #copy x across columns. copy y across rows
        y = _y[:, self.num_added_equation_constraints+self.num_added_initial_constraints: self.num_added_equation_constraints+self.num_added_initial_constraints+self.num_added_derivative_constraints]
        x = _x[:, :self.var_set.num_vars+self.var_set.num_added_eps_vars]


        #ipdb.set_trace()
        ####### dense
        #x = x.reshape(b, 1, -1)
        #y = y.reshape(b, -1, 1)

        #dA = y*x#.reshape(b, -1, 1)
        #return dA
        ##########

        x = x.reshape(b,-1)
        y = y.reshape(b,-1)

        derivative_row_counts = self.derivative_row_counts.to(x.device)
        derivative_column_counts = self.derivative_column_counts.to(x.device)

        y_repeat = torch.repeat_interleave(y, derivative_row_counts, dim=-1)
        x_repeat = torch.repeat_interleave(x, derivative_column_counts, dim=-1)

        x_repeat = x_repeat.reshape(-1)
        y_repeat = y_repeat.reshape(-1)

        X = torch.sparse_coo_tensor(self.derivative_row_sorted, x_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       dtype=self.dtype, device=x.device)

        Y = torch.sparse_coo_tensor(self.derivative_column_sorted, y_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       dtype=self.dtype, device=x.device)

        #Bug: multiplying sparse matrices pointwise raises floating point exception for large 3D inputs
        #dD = X*Y

        #Workaround
        X = X.coalesce()
        Y = Y.coalesce()
        #assert((X._indices()==Y._indices()).all())

        dD_values = X._values()*Y._values()

        dD = torch.sparse_coo_tensor(X._indices(), dD_values, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       #size=(self.bs, self.num_added_equation_constraints, 
                                       #     total_vars),
                                       dtype=self.dtype, device=x.device)

        return dD

    def sparse_grad_eq_constraint(self, x, y, mask):
        """ sparse x y' for eq constraint"""
        #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
        #dA = dx*nu.reshape(bs, 1,num_coeffs)
        #correct x, y shapes



        b = x.shape[0]
        #copy x across columns. copy y across rows
        y = y[:, 0:self.num_added_equation_constraints]
        #y = y[:, 1:self.num_vars]
        #_x = x[:, 0:self.num_vars+self.num_added_eps_vars]
        #x = x[:, 0:self.num_vars+self.n_step]
        x = x[:, 0:self.var_set.num_vars]

        mx = mask*x.reshape(b, 1, -1)
        my = mask*y.reshape(b, -1, 1)

        dD = mx*my
        return dD

        
        ########dense
        x = x.reshape(b, 1, -1)
        y = y.reshape(b, -1, 1)

        dA_dense = y*x#.reshape(b, -1, 1)
        #return dA_dense
        #######3

        x = x.reshape(b,-1)
        y = y.reshape(b,-1)

        eq_row_counts = self.eq_row_counts.to(x.device)
        eq_column_counts = self.eq_column_counts.to(x.device)

        y_repeat = torch.repeat_interleave(y, eq_row_counts, dim=-1)
        x_repeat = torch.repeat_interleave(x, eq_column_counts, dim=-1)

        x_repeat = x_repeat.reshape(-1)
        y_repeat = y_repeat.reshape(-1)


        total_vars = self.var_set.num_vars + self.var_set.num_added_eps_vars

        X = torch.sparse_coo_tensor(self.eq_row_sorted, x_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       #size=(self.bs, self.num_added_equation_constraints, total_vars),
                                       size=(self.bs, self.num_added_equation_constraints, self.var_set.num_vars),
                                       dtype=self.dtype, device=x.device)

        Y = torch.sparse_coo_tensor(self.eq_column_sorted, y_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       size=(self.bs, self.num_added_equation_constraints, self.var_set.num_vars),
                                       #     total_vars),
                                       dtype=self.dtype, device=y.device)


        #Y = torch.sparse_coo_tensor(self.eq_column_sorted, y_repeat, 
        #                               #size=(self.num_added_derivative_constraints, self.num_vars), 
        #                               size=(self.bs, self.num_added_equation_constraints, 
        #                                    total_vars),
        #                               dtype=self.dtype, device=y.device)

        #Bug: multiplying sparse matrices pointwise raises floating point exception for large 3D inputs
        #ipdb.set_trace()
        #dD1 = Y*X

        X = X.coalesce()
        Y = Y.coalesce()
        #assert((X._indices()==Y._indices()).all())

        dD_values = X._values()*Y._values()

        dD = torch.sparse_coo_tensor(X._indices(), dD_values, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       size=(self.bs, self.num_added_equation_constraints, 
                                            total_vars),
                                       dtype=self.dtype, device=x.device)

        return dD, dA_dense
    
    #def sparse_grad_derivative_constraint(self, x, y, dense=False):
    #    """ sparse x y' for derivative constraint"""
    #    #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
    #    #dA = dx*nu.reshape(bs, 1,num_coeffs)
    #    #correct x, y shapes

    #    b = x.shape[0]
    #    #copy x across columns. copy y across rows
    #    x = x[:, self.num_added_equation_constraints+self.num_added_initial_constraints: self.num_added_equation_constraints+self.num_added_initial_constraints+self.num_added_derivative_constraints]
    #    y = y[:, :self.var_set.num_vars]

    #    x = x.reshape(b,-1)
    #    y = y.reshape(b,-1)

    #    if dense:
    #        x = x.unsqueeze(2)
    #        y = y.unsqueeze(1)
    #        outer = x*y
    #        return outer

    #    #x = x.reshape(b, -1, 1)
    #    #y = y.reshape(b, 1, -1)

    #    #dA = x*y.reshape(b, 1,-1)
    #    #return dA


    #    self.derivative_row_counts = self.derivative_row_counts.to(x.device)
    #    self.derivative_column_counts = self.derivative_column_counts.to(x.device)

    #    #x_repeat = torch.repeat_interleave(x, self.derivative_row_counts, dim=-1)
    #    #y_repeat = torch.repeat_interleave(y, self.derivative_column_counts, dim=-1)


    #    x_repeat = torch.repeat_interleave(x, self.derivative_column_counts, dim=-1)
    #    y_repeat = torch.repeat_interleave(y, self.derivative_row_counts, dim=-1)

    #    x_repeat = x_repeat.reshape(-1)
    #    y_repeat = y_repeat.reshape(-1)

    #    X = torch.sparse_coo_tensor(self.derivative_row_sorted, x_repeat, 
    #                                   #size=(self.num_added_derivative_constraints, self.num_vars), 
    #                                   dtype=self.dtype, device=x.device)

    #    Y = torch.sparse_coo_tensor(self.derivative_column_sorted, y_repeat, 
    #                                   #size=(self.num_added_derivative_constraints, self.num_vars), 
    #                                   dtype=self.dtype, device=x.device)

    #    #ipdb.set_trace()

    #    dD = X*Y

    #    return dD

    #def sparse_grad_eq_constraint(self, x, y):
    #    """ sparse x y' for eq constraint"""
    #    #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
    #    #dA = dx*nu.reshape(bs, 1,num_coeffs)
    #    #correct x, y shapes

    #    b = x.shape[0]
    #    #copy x across columns. copy y across rows
    #    x = x[:, 0:self.num_added_equation_constraints]
    #    #y = y[:, 1:self.var_set.num_vars]
    #    y = y[:, 0:self.var_set.num_vars]

    #    #y = y[:, 0:self.num_added_equation_constraints]
    #    #x = x[:, 0:self.num_vars]

    #    #x = x.reshape(b, -1, 1)
    #    #y = y.reshape(b, 1, -1)

    #    #dA = x*y.reshape(b, 1,-1)
    #    #return dA

    #    x = x.reshape(b,-1)
    #    y = y.reshape(b,-1)

    #    self.eq_row_counts = self.eq_row_counts.to(x.device)
    #    self.eq_column_counts = self.eq_column_counts.to(x.device)

    #    #x_repeat = torch.repeat_interleave(x, self.eq_row_counts, dim=-1)
    #    #y_repeat = torch.repeat_interleave(y, self.eq_column_counts, dim=-1)

    #    y_repeat = torch.repeat_interleave(y, self.eq_row_counts, dim=-1)
    #    x_repeat = x #torch.repeat_interleave(x, self.eq_column_counts, dim=-1)

    #    x_repeat = x_repeat.reshape(-1)
    #    y_repeat = y_repeat.reshape(-1)

    #    X = torch.sparse_coo_tensor(self.eq_row_sorted, x_repeat, 
    #                                   #size=(self.num_added_derivative_constraints, self.num_vars), 
    #                                   dtype=self.dtype, device=x.device)

    #    Y = torch.sparse_coo_tensor(self.eq_column_sorted, y_repeat, 
    #                                   #size=(self.num_added_derivative_constraints, self.num_vars), 
    #                                   dtype=self.dtype, device=x.device)

    #    #ipdb.set_trace()

    #    dD = X*Y

    #    return dD


def test_perm():
    coord_dims = (8,9)
    bs = 1
    step = 0.1
    pde = PDESYSLP(bs=bs, coord_dims=coord_dims, n_iv=1, step_size=step, order=2, n_iv_steps=1, 
                step_list = None, build=True)


def test_mat_eq():
    coord_dims = (8,9)
    bs = 1
    step = 0.1
    pde = PDESYSLP(bs=bs, coord_dims=coord_dims, n_iv=1, step_size=step, order=2, n_iv_steps=1, 
                step_list = None, build=False)

    pde.build_constraints()
    #steps shape: (batch, n_coord, dim0-1, dim1-1, dim2-1)
    #steps0 = 0.25*torch.ones(bs, coord_dims[0]-1, coord_dims[1])
    #steps0 = 0.25*torch.ones(bs, *pde.step_grid_shape[0])
    #steps1 = 0.25*torch.ones(bs, coord_dims[0], coord_dims[1]-1)

    steps0 = step*torch.ones(bs, coord_dims[0]-1)
    steps1 = step*torch.ones(bs, coord_dims[1]-1)
    #steps2 = 0.25*torch.ones(bs, coord_dims[2]-1)
    #steps3 = 0.25*torch.ones(bs, coord_dims[3]-1)

    steps_list = [steps0, steps1]
    #pde.tc_tensor()
    derivative_A = pde.derivative_A#.todense()


    fill_A = pde.build_derivative_tensor(steps_list)
    vd = derivative_A._values()
    vf = fill_A._values()

    print(fill_A.shape, derivative_A.shape, fill_A._nnz(), derivative_A._nnz())
    #print(fill_A._indices())
    #print(derivative_A._indices())
    diff = vd-vf
    print((diff))
    print(diff.mean(), diff.abs().max())

    #print(X[0, 198,318])

    #print((fi==di).all())

    #dda = derivative_A.coalesce().to_dense()
    #dfa = fill_A.coalesce().to_dense()
    #print(dda, dda.shape)
    #print(dfa, dfa.shape)
    #print(dda[:,:,-1])
    #print((dda[:,:,:142]-dfa).sum())
    #print(vd)
    #print(vf)
    #print(list(zip(vd,vf)))

    #repr = pde.repr_taylor(values = vf, print_row=True)
    #print(repr)
    ##print(vf)

    #print("********")
    #repr = pde.repr_taylor(values = vd, print_row=True)
    #print(repr)

def test_taylor_repr():
    pde = PDESYSLP(bs=1, coord_dims=(5,6), n_iv=1, step_size=0.25, order=2, 
                n_iv_steps=1, step_list = None, build=False)
    #pde.build_equation_constraints()
    #pde.build_derivative_constraints()
    #pde.forward_constraints()
    #pde.backward_constraints()
    pde.build_constraints()
    #pde.central_constraints()
    #pde.tc_tensor()
    repr = pde.repr_taylor()
    print(repr)

def test_initial():
    init_list = [(0,0,[0,3],[0,5]), (1,0,[1,0],[3,0])]
    pde = PDESYSLP(bs=1, coord_dims=(4,6), n_iv=1, step_size=0.25, order=2, 
                   init_index_mi_list=init_list, n_iv_steps=1, step_list = None, build=True)
    #pde.build_equation_constraints()
    #pde.build_initial_constraints()
    #pde.build_constraints()
    #pde.build_derivative_constraints()
    #pde.tc_tensor()
    repr = pde.repr_eq(type=ConstraintType.Initial)
    print(repr)

def test_eq2():
    pde = PDESYSLP(bs=1, coord_dims=(5,6), n_iv=1, step_size=0.25, order=2, 
                n_iv_steps=1, step_list = None, build=False)
    #pde.build_equation_constraints()
    #pde.build_initial_constraints()
    pde.build_constraints()
    #pde.tc_tensor()
    coeffs = torch.arange(pde.var_set.num_pde_vars).reshape(pde.var_set.grid_size, len(pde.var_set.mi_list))
    #coeffs[:, 0] = 1
    #coeffs[:, 4] = 0
    rhs = torch.ones(pde.var_set.grid_size)
    G = pde.build_equation_tensor(coeffs)
    _values = G._values()
    repr = pde.repr_eq(values =_values.numpy(),rhs=rhs, type=ConstraintType.Equation)
    print(repr)

def test_eq():
    pde = PDESYSLP(bs=1, coord_dims=(4,6), n_iv=1, step_size=0.25, order=2, 
                n_iv_steps=1, step_list = None, build=False)
    pde.build_equation_constraints()
    pde.build_derivative_constraints()
    pde.tc_tensor()
    repr = pde.repr_eq()
    print(repr)

def test_grid():
    pde = PDESYSLP(bs=1, coord_dims=(4,6), n_iv=1, step_size=0.25, order=2, n_iv_steps=1, step_list = None, build=True)
    #pde.build_equation_constraints()
    #pde.build_derivative_constraints()
    #pde.tc_tensor()
    #step0 = torch.ones((3,1).repeat((1,6))
    step0 = torch.arange(3).unsqueeze(1).repeat((1,6))
    step1 = torch.arange(5).unsqueeze(0).repeat((4,1))
    #repr = pde.repr_eq()
    #print(repr)
    print(pde.var_set.grid_indices)
    print(step0.reshape(-1))
    print(step1.reshape(-1))

def test():
    n_step = 10
    dim =1
    #steps = 0.1*torch.ones(1,n_step-1,dim)
    _steps = 0.01+ np.random.random(n_step-1)
    steps = torch.tensor(_steps).reshape(1,n_step-1,1)

    ode = ODESYSLP(bs=1, n_dim=dim, n_equations=1, n_auxiliary=0, n_step=n_step, step_size=0.1, order=2, n_iv=1, device='cpu', step_list=_steps)

    derivative_constraints,deriv_values = ode.build_derivative_tensor(steps)
    #eq_constraints = self.ode.build_equation_tensor(coeffs)

    fix_values = ode.value_dict[ConstraintType.Derivative]

    print('A',deriv_values)
    print('B', fix_values)

    #print(ode.value_dict[ConstraintType.Derivative])
    diff = deriv_values - torch.tensor(fix_values)
    print(diff)
    print(diff.mean())

if __name__=="__main__":
    #ODESYSLP().ode()
    #test_eq()
    #test_eq2()
    #test_taylor_repr()
    #test_mat_eq()
    #test_grid()
    #test_initial()
    test_perm()
