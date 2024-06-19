import torch

interval_bs = 10

#TODO equality and inequality constraints
#equality constraints have unbounded dual vars
#inequality constraints with dual vars bounded below by 0, unbounded above.
#gradient

def variable_roll(X, shifts):
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    arange1 = torch.arange(n_cols, device='cuda').view(( 1,n_cols)).repeat((n_rows,1))
    arange2 = (arange1 - shifts) % n_cols
    return torch.gather(X, 1, arange2)

@torch.no_grad()
def gradient_projection(A_in, b_in, gamma, d_in):
    """solve dual QP using gradient projection 
    
    min -1/2 y^t (A Ginv A^t) y + y^t (-A Ginv d - b)
    given y \ge 0

    Ginv = 1/gamma * I

    A is a batch of sparse constraints, rhs b: Ax \ge b
    d = [1, 0s]
    
    y: current y
    """

    # G = A_in 1/gamma A_in^t
    # d = -A_in 1/gamma d_in - b_in

    
    #1. choose initial feasible y

    #2. compute cauchy point.

    #2.5 optional. apply subspace reduction

    #3. check kkt

    #4. next


@torch.no_grad()
def compute_cauchy_point(G, c, y, n_eq, n_ineq):
    """ Compute Cauchy point
    min 1/2 x^t G x + x^t c
    x \ge l
    G: sparse tensor
    y: current y: shape(batch, dimension)
    Assume dimension is num equality constraints + num inequality constraints in that order
    """

    N = y.shape[1]
    #TODO add equality constraints

    #1. Compute gradient 

    g = G@y + c

    #2. Compute breaks for each coordinate i.
    # t_i = (yi - ui)/gi if gi < 0 ui < inf. for this problem ui is always inf.
    # t_i = (y_i - li)/gi if gi>0, li > -inf. for this problem li is always 0.
    # t_i = inf otherwise.

    # TODO fix for unbounded below
    t_rhs_2 = y/g
    t_breaks = torch.where((g>0), t_rhs_2, torch.inf)
    #Set inf for equality constraints
    t_breaks[:, :n_eq] = torch.inf

    ############
    # remove duplicate breaks and inf breaks. roll to move them at the end.
    num_zero = torch.isclose(t_breaks, torch.tensor([0.])).int().sum(dim=-1).long()
    num_inf = torch.isinf(t_breaks).int().sum(dim=-1)

    #TODO handle zero infs
    num_intervals = t_breaks.shape[0] - num_zero - num_inf + 1

    max_num_intervals = num_intervals.max()

    #3. Get sorted and unique breaks. 0<t1<t2 ... <tn
    # For this specific problem, only 0's and infs can be duplicates.
    # Count number of 0 break points for each y
    # sort break points.
    # roll tensor (each row) by number of 0s -1
    # use the max number of intervals (over rows) for the number of intervals.

    sorted_t_breaks,_ = torch.sort(t_breaks, dim=-1)

    t_breaks_rolled = variable_roll(sorted_ts, num_zero)

    #TODO: add intial 0.
    time_intervals = t_breaks_rolled
    #############

    #TODO add zero column


    #. pad with inf to get equal intervals per batch

    #4. Loop over batches of intervals. [t_{j-1}, t_j]

        #5. compute y(t_{j-1}), p^{j-1} 
        
        #6. compute f_j-1, f'_{j-1}, f''_{j-1}

        #7. compute delta_t = -f'/f''

        #8. check if minimum exists in interval. Find interval and delta_t

    #t_breaks shape [batch, dimension]
    #TODO remove last dims
    #reduced_breaks shape [batch, n_intervals] 
    found = torch.zeros(y.shape[0]).type_as(y)
    min_t = torch.zeros(y.shape[0]).type_as(y)

    for begin in range(0, max_num_intervals, interval_bs):
        end = begin + max(begin+interval_bs, max_num_intervals)

        #get batch of breaks [batch, dimension, interval_set]
        #batch, 1,  interval_batch
        ti_begin_batch= time_intervals[:, None, begin:end]
        ti_end_batch= time_intervals[:, None, begin+1:end+1]
        #t_breaks: batch, N
        
        #g, y shape [batch, N]
        y_j = y[...,None]
        g_j = g[...,None]

        y_j = torch.where(ti_begin_batch < t_breaks[...,None], y_j - ti_begin_batch*g_j, y_j - t_breaks[...,None]*g_j)
        p_j = torch.where(ti_begin_batch < t_breaks[...,None], -g_j, torch.zeros_like(g_j))

        #y_j shape [batch, dimension, interval_set ]
        #p_j shape [batch, dimension, interval_set ]

        #y(t) for each t in t_list

        #output shapes: batch, G-dim[0], num_intervals
        Gy = torch.bmm(G, y_j)
        Gp = torch.bmm(G, p_j)

        ## c^t y
        cy = c.unsqueeze(1)
        #yhape: b, 1, n_interval
        cy = torch.bmm(cy, y_j)
        #shape: b, n_interval
        cy = cy.squeeze(1)

        ## c^t p
        cp = c.unsqueeze(1)
        #shape: b, 1, n_interval
        cp = torch.bmm(cp, p_j)
        #shape: b, n_interval
        cp = cp.squeeze(1)

        ## f_j
        ## c^t y + 1/2 y^tGy 
        #shape: b,N,n_interval * b,N,n_interval
        f_j = 0.5* (y*Gy).sum(dim=2)
        #: b,n_interval + b, 
        f_j = cy + f_j

        ## fp_j
        ## c^t p + 1/2 y^tGp 
        fp_j = (y*Gp).sum(dim=2)
        #: b,n_interval + b, 
        fp_j = cp + fp_j

        ## fpp_j
        ## p^tGp 
        fpp_j = (p_j*Gp).sum(dim=2)

        # Compute minimizing delta t
        # shape b, n_interval
        delta_t = -fp_j/fpp_j
        assert((delta_t >=0).all())

        #check if minimum is in one of intervals
        #if fp_j > 0 -> min at begin
        #else if delta_t < t_break[i+1]-t_break[i] -> min at t[i] + delta_t
        #else min not in interval
        #first interval with min

        #min time found in this batch
        fp_gt_0 = (fp_j>0)
        delta_t_in_int = (delta_t < (ti_end_batch - ti_begin_batch).squeeze())

        batch_min_t = fp_gt_0.float()*ti_begin_batch + (1-fp_gt_0.float())*delta_t_in_int.float()*delta_t
        #whether we found a min t in this batch fp_j > 0 or 
        batch_found = fp_gt_0 | delta_t_in_int

        #save previous found status
        prev_found = found
        #update found status only if we found a min in this batch
        found = (1-found)*batch_found

        #update min t only if we hadn't a previous min
        min_t = (1-prev_found)*found*batch_min_t

    #for the first interval with minimum
    #9. y = y(tj-1) + delta_t p^{j-1}

    #return cauchy point
    

############## Test
