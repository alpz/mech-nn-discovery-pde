import torch

interval_bs = 10

def variable_roll(X, shifts):
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    arange1 = torch.arange(n_cols, device='cuda').view(( 1,n_cols)).repeat((n_rows,1))
    arange2 = (arange1 - shifts) % n_cols
    return torch.gather(X, 1, arange2)

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



def compute_cauchy_point(A_in, b_in, gamma, d_in, y):
    """ Compute Cauchy point
    A is a batch of sparse constraints, rhs b: Ax \ge b
    d = [1, 0s], min eps
    
    y: current y
    """
    #TODO add equality constraints

    #1. Compute gradient 
    #g = Gy + d 

    g = G@y + d

    #2. Compute breaks for each coordinate i.
    # t_i = (yi - ui)/gi if gi < 0 ui < inf. for this problem ui is always inf.
    # t_i = (y_i - li)/gi if gi>0, li > -inf. for this problem li is always 0.
    # t_i = inf otherwise.

    t_rhs_2 = y/g
    t_breaks = torch.where((g>0), t_rhs_2, torch.inf)

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

    #TODO add zero column


    #. pad with inf to get equal intervals per batch

    #4. Loop over batches of intervals. [t_{j-1}, t_j]

        #5. compute y(t_{j-1}), p^{j-1} 
        
        #6. compute f_j-1, f'_{j-1}, f''_{j-1}

        #7. compute delta_t = -f'/f''

        #8. check if minimum exists in interval. Find interval and delta_t

    for begin in range(0, max_num_intervals, interval_bs):
        end = begin + max(begin+interval_bs, max_num_intervals)

        #batch, interval
        t_interval = t_breaks_rolled[:, None, begin:end]
        #t_breaks: batch, N
        
        y_j = y[...,None]
        g_j = g[...,None]

        y_j = torch.where(t_interval < t_breaks[...,None], y_t - t_interval*g_j, y_t - t_breaks[...,None]*g_j  )
        p_j = torch.where(t_interval < t_breaks[...,None], -g_j, torch.zeros_like(g_j))

        #y(t) for each t in t_list

    #for the first interval with minimum
    #9. y = y(tj-1) + delta_t p^{j-1}

    #return cauchy point
    
