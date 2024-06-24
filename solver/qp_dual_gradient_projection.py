import torch
import numpy as np

from solver.ode_layer import ODEINDLayerTest

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

def get_primal_vars(A, H, gamma, c, y, n_eq):
    """
    x = G^-1(At yeq + Ht yineq -c) 
    """

    AT = A.transpose(1,2)
    HT = H.transpose(1,2)

    phi = y[:, :n_eq].unsqueeze(2)
    lam = y[:, n_eq:].unsqueeze(2)

    x = torch.bmm(AT, phi)
    x = x + torch.bmm(HT, lam)
    x = x - c
    x = x/gamma

    return x

@torch.no_grad()
def gradient_projection(A, H, b, gamma, d):
    """
    Input QP:
    min 1/2 x^t G x + x^t d
    Given  Ax == b 
           Hx \ge 0
    G = gamma*I
    
    solve dual QP using gradient projection 
    
    min -1/2 y^t (A Ginv A^t) y + y^t (-A Ginv d - b)
    given y \ge 0

    Ginv = 1/gamma * I

    A is a batch of sparse constraints, rhs b: Ax \ge b
    d = [1, 0s]
    
    y: current y
    """

    n_eq = A.shape[1]
    # G = A_in 1/gamma A_in^t
    # d = -A_in 1/gamma d_in - b_in

    
    #1. choose initial feasible y

    #2. compute cauchy point.

    #2.5 optional. apply subspace reduction

    #3. check kkt

    #4. next

def get_gradient(AH, gamma, cp, y):
    """
    G' = [A,  H] 1/gamma*I * [At Ht]
    c' = 1/gamma I * c + [b 0]
    g = G'y + c
    """
    # Concat along row
    
    assert(len(y.shape)==3)
    #g = AH.T @ y 
    AHT = AH.transpose(1,2)
    #g = AH.T @ y 
    g = torch.bmm(AHT, y) 
    #g = AH @ g
    g = torch.bmm(AH, g)
    g = (1/gamma)*g 

    g = g + cp

    return g

def compute_delta_t(AH, gamma, cp, yj, pj):

    # G' = [A, H]1/gamma I [At Ht]
    AHT = AH.transpose(1,2)
    AHy = torch.bmm(AHT, yj)
    AHp = torch.bmm(AHT, pj)

    AHy = torch.bmm(AH, AHy)
    AHp = torch.bmm(AH, AHp)

    AHy = AHy/gamma
    AHp = AHp/gamma

    ## c^t y
    #y shape: b, 1, n_interval
    cpy = (cp*yj).sum(dim=1)
    ##shape: b, n_interval
    #cy = cy.squeeze(1)

    ## c^t p
    #shape: b, n_interval
    cpp = (cp*pj).sum(dim=1)

    ## f_j
    ## c^t y + 1/2 y^tGy 
    #shape: b,N,n_interval * b,N,n_interval
    f_j = (yj*AHy).sum(dim=1)
    f_j = 0.5*f_j/gamma
    #: b,n_interval + b, 
    f_j = cpy + f_j

    ## fp_j
    ## c^t p + 1/2 y^tGp 
    fp_j = (yj*AHp).sum(dim=1)
    fp_j = fp_j/gamma
    #: b,n_interval + b, 
    fp_j = cpp + fp_j

    ## fpp_j
    ## p^tGp 
    fpp_j = (pj*AHp).sum(dim=1)
    fpp_j = fpp_j/gamma

    # Compute minimizing delta t
    # shape b, n_interval
    delta_t = -fp_j/fpp_j
    #assert((delta_t >=0).all())

    return delta_t, fp_j

@torch.no_grad()
def compute_cauchy_point(A, H, A_rhs, H_rhs, gamma, c, y, n_eq):
    """ Compute Cauchy point
    min 1/2 x^t G x + x^t c
    x \ge l
    G: sparse tensor
    y: current y: shape(batch, dimension)
    Assume dimension is num equality constraints + num inequality constraints in that order
    """

    
    assert(len(y.shape)==2)
    y = y.unsqueeze(2)
    c = c.unsqueeze(2)
    A_rhs = A_rhs.unsqueeze(2)
    H_rhs = H_rhs.unsqueeze(2)
    N = y.shape[1]

    AH = torch.cat([A, H], dim=1)
    rhs = torch.cat([A_rhs, H_rhs], dim=1)


    #c' = -(1/gamma I * c + [b 0])
    cp = torch.bmm(AH, c)
    cp = cp/gamma
    cp = cp+rhs
    cp = -cp

    #TODO add equality constraints

    #1. Compute gradient 
    #g = G@y + c
    g = get_gradient(AH, gamma, cp, y)

    #2. Compute breaks for each coordinate i.
    # t_i = (yi - ui)/gi if gi < 0 ui < inf. for this problem ui is always inf.
    # t_i = (y_i - li)/gi if gi>0, li > -inf. for this problem li is always 0.
    # t_i = inf otherwise.

    # Only lower bound is valid for inequalities \lambda >= 0
    t_rhs_2 = y/g
    t_breaks = torch.where((g>0), t_rhs_2, torch.inf)
    # break at inf whenever g==0.
    t_breaks = t_breaks.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=torch.inf)

    #Set inf for equality constraints
    t_breaks[:, :n_eq] = torch.inf
    t_breaks = t_breaks.squeeze(2)

    ############
    # remove duplicate breaks and inf breaks. roll to move them at the end.

    #3. Get sorted and unique breaks. 0<t1<t2 ... <tn
    # For this specific problem, only 0's and infs can be duplicates.
    # Count number of 0 break points for each y
    # sort break points.
    # roll tensor (each row) by number of 0s -1
    # use the max number of intervals (over rows) for the number of intervals.

    sorted_t_breaks,_ = torch.sort(t_breaks, dim=1)

    #TODO check for other duplicates
    #First index of max element for each example. infs may be repeated.
    num_intervals = sorted_t_breaks.argmax(dim=1)
    max_num_intervals = num_intervals.max()+1

    #t_breaks_rolled = variable_roll(sorted_ts, num_zero)
    #add intial 0.
    zz = torch.zeros((sorted_t_breaks.shape[0],1), device=sorted_t_breaks.device)
    time_intervals = torch.cat([zz, sorted_t_breaks], dim=1)
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
    found = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
    opt_y = torch.zeros_like(y.squeeze(2))
    #opt_init_times = torch.zeros_like(y)
    #opt_delta = torch.zeros_like(y)
    r = torch.arange(y.shape[0], device=y.device)

    for begin in range(0, max_num_intervals, interval_bs):
        end = min(begin+interval_bs, max_num_intervals)

        #get batch of breaks [batch, dimension, interval_set]
        #batch, 1,  interval_batch
        #ti_begin_batch= time_intervals[:, None, begin:end]
        #ti_end_batch= time_intervals[:, None, begin+1:end+1]

        ti_begin_batch= time_intervals[:, None, begin:end]
        ti_end_batch= time_intervals[:, None, begin+1:end+1]
        #t_breaks: batch, N
        
        #g, y shape [batch, N]
        y_j = y#[...,None]
        g_j = g#[...,None]

        y_j = torch.where(ti_begin_batch < t_breaks[...,None], y_j - ti_begin_batch*g_j, y_j - t_breaks[...,None]*g_j)
        p_j = torch.where(ti_begin_batch < t_breaks[...,None], -g_j, torch.zeros_like(g_j))


        #y_j shape [batch, dimension, interval_set ]
        #p_j shape [batch, dimension, interval_set ]


        #y(t) for each t in t_list

        delta_t, fp_j = compute_delta_t(AH, gamma, cp, y_j, p_j)

        #check if minimum is in one of intervals
        #if fp_j > 0 -> min at begin
        #else if delta_t < t_break[i+1]-t_break[i] -> min at t[i] + delta_t
        #else min not in interval
        #first interval with min

        ti_begin_batch= ti_begin_batch.squeeze(1)
        ti_end_batch= ti_end_batch.squeeze(1)

        #min time found in this batch
        fp_gt_0 = (fp_j>0)
        #delta_t_in_int = (delta_t < (ti_end_batch - ti_begin_batch).squeeze(1))
        #delta_t_in_int = (delta_t < (ti_end_batch - ti_begin_batch))
        delta_t_in_int = ((delta_t < (ti_end_batch - ti_begin_batch)) & (delta_t >= 0))

        #_batch_min_t = delta_t
        batch_min_t = fp_gt_0.float()*0. + (1-fp_gt_0.float())*delta_t_in_int.float()*delta_t

        #save previous found status
        prev_found = found

        #whether we found a min t in this batch fp_j > 0 or 
        batch_found = fp_gt_0 | delta_t_in_int
        #batch_found = torch.where((~prev_found).unsqueeze(2), fp_gt_0 | delta_t_in_int, batch_found)

        #update found status only if we found a min in this batch
        #found = (1-found)*batch_found + found
        found = torch.where(found==False, batch_found.any(dim=1), found)

        #index of first time inteval
        found_index = torch.argmax(batch_found.int(), dim=1)
        #time for which opt is found
        batch_opt_init_times = ti_begin_batch[r, found_index].unsqueeze(1)
        batch_opt_delta_t = batch_min_t[r, found_index]


        #TODO: move outside loop
        #Compute y and p for optimal interval
        y_j = y.squeeze(2)#[...,None]
        g_j = g.squeeze(2)#[...,None]
        y_j = torch.where(batch_opt_init_times < t_breaks, y_j - batch_opt_init_times*g_j, y_j - t_breaks*g_j)
        p_j = torch.where(batch_opt_init_times < t_breaks, -g_j, torch.zeros_like(g_j))

        #update min t only if we hadn't a previous min
        #Compute new Cauchy point given optimal delta t
        _opt = y_j + batch_opt_delta_t * p_j
        #Update only ones for which optimal interval was found
        opt_y = torch.where((~prev_found & found).unsqueeze(1), _opt, opt_y)

        #opt_init_times = torch.where((~prev_found & found).unsqueeze(2), batch_opt_init_times, opt_init_times)
        #opt_delta_t = torch.where((~prev_found & found).unsqueeze(2), batch_opt_delta_t, opt_delta_t)

        if found.all():
            break

    #for the first interval with minimum
    #9. y = y(tj-1) + delta_t p^{j-1}

    x = get_primal_vars(A, H, gamma, c, opt_y, n_eq)
    return opt_y
    #return cauchy point
    

############## Test

def test():
    step_size = 0.05
    end = 3*step_size
    n_step = int(end/step_size)
    order=2

    steps = step_size*np.ones((n_step-1,))
    steps = torch.tensor(steps)

    #coeffs are c_2 = 1, c_1 = 0, c_0 = 0
    _coeffs = np.array([[1,0,1]], dtype='float32')
    _coeffs = np.repeat(_coeffs, n_step, axis=0)
    _coeffs = torch.tensor(_coeffs)

    _rhs = torch.tensor(0.)
    rhs = _rhs.repeat(n_step)

    # initial values at time t=0. 
    iv = torch.tensor([0,1], dtype=torch.float32)

    coeffs = _coeffs.reshape(1, n_step, order+1)
    rhs = rhs.unsqueeze(0)
    iv = iv.unsqueeze(0)

    ode = ODEINDLayerTest(bs=1,order=order,n_ind_dim=1,n_iv=2,n_step=n_step,n_iv_steps=1)

    u0,u1,u2,eps,_, eq, initial, derivative, eps_tensor = ode(_coeffs, rhs, iv, steps)

    #derivative.to_dense()

    b = np.array([-1]*derivative.shape[-1])
    b[0] = 1
    b = torch.tensor(b)[None, None, ...]
    derivative_neg = derivative*b
    #print(derivative_neg.to_dense())
    
    A = torch.cat([eq, initial], dim=1)
    H = torch.cat([derivative, derivative_neg, eps_tensor], dim=1)

    print(A.shape)
    print(H.shape)

    A_rhs = torch.cat([rhs, iv], dim=1)
    H_rhs = torch.zeros(H.shape[0:2]).type_as(rhs)

    print(A_rhs.shape)
    print(H_rhs.shape)

    #c = torch.cat([A_rhs, H_rhs], dim=1)
    c = torch.zeros((1,A.shape[2]), device=coeffs.device).double()
    c[:,0] = 1
    y_init = torch.rand((coeffs.shape[0], A.shape[1]+H.shape[1])).double()

    compute_cauchy_point(A, H, A_rhs, H_rhs, 1, c, y_init, A.shape[1])


test()