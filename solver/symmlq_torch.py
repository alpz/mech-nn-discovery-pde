import torch
import numpy as np
import solver.minres_torch as MR

def machine_epsilon():
    """Return the machine epsilon in double precision."""
    return np.finfo(np.double).eps

def apply_block_jacobi_M(Blocks, x:torch.Tensor, upper:bool=False, block_size:int=100, stride:int=100):

    LU, pivots = Blocks
    Blocks = LU
    #Blocks = Blocks.transpose(-1,-2)
    #upper=True
    #block_size = 300
    #stride=block_size//2
    step_size = stride

    n_blocks = Blocks.shape[1]
    block_size = Blocks.shape[2]

    #len_x = (n_blocks-2)*step_size + 2*step_size
    #len_x = (n_blocks-1)*step_size + block_size-step_size #+ 2*step_size
    len_x = (n_blocks-1)*step_size #+ block_size-step_size #+ 2*step_size
    Blocks_x = Blocks[:, :-1]

    #len_x = (n_blocks)*step_size #+ 2*step_size
    #Blocks_x = Blocks#[:, :-1]

    r = x.clone()
    z = torch.zeros_like(x)
    r_unfold = r[:,:len_x].unfold(dimension=1, size=block_size, step=step_size)

    #print(Blocks.shape, r_unfold.shape, block_size, step_size)
    #torch.linalg.solve_ex(Blocks_x, r_unfold, check_errors=True)
    #print('runfold', r_unfold.shape)
    #y = torch.linalg.solve_triangular(Blocks_x, r_unfold.unsqueeze(-1), upper=upper).squeeze(-1)
    y = torch.cholesky_solve(r_unfold.unsqueeze(-1), Blocks_x, upper=upper).squeeze(-1)
    #y = torch.linalg.lu_solve(LU[:,:-1], pivots[:, :-1], r_unfold.unsqueeze(-1)).squeeze(-1)

    y = y.permute(0,2,1)

    #print('y', y.shape)
    d = torch.nn.functional.fold(input=y, output_size=(1,len_x), kernel_size=(1,block_size), 
                                    stride=(1,step_size))

    d = d.reshape(-1, len_x)
    #upto last block
    z[:, :len_x] = z[:, :len_x] + d

    ##last block
    ##torch.linalg.solve_ex(Blocks[:, -1], r[:, -block_size:], check_errors=True)
    #dl = torch.linalg.solve_triangular(Blocks[:, -1], r[:, -block_size:].unsqueeze(-1), 
    dl = torch.cholesky_solve(r[:, -block_size:].unsqueeze(-1), Blocks[:, -1],  
                                       upper=upper).squeeze(-1)

    #dl = torch.linalg.lu_solve(Blocks[:, -1], pivots[:, -1], r[:, -block_size:].unsqueeze(-1)).squeeze(-1)
    ##print('dl ', dl.shape)

    z[:, -block_size:] = z[:, -block_size:] + dl
    

    return z


def get_block_indices(n_row, block_size=100, stride=100):

    #block_size = 300
    #stride=block_size//2
    step_size = stride

    #begin_index = torch.arange(0, n_row, step=block_size)
    begin_index = (list(range(0, n_row,stride)))
    #end = begin_index + block_size
    end = [idx + block_size for idx in begin_index if idx + block_size <= n_row]

    begin_index = begin_index[: len(end)]
    begin_index[-1] = n_row-block_size
    end[-1] = n_row

    #n_block = len(begin_index)
    #mask = torch.zeroes((n_row, n_block))

    #for i in range(n_block):
    #    mask[begin_index[i]:end[i], i] = 1

    return begin_index, end

def get_dense_block(block, block_size=100, stride=100):
    """dense block@block.T"""

    #print('dense ', block.shape)
    #block_size = 300
    #stride=block_size//2
    step_size = stride
    #shape batch, row, col
    D = block.unsqueeze(2)
    E = block.unsqueeze(1)

    D = torch.cat([D]*block_size, dim=2)
    E = torch.cat([E]*block_size, dim=1)

    #batch, row, row, col
    DD = D*E
    DD = DD.sum(dim=3)
    DD = DD.to_dense()

    return DD

def get_blocks(M, block_size=100, stride=100):
    """get blocks given sparse M"""

    #print('o print spd', torch.diagonal(M[0].to_dense()))
    M = M.coalesce()
    indices = M.indices().clone()
    values = M.values().clone()

    row_idx = indices[1]
    #col_idx = indices[2]

    index_begin, index_end = get_block_indices(M.shape[1], block_size=block_size, stride=stride)
    blocks = []

    for begin,end in zip(index_begin, index_end):
        # get rows[begin:end] cols[begin:end] and vlaues

        mask = (begin <= row_idx)  & (row_idx < end)
        #col_mask = (begin <= col_idx)  & (col_idx < end)
        nrow = end-begin

        #mask = row_mask*col_mask

        block_indices = indices[:, mask].clone()
        #shift row indices
        #block_indices[1:] = block_indices[1:] - begin
        block_indices[1] = block_indices[1] - begin
        block_values = values[mask]

        block = torch.sparse_coo_tensor(block_indices, block_values, size=(1, nrow, M.shape[2]), 
                                        check_invariants=True)

        block = get_dense_block(block, block_size=block_size, stride=stride)
        blocks.append(block)

    blocks = torch.stack(blocks, dim=1)

    (L, info) = torch.linalg.cholesky_ex(blocks, check_errors=True)

    #(LU, pivots, info) = torch.linalg.lu_factor_ex(blocks, check_errors=False)

    #return (LU, pivots)
    return (L, info)


#def symmlq(A, rhs, x0, M=None, rtol=1e-6, maxiter=100,
#           block_size:int=100, stride:int=100
#           ):

def symmlq(A, rhs, x0, M1=None, M2=None, mlens=None,
           rtol=1e-4, maxiter=100,
           #perm=None, perminv=None,
           block_size:int=100, stride:int=100, show=False
           ):

    num_var,num_constraint = mlens
    bs = rhs.shape[0]
    n = rhs.shape[1]
    nMatvec = 0

    eps = machine_epsilon()

    istop  = torch.zeros(bs, device=rhs.device) 
    ynorm  = torch.zeros(bs, device=rhs.device)
    w = torch.zeros((bs,n), device=rhs.device)
    acond  = torch.zeros(bs, device=rhs.device)
    itn    = 0 #torch.zeros(bs, device=rhs.device)
    xnorm  = torch.zeros(bs, device=rhs.device)
    x = torch.zeros((bs,n), device=rhs.device)
    done=False 
    anorm  = torch.zeros(bs, device=rhs.device)
    rnorm  = torch.zeros(bs, device=rhs.device)
    v = torch.zeros((bs,n), device=rhs.device)

    #r1 = rhs.copy()
    r1 = rhs.clone()
    #if M is not None:
    #    #y = self.precon * r1
    #    #apply M
    #    #pass

    #    #y = apply_block_jacobi_M(M.transpose(-1,-2), r1, upper=True, 
    #    y = apply_block_jacobi_M(M, r1, upper=False, 
    #                              block_size=block_size, stride=stride)
    #else:
    #    #y = rhs.copy()
    #    y = rhs.clone()

    if M1 is not None:
        _r1 = r1[:, num_var:]
        #_r1 = _r1[:, perm]
        y2 = MR.apply_block_jacobi_M(M1,_r1 , upper=False, 
                                  block_size=block_size, stride=stride)
        #y2 = y2[:, perminv]

        y1 = r1[:, :num_var]/M2
        y = torch.cat([y1,y2], dim=1)
        #y = r1*1/M
    else:
        y = r1
        

    b1 = y[:, 0].unsqueeze(1)
    #beta1 = np.dot(r1, y)
    beta1 = (r1*y).sum(dim=-1)
    
    done = torch.where(beta1<=0, 1, 0)
    istop = torch.where(beta1<=0, 80, istop)

    assert(beta1[0]>=0)
    #if beta1 > 0:
    beta1 = torch.sqrt(beta1)
    s     = 1.0 / beta1
    v     = s.unsqueeze(1) * y

    #y = self.matvec(v) ; nMatvec += 1
    y = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    nMatvec += 1

    #if shift is not None: y -= shift * v
    #alfa = np.dot(v, y)
    alfa = (v*y).sum(dim=1) #dot
    #y -= (alfa / beta1) * r1
    y = y - (alfa / beta1).unsqueeze(1) * r1

    # Make sure  r2  will be orthogonal to the first  v.

    #z  = np.dot(v, y)
    #s  = np.dot(v, v)

    z  = (v* y).sum(dim=1)
    s  = (v* v).sum(dim=1)
    y = y - (z / s).unsqueeze(1) * v
    r2 = y.clone()

    #if M is not None: 
    #    #y = self.precon * r2
    #    #pass

    #    #y = apply_block_jacobi_M(M.transpose(-1,-2), r2, upper=True, 
    #    y = apply_block_jacobi_M(M, r2, upper=False, 
    #                              block_size=block_size, stride=stride)

    if M1 is not None:
        _r2 = r2[:, num_var:]
        #_r1 = _r1[:, perm]
        y2 = MR.apply_block_jacobi_M(M1,_r2 , upper=False, 
                                  block_size=block_size, stride=stride)
        #y2 = y2[:, perminv]

        y1 = r2[:, :num_var]/M2
        y = torch.cat([y1,y2], dim=1)
        #y = r1*1/M
    else:
        y = r2

    oldb   = beta1
    #beta   = np.dot(r2, y)
    beta   = (r2* y).sum(dim=-1)
    #if beta < 0:
    #    istop = 8
    #    done = True
    done = torch.where(beta<=0, 1, 0)
    istop = torch.where(beta<=0, 8, istop)

    #  Cause termination (later) if beta is essentially zero.

    #beta = np.sqrt(beta)
    beta = torch.sqrt(beta)
    #if beta <= eps:
    #    istop = -1
    istop = torch.where(beta<=eps, -1, istop)

    #  See if the local reorthogonalization achieved anything.

    #denom = np.sqrt(s) * np.linalg.norm(r2) + eps
    denom = torch.sqrt(s) * torch.linalg.vector_norm(r2, dim=-1) + eps
    s = z / denom
    #t = np.dot(v, r2)
    t = (v* r2).sum(dim=1)
    t = t / denom

    #  Initialize other quantities.
    cgnorm = beta1 ; rhs2   = torch.zeros_like(beta1) ; tnorm  = alfa**2 + beta**2
    gbar   = alfa  ; bstep  = torch.zeros_like(ynorm) ; ynorm2 = torch.zeros_like(ynorm)
    #dbar   = beta  ; snprod = 1 ; gmax   = np.abs(alfa) + eps
    dbar   = beta  ; snprod = torch.ones_like(beta1) ; gmax   = torch.abs(alfa) + eps
    rhs1   = beta1 ; x1cg   = 0 ; gmin   = gmax
    qrnorm = beta1
    #end if beta1 > 0:

    
    #while nMatvec < matvec_max:
    while True:
        itn    = itn  +  1
        anorm  = torch.sqrt(tnorm)
        ynorm  = torch.sqrt(ynorm2)
        epsa   = anorm * eps
        epsx   = anorm * ynorm * eps
        epsr   = anorm * ynorm * rtol
        diag   = gbar

        #if diag == 0: 
        #    diag = epsa
        diag = torch.where(diag==0, epsa, diag)

        lqnorm = torch.sqrt(rhs1**2 + rhs2**2)
        qrnorm = snprod * beta1
        cgnorm = qrnorm * beta / torch.abs(diag)

        #if lqnorm < cgnorm:
        acond1  = gmax / gmin
        #else:
        #denom2  = min(gmin, torch.abs(diag))
        denom2  = torch.minimum(gmin, torch.abs(diag))
        acond2  = gmax / denom2

        acond = torch.where(lqnorm < cgnorm, acond1, acond2)
        denom = torch.where(lqnorm < cgnorm, denom, denom2)

        zbar = rhs1 / diag
        z    = (snprod.unsqueeze(1) * zbar + bstep) / beta1.unsqueeze(1)
        x1lq = x[:,0] + b1 * bstep / beta1
        x1cg = x[:, 0] + w[:, 0] * zbar  +  b1 * z

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above
        # (Abar = const * I).

        if itn %100 == 0: 
            print(itn, cgnorm[0].item(), qrnorm[0].item())
        if itn >= maxiter:
            break

        s = 1/beta
        v = s.unsqueeze(1) * y
        #y = self.op * v ; nMatvec += 1
        y = torch.bmm(A, v.unsqueeze(2)).squeeze(2)
        nMatvec += 1
        #if shift is not None: y -= shift * v

        #y -= (beta / oldb) * r1
        y = y - (beta / oldb).unsqueeze(1) * r1
        #alfa = np.dot(v, y)
        alfa = (v* y).sum(dim=1)
        #y -= (alfa / beta) * r2
        y = y - (alfa / beta).unsqueeze(1) * r2
        r1 = r2.clone()
        r2 = y.clone()
        #if self.precon is not None: y = self.precon * r2
        #if M is not None: 
        #    #y = self.precon * r2
        #    #y = apply_block_jacobi_M(M.transpose(-1,-2), r2, upper=True, 
        #    y = apply_block_jacobi_M(M, r2, upper=False, 
        #                          block_size=block_size, stride=stride)

        if M1 is not None:
            _r2 = r2[:, num_var:]
            #_r1 = _r1[:, perm]
            y2 = MR.apply_block_jacobi_M(M1,_r2 , upper=False, 
                                    block_size=block_size, stride=stride)
            #y2 = y2[:, perminv]

            y1 = r2[:, :num_var]/M2
            y = torch.cat([y1,y2], dim=1)
            #y = r1*1/M
        else:
            y = r2
        oldb = beta
        #beta = np.dot(r2, y)
        beta = (r2* y).sum(dim=1)

        beta  = torch.sqrt(beta)
        tnorm = tnorm  +  alfa**2  +  oldb**2  +  beta**2

        # Compute the next plane rotation for Q.

        #gamma  = np.sqrt(gbar**2 + oldb**2)
        gamma  = torch.sqrt(gbar**2 + oldb**2)
        cs     = gbar / gamma
        sn     = oldb / gamma
        delta  = cs * dbar  +  sn * alfa
        gbar   = sn * dbar  -  cs * alfa
        epsln  = sn * beta
        dbar   =            -  cs * beta

        # Update  X.

        z = rhs1 / gamma
        s = z*cs
        t = z*sn
        x = x + s.unsqueeze(1)*w + t.unsqueeze(1)*v
        w = w*sn.unsqueeze(1) ; w = w- cs.unsqueeze(1)*v
        

        bstep  = snprod * cs * z  +  bstep
        snprod = snprod * sn
        gmax   = torch.maximum(gmax, gamma)
        gmin   = torch.minimum(gmin, gamma)
        ynorm2 = z**2  +  ynorm2
        rhs1   = rhs2  -  delta * z
        rhs2   =       -  epsln * z

    ####### end while

    # Move to the CG point if it seems better.
    # In this version of SYMMLQ, the convergence tests involve
    # only cgnorm, so we're unlikely to stop at an LQ point,
    # EXCEPT if the iteration limit interferes.

    #if cgnorm < lqnorm:
    cond = cgnorm < lqnorm
    zbar = torch.where(cond, rhs1 / diag, zbar)
    bstep = torch.where(cond, snprod * zbar + bstep, bstep)
    ynorm  = torch.where(cond, torch.sqrt(ynorm2 + zbar**2), ynorm)
    x    = torch.where(cond, x+ zbar.unsqueeze(1) * w, x)

    # Add the step along b.

    bstep  = bstep / beta1
    #if self.precon is not None:
    #if M is not None:
    #    #y = self.precon * rhs
    #    #y = apply_block_jacobi_M(M.transpose(-1,-2), r2, upper=True, 
    #    y = apply_block_jacobi_M(M, r2, upper=False, 
    #                              block_size=block_size, stride=stride)


    #else:
    #    y = rhs.clone()

    r1 = rhs.clone()
    if M1 is not None:
        _r1 = r1[:, num_var:]
        #_r1 = _r1[:, perm]
        y2 = MR.apply_block_jacobi_M(M1,_r1 , upper=False, 
                                  block_size=block_size, stride=stride)
        #y2 = y2[:, perminv]

        y1 = r1[:, :num_var]/M2
        y = torch.cat([y1,y2], dim=1)
        #y = r1*1/M
    else:
        y = r1

    #x += bstep * y
    x = x + bstep.unsqueeze(1) * y

    # Compute the final residual,  r1 = b - (A - shift*I)*x.

    #y = self.op * x ; nMatvec += 1
    y = torch.bmm(A, x.unsqueeze(2)).squeeze(2)
    nMatvec += 1
    #if shift is not None: y -= shift * x
    r1 = rhs - y
    #rnorm = np.linalg.norm(r1)
    #xnorm = np.linalg.norm(x)

    rnorm = torch.linalg.vector_norm(r1, dim=1)
    xnorm = torch.linalg.vector_norm(x, dim=1)

    return x, rnorm,itn