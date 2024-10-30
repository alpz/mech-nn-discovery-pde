import torch
import numpy as np
from typing import Tuple
import time

def _get_tensor_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).eps,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")

def _get_tensor_max(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).max,
    eps32: float = torch.finfo(torch.float32).max,
    eps64: float = torch.finfo(torch.float64).max,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")

#@torch.jit.script
#def machine_epsilon():
#    """Return the machine epsilon in double precision."""
#    #return torch.finfo(torch.double).eps
#    return 1e

@torch.jit.script
def apply_block_jacobi_M(Blocks:torch.Tensor, x:torch.Tensor, upper:bool=False, block_size:int=100, stride:int=100):


    #LU, pivots = Blocks
    #Blocks = _Blocks[0]
    #pivots = _Blocks[1]
    #Blocks = Blocks.transpose(-1,-2)
    #upper=True
    #block_size = 300
    #stride=block_size//2
    step_size = stride

    n_blocks = Blocks.shape[1]
    block_size = Blocks.shape[2]

    #len_x = (n_blocks-2)*step_size + 2*step_size
    len_x = (n_blocks-1)*step_size + block_size-step_size #+ 2*step_size
    #len_x = (n_blocks-1)*step_size #+ block_size-step_size #+ 2*step_size
    #Blocks_x = Blocks[:, :-1]

    #len_x = (n_blocks)*step_size #+ 2*step_size
    #Blocks_x = Blocks#[:, :-1]

    r = x.clone()
    z = torch.zeros_like(x)
    r_unfold = r[:,:len_x].unfold(dimension=1, size=block_size, step=step_size)

    r_agg = torch.cat([r_unfold, r[:, -block_size:].unsqueeze(1)], dim=1)
    #print(r_agg.shape)
    #print(Blocks.shape, r_unfold.shape, block_size, step_size)
    #torch.linalg.solve_ex(Blocks_x, r_unfold, check_errors=True)
    #print('runfold', r_unfold.shape)
    #y = torch.linalg.solve_triangular(Blocks_x, r_unfold.unsqueeze(-1), upper=upper).squeeze(-1)
    #y = torch.cholesky_solve(r_unfold.unsqueeze(-1), Blocks_x, upper=upper).squeeze(-1)
    _y = torch.cholesky_solve(r_agg.unsqueeze(-1), Blocks, upper=upper).squeeze(-1)
    #y = torch.linalg.lu_solve(LU[:,:-1], pivots[:, :-1], r_unfold.unsqueeze(-1)).squeeze(-1)
    #_y = torch.linalg.ldl_solve(Blocks, pivots, r_agg.unsqueeze(-1)).squeeze(-1)

    y = _y[:,:-1]
    y = y.permute(0,2,1)

    #print('y', y.shape)
    d = torch.nn.functional.fold(input=y, output_size=(1,len_x), kernel_size=(1,block_size), 
                                    stride=(1,step_size))

    d = d.reshape(-1, len_x)
    #weight for overlap
    #d[:, step_size:-step_size] = 1/2*d[:, step_size:-step_size]
    #d[:, step_size:] = 1/2*d[:, step_size:]
    #upto last block
    z[:, :len_x] = z[:, :len_x] + d

    ##last block
    ##torch.linalg.solve_ex(Blocks[:, -1], r[:, -block_size:], check_errors=True)
    #dl = torch.linalg.solve_triangular(Blocks[:, -1], r[:, -block_size:].unsqueeze(-1), 
    #dl = torch.cholesky_solve(r[:, -block_size:].unsqueeze(-1), Blocks[:, -1],  
    #                                   upper=upper).squeeze(-1)

    dl = _y[:, -1]

    #dl = torch.linalg.lu_solve(Blocks[:, -1], pivots[:, -1], r[:, -block_size:].unsqueeze(-1)).squeeze(-1)
    ##print('dl ', dl.shape)
    rest = z.shape[1]-len_x
    z[:, -block_size:] = z[:, -block_size:] + dl
    #z[:, -block_size:-rest] = z[:, -block_size:-rest]/2
    #weight overlapping
    #z[:, -block_size:-step_size] = z[:, -block_size:-step_size]/2
    

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

        block = torch.sparse_coo_tensor(block_indices, block_values, size=(M.shape[0], nrow, M.shape[2]), 
                                        check_invariants=False)

        block = get_dense_block(block, block_size=block_size, stride=stride)
        blocks.append(block)

    blocks = torch.stack(blocks, dim=1)

    (L, info) = torch.linalg.cholesky_ex(blocks, check_errors=False)
    #(L, pivots, info) = torch.linalg.ldl_factor_ex(blocks, check_errors=False)
    #(LU, pivots, info) = torch.linalg.lu_factor_ex(blocks, check_errors=False)

    return (L, info)


#def minres(A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor, M=None, rtol=1e-6, maxiter=100,
#           block_size:int=100, stride:int=100, show=False
#           ):

#@torch.jit.script
def minres(Aperm:torch.Tensor, A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor, M1:torch.Tensor=None, M2:torch.Tensor=None, 
           mlens:Tuple[int, int]=(100,100),
           rtol:float=1e-4, maxiter:int=100,
           #perm=None, perminv=None,
           block_size:int=100, stride:int=100
           ):
#def minres(A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor,rtol:float=1e-6, maxiter:int=100,
#           _max:float=0):
#def minres(A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor, M:Tuple[torch.Tensor,torch.Tensor]=None, rtol:float=1e-6, maxiter:int=100,
#           block_size:int=100, stride:int=100):

    At = Aperm.transpose(1,2)

    num_var,num_constraint = mlens
    #A = A[0]
    #b = b[0]
    #x0 = x0[0]

    bs = b.shape[0]
    n = b.shape[1]

    istop = 0
    itn = 0
    Anorm = torch.zeros(bs, device=b.device)
    Acond =torch.zeros(bs, device=b.device) 
    rnorm = torch.zeros(bs, device=b.device)
    ynorm = torch.zeros(bs, device=b.device)

    done = torch.zeros(bs,dtype=torch.int32, device=b.device)


    eps = torch.tensor(_get_tensor_eps(b), device=b.device)

    x = torch.zeros_like(x0)
    
    if x0 is None:
        r1 = b.clone()
    else:
        #r1 = b - A@x
        print(A.shape, x.shape)

        #mm = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        r1 = b - torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        #r1 = b - torch.mm(A1, x[0].unsqueeze(-1)).unsqueeze(0).squeeze(-1)
    
    #if M is not None:

    #    #y = r1/M
    #    y = apply_block_jacobi_M(M, r1, upper=False, 
    #                              block_size=block_size, stride=stride)
    #else:
    #y = r1
    #y = psolve(r1)
    #precond
    if M1 is not None:

        _r1 = r1[:, num_var:]
        #_r1 = _r1[:, perminv]
        y2 = apply_block_jacobi_M(M1,_r1 , upper=False, 
                                  block_size=block_size, stride=stride)
        #y2 = y2[:, perm]

        y1 = r1[:, :num_var]/M2
        y = torch.cat([y1,y2], dim=1)
        #y = r1*1/M
    else:
        y = r1

    #beta1 = inner(r1, y)
    beta1 = (r1*y).sum(dim=-1)

    if (beta1<=0).any():
        raise ValueError('indefinite preconditioner or 0 beta')

    #bnorm = norm(b)
    bnorm = torch.linalg.vector_norm(b, dim=-1)
    #if bnorm == 0:
    #    x = b
    #    return #(postprocess(x), 0)
    #done = torch.where(bnorm==0, 1, 0)
    if (bnorm ==0).any():
        raise ValueError('bnorm 0')


    beta1 = torch.sqrt(beta1)

     # Initialize other quantities
    oldb = torch.zeros(bs, device=b.device)
    beta = beta1
    dbar = torch.zeros(bs, device=b.device)
    epsln = torch.zeros(bs, device=b.device)
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = torch.zeros_like(rhs1)
    tnorm2 = torch.zeros(bs, device=b.device)
    #gmax = torch.tensor(0, device=b.device)
    #gmin = torch.tensor(np.finfo(np.float64).max, device=b.device)
    #gmin = torch.tensor((_get_tensor_max(b)), device=b.device)
    #gmin = torch.tensor(_max, device=b.device)
    cs = -1*torch.ones(bs, device=b.device)
    sn = torch.zeros(bs, device=b.device)
    w = torch.zeros_like(b)
    w2 = torch.zeros_like(b)
    r2 = r1


    while itn < maxiter:
        itn += 1

        s = 1.0/beta
        v = s.unsqueeze(1)*y

        #y = matvec(v)

        y = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        #y = y - shift * v

        if itn >= 2:
            y = y - (beta/oldb).unsqueeze(1)*r1

        #alfa = inner(v,y)
        alfa = (v*y).sum(dim=-1)
        y = y - (alfa/beta).unsqueeze(1)*r2
        r1 = r2
        r2 = y
        #y = psolve(r2)
        #if M is not None:
        #    #pass
        #    y = apply_block_jacobi_M(M, r2, upper=False, 
        #                          block_size=block_size, stride=stride)
        #    #y = r2/M
        #else:
        #y = r2

        if M1 is not None:
            #pass
            #y = r2*1/M
            #y = apply_block_jacobi_M(M, r2, upper=False, 
            #                      block_size=block_size, stride=stride)
            #y = r2/M
            _r2 = r2[:, num_var:]
            #_r2 = _r2[:, perminv]
            y2 = apply_block_jacobi_M(M1,_r2 , upper=False, 
                                    block_size=block_size, stride=stride)
            #y2 = y2[:, perm]
            y1 = r2[:, :num_var]/M2
            y = torch.cat([y1,y2], dim=1)
        else:
            y = r2

        oldb = beta
        #beta = inner(r2,y)
        beta = (r2*y).sum(dim=-1)
        #if beta < 0:
        if (beta < 0).any():
            raise ValueError('non-symmetric matrix')
        beta = torch.sqrt(beta)
        tnorm2 = tnorm2+ alfa**2 + oldb**2 + beta**2

        #if itn == 1:
        #    if beta/beta1 <= 10*eps:
        #        istop = -1  # Terminate later

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
        epsln = sn * beta     # epsln2 = 0         epslnk+1
        dbar = - cs * beta   # dbar 2 = beta2     dbar k+1
        #root = norm([gbar, dbar])
        #root = torch.linalg.vector_norm(torch.tensor([gbar, dbar], device=b.device), dim=-1)
        root = torch.linalg.vector_norm(torch.stack([gbar, dbar], dim=1), dim=-1)
        Arnorm = phibar * root

        # Compute the next plane rotation Qk

        #gamma = norm([gbar, beta])       # gammak
        gamma = torch.linalg.vector_norm(torch.stack([gbar, beta], dim=1), dim=-1)
        gamma = torch.maximum(gamma, eps)
        #gamma = torch.tensor(gamma, device=b.device)
        cs = gbar / gamma             # ck
        sn = beta / gamma             # sk
        phi = cs * phibar              # phik
        phibar = sn * phibar              # phibark+1

        # Update  x.

        denom = 1.0/gamma
        w1 = w2
        w2 = w
        w = (v - oldeps.unsqueeze(1)*w1 - delta.unsqueeze(1)*w2) * (denom.unsqueeze(1))
        x = x + phi.unsqueeze(1)*w

        # Go round again.

        #gmax = torch.maximum(torch.tensor(gmax, device=b.device), gamma)
        #gmin = torch.minimum(torch.tensor(gmin, device=b.device), gamma)

        #gmax = torch.maximum(gmax, gamma)
        #gmin = torch.minimum(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta*z
        rhs2 = - epsln*z

        # Estimate various norms and test for convergence.

        Anorm = torch.sqrt(tnorm2)
        ynorm = torch.linalg.vector_norm(x, dim=-1)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * rtol
        diag = gbar

        #if diag == 0:
        #    diag = epsa
        diag = torch.where(diag==0, epsa, diag)

        qrnorm = phibar
        rnorm = qrnorm
        #if ynorm == 0 or Anorm == 0:
        #    test1 = torch.inf
        #else:
        test1 = rnorm / (Anorm*ynorm)    # ||r||  / (||A|| ||x||)
        #if Anorm == 0:
        #    test2 = torch.inf
        #else:
        test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q @ H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        #Acond = gmax/gmin

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        #if istop == 0:
        #    t1 = 1 + test1      # These tests work if rtol < eps
        #    t2 = 1 + test2
        #    if t2 <= 1:
        #        istop = 2
        #    if t1 <= 1:
        #        istop = 1

        #    if itn >= maxiter:
        #        istop = 6
        #    if Acond >= 0.1/eps:
        #        istop = 4
        #    if epsx >= beta1:
        #        istop = 3
        #    # if rnorm <= epsx   : istop = 2
        #    # if rnorm <= epsr   : istop = 1
        #    if test2 <= rtol:
        #        istop = 2
        #    if test1 <= rtol:
        #        istop = 1

        # See if it is time to print something.

        #prnt = False
        #if n <= 40:
        #    prnt = True
        #if itn <= 10:
        #    prnt = True
        #if itn >= maxiter-10:
        #    prnt = True
        #if itn % 10 == 0:
        #    prnt = True
        #if qrnorm <= 10*epsx:
        #    prnt = True
        #if qrnorm <= 10*epsr:
        #    prnt = True
        #if Acond <= 1e-2/eps:
        #    prnt = True
        #if istop != 0:
        #    prnt = True

        #if show and prnt:
        #    str1 = f'{itn:6g} {x[0]:12.5e} {test1:10.3e}'
        #    str2 = f' {test2:10.3e}'
        #    str3 = f' {Anorm:8.1e} {Acond:8.1e} {gbar/Anorm:8.1e}'

        #    print(str1 + str2 + str3)

        #    if itn % 10 == 0:
        #        print()


        #if istop != 0:
        #    break  # TODO check this
        if itn >= maxiter:
            break

        if itn % 100 == 0:
            print(itn, rnorm[0].item(), qrnorm[0].item(), Anorm[0].item(), test2.max().item(), test1.max().item())

        ##if itn>=5 and ((test2 <= rtol).all() and (test1 <= rtol).all()):
        #if itn>=5 and ((test2 <= rtol).all()):
        #if itn>=5 and ((test2 <= 1e-4).all()):
        if itn>=5 and ((test1 <= 1e-9).all()):
            print('break ', test2, test1)
            break

    #last='exit'
    #if show:
    #    print()
    #    print(last + f' istop   =  {istop:3g}               itn   ={itn:5g}')
    #    print(last + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
    #    print(last + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
    #    print(last + f' Arnorm  =  {Arnorm:12.4e}')
    #    #print(last + msg[istop+1])

    #if istop == 6:
    #    info = maxiter
    #else:
    #    info = 0
    
    
    #x = x.unsqueeze(0)
    return x, rnorm, itn