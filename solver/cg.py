#Batched sparse conjugate gradient adapted from cupy cg

import torch
import numpy as np
import ipdb

def block_mv(A, x):
    """shape x: (b, d), A sparse block"""
    b = x.shape[0]
    x = x.reshape(-1)

    y = torch.mv(A, x)
    y = y.reshape(b, -1)
    return y

def batch_mat_vec(As, x):
    A = As[0].transpose(1,2)
    #Pinv = As[1]
    Pinv_diag = 1/As[1]
    #At = As[2]
    At = A.transpose(1,2)

    x = x.unsqueeze(2)
    x = torch.bmm(At, x)
    Atp = x.squeeze(2)

    x = x.squeeze(2)
    x = Pinv_diag*x
    GAtp = x

    x = x.unsqueeze(2)
    x = torch.bmm(A, x)
    x = x.squeeze(2)

    pAAtp = (Atp*GAtp).sum(dim=1)


    return x, pAAtp

def get_M(As):
    A = As[0]

    Pinv = As[1].unsqueeze(1)
    #print(A.shape, Pinv.shape)
    #diag = (A*Pinv*A).sum(dim=2)
    diag = (A*A).sum(dim=2)

    return (diag).to_dense()

@torch.no_grad()
def cg_matvec(As, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None, atol=None,
              MG=None, MG_args=None, back=False):
    #A, M, x, b = _make_system(A, M, x0, b)

    #matvec = A.matvec
    #psolve = M.matvec

    #n = A.shape[0]
    #n = A.shape[1]
    #if maxiter is None:
    #    maxiter = n * 10
    #if n == 0:
    #    return cupy.empty_like(b), 0
    #TODO fix this. check all norms and use masks 
    #b_norm = torch.linalg.norm(b[0])
    #b_norm = torch.linalg.vector_norm(b[0],dim=-1)
    b_norm = torch.linalg.vector_norm(b,dim=-1)
    #if b_norm == 0:
    cont_mask = (b_norm>1e-9)
    #if b_norm < 1e-10:
    if not cont_mask.any():
        print('zero return')
        return b, 0

    cont_mask = cont_mask.float()
    #if atol is None:
    #    #atol = tol * float(b_norm)
    #    atol = tol * b_norm
    #else:
    #    atol = max(float(atol), tol * float(b_norm))
        #atol = float(atol)
    atol = tol


    #r = b - matvec(x)
    #b = b.unsqueeze(-1)
    x = x0 if x0 is not None else torch.zeros_like(b)
    #x = torch.rand_like(b)
    #r = b - torch.bmm(A,x)#.reshape(b.shape)
    #r = b - block_mv(A,x)#.reshape(b.shape)
    r = b - batch_mat_vec(As,x)[0]#.reshape(b.shape)
    iters = 0
    rho = 0
    resid = 0
    #ipdb.set_trace()
    while iters < maxiter:
        #z = psolve(r)
        z = r #psolve(r)
        if MG is None:
            z = r #psolve(r)
        else:
            z = apply_MG(MG, MG_args, r, back=back)
        #z = M*r

        rho1 = rho
        #rho = cublas.dotc(r, z)
        rho = (r*z).sum(dim=1)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

            beta = beta.unsqueeze(1)
            p = z + beta * p
        #q = matvec(p)
        #q = torch.bmm(A, p)
        #q = bmm_fix(A, p)
        #q = block_mv(A, p)
        q, ptAAtp = batch_mat_vec(As, p)
        #alpha = rho / cublas.dotc(p, q)
        #alpha = rho / (p*q).sum(dim=1)
        alpha = rho / ptAAtp

        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha*cont_mask

        alpha = alpha.unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        #if callback is not None:
        #    callback(x)
        #resid = cublas.nrm2(r)
        #resid = cublas.nrm2(r)
        resid = torch.linalg.vector_norm(r, dim=1)
        res_mask = (resid > atol).float()
        cont_mask = cont_mask*res_mask
        if resid.max() <= atol:
            print('break ', resid.max(), atol)
            break

    info = 0
    if iters == maxiter and not (resid.max() <= atol):
        info = iters

    return x, (info, iters, resid)

def apply_M(M, x):
    #apply preconditoners on cpu. M is a list
    device = x.device
    x = x.cpu().numpy()
    mx_list = []
    for idx in range(len(M)):
        #mx = M.solve(x[0].cpu().numpy())
        mx = M[idx].solve(x[idx])
        mx = torch.tensor(mx).unsqueeze(0)
        mx_list.append(mx)
    mx = torch.cat(mx_list, dim=0)
    mx = mx.to(device)
    return mx

def apply_MG_kkt(MG, MG_args, bin, back=False):

    m = MG_args[-1][0]
    n = MG_args[-1][1]
    A = MG_args[0][0][0]
    G = MG_args[0][0][1]

    b = bin[:, m:]
    x = MG.v_cycle_jacobi_start(MG_args[0], [b], MG_args[1],MG_args[2], n_step=1, back=back)
    #x = MG.full_multigrid_jacobi_start(MG_args[0], [b], MG_args[1],MG_args[2])

    y = torch.bmm(A, x.unsqueeze(2)).squeeze(2)
    y= -(1/G)*y

    z = torch.cat([y,x], dim=1)

    return z


def apply_MG(MG, MG_args, b, back=False):

    #x = MG.v_cycle_jacobi_start(MG_args[0], [b], MG_args[1],MG_args[2], n_step=1, back=back)
    bs = b.shape[0]
    b = b.reshape(-1)
    x = MG.v_cycle_gs_start(MG_args[0], b, MG_args[1],MG_args[2], MG_args[3], n_step=1, back=back)
    x = x.reshape(bs, -1)
    #x = MG.full_multigrid_jacobi_start(MG_args[0], [b], MG_args[1],MG_args[2])
    return x
    
@torch.no_grad()
def gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, MG=None, MG_args=None,
          callback=None, atol=1e-5, callback_type=None, back=False):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an argument.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #x0 = torch.zeros_like(b)
    A, M, x, b = A, M, x0, b
    #matvec = A.matvec
    #psolve = M.matvec

    bs = A.shape[0]
    n = A.shape[1]
    if n == 0:
        #return cupy.empty_like(b), 0
        return torch.empty_like(b), 0
    #b_norm = cupy.linalg.norm(b)
    b_norm = torch.linalg.norm(b, dim=1)
    if (b_norm==0).all():
        return b, 0
    #if atol is None:
    #    atol = tol * float(b_norm)
    #else:
    #    atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = 'pr_norm'
    if callback_type not in ('x', 'pr_norm'):
        raise ValueError('Unknown callback_type: {}'.format(callback_type))
    if callback is None:
        callback_type = None

    #V = torch.empty((n, restart), dtype=A.dtype, order='F')
    V = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    Z = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    #H = torch.zeros((restart+1, restart), dtype=A.dtype, order='F')
    H = torch.zeros((bs, restart+1, restart), dtype=A.dtype, device=x.device)
    #e = np.zeros((restart+1,), dtype=A.dtype)
    e = torch.zeros((bs, restart+1,), dtype=A.dtype, device=x.device)

    def compute_hu(VV, u, j):
        S = VV[:, :, :j+1]
        #h = torch.mm(S.T, u.unsqueeze(1)).squeeze(1)
        h = torch.bmm(S.transpose(1,2), u.unsqueeze(2)).squeeze(2)
        u = u - torch.bmm(S, h.unsqueeze(2)).squeeze(2)
        return h, u

    #compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        #mx = psolve(x)
        mx = x
        #if M is not None:
        if MG is not None:
            #mx = M.solve(x[0].cpu().numpy())
            #mx = torch.tensor(mx).unsqueeze(0)
            #mx = mx.to(x.device)
            mx = apply_MG(MG, MG_args, mx, back=back)
        #r = b - matvec(mx)
        #r = b - torch.mm(A, mx.unsqueeze(1)).squeeze(1)
        #print(A, mx.shape)
        r = b - torch.bmm(A, mx.unsqueeze(2)).squeeze(2)
        #r_norm = cublas.nrm2(r)
        r_norm = torch.linalg.norm(r, dim=1)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        #if r_norm <= atol or iters >= maxiter:
        if (r_norm <= atol).all() or iters >= maxiter:
            break
        v = r / r_norm.unsqueeze(1)
        V[:, :, 0] = v
        e[:, 0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            #z = psolve(v)
            z = v #psolve(v)
            #if M is not None:
            if MG is not None:
                #z = M.solve(z[0].cpu().numpy())
                #z = torch.tensor(z).unsqueeze(0)
                #z = z.to(x.device)
                #z = apply_M(M, z)
                z = apply_MG(MG, MG_args, z, back=back)
                #Z[:, :, j] = z.clone()
            #u = matvec(z)
            #u = torch.mm(A, z.unsqueeze(1)).squeeze(1)
            u = torch.bmm(A, z.unsqueeze(2)).squeeze(2)
            #H[:j+1, j], u = compute_hu(V, u, j)
            H[:, :j+1, j], u = compute_hu(V, u, j)
            #cublas.nrm2(u, out=H[j+1, j])
            torch.linalg.norm(u, out=H[:, j+1, j], dim=-1)
            if j+1 < restart:
                v = u / H[:, j+1, j].unsqueeze(1)
                V[:, :, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        #ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        ret = torch.linalg.lstsq(H, e.unsqueeze(2))
        #y = cupy.array(ret[0])
        y = (ret[0].squeeze(2))
        #x += V @ y
        #x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
        x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
        #x = mx + torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        iters += restart

    #info = 0
    #if iters == maxiter and not (r_norm <= atol):
    #    info = iters
    return mx, (iters, r_norm)

@torch.no_grad()
def lgmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, MG=None, MG_args=None,
          callback=None, atol=1e-5, callback_type=None, back=False):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an argument.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #x0 = torch.zeros_like(b)
    A, M, x, b = A, M, x0, b
    #matvec = A.matvec
    #psolve = M.matvec

    bs = A.shape[0]
    n = A.shape[1]
    if n == 0:
        #return cupy.empty_like(b), 0
        return torch.empty_like(b), 0
    #b_norm = cupy.linalg.norm(b)
    b_norm = torch.linalg.norm(b, dim=1)
    if (b_norm==0).all():
        return b, 0
    #if atol is None:
    #    atol = tol * float(b_norm)
    #else:
    #    atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = 'pr_norm'
    if callback_type not in ('x', 'pr_norm'):
        raise ValueError('Unknown callback_type: {}'.format(callback_type))
    if callback is None:
        callback_type = None

    #V = torch.empty((n, restart), dtype=A.dtype, order='F')
    V = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    Z = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    #H = torch.zeros((restart+1, restart), dtype=A.dtype, order='F')
    H = torch.zeros((bs, restart+1, restart), dtype=A.dtype, device=x.device)
    #e = np.zeros((restart+1,), dtype=A.dtype)
    e = torch.zeros((bs, restart+1,), dtype=A.dtype, device=x.device)

    def compute_hu(VV, u, j):
        S = VV[:, :, :j+1]
        #h = torch.mm(S.T, u.unsqueeze(1)).squeeze(1)
        h = torch.bmm(S.transpose(1,2), u.unsqueeze(2)).squeeze(2)
        u = u - torch.bmm(S, h.unsqueeze(2)).squeeze(2)
        return h, u

    #compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        #mx = psolve(x)
        mx = x
        #if M is not None:
        #if MG is not None:
        #    #mx = M.solve(x[0].cpu().numpy())
        #    #mx = torch.tensor(mx).unsqueeze(0)
        #    #mx = mx.to(x.device)
        #    mx = apply_MG(MG, MG_args, mx, back=back)
        #r = b - matvec(mx)
        #r = b - torch.mm(A, mx.unsqueeze(1)).squeeze(1)
        #print(A, mx.shape)
        r = b - torch.bmm(A, mx.unsqueeze(2)).squeeze(2)
        if MG is not None:
            #z = M.solve(z[0].cpu().numpy())
            #z = torch.tensor(z).unsqueeze(0)
            #z = z.to(x.device)
            #z = apply_M(M, z)
            r = apply_MG(MG, MG_args, r, back=back)
            #Z[:, :, j] = z.clone()
        #r_norm = cublas.nrm2(r)
        r_norm = torch.linalg.norm(r, dim=1)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        #if r_norm <= atol or iters >= maxiter:
        if (r_norm <= atol).all() or iters >= maxiter:
            break
        v = r / r_norm.unsqueeze(1)
        V[:, :, 0] = v
        e[:, 0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            #z = psolve(v)
            z = v #psolve(v)
            u = torch.bmm(A, z.unsqueeze(2)).squeeze(2)
            #if M is not None:
            if MG is not None:
                #z = M.solve(z[0].cpu().numpy())
                #z = torch.tensor(z).unsqueeze(0)
                #z = z.to(x.device)
                #z = apply_M(M, z)
                u = apply_MG(MG, MG_args, u, back=back)
                #Z[:, :, j] = z.clone()
            #u = matvec(z)
            #u = torch.mm(A, z.unsqueeze(1)).squeeze(1)
            #H[:j+1, j], u = compute_hu(V, u, j)
            H[:, :j+1, j], u = compute_hu(V, u, j)
            #cublas.nrm2(u, out=H[j+1, j])
            torch.linalg.norm(u, out=H[:, j+1, j], dim=-1)
            if j+1 < restart:
                v = u / H[:, j+1, j].unsqueeze(1)
                V[:, :, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        #ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        ret = torch.linalg.lstsq(H, e.unsqueeze(2))
        #y = cupy.array(ret[0])
        y = (ret[0].squeeze(2))
        #x += V @ y
        x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
        #_v =  torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        #if MG is not None:
        #    _v = apply_MG(MG, MG_args, _v, back=back)
        #x = x + _v
        #x = mx + torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        iters += restart

    #info = 0
    #if iters == maxiter and not (r_norm <= atol):
    #    info = iters
    return mx, (iters, r_norm)


@torch.no_grad()
def fgmres_matvec(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, MG=None, MG_args=None,
          callback=None, atol=1e-5, callback_type=None, back=False):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an argument.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #x0 = torch.zeros_like(b)
    A, M, x, b = A, M, x0, b
    At = A.transpose(1,2)
    #matvec = A.matvec
    #psolve = M.matvec

    bs = A.shape[0]
    n = A.shape[2]
    if n == 0:
        #return cupy.empty_like(b), 0
        return torch.empty_like(b), 0
    #b_norm = cupy.linalg.norm(b)
    b_norm = torch.linalg.norm(b, dim=1)
    if (b_norm==0).all():
        return b, 0
    #if atol is None:
    #    atol = tol * float(b_norm)
    #else:
    #    atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = 'pr_norm'
    if callback_type not in ('x', 'pr_norm'):
        raise ValueError('Unknown callback_type: {}'.format(callback_type))
    if callback is None:
        callback_type = None

    #V = torch.empty((n, restart), dtype=A.dtype, order='F')
    V = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    Z = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    #H = torch.zeros((restart+1, restart), dtype=A.dtype, order='F')
    H = torch.zeros((bs, restart+1, restart), dtype=A.dtype, device=x.device)
    #e = np.zeros((restart+1,), dtype=A.dtype)
    e = torch.zeros((bs, restart+1,), dtype=A.dtype, device=x.device)

    def compute_hu(VV, u, j):
        S = VV[:, :, :j+1]
        #h = torch.mm(S.T, u.unsqueeze(1)).squeeze(1)
        h = torch.bmm(S.transpose(1,2), u.unsqueeze(2)).squeeze(2)
        u = u - torch.bmm(S, h.unsqueeze(2)).squeeze(2)
        return h, u

    #compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        #mx = psolve(x)
        mx = x
        #if M is not None:
        #if MG is not None:
        #    #mx = M.solve(x[0].cpu().numpy())
        #    #mx = torch.tensor(mx).unsqueeze(0)
        #    #mx = mx.to(x.device)
        #    mx = apply_MG(MG, MG_args, mx, back=back)
        #r = b - matvec(mx)
        #r = b - torch.mm(A, mx.unsqueeze(1)).squeeze(1)
        #print(A, mx.shape)
        r = b - torch.bmm(At, torch.bmm(A, mx.unsqueeze(2))).squeeze(2)
        #r_norm = cublas.nrm2(r)
        r_norm = torch.linalg.norm(r, dim=1)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        #if r_norm <= atol or iters >= maxiter:
        if (r_norm <= atol).all() or iters >= maxiter:
            break
        v = r / r_norm.unsqueeze(1)
        V[:, :, 0] = v
        e[:, 0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            #z = psolve(v)
            z = v #psolve(v)
            #if M is not None:
            if MG is not None:
                #z = M.solve(z[0].cpu().numpy())
                #z = torch.tensor(z).unsqueeze(0)
                #z = z.to(x.device)
                #z = apply_M(M, z)
                z = apply_MG(MG, MG_args, z, back=back)
                Z[:, :, j] = z#.clone()
            #u = matvec(z)
            #u = torch.mm(A, z.unsqueeze(1)).squeeze(1)
            u = torch.bmm(At,torch.bmm(A, z.unsqueeze(2))).squeeze(2)
            #H[:j+1, j], u = compute_hu(V, u, j)
            H[:, :j+1, j], u = compute_hu(V, u, j)
            #cublas.nrm2(u, out=H[j+1, j])
            torch.linalg.norm(u, out=H[:, j+1, j], dim=-1)
            if j+1 < restart:
                v = u / H[:, j+1, j].unsqueeze(1)
                V[:, :, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        #ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        ret = torch.linalg.lstsq(H, e.unsqueeze(2))
        #y = cupy.array(ret[0])
        y = (ret[0].squeeze(2))
        #x += V @ y
        #x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
        _v =  torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        #if MG is not None:
        #    _v = apply_MG(MG, MG_args, _v, back=back)
        x = x + _v
        #x = mx + torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        iters += restart

    #info = 0
    #if iters == maxiter and not (r_norm <= atol):
    #    info = iters
    return mx, (iters, r_norm)

@torch.no_grad()
def fgmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, MG=None, MG_args=None,
          callback=None, atol=1e-5, callback_type=None, back=False):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an argument.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #x0 = torch.zeros_like(b)
    A, M, x, b = A, M, x0, b
    #matvec = A.matvec
    #psolve = M.matvec

    bs = A.shape[0]
    n = A.shape[1]
    if n == 0:
        #return cupy.empty_like(b), 0
        return torch.empty_like(b), 0
    #b_norm = cupy.linalg.norm(b)
    b_norm = torch.linalg.norm(b, dim=1)
    if (b_norm==0).all():
        return b, 0
    #if atol is None:
    #    atol = tol * float(b_norm)
    #else:
    #    atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = 'pr_norm'
    if callback_type not in ('x', 'pr_norm'):
        raise ValueError('Unknown callback_type: {}'.format(callback_type))
    if callback is None:
        callback_type = None

    #V = torch.empty((n, restart), dtype=A.dtype, order='F')
    V = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    Z = torch.empty((bs, n, restart), dtype=A.dtype, device=x.device)
    #H = torch.zeros((restart+1, restart), dtype=A.dtype, order='F')
    H = torch.zeros((bs, restart+1, restart), dtype=A.dtype, device=x.device)
    #e = np.zeros((restart+1,), dtype=A.dtype)
    e = torch.zeros((bs, restart+1,), dtype=A.dtype, device=x.device)

    def compute_hu(VV, u, j):
        S = VV[:, :, :j+1]
        #h = torch.mm(S.T, u.unsqueeze(1)).squeeze(1)
        h = torch.bmm(S.transpose(1,2), u.unsqueeze(2)).squeeze(2)
        u = u - torch.bmm(S, h.unsqueeze(2)).squeeze(2)
        return h, u

    #compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        #mx = psolve(x)
        mx = x
        #if M is not None:
        #if MG is not None:
        #    #mx = M.solve(x[0].cpu().numpy())
        #    #mx = torch.tensor(mx).unsqueeze(0)
        #    #mx = mx.to(x.device)
        #    mx = apply_MG(MG, MG_args, mx, back=back)
        #r = b - matvec(mx)
        #r = b - torch.mm(A, mx.unsqueeze(1)).squeeze(1)
        #print(A, mx.shape)
        r = b - torch.bmm(A, mx.unsqueeze(2)).squeeze(2)
        #r_norm = cublas.nrm2(r)
        r_norm = torch.linalg.norm(r, dim=1)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        #if r_norm <= atol or iters >= maxiter:
        if (r_norm <= atol).all() or iters >= maxiter:
            break
        v = r / r_norm.unsqueeze(1)
        V[:, :, 0] = v
        e[:, 0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            #z = psolve(v)
            z = v #psolve(v)
            #if M is not None:
            if MG is not None:
                #z = M.solve(z[0].cpu().numpy())
                #z = torch.tensor(z).unsqueeze(0)
                #z = z.to(x.device)
                #z = apply_M(M, z)
                z = apply_MG(MG, MG_args, z, back=back)
                Z[:, :, j] = z.clone()
            #u = matvec(z)
            #u = torch.mm(A, z.unsqueeze(1)).squeeze(1)
            u = torch.bmm(A, z.unsqueeze(2)).squeeze(2)
            #H[:j+1, j], u = compute_hu(V, u, j)
            H[:, :j+1, j], u = compute_hu(V, u, j)
            #cublas.nrm2(u, out=H[j+1, j])
            torch.linalg.norm(u, out=H[:, j+1, j], dim=-1)
            if j+1 < restart:
                v = u / H[:, j+1, j].unsqueeze(1)
                V[:, :, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        #ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        ret = torch.linalg.lstsq(H, e.unsqueeze(2))
        #y = cupy.array(ret[0])
        y = (ret[0].squeeze(2))
        #x += V @ y
        #x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
        _v =  torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        #if MG is not None:
        #    _v = apply_MG(MG, MG_args, _v, back=back)
        x = x + _v
        #x = mx + torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        iters += restart

    #info = 0
    #if iters == maxiter and not (r_norm <= atol):
    #    info = iters
    return mx, (iters, r_norm)

def cg_block(A, b, x0=None, tol=1e-12, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

        A block sparse, b: (b, d)
    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``. ``A`` must be a hermitian,
            positive definitive matrix with type of :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.cg`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #matvec = A.matvec
    #psolve = M.matvec

    #n = A.shape[0]
    #n = A.shape[1]
    #if maxiter is None:
    #    maxiter = n * 10
    #if n == 0:
    #    return cupy.empty_like(b), 0
    #TODO fix this. check all norms and use masks 
    #b_norm = torch.linalg.norm(b[0])
    #b_norm = torch.linalg.vector_norm(b[0],dim=-1)
    b_norm = torch.linalg.vector_norm(b,dim=-1)
    #if b_norm == 0:
    cont_mask = (b_norm>1e-9)
    #if b_norm < 1e-10:
    if not cont_mask.any():
        print('zero return')
        return b, 0

    cont_mask = cont_mask.float()
    if atol is None:
        #atol = tol * float(b_norm)
        atol = tol * b_norm
    else:
        atol = max(float(atol), tol * float(b_norm))
        #atol = float(atol)

    #r = b - matvec(x)
    #b = b.unsqueeze(-1)
    x = torch.zeros_like(b)
    #r = b - torch.bmm(A,x)#.reshape(b.shape)
    r = b - block_mv(A,x)#.reshape(b.shape)
    iters = 0
    rho = 0
    #ipdb.set_trace()
    while iters < maxiter:
        #z = psolve(r)
        z = r #psolve(r)
        rho1 = rho
        #rho = cublas.dotc(r, z)
        rho = (r*z).sum(dim=1)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

            beta = beta.unsqueeze(1)
            p = z + beta * p
        #q = matvec(p)
        #q = torch.bmm(A, p)
        #q = bmm_fix(A, p)
        q = block_mv(A, p)
        #alpha = rho / cublas.dotc(p, q)
        alpha = rho / (p*q).sum(dim=1)

        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha*cont_mask

        alpha = alpha.unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        #resid = cublas.nrm2(r)
        #resid = cublas.nrm2(r)
        resid = torch.linalg.vector_norm(r, dim=1)
        res_mask = (resid > atol).float()
        cont_mask = cont_mask*res_mask
        #if resid.max() <= atol:
        #    break

    info = 0
    #if iters == maxiter and not (resid.max() <= atol):
    #    info = iters

    return x, info

def bmm_fix(A, x):
    b = A.shape[0]
    r = [A[i]@x[i] for i in range(b)]
    r = torch.stack(r, dim=0)
    return r


def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``. ``A`` must be a hermitian,
            positive definitive matrix with type of :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.cg`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #matvec = A.matvec
    #psolve = M.matvec

    #n = A.shape[0]
    n = A.shape[1]
    if maxiter is None:
        maxiter = n * 10
    #if n == 0:
    #    return cupy.empty_like(b), 0
    #TODO fix this. check all norms and use masks 
    #b_norm = torch.linalg.norm(b[0])
    b_norm = torch.linalg.vector_norm(b[0],dim=-1)
    #if b_norm == 0:
    if b_norm < 1e-8:
        return b, 0
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
        #atol = float(atol)

    #r = b - matvec(x)
    b = b.unsqueeze(-1)
    x = torch.zeros_like(b)
    #r = b - torch.bmm(A,x)#.reshape(b.shape)
    r = b - bmm_fix(A,x)#.reshape(b.shape)
    iters = 0
    rho = 0
    #ipdb.set_trace()
    while iters < maxiter:
        #z = psolve(r)
        z = r #psolve(r)
        rho1 = rho
        #rho = cublas.dotc(r, z)
        rho = (r*z).sum(dim=1)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            beta = beta.unsqueeze(1)
            p = z + beta * p
        #q = matvec(p)
        #q = torch.bmm(A, p)
        q = bmm_fix(A, p)
        #alpha = rho / cublas.dotc(p, q)
        alpha = rho / (p*q).sum(dim=1)
        alpha = alpha.unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        #resid = cublas.nrm2(r)
        #resid = cublas.nrm2(r)
        resid = torch.linalg.vector_norm(r, dim=(1,2))
        if resid.max() <= atol:
            break

    info = 0
    if iters == maxiter and not (resid.max() <= atol):
        info = iters

    return x.squeeze(-1), info

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

def minres(A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor, M1:torch.Tensor=None, M2:torch.Tensor=None, 
           rtol:float=1e-4, maxiter:int=100, MG=None, MG_args=None, back=False
           #perm=None, perminv=None,
           ):
#def minres(A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor,rtol:float=1e-6, maxiter:int=100,
#           _max:float=0):
#def minres(A:torch.Tensor, b:torch.Tensor, x0:torch.Tensor, M:Tuple[torch.Tensor,torch.Tensor]=None, rtol:float=1e-6, maxiter:int=100,
#           block_size:int=100, stride:int=100):

    #At = Aperm.transpose(1,2)

    #num_var,num_constraint = mlens
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

    if MG is not None:
        y = apply_MG(MG, MG_args, r1, back=back)
    #if MG is not None:

    #    _r1 = r1[:, num_var:]
    #    #_r1 = _r1[:, perminv]
    #    y2 = apply_block_jacobi_M(M1,_r1 , upper=False, 
    #                              block_size=block_size, stride=stride)
    #    #y2 = y2[:, perm]

    #    y1 = r1[:, :num_var]/M2
    #    y = torch.cat([y1,y2], dim=1)
    #    #y = r1*1/M
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

        #if MG is not None:
        #    #pass
        #    #y = r2*1/M
        #    #y = apply_block_jacobi_M(M, r2, upper=False, 
        #    #                      block_size=block_size, stride=stride)
        #    #y = r2/M
        #    _r2 = r2[:, num_var:]
        #    #_r2 = _r2[:, perminv]
        #    y2 = apply_block_jacobi_M(M1,_r2 , upper=False, 
        #                            block_size=block_size, stride=stride)
        #    #y2 = y2[:, perm]
        #    y1 = r2[:, :num_var]/M2
        #    y = torch.cat([y1,y2], dim=1)

        if MG is not None:
            y = apply_MG(MG, MG_args, r2, back=back)
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
        #if itn>=5 and ((test1 <= 1e-9).all()):
        #    print('break ', test2, test1)
        #    break

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
    
    