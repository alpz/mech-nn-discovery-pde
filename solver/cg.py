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
    A = As[0]
    #Pinv = As[1]
    Pinv_diag = As[1]
    At = As[2]

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
def cg_matvec(As, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None, atol=None):
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
    
@torch.no_grad()
def gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None,
          callback=None, atol=1e-5, callback_type=None):
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
    if b_norm == 0:
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
    V = torch.empty((bs, n, restart), dtype=A.dtype)
    #H = torch.zeros((restart+1, restart), dtype=A.dtype, order='F')
    H = torch.zeros((bs, restart+1, restart), dtype=A.dtype)
    #e = np.zeros((restart+1,), dtype=A.dtype)
    e = torch.zeros((bs, restart+1,), dtype=A.dtype)

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
        if M is not None:
            mx = M.solve(x[0].cpu().numpy())
            mx = torch.tensor(mx).unsqueeze(0)
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
        v = r / r_norm
        V[:, :, 0] = v
        e[:, 0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            #z = psolve(v)
            z = v #psolve(v)
            if M is not None:
                #z = z*M
                z = M.solve(z[0].cpu().numpy())
                z = torch.tensor(z).unsqueeze(0)
            #u = matvec(z)
            #u = torch.mm(A, z.unsqueeze(1)).squeeze(1)
            u = torch.bmm(A, z.unsqueeze(2)).squeeze(2)
            #H[:j+1, j], u = compute_hu(V, u, j)
            H[:, :j+1, j], u = compute_hu(V, u, j)
            #cublas.nrm2(u, out=H[j+1, j])
            torch.linalg.norm(u, out=H[:, j+1, j], dim=-1)
            if j+1 < restart:
                v = u / H[:, j+1, j]
                V[:, :, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        #ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        ret = torch.linalg.lstsq(H, e.unsqueeze(2))
        #y = cupy.array(ret[0])
        y = (ret[0].squeeze(2))
        #x += V @ y
        x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
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