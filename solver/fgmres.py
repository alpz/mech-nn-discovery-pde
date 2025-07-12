
import torch
import numpy as np
import ipdb



def apply_MG(MG, MG_args, b, back=False):

    #x = MG.v_cycle_jacobi_start(MG_args[0], [b], MG_args[1],MG_args[2], n_step=1, back=back)
    #bs = b.shape[0]
    #b = b.reshape(-1)
    x = MG.v_cycle_gs_start(MG_args[0], b, MG_args[1],MG_args[2], MG_args[3], n_step=1, back=back)
    #x = x.reshape(bs, -1)
    #x = MG.full_multigrid_jacobi_start(MG_args[0], [b], MG_args[1],MG_args[2])
    return x

# adapted from scipy GMRES
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html

@torch.no_grad()
def fgmres_matvec(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, MG=None, MG_args=None,
          callback=None, atol=1e-5, callback_type=None, back=False, bs=1):
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
    #At = A.transpose(0,2)
    #matvec = A.matvec
    #psolve = M.matvec

    #bs = A.shape[0]
    n = A.shape[1]
    if n == 0:
        #return cupy.empty_like(b), 0
        return torch.empty_like(b), 0
    #b_norm = cupy.linalg.norm(b)
    b_norm = torch.linalg.norm(b, dim=0)
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
    V = torch.empty((n, restart), dtype=A.dtype, device=x.device)
    Z = torch.empty((n, restart), dtype=A.dtype, device=x.device)
    #H = torch.zeros((restart+1, restart), dtype=A.dtype, order='F')
    H = torch.zeros((restart+1, restart), dtype=A.dtype, device=x.device)
    #e = np.zeros((restart+1,), dtype=A.dtype)
    e = torch.zeros((restart+1,), dtype=A.dtype, device=x.device)

    def compute_hu(VV, u, j):
        S = VV[:, :j+1]
        #h = torch.mm(S.T, u.unsqueeze(1)).squeeze(1)
        #h = torch.bmm(S.transpose(0,1), u.unsqueeze(1)).squeeze(1)
        h = torch.mm(S.transpose(0,1), u.unsqueeze(1)).squeeze(1)
        u = u - torch.mm(S, h.unsqueeze(1)).squeeze(1)
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
        r = b - torch.mm(A, mx.unsqueeze(1)).squeeze(1)
        #print(A, mx.shape)
        #r = b - torch.bmm(At, torch.bmm(A, mx.unsqueeze(2))).squeeze(2)
        #r_norm = cublas.nrm2(r)
        r_norm = torch.linalg.norm(r, dim=0)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        #if r_norm <= atol or iters >= maxiter:
        if (r_norm <= atol).all() or iters >= maxiter:
            break
        v = r / r_norm#.unsqueeze(1)
        V[:, 0] = v
        e[0] = r_norm

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
                Z[:, j] = z#.clone()
            #u = matvec(z)
            u = torch.mm(A, z.unsqueeze(1)).squeeze(1)
            #u = torch.bmm(At,torch.bmm(A, z.unsqueeze(2))).squeeze(2)
            #H[:j+1, j], u = compute_hu(V, u, j)
            H[:j+1, j], u = compute_hu(V, u, j)
            #cublas.nrm2(u, out=H[j+1, j])
            torch.linalg.norm(u, out=H[j+1, j], dim=-1)
            if j+1 < restart:
                v = u / H[j+1, j]#.unsqueeze(1)
                V[:, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        #ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        ret = torch.linalg.lstsq(H, e.unsqueeze(1))
        #y = cupy.array(ret[0])
        y = (ret[0].squeeze(1))
        #x += V @ y
        #x = x + torch.bmm(V, y.unsqueeze(2)).squeeze(2)
        #_v =  torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        _v =  torch.mm(Z, y.unsqueeze(1)).squeeze(1)
        #if MG is not None:
        #    _v = apply_MG(MG, MG_args, _v, back=back)
        x = x + _v
        #x = mx + torch.bmm(Z, y.unsqueeze(2)).squeeze(2)
        iters += restart

    #info = 0
    #if iters == maxiter and not (r_norm <= atol):
    #    info = iters
    return mx, (iters, r_norm)