
# %%
import numpy
import numpy as np

import torch
import osqp
import scipy
import scipy.sparse as SPS
import scipy.sparse.linalg as SPSLG
from scipy.integrate import odeint

from solver.ode_layer import ODEINDLayerTest
from solver.cg import cg_matvec

import matplotlib.pyplot as plt

numpy.set_printoptions(linewidth=200, precision=3)

siters = 0
def nonlocal_iterate(arr):
    #nonlocal siters
    global siters
    siters+=1

def test_osqp_dual_relaxation():
    step_size = 0.1
    #end = 3*step_size
    end = 600*step_size
    n_step = int(end/step_size)
    order=2

    steps = step_size*np.ones((n_step-1,))
    steps = torch.tensor(steps)

    #coeffs are c_2 = 1, c_1 = 0, c_0 = 0
    #_coeffs = np.array([[0,1,0,1]], dtype='float32')
    _coeffs = np.array([[1,0,1]], dtype='float64')
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
    #H = torch.cat([derivative, derivative_neg], dim=1)
    #H = torch.cat([derivative, derivative_neg], dim=1)
    #H = torch.cat([derivative], dim=1)

    print(A.shape)
    print(H.shape)

    A_rhs = torch.cat([rhs, iv], dim=1)
    H_rhs = torch.zeros(H.shape[0:2]).type_as(rhs)

    print(A_rhs.shape)
    print(H_rhs.shape)

    #c = torch.cat([A_rhs, H_rhs], dim=1)
    c = torch.zeros((1,A.shape[2]), device=coeffs.device).double()
    c[:,0] = 100

    gamma =0 # 0.001
    #eps_gamma = 0.1

    #A = a_rhs -> A>=a_rhs, A<=a_rhs -> -A>=-a_rhs, H \ge 0

    # A>= A_rhs
    # H>= 0
    #C = torch.cat([A,-A, H], dim=1)
    C = torch.cat([A, H], dim=1)
    CT = C.transpose(1,2)

    #eye = torch.sparse.spdiags(torch.ones(CT.shape[2]), torch.tensor(0), (CT.shape[2], CT.shape[2]))
    eye_H = torch.sparse.spdiags(torch.ones(H.shape[1]), torch.tensor(0), (H.shape[1], H.shape[1]))
    zero_A = torch.sparse_coo_tensor([[],[]], [], (H.shape[1], A.shape[1]))
    var_constraints = torch.cat([zero_A, eye_H], dim=1)
    #CTX = torch.cat([CT, eye.unsqueeze(0)], dim=1)
    CTX = torch.cat([CT, var_constraints.unsqueeze(0)], dim=1)

    #rhs_x = torch.zeros(CT.shape[2])
    rhs_x = torch.zeros(H.shape[1])
    rhs = torch.cat([c[0], rhs_x], dim=0)

    rhs_x_u = torch.ones_like(rhs_x)*torch.inf
    rhs_u = torch.cat([c[0], rhs_x_u])
    #Take dual 
    
    #rhs_dual = torch.cat([A_rhs,-A_rhs, H_rhs], dim=1)
    rhs_dual = torch.cat([A_rhs, H_rhs], dim=1)

    q = -rhs_dual

    A = CTX[0]
    l = rhs#[0]
    u = rhs_u#[0]
    q = q[0]

    l = l.cpu().numpy()
    u = u.cpu().numpy()
    q = q.cpu().numpy()
    #d = c[0].cpu().numpy()
    #G = gamma*np.eye(A.shape[2])
    #E = A[0].to_dense().cpu().numpy()
    #C = H[0].to_dense().cpu().numpy()
    #e = A_rhs[0].cpu().numpy()
    #c = H_rhs[0].cpu().numpy()

    #build scipy coo
    indices = A._indices().cpu().numpy()
    values = A._values().cpu().numpy()
    shape = list(A.shape)

    A_s = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)
    ones = gamma*np.ones(A_s.shape[1])
    #ones[-1] = eps_gamma
    #P = gamma*SPS.eye(A_s.shape[1])
    P = SPS.diags(ones)

    A_s = A_s.tocsc()
    P = P.tocsc()

    print(A_s.T.toarray()[-5:, 0:20])

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A_s, l=l, u=u)
    res = m.solve()
    #lambdaphi=np.zeros(A.shape[1]+H.shape[1])
    #x = (quadraticdual(lambdaphi,d,G,E,e,C,c))
    print(res.y)

    y = res.y[:n_step*(order+1)+1]
    u = y[1:]
    u = u.reshape(n_step, order+1)
    #shape: batch, step, vars, order
    #u = u.permute(0,2,1,3)

    u0 = u[:,0]
    u1 = u[:,1]
    #u2 = u[:,:,:,2]
    return y[0], u0, u1


def test_primal_equality_cg():
    step_size = 0.1
    #end = 3*step_size
    end = 100*step_size
    n_step = int(end/step_size)
    order=2

    steps = step_size*np.ones((n_step-1,))
    steps = torch.tensor(steps)

    #coeffs are c_2 = 1, c_1 = 0, c_0 = 0
    #_coeffs = np.array([[0,1,0,1]], dtype='float32')

    _coeffs = np.array([[10,0.0,1]], dtype='float64')
    #_coeffs = np.array([[20,0.1,0.1]], dtype='float64')
    #_coeffs = np.array([[10,0.1,0.1]], dtype='float64')
    #_coeffs = np.array([[-0.1,0.1, 10]], dtype='float64')
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
    #derivative_neg = derivative*b
    #print(derivative_neg.to_dense())
    
    A = torch.cat([eq, initial], dim=1)
    #H = torch.cat([derivative, derivative_neg, eps_tensor], dim=1)
    H = derivative
    #H = torch.cat([derivative, derivative_neg], dim=1)
    #H = torch.cat([derivative, derivative_neg], dim=1)
    #H = torch.cat([derivative], dim=1)

    rhs = rhs[0]
    iv = iv[0]
    A = A[0]
    H = H[0]


    num_eps = H.shape[0]
    num_var = H.shape[1]
    zero_A = torch.sparse_coo_tensor([[],[]], [], (A.shape[0], num_eps))
    eps_H = torch.sparse.spdiags(-1*torch.ones(H.shape[0]), torch.tensor(0), (H.shape[0], H.shape[0]))

    A = torch.cat([zero_A, A], dim=1)
    H = torch.cat([eps_H, H], dim=1)
    
    print(A.shape)
    print("H ", H.shape)

    A_rhs = torch.cat([rhs, iv], dim=0)
    H_rhs = torch.zeros(H.shape[0]).type_as(rhs)

    print(A_rhs.shape)
    print(H_rhs.shape)

    #c = torch.cat([A_rhs, H_rhs], dim=1)
    c = torch.zeros(A.shape[1], device=coeffs.device).double()
    #c[:,0] = 100

    #gamma =0 # 0.001
    #eps_gamma = 0.1

    #A = a_rhs -> A>=a_rhs, A<=a_rhs -> -A>=-a_rhs, H \ge 0

    # A>= A_rhs
    # H>= 0
    #C = torch.cat([A,-A, H], dim=1)
    C = torch.cat([A, H], dim=0)
    l = torch.cat([A_rhs, H_rhs], dim=0)
    u = l
    P_diag = torch.ones(num_eps)*1e8
    P_zeros = torch.zeros(num_var) +1e-8
    P_diag = torch.cat([P_diag, P_zeros])

    P= torch.sparse.spdiags(P_diag, torch.tensor(0), (num_eps+num_var, num_eps+num_var))

    print(C.shape)
    #q = -rhs_dual

    #A = CTX[0]
    #l = rhs#[0]
    #u = rhs_u#[0]
    #q = q[0]

    l = l.cpu().numpy()
    u = u.cpu().numpy()
    #q = q.cpu().numpy()
    q = c.cpu().numpy()
    ##G = gamma*np.eye(A.shape[2])
    ##E = A[0].to_dense().cpu().numpy()
    ##C = H[0].to_dense().cpu().numpy()
    ##e = A_rhs[0].cpu().numpy()
    ##c = H_rhs[0].cpu().numpy()

    ##build scipy coo
    indices = C._indices().cpu().numpy()
    values = C._values().cpu().numpy()
    shape = list(C.shape)
    A_s = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)

    indices = P._indices().cpu().numpy()
    values = P._values().cpu().numpy()
    shape = list(P.shape)
    P_s = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)

    Pinv_s = SPS.coo_matrix((1/values, (indices[0], indices[1]) ), shape = shape)

    #An = A_s.toarray()
    #Pn = P_s.toarray()
    #Pninv = P_s.toarray()

    #eq_rhs = np.concatenate([-q, l], axis=0)
    eq_rhs = np.concatenate([q, -l], axis=0)

    #ZZ = np.concatenate([U,D], axis=0)

    pdmat = A_s@Pinv_s@A_s.T
    pd_rhs = A_s@Pinv_s@q[:,None] + l[:,None]

    dd = pdmat.diagonal()
    ddmat = SPS.spdiags(dd, np.array([0]), dd.shape[0], dd.shape[0])

    #tr = SPS.tril(pdmat, k=-1) + ddmat
    #tr = SPS.tril(pdmat, k=-1) + ddmat
    #tr = SPSLG.inv(tr)
    #cc = SPSLG.spilu(pdmat)


    lam,info = SPSLG.cg(pdmat, pd_rhs, callback=nonlocal_iterate)
    #lam,info = SPSLG.cg(pdmat, pd_rhs, M=ddmat, callback=nonlocal_iterate)
    #lam,info = SPSLG.cg(pdmat, pd_rhs, M=tr, callback=nonlocal_iterate)
    #lam,info = SPSLG.cg(pdmat, pd_rhs, M=cc, callback=nonlocal_iterate)
    print( 'cg info', info, siters)
    #lam,info = SPSLG.lgmres(pdmat, pd_rhs)
    xl = -Pinv_s@(A_s.T@lam -q)

    #ZZ2 = SPS.bmat([[P_s, A_s.T],[A_s, None]])
    #ii = np.linalg.inv(ZZ)
    #eq_rhs = eq_rhs[...,np.newaxis]
    #xl = ii@eq_rhs

    #xl,info = SPSLG.gmres(ZZ2, eq_rhs, atol=1e-8)
    #xl,info = SPSLG.lgmres(ZZ2, eq_rhs, atol=1e-8)
    #xl,info = SPSLG.bicg(ZZ2, eq_rhs)
    #xl,info = SPSLG.qmr(ZZ2, eq_rhs)
    #xl,info = SPSLG.gcrotmk(ZZ2, eq_rhs)

    sol = -xl[:num_eps+num_var]
    #return None, 
    #r = ZZ @ii
    #d = np.diagonal(r)
    #ones = gamma*np.ones(A_s.shape[1])
    ##ones[-1] = eps_gamma
    ##P = gamma*SPS.eye(A_s.shape[1])
    #P = SPS.diags(ones)

    #A_s = A_s.tocsc()
    #P = P.tocsc()

    #print(A_s.T.toarray()[-5:, 0:20])

    #m = osqp.OSQP()
    #m.setup(P=P_s, q=q, A=A_s, l=l, u=u)
    #res = m.solve()
    #print(res.x)
    #sol = res.x

    eps = sol[:num_eps]
    u = sol[num_eps:num_eps+n_step*(order+1)]
    #u = y[1:]
    u = u.reshape(n_step, order+1)
    ##shape: batch, step, vars, order
    ##u = u.permute(0,2,1,3)

    u0 = u[:,0]
    u1 = u[:,1]
    ##u2 = u[:,:,:,2]
    return eps, u0, u1

def test_primal_equality_cg_torch():
    step_size = 0.1
    #end = 3*step_size
    end = 500*step_size
    n_step = int(end/step_size)
    order=2

    steps = step_size*np.ones((n_step-1,))
    steps = torch.tensor(steps)

    #coeffs are c_2 = 1, c_1 = 0, c_0 = 0
    _coeffs = np.array([[1,0,1]], dtype='float32')

    #_coeffs = np.array([[10,0.0,1]], dtype='float64')
    #_coeffs = np.array([[20,0.1,0.1]], dtype='float64')
    #_coeffs = np.array([[10,0.1,0.1]], dtype='float64')
    #_coeffs = np.array([[-0.1,0.1, 10]], dtype='float64')
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
    #derivative_neg = derivative*b
    #print(derivative_neg.to_dense())
    
    A = torch.cat([eq, initial], dim=1)
    #H = torch.cat([derivative, derivative_neg, eps_tensor], dim=1)
    H = derivative
    #H = torch.cat([derivative, derivative_neg], dim=1)
    #H = torch.cat([derivative, derivative_neg], dim=1)
    #H = torch.cat([derivative], dim=1)

    rhs = rhs[0]
    iv = iv[0]
    A = A[0]
    H = H[0]


    num_eps = H.shape[0]
    num_var = H.shape[1]
    zero_A = torch.sparse_coo_tensor([[],[]], [], (A.shape[0], num_eps))
    eps_H = torch.sparse.spdiags(-1*torch.ones(H.shape[0]), torch.tensor(0), (H.shape[0], H.shape[0]))

    A = torch.cat([zero_A, A], dim=1)
    H = torch.cat([eps_H, H], dim=1)
    
    print(A.shape)
    print("H ", H.shape)

    A_rhs = torch.cat([rhs, iv], dim=0)
    H_rhs = torch.zeros(H.shape[0]).type_as(rhs)

    print(A_rhs.shape)
    print(H_rhs.shape)

    #c = torch.cat([A_rhs, H_rhs], dim=1)
    c = torch.zeros(A.shape[1], device=coeffs.device).double()
    #c[:,0] = 100

    #gamma =0 # 0.001
    #eps_gamma = 0.1

    #A = a_rhs -> A>=a_rhs, A<=a_rhs -> -A>=-a_rhs, H \ge 0

    # A>= A_rhs
    # H>= 0
    #C = torch.cat([A,-A, H], dim=1)
    C = torch.cat([A, H], dim=0)
    l = torch.cat([A_rhs, H_rhs], dim=0)
    u = l
    P_diag = torch.ones(num_eps)*1e8
    P_zeros = torch.zeros(num_var) +1e-8
    P_diag = torch.cat([P_diag, P_zeros])
    P_diag_inv = 1/P_diag

    P= torch.sparse.spdiags(P_diag, torch.tensor(0), (num_eps+num_var, num_eps+num_var))

    print(C.shape)
    #q = -rhs_dual

    #A = CTX[0]
    #l = rhs#[0]
    #u = rhs_u#[0]
    #q = q[0]

    #l = l.cpu().numpy()
    #u = u.cpu().numpy()
    #q = c.cpu().numpy()


    ##build scipy coo
    indices = C._indices().cpu().numpy()
    values = C._values().cpu().numpy()
    shape = list(C.shape)
    A_s = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)

    indices = P._indices().cpu().numpy()
    values = P._values().cpu().numpy()
    shape = list(P.shape)
    P_s = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)

    Pinv_s = SPS.coo_matrix((1/values, (indices[0], indices[1]) ), shape = shape)

    #An = A_s.toarray()
    #Pn = P_s.toarray()
    #Pninv = P_s.toarray()

    #eq_rhs = np.concatenate([-q, l], axis=0)
    #eq_rhs = np.concatenate([q, -l], axis=0)

    #ZZ = np.concatenate([U,D], axis=0)

    #pdmat = A_s@Pinv_s@A_s.T

    A = C.unsqueeze(0)
    At = C.T.unsqueeze(0)
    c = c.unsqueeze(0)
    l = l.unsqueeze(0)
    P_diag_inv = P_diag_inv.unsqueeze(0)

    #pd_rhs = A_s@Pinv_s@q[:,None] + l[:,None]
    rhs = c
    rhs = P_diag_inv*rhs
    rhs = torch.bmm(A, rhs.unsqueeze(2))
    rhs = rhs.squeeze(2) + l

    #lam,info = SPSLG.cg(pdmat, pd_rhs)
    lam, info = cg_matvec([A, P_diag_inv, At], rhs, maxiter=8000)
    print('torch cg info ', info)
    #lam,info = SPSLG.lgmres(pdmat, pd_rhs)
    #xl = -Pinv_s@(A_s.T@lam -q)

    #xl = -Pinv_s@(A_s.T@lam -c)
    xl = lam.unsqueeze(2)
    xl = torch.bmm(At, xl)
    xl = -P_diag_inv*(xl.squeeze(2) - c)
    #xl = xl.squeeze(2)
    xl = xl[0]


    sol = -xl[:num_eps+num_var]

    eps = sol[:num_eps]
    u = sol[num_eps:num_eps+n_step*(order+1)]
    #u = y[1:]
    u = u.reshape(n_step, order+1)
    ##shape: batch, step, vars, order
    ##u = u.permute(0,2,1,3)

    u0 = u[:,0]
    u1 = u[:,1]
    ##u2 = u[:,:,:,2]
    return eps, u0, u1

def ode_solve():
    def f(state, t):
        x, y= state

        #ax'' + bx' + cx + d = 0
        a = 0.1
        b = 0.1
        c = 10
        d = 0

        dx = y
        dy = -b*y -c*x -d
        dy = dy/a

        return dx, dy
        
    STEP=0.1
    T=500
    state0 = [0.0, 1.0]
    time_steps = np.linspace(0, T*STEP, T)

    x_sim = odeint(f, state0, time_steps)
    return x_sim

#test_osqp()
# %%
#eps, u0, u1 = test_osqp_dual_relaxation()
#eps, u0, u1 = test_primal_equality_cg()
eps, u0, u1 = test_primal_equality_cg_torch()
#eps, u0, u1 = test_osqp()
#eps, u0, u1 = test_py()
#res = ode_solve()
#u0 = res[:, 0]
#u1 = res[:, 1]
#print(eps)

#test_py()

# %%
f, axis = plt.subplots(1,2, figsize=(16,3))
axis[0].plot(u0.squeeze())
axis[1].plot(u1.squeeze())
#axis[2].plot(u2.squeeze())
# %%
u0

# %%
np.abs(eps).max()

# %%
