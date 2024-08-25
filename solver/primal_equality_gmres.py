
# %%
import numpy
import numpy as np

import torch
import osqp
import scipy
import scipy.sparse as SPS
import scipy.sparse.linalg as SPSLG
from scipy.integrate import odeint

from solver.ode_layer import ODEINDLayerTest#EPS
from solver.cg import gmres, cg_matvec

import matplotlib.pyplot as plt

numpy.set_printoptions(linewidth=200, precision=3)

siters = 0
def nonlocal_iterate(arr):
    #nonlocal siters
    global siters
    siters+=1


def test_primal_equality_cg_torch():
    step_size = 0.1
    #end = 3*step_size
    end = 100*step_size
    n_step = int(end/step_size)
    order=2

    steps = step_size*np.ones((n_step-1,))
    steps = torch.tensor(steps)

    #coeffs are c_2 = 1, c_1 = 0, c_0 = 0
    _coeffs = np.array([[5,0,1]], dtype='float64')

    #_coeffs = np.array([[1,0.0,1]], dtype='float64')
    #_coeffs = np.array([[2,0.1,0.1]], dtype='float64')
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

    ode = ODEINDLayerTest(bs=1,order=order,n_ind_dim=1,n_iv=2,n_step=n_step,n_iv_steps=1, step_size=step_size)

    #u0,u1,u2,eps,_, eq, initial, derivative, eps_tensor = ode(_coeffs, rhs, iv, steps)
    u0,u1,u2,eps,_, C,l = ode(_coeffs, rhs, iv, steps)

    #derivative.to_dense()

    num_eps = ode.num_eps
    num_var = ode.num_var
    u = l
    P_diag = torch.ones(num_eps)*1e2
    P_zeros = torch.zeros(num_var) +1e-5
    P_diag = torch.cat([P_zeros, P_diag])
    P_diag_inv = 1/P_diag

    #c = torch.zeros(num_var+num_eps, device=coeffs.device).double()
    c0 = torch.zeros(num_var, device=coeffs.device).double()
    c1 = 0*torch.ones(num_eps, device=coeffs.device).double()
    c = torch.cat([c0,c1])

    P= torch.sparse.spdiags(P_diag, torch.tensor(0), (num_eps+num_var, num_eps+num_var))

    print(C.shape)

    A = C#.unsqueeze(0)
    At = C.transpose(1,2)#.unsqueeze(0)
    c = c.unsqueeze(0)
    l = l#.unsqueeze(0)
    P_diag_inv = P_diag_inv.unsqueeze(0)

    #pd_rhs = A_s@Pinv_s@q[:,None] + l[:,None]
    rhs = c
    rhs = P_diag_inv*rhs
    rhs = torch.bmm(A, rhs.unsqueeze(2))
    rhs = rhs.squeeze(2) + l

    #mz = torch.zeros((A.shape[1])) + 1e-4
    #M = None #torch.cat([P_diag, mz]).unsqueeze(0)
    
    G = torch.sparse.spdiags(P_diag, torch.tensor([0]), (P_diag.shape[0], P_diag.shape[0]), 
                             layout=torch.sparse_coo)
    G = G.unsqueeze(0)
    GA = torch.cat([G, A], dim=1)
    Z = torch.sparse_coo_tensor(torch.empty([2,0]), [], size=(A.shape[1], A.shape[1]), dtype=A.dtype)
    Z = Z.unsqueeze(0)

    AtZ = torch.cat([A.transpose(1,2), Z], dim =1)
    KKT = torch.cat([GA, AtZ], dim =2)


    KKTs = KKT[0]
    print(KKTs._indices().shape, KKTs._values().shape)
    indices = KKTs._indices().cpu().numpy()
    values = KKTs._values().cpu().numpy()
    shape = list(KKTs.shape)
    KKTs = SPS.coo_matrix((values, (indices[0], indices[1]) ), shape = shape)

    M = SPSLG.spilu(KKTs, fill_factor=4.0)
    print('nnz ', M.nnz, KKTs.nnz)

    #Ad = A.to_dense()[0]
    #Atd = At.to_dense()[0]
    #PAtd = P_diag_inv[0].unsqueeze(1)*Atd
    #APAtd = torch.mm(Ad, PAtd)
    ##lam,info = SPSLG.cg(pdmat, pd_rhs, callback=nonlocal_iterate)
    #G = torch.diag(P_diag)
    #GA = torch.cat([G, Ad], dim=0)
    #Z = torch.zeros((Ad.shape[0], Ad.shape[0])).type_as(G)
    #AtZ = torch.cat([Atd, Z], dim =0)
    #KKT = torch.cat([GA, AtZ], dim =1)
    R = torch.cat([torch.zeros(G.shape[1]), -rhs[0]])
    #lam, info = gmres(APAtd, rhs[0], x0=torch.zeros_like(rhs[0]), maxiter=500, restart=40)
    sol, info = gmres(KKT, R.unsqueeze(0), x0=torch.zeros_like(R).unsqueeze(0), M=M, 
                      maxiter=20, restart=10)
    #sol = sol.unsqueeze(0)
    sol = -sol.squeeze()

    #lam, info = cg_matvec([A, P_diag_inv, At], rhs, maxiter=15000)
    print('torch gmres info ', info)
    #lam,info = SPSLG.lgmres(pdmat, pd_rhs)
    #xl = -Pinv_s@(A_s.T@lam -q)

    #xl = -Pinv_s@(A_s.T@lam -c)
    #xl = lam.unsqueeze(2)
    #xl = torch.bmm(At, xl)
    #xl = -P_diag_inv*(xl.squeeze(2) - c)
    ##xl = xl.squeeze(2)
    #xl = xl[0]


    #sol = -xl[:num_eps+num_var]

    eps = sol[num_var:num_var+num_eps]
    #u = sol[num_eps:num_eps+n_step*(order+1)]
    u = sol[:n_step*(order+1)]
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
eps
eps.abs().max()

# %%
