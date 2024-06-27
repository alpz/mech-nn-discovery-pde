
# %%
import numpy
import numpy as np

import scipy.optimize
import torch
import osqp
import scipy
import scipy.sparse as SPS

from solver.ode_layer import ODEINDLayerTest

import matplotlib.pyplot as plt


def test_lp():
    step_size = 0.1
    #end = 3*step_size
    end = 10*step_size
    n_step = int(end/step_size)
    order=2

    steps = step_size*np.ones((n_step-1,))
    steps = torch.tensor(steps)

    #coeffs are c_2 = 1, c_1 = 0, c_0 = 0
    #_coeffs = np.array([[0,1,0,1]], dtype='float32')
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

    ode = ODEINDLayerTest(bs=1,order=order,n_ind_dim=1,n_iv=2,n_step=n_step,n_iv_steps=1, step_size=step_size)

    u0,u1,u2,eps,_, eq, initial, derivative, eps_tensor = ode(_coeffs, rhs, iv, steps)

    #derivative.to_dense()

    b = np.array([-1]*derivative.shape[-1])
    b[0] = 1
    b = torch.tensor(b)[None, None, ...]
    derivative_neg = derivative*b
    #print(derivative_neg.to_dense())
    
    A = torch.cat([eq, initial], dim=1)
    H = torch.cat([derivative, derivative_neg, eps_tensor], dim=1)

    A_rhs = torch.cat([rhs, iv], dim=1)

    


    print(A.shape)
    print(H.shape)

    H_rhs = torch.zeros(H.shape[0:2]).type_as(rhs)

    print(A_rhs.shape)
    print(H_rhs.shape)

    #c = torch.cat([A_rhs, H_rhs], dim=1)
    c = torch.zeros((1,A.shape[2]), device=coeffs.device).double()
    c[:,0] = 1

    c = c.cpu().numpy()



    A = A[0].to_dense().cpu().numpy()
    H = -H[0].to_dense().cpu().numpy()
    h = -H_rhs[0].cpu().numpy()
    a = A_rhs[0].cpu().numpy()


    res = scipy.optimize.linprog(c, A_eq=A, b_eq=a, A_ub=H, b_ub=h, bounds=(None, None))

    print(res.status)

    y = res.x
    u = y[1:]
    u = u.reshape(n_step, order+1)
    #shape: batch, step, vars, order
    #u = u.permute(0,2,1,3)

    u0 = u[:,0]
    u1 = u[:,1]
    #u2 = u[:,:,:,2]
    return y[0], u0, u1

def test_osqp_dual_relaxation():
    step_size = 0.1
    #end = 3*step_size
    end = 100*step_size
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
    #H = torch.cat([derivative], dim=1)

    print(A.shape)
    print(H.shape)

    A_rhs = torch.cat([rhs, iv], dim=1)
    H_rhs = torch.zeros(H.shape[0:2]).type_as(rhs)

    print(A_rhs.shape)
    print(H_rhs.shape)

    #c = torch.cat([A_rhs, H_rhs], dim=1)
    c = torch.zeros((1,A.shape[2]), device=coeffs.device).double()
    c[:,0] = 10

    gamma = 0.1
    eps_gamma = 0.1

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
    ones[-1] = eps_gamma
    #P = gamma*SPS.eye(A_s.shape[1])
    P = SPS.diags(ones)

    A_s = A_s.tocsc()
    P = P.tocsc()

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

#test_osqp()
# %%
eps, u0, u1 = test_lp()
#eps, u0, u1 = test_osqp()
print(eps)

#test_py()

# %%
f, axis = plt.subplots(1,2, figsize=(16,3))
axis[0].plot(u0.squeeze())
axis[1].plot(u1.squeeze())
#axis[2].plot(u2.squeeze())

# %%
eps

# %%
