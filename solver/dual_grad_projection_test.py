
# %%
import numpy
import numpy as np

import torch
import osqp
import scipy
import scipy.sparse as SPS

from solver.ode_layer import ODEINDLayerTest

import matplotlib.pyplot as plt

def quadraticdual(lambdaphi,d,G,E,e,C,c):
    n=len(d)
    j=len(c)
    k=len(e)
    if j==0:
        C=numpy.zeros((0,n))
    if k==0:
        E=numpy.zeros((0,n))
    A=numpy.concatenate((C,E),0)
    a=numpy.concatenate((c,e))
    B=numpy.matmul(A,numpy.linalg.solve(G,numpy.transpose(A)))
    b=-(numpy.matmul(A,numpy.linalg.solve(G,d))+a)
    l=numpy.concatenate((numpy.zeros(j),-numpy.inf*numpy.ones(k)))
    u=numpy.inf*numpy.ones(j+k)
    lambdaphi=gradientprojection(lambdaphi,b,B,l,u)
    x=numpy.linalg.solve(G,numpy.matmul(numpy.transpose(A),lambdaphi)-d)
    return x,lambdaphi

def gradientprojection(x,d,G,l,u):
    tol=1e-14*numpy.max(numpy.max(numpy.abs(G)))
    n=len(x)
    x=numpy.array(x)
    d=numpy.array(d)
    G=numpy.array(G)
    l=numpy.array(l)
    u=numpy.array(u)
    g=numpy.matmul(G,x)+d
    while True:
        tb=numpy.inf*numpy.ones(n)
        tb[(g<0)&(u<numpy.inf)]=((x-u)/g)[(g<0)&(u<numpy.inf)]
        tb[(g>0)&(l>-numpy.inf)]=((x-l)/g)[(g>0)&(l>-numpy.inf)]
        tbi=numpy.argsort(tb)
        tbs=tb[tbi]
        xt=lambda t:x-numpy.minimum(tb,t)*g
        xm=numpy.array(x)
        gm=numpy.array(g)
        tminf=0
        for k in range(n):
            gGg=numpy.dot(gm,numpy.matmul(G,gm))
            mc=numpy.dot(gm,d)+numpy.dot(xm,numpy.matmul(G,gm))
            tmid=mc/gGg
            if tmid>0 and tmid<tbs[k] and gGg>0:
                tminf=tmid
                break
            if mc<0:
                break
            xm=xt(tbs[k])
            gm[tbi[k]]=0
            tminf=tbs[k]
        x=xt(tminf)
        g=numpy.matmul(G,x)+d
        onbound=onbound=1.*(numpy.abs(x-l)<10.*numpy.spacing(l))-1.*(numpy.abs(x-u)<10.*numpy.spacing(u))
        A=numpy.diag(onbound)
        keep=onbound!=0
        m=numpy.sum(keep)
        A=A[keep,:]
        if m>0:
            Q,R=numpy.linalg.qr(numpy.transpose(A),'complete')
            Z=Q[:,m:]
            ll=numpy.array(l)
            uu=numpy.array(u)
            ll[numpy.isinf(ll)]=-.9*numpy.finfo('d').max
            uu[numpy.isinf(uu)]=.9*numpy.finfo('d').max
            ZGZ=numpy.matmul(numpy.transpose(Z),numpy.matmul(G,Z))
            ulZ=numpy.matmul(numpy.transpose(Z),numpy.column_stack((ll-x,uu-x)))
            lZ=numpy.min(ulZ,1)
            uZ=numpy.max(ulZ,1)
            p=steihaugssquare(-numpy.matmul(numpy.transpose(Z),g),ZGZ,lZ,uZ)
            x=x+numpy.matmul(Z,p)
        else:
            p=steihaugssquare(-g,G,l-x,u-x)
            x=x+p
        g=numpy.matmul(G,x)+d
        if numpy.any(numpy.isinf(x)):
            print('Unbounded')
            return x
        if m>0:
            lambd=numpy.linalg.solve(numpy.matmul(A,numpy.transpose(A)),numpy.matmul(A,g))
            dLdx=g-numpy.matmul(numpy.transpose(A),lambd)
        else:
            dLdx=g
        if numpy.dot(dLdx,dLdx)**.5<tol:
            return x

def steihaugssquare(g,B,l,u):
    A=B
    b=-numpy.array(g)
    tol=1e-8*numpy.dot(g,g)**.5
    n=len(g)
    x=numpy.zeros(n)
    r=-numpy.array(g)
    p=-numpy.array(r)
    rr=numpy.dot(r,r)
    for j in range(n):
        Ap=numpy.matmul(A,p)
        pAp=numpy.dot(p,Ap)
        if pAp<=0:
            return x
        alpha=rr/pAp
        nx=x+alpha*p
        if numpy.any(nx<l-1e-8) or numpy.any(nx>u+1e-8):
            t=numpy.concatenate(((l-x)/p,(u-x)/p))
            tindom=t[(t>=-1e-14)&(t<alpha)]
            if len(tindom)>0:
                t=numpy.min(tindom)
            else:
                t=0
            return x+t*p
        x=nx
        rn=r+alpha*Ap
        rnrn=numpy.dot(rn,rn)
        if rnrn**.5<tol:
            return x
        beta=rnrn/rr
        p=-rn+beta*p
        r=rn
        rr=rnrn
    return x

#if __name__ == '__main__':
#    x=[0.,0.]
#    d=[-2.,-5.]
#    G=[[2.,0.],[0.,2.]]
#    l=[0.,-1.]
#    u=[2.,2.4]
#    print(gradientprojection(x,d,G,l,u))
#
#    E=[]
#    e=[]
#    C=[[1.,-2.],[-1.,-2.],[-1.,2.],[1.,0.],[0.,1.]]
#    c=[-2.,-6.,-2.,0.,0.]
#    lambdaphi=[0,0,0,0,0]
#    print(quadraticdual(lambdaphi,d,G,E,e,C,c))

def test_py():
    step_size = 0.1
    #end = 3*step_size
    end = 3*step_size
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

    gamma = 0.01
    d = c[0].cpu().numpy()
    G = gamma*np.eye(A.shape[2])
    E = A[0].to_dense().cpu().numpy()
    C = H[0].to_dense().cpu().numpy()
    e = A_rhs[0].cpu().numpy()
    c = H_rhs[0].cpu().numpy()

    lambdaphi=np.zeros(A.shape[1]+H.shape[1])
    x = (quadraticdual(lambdaphi,d,G,E,e,C,c))
    print(x)

def test_osqp():
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

    gamma = 0.1

    A = torch.cat([A,H], dim=1)
    l = torch.cat([A_rhs, H_rhs], dim=1)

    H_u = torch.ones_like(H_rhs)*torch.inf
    u = torch.cat([A_rhs, H_u ], dim=1)
    q = c

    A = A[0]
    l = l[0]
    u = u[0]
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
    P = gamma*SPS.eye(A_s.shape[1])

    A_s = A_s.tocsc()
    P = P.tocsc()

    print("AAA ", A_s.shape)

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A_s, l=l, u=u)
    res = m.solve()
    #lambdaphi=np.zeros(A.shape[1]+H.shape[1])
    #x = (quadraticdual(lambdaphi,d,G,E,e,C,c))
    print(res.x)

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
    end = 50*step_size
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

    gamma = 0.1

    #A = a_rhs -> A>=a_rhs, A<=a_rhs -> -A>=-a_rhs, H \ge 0

    # A>= A_rhs
    # H>= 0
    C = torch.cat([A,-A, H], dim=1)
    CT = C.transpose(1,2)
    eye = torch.sparse.spdiags(torch.ones(CT.shape[2]), torch.tensor(0), (CT.shape[2], CT.shape[2]))
    CTX = torch.cat([CT, eye.unsqueeze(0)], dim=1)

    rhs_x = torch.zeros(CT.shape[2])
    rhs = torch.cat([c[0], rhs_x], dim=0)

    rhs_x_u = torch.ones_like(rhs_x)*torch.inf
    rhs_u = torch.cat([c[0], rhs_x_u])
    #Take dual 
    
    rhs_dual = torch.cat([A_rhs,-A_rhs, H_rhs], dim=1)

    q = rhs_dual

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
    P = gamma*SPS.eye(A_s.shape[1])

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
eps, u0, u1 = test_osqp_dual_relaxation()
#eps, u0, u1 = test_osqp()
print(eps)

#test_py()

# %%
f, axis = plt.subplots(1,2, figsize=(16,3))
axis[0].plot(u0.squeeze())
axis[1].plot(u1.squeeze())
#axis[2].plot(u2.squeeze())
# %%
u0

# %%
