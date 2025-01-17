# %%
import matplotlib.pyplot as plt
import pyro.util.io_pyro as io
import os

dirname = os.path.expanduser('~')
dirname = os.path.join(dirname, 'tmp/visc')

all_files = os.listdir(dirname)

files = []
for file in all_files:
    if file.endswith('.h5'):
        files.append(file)
files = sorted(files)   

# %%
gradp_xs=[]
gradp_ys=[]
us = []
vs = []
ts = []
dxs = []
dys = []
for file in files:
    sim = io.read(os.path.join(dirname, file))
    print(sim.cc_data.grid.dx)
    print(sim.cc_data.t)

    gpx = sim.cc_data.get_var('gradp_x')
    gpy = sim.cc_data.get_var('gradp_y')
    u = sim.cc_data.get_var('x-velocity')
    v = sim.cc_data.get_var('y-velocity')

    gradp_xs.append(gpx)
    gradp_ys.append(gpy)
    us.append(u)
    vs.append(v)

    ts.append(sim.cc_data.t)
    dxs.append(sim.cc_data.grid.dx)
    dys.append(sim.cc_data.grid.dy)
# %%
dys
# %%
sim.cc_data.vars
# %%

files
# %%

import numpy as np

u = np.stack(us, axis=0)
v = np.stack(vs, axis=0)
gradp_x = np.stack(gradp_xs, axis=0)
gradp_y = np.stack(gradp_ys, axis=0)
ts = np.array(ts)
dx = np.array(dxs)
dy = np.array(dys)
# %%
u.shape
v.shape
gradp_x.shape

# %%

u = u[:, 4:-4, 4:-4]
v = v[:, 4:-4, 4:-4]
gradp_x = gradp_x[:, 4:-4, 4:-4]
gradp_y = gradp_y[:, 4:-4, 4:-4]

# %%
np.save('euler_u.npy', u)
np.save('euler_v.npy', v)
np.save('euler_gradp_x.npy', gradp_x)
np.save('euler_gradp_y.npy', gradp_y)
np.save('euler_t.npy', ts)
np.save('euler_dx.npy', dx)
np.save('euler_dy.npy', dy)

# %%

import matplotlib.pyplot as plt
plot = plt.pcolormesh(u[190], cmap='viridis', shading='gouraud')
# %%

u[:, 4:-4, 4:-4].shape
# %%

ut = np.gradient(u, 0.005, axis=0)
ux = np.gradient(u, 0.0078, axis=1 )
uy = np.gradient(u, 0.0078, axis=2 )
uxx = np.gradient(ux, 0.0078, axis=1)
uyy = np.gradient(uy, 0.0078, axis=2)

#ut = np.gradient(u, axis=0)
#ux = np.gradient(u,  axis=1 )
#uy = np.gradient(u,  axis=2 )
#uxx = np.gradient(ux, axis=1)
#uyy = np.gradient(uy, axis=2)
# %%

rhs = 0.001*(uxx + uyy)
lhs = ut + u*ux + v*uy + gradp_x
# %%

diff = rhs-lhs
# %%
diff[150]
# %%
ut

# %%

u
# %%
