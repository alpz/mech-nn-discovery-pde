# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from functions_python import *

# %%
filename = "outputs/GL_beta1.5/MeshDump-N620.bin"

fields = ReadDataFile(filename)

x = fields["x"]
y = fields["y"]
Ai = fields["Ai"]
#A2 = fields["A2"]
Ar = fields["Ar"]

print(x.shape, y.shape)
#plt.pcolormesh(x, y, Ai, cmap="jet")
#plt.pcolormesh(x, y, Ar, cmap="jet")
plt.pcolormesh(x, y, Ar**2 + Ai**2, cmap="jet")
plt.show()

# %%
#filename = "outputs/BR_mu0.04/"
filename = "outputs/euler1/"
times, fields = ReadAllDataFilesInFolder(filename)
# %%
times['x'][2] - times['x'][1]

# %%
times['C1'].shape

# %%
C1 =np.transpose(times['C1'], (2,0,1))
C2 =np.transpose(times['C2'], (2,0,1))
# %%
C1.shape
C2.shape
# %%
c1t = np.gradient(C1,1, axis=0)
c1xx = np.gradient(np.gradient(C1,0.5, axis=1),0.5 ,axis=1)
c1yy = np.gradient(np.gradient(C1,0.5, axis=2),0.5, axis=2)

# %%
mu = 0.04
k = 1
ka = 4.5
D = 8
nu = np.sqrt(1/D)
kbcrit = np.sqrt(1+ka*nu)
kb = kbcrit*(1+mu)

rhs = c1xx + c1yy + (ka - (kb+1)*C1 + np.power(C1,2)*C2)

# %%
diff = c1t[10:100] - rhs[11:101]

# %%
diff[5]#.max()

# %%
#np.save('C1_mu_0.1_128_64_0.1.npy', C1)
#np.save('C2_mu_0.1_128_64_0.1.npy', C2)
# %%
