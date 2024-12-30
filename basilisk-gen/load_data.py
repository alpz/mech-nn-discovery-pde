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
filename = "outputs/GL_beta1.5/"
times, fields = ReadAllDataFilesInFolder(filename)
# %%
times['x'][2] - times['x'][1]

# %%
fields

# %%
Ai =np.transpose(times['Ai'], (2,0,1))
Ar =np.transpose(times['Ar'], (2,0,1))
# %%
Ai.shape
Ar.shape
# %%
np.save('Ai_256.npy', Ai)
np.save('Ar_256.npy', Ar)
# %%
