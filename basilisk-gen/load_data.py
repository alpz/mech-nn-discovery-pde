import numpy as np
import matplotlib.pyplot as plt
import os
from functions_python import *

filename = "outputs/GL_beta1.5/MeshDump-N620.bin"

fields = ReadDataFile(filename)

x = fields["x"]
y = fields["y"]
Ai = fields["Ai"]
A2 = fields["A2"]

print(x.shape, y.shape)
plt.pcolormesh(x, y, Ai, cmap="jet")
plt.show()