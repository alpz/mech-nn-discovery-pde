#%%

from pde import DiffusionPDE, ScalarField, UnitGrid, MemoryStorage, CartesianGrid

#%%
grid = UnitGrid([64, 64])  # generate grid
#grid = CartesianGrid([(0,63),(0,63)],[64, 64])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition
print(state.data.shape)
#%%

eq = DiffusionPDE(diffusivity=0.1)  # define the pde
#result = eq.solve(state, t_range=10)

storage = MemoryStorage()  # store intermediate information of the simulation
result = eq.solve(state, 128, dt=5e-2, tracker=storage.tracker(1))  # solve the PDE
#result.plot()
# %%


result.plot()
result.data.shape
# %%

len(storage.data)
# %%
import matplotlib.pyplot as plt

from IPython.display import HTML
from matplotlib.animation import FuncAnimation


data_list = storage.data
plot = plt.pcolormesh(data_list[20], cmap='viridis', shading='gouraud')
# %%

import numpy as np
fig, ax = plt.subplots(1,1)


x = np.arange(64)
y = np.arange(64)
cax0 = ax.pcolormesh(data_list[-1], cmap='viridis', shading='gouraud')

def animate(i):
    #cax0.set_array(data_list[i][:-1,:-1].ravel())
    cax0.set_array(data_list[i].ravel())
    #cax1.set_array(func_list[i].reshape(*coord_dims).flatten())

anim = FuncAnimation(fig, animate, interval=40, frames=100)
HTML(anim.to_html5_video())
# %%

grid.cell_volume_data
data_c = np.stack(data_list, axis=0)
data_c.shape
# %%
import numpy as np
np.save('data/diffusion_01.npy', data_c)
# %%
