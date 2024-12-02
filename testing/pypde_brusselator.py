# %%
from pde import PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid, MemoryStorage

# define the PDE
a, b = 1, 3
d0, d1 = 1, 0.1
eq = PDE(
    {
        "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
        "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
    }
)

# initialize state
grid = UnitGrid([64, 64])
u = ScalarField(grid, a, label="Field $u$")
v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
state = FieldCollection([u, v])

# simulate the pde
#tracker = PlotTracker(interrupts=1, plot_args={"vmin": 0, "vmax": 5})

storage = MemoryStorage()  # store intermediate information of the simulation
#sol = eq.solve(state, t_range=20, dt=1e-3, tracker=tracker)
sol = eq.solve(state, t_range=100, dt=1e-3, tracker=storage.tracker(1))
# %%
print(sol.data.shape)
# %%
len(storage.data)
# %%
# %%
import matplotlib.pyplot as plt

from IPython.display import HTML
from matplotlib.animation import FuncAnimation


data_list = storage.data
data_list[0].shape
plot = plt.pcolormesh(data_list[20][1], cmap='viridis', shading='gouraud')
# %%

import numpy as np
fig, ax = plt.subplots(1,1)


x = np.arange(64)
y = np.arange(64)
cax0 = ax.pcolormesh(data_list[-1][1], cmap='viridis', shading='gouraud')

def animate(i):
    #cax0.set_array(data_list[i][:-1,:-1].ravel())
    cax0.set_array(data_list[i][1].ravel())
    #cax1.set_array(func_list[i].reshape(*coord_dims).flatten())

anim = FuncAnimation(fig, animate, interval=100, frames=100)
HTML(anim.to_html5_video())
# %%

grid.cell_volume_data
data_c = np.stack(data_list, axis=0)
data_c.shape
# %%
import numpy as np
np.save('data/brusselator_01_1en3.npy', data_c)
# %%
