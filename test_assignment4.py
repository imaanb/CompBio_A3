
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm

nx = ny = 50
dx = dy = 1.0
D_v = 1.0
mu_v = 0.1
dt = 1.0
steps = 100

infection_threshold = 0.2
infection_prob = 0.2
emission_rate = 1.0
removal_prob = 0.1

# Initialize grid
sir_grid = np.full((nx, ny), 0)
num_initial_infected = 10

random_indices = np.random.choice(nx * ny, size=num_initial_infected, replace=False)

for idx in random_indices:
    i, j = divmod(idx, ny)
    sir_grid[i, j] = 1


# Setup FVM
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)

x, y = mesh.cellCenters

eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)

# Plot figures
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


custom =  ListedColormap(['lightgreen', 'orangered', 'black'])
im1 = ax1.imshow(sir_grid, cmap=custom, origin='lower', vmin=0, vmax=2)
ax1.set_title("ABM, t=0")


virus_data = V.value.reshape((nx, ny)).T
im2 = ax2.imshow(virus_data, origin='lower', cmap='viridis', vmin=0, vmax=3.5)
ax2.set_title("PDE")

cbar = fig.colorbar(im2, ax=ax2, label='Virus concentration')

def plot_state(t):
    # Update ABM
    im1.set_data(sir_grid)
    ax1.set_title(f"ABM, t={t}")

    # Update virus field
    virus_data = V.value.reshape((nx, ny)).T
    im2.set_data(virus_data)

    # Refresh plot
    plt.pause(0.01)


# Calculations
for step in range(steps):
    t = step * dt
    
    # Agent → virus field
    for i in range(nx):
        for j in range(ny):
            if sir_grid[i, j] == 1:
                idx = i * ny + j
                V[idx] += emission_rate * dt

    # Diffusion step
    V.updateOld()
    eq.solve(var=V, dt=dt)

    # Update ABM
    for i in range(nx):
        for j in range(ny):
            idx = i + j * nx
            if sir_grid[i, j] == 0 and V[idx] > infection_threshold:
                if np.random.rand() < infection_prob:
                    sir_grid[i, j] = 1
                    
            if sir_grid[i, j] == 1:
                if np.random.rand() < removal_prob:
                    sir_grid[i, j] = 2

    # Plotting
    if step % 5 == 0:
        plot_state(t)

plt.ioff()
plt.show()
