
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm

# Parameters
nx = ny = 50
dx = dy = 1.0
D_v = 1.0
mu_v = 0.1
dt = 1.0
steps = 100

infection_threshold = 0.2
infection_prob = 0.2
emission_rate = 1
removal_prob = 0.1
degradation_rate = 1.5  # macrophage virus degradation

# ABM states: 0 = S, 1 = I, 2 = R
sir_grid = np.full((nx, ny), 0)
num_initial_infected = 15
num_macrophages = 55

# Infect some agents
infected_indices = np.random.choice(nx * ny, size=num_initial_infected, replace=False)
for idx in infected_indices:
    i, j = divmod(idx, ny)
    sir_grid[i, j] = 1

# Initialize macrophage positions
macrophage_positions = []
macrophage_indices = np.random.choice(nx * ny, size=num_macrophages, replace=False)
for idx in macrophage_indices:
    i, j = divmod(idx, ny)
    macrophage_positions.append((i, j))

# PDE setup
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)
x, y = mesh.cellCenters
eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)

# Plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cmap = ListedColormap(['lightgreen', 'orangered', 'black'])  # 0=S, 1=I, 2=R
im1 = ax1.imshow(sir_grid, cmap=cmap, origin='lower', vmin=0, vmax=2)
ax1.set_title("ABM, t=0")

virus_data = V.value.reshape((nx, ny)).T
im2 = ax2.imshow(virus_data, origin='lower', cmap='viridis', vmin=0, vmax=3.5)
ax2.set_title("Virus Field")
cbar = fig.colorbar(im2, ax=ax2, label='Virus concentration')

macrophage_plot, = ax1.plot([], [], 'wo', markersize=5)

def plot_state(t):
    im1.set_data(sir_grid)
    ax1.set_title(f"ABM, t={t}")
    virus_data = V.value.reshape((nx, ny)).T
    im2.set_data(virus_data)

    # Plot macrophages as white circles
    ys, xs = zip(*macrophage_positions)
    macrophage_plot.set_data(xs, ys)

    plt.pause(0.01)

# Helper to get von Neumann neighbors
def get_neighbors(i, j):
    return [(i + di, j + dj) for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]
            if 0 <= i + di < nx and 0 <= j + dj < ny]

# Main loop
for step in range(steps):
    t = step * dt

    # Emit virus from infected
    for i in range(nx):
        for j in range(ny):
            if sir_grid[i, j] == 1:
                idx = j * nx + i
                V[idx] += emission_rate * dt

    # Virus diffusion
    V.updateOld()
    eq.solve(var=V, dt=dt)

    # Infection + Recovery
    for i in range(nx):
        for j in range(ny):
            idx = j * nx + i
            if sir_grid[i, j] == 0 and V[idx] > infection_threshold:
                if np.random.rand() < infection_prob:
                    sir_grid[i, j] = 1
            elif sir_grid[i, j] == 1:
                if np.random.rand() < removal_prob:
                    sir_grid[i, j] = 2

    # Macrophage movement + virus degradation
    new_macrophage_positions = []
    for i, j in macrophage_positions:
        best_i, best_j = i, j
        max_val = V[j * nx + i]

        for ni, nj in get_neighbors(i, j):
            idx = nj * nx + ni
            if V[idx] > max_val:
                best_i, best_j = ni, nj
                max_val = V[idx]

        # Degrade virus at new position
        idx = best_j * nx + best_i
        V[idx] = max(V[idx] - degradation_rate * dt, 0.0)

        new_macrophage_positions.append((best_i, best_j))

    macrophage_positions = new_macrophage_positions

    # Plotting
    if step % 5 == 0:
        plot_state(t)

plt.ioff()
plt.show()
