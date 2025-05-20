import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm

# --- Parameters ---
SEED_VALUE = 42 # For reproducibility
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

num_initial_infected = 15

# --- Initialization ---
np.random.seed(SEED_VALUE)

sir_grid = np.full((ny, nx), 0) # (rows, cols) -> (ny, nx)
infected_indices = np.random.choice(nx * ny, size=num_initial_infected, replace=False)
for idx in infected_indices:
    r_idx, c_idx = divmod(idx, nx) # row, col
    sir_grid[r_idx, c_idx] = 1

mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)
eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)

# --- Plotting Setup ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap_sir = ListedColormap(['lightgreen', 'orangered', 'black'])
im1 = ax1.imshow(sir_grid, cmap=cmap_sir, origin='lower', vmin=0, vmax=2)
ax1.set_title("ABM (No Macrophages), t=0")

# FiPy's V.value is 1D (x varies fastest). Reshape to (ny, nx) for (rows, cols)
virus_data_display = V.value.reshape((ny, nx))
im2 = ax2.imshow(virus_data_display, origin='lower', cmap='viridis', vmin=0, vmax=3.5)
ax2.set_title("Virus Field")
cbar = fig.colorbar(im2, ax=ax2, label='Virus concentration')

def plot_state(t_current):
    im1.set_data(sir_grid)
    ax1.set_title(f"ABM (No Macrophages), t={t_current:.1f}")
    virus_data_display = V.value.reshape((ny, nx))
    im2.set_data(virus_data_display)
    current_max_virus = np.max(virus_data_display)
    if current_max_virus > 0:
        im2.set_clim(vmin=0, vmax=max(current_max_virus, 0.1))
    plt.pause(0.01)

# --- Main Loop ---
for step in range(steps):
    t = step * dt

    for r_idx in range(ny):
        for c_idx in range(nx):
            if sir_grid[r_idx, c_idx] == 1:
                fipy_idx = c_idx + r_idx * nx
                V[fipy_idx] += emission_rate * dt

    V.updateOld()
    eq.solve(var=V, dt=dt)
    V.setValue(np.maximum(0, V.value), where=V.value < 0)

    for r_idx in range(ny):
        for c_idx in range(nx):
            fipy_idx = c_idx + r_idx * nx
            current_virus_level = V[fipy_idx]
            if sir_grid[r_idx, c_idx] == 0 and current_virus_level > infection_threshold:
                if np.random.rand() < infection_prob:
                    sir_grid[r_idx, c_idx] = 1
            elif sir_grid[r_idx, c_idx] == 1:
                if np.random.rand() < removal_prob:
                    sir_grid[r_idx, c_idx] = 2

    if step % 5 == 0 or step == steps - 1:
        plot_state(t)

plt.ioff()
plt.show()
