
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
import tqdm
import pickle
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

"""
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

"""
neighbor_dict = {}
for i in range(nx):
    for j in range(ny):
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                neighbors.append((ni, nj))
        neighbor_dict[(i, j)] = neighbors



runs = 20 


score_dict = {"S":[], "I": [], "R": []}


for run in range(runs): 
    S, I, R = [], [], [] 
    print("Run ", run )
    # ABM states: 0 = S, 1 = I, 2 = R
    sir_grid = np.full((nx, ny), 0)
    num_initial_infected = 15
    num_macrophages = 20

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


    for step in range(steps):
        S.append(np.sum(sir_grid == 0))
        I.append(np.sum(sir_grid == 1))
        R.append(np.sum(sir_grid == 2))
        


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

            for ni, nj in neighbor_dict[(i, j)]:
                idx = nj * nx + ni
                if V[idx] > max_val:
                    best_i, best_j = ni, nj
                    max_val = V[idx]

            # Degrade virus at new position
            idx = best_j * nx + best_i
            V[idx] = max(V[idx] - degradation_rate * dt, 0.0)

            new_macrophage_positions.append((best_i, best_j))

        macrophage_positions = new_macrophage_positions

    score_dict["S"].append(S)
    score_dict["I"].append(I)
    score_dict["R"].append(R)

with open("sirdict.pkl", "wb") as f:
    pickle.dump(score_dict, f)



with open("sirdict.pkl", "rb") as f:
    score_dict = pickle.load(f)



time_axis = np.linspace(0, steps, steps)
plt.figure(figsize=(10,6))

for i in range(runs):
    plt.plot(time_axis, score_dict["S"][i],alpha = .1, color = "green")
    plt.plot(time_axis, score_dict["I"][i], alpha = .1, color = "red" )
    plt.plot(time_axis, score_dict["R"][i], alpha = .1, color = "black")


plt.plot(time_axis, np.mean(score_dict["S"], axis=0), label="S", color="green", linewidth=2)
plt.plot(time_axis, np.mean(score_dict["I"], axis=0), label="I", color="red", linewidth=2)
plt.plot(time_axis, np.mean(score_dict["R"], axis=0), label="R", color="black", linewidth=2)
plt.legend()
plt.title(f"Susceptible, Infected and Removed cells over time for {num_macrophages} macrophages")
plt.grid()
plt.savefig(f"sir_plot_{num_macrophages}_macrophages", dpi = 300)