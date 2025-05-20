
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
macrophages_numbers = np.linspace(1, 201, 11).astype(int)

score_dict = {}
print(score_dict)


"""
for macrophage_number in macrophages_numbers: 
    score_dict[macrophage_number] = []
    for run in tqdm.tqdm(range(runs), desc=f"Macrophages: {macrophage_number}"): 
        S =  [] 
        # ABM states: 0 = S, 1 = I, 2 = R
        sir_grid = np.full((nx, ny), 0)
        num_initial_infected = 15
        num_macrophages = macrophage_number

        # Infect some agents
        infected_indices = np.random.choice(nx * ny, size=num_initial_infected, replace=False)
        for idx in infected_indices:
            i, j = divmod(idx, ny)
            sir_grid[i, j] = 1

        # Initialize macrophage positions
        macrophage_positions = []
        macrophage_indices = np.random.choice(nx * ny, size=int(num_macrophages), replace=False)
       
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

        score_dict[macrophage_number].append(S)

with open("score_dict.pkl", "wb") as f:
    pickle.dump(score_dict, f)
"""


with open("score_dict.pkl", "rb") as f:
    score_dict = pickle.load(f)

time_axis = np.linspace(0, steps, steps)
plt.figure(figsize=(10,6))

for macrophage_number in macrophages_numbers:
    S_runs = np.array(score_dict[macrophage_number])  
    mean_S = np.mean(S_runs, axis=0)
    plt.plot(time_axis, mean_S, label=f"{macrophage_number}", color=plt.cm.coolwarm((macrophage_number - macrophages_numbers.min()) / (macrophages_numbers.max() - macrophages_numbers.min()))
)

plt.xlabel("Timesteps")
plt.ylabel("Number of Susceptible cells")
plt.title("Susceptible cells over time for different numbers of macrophages")
plt.legend(title=f"Number of Macrophages in the system ")
plt.grid()
plt.tight_layout()
plt.savefig("sir_plot_S_vs_macrophages.png", dpi=300)
plt.show()