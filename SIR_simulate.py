import tqdm
import pickle
from SIR_ABM_macrophages import run_simulation_core 
import argparse

# Parameters
nx = 50 
ny = 50 
dx = 1
dy = 1
D_v = 1
mu_v = 0.1
dt = 1
steps = 100

I_T = 0.2
P_I = 0.2
f_ij = 1.0
P_R = 0.1
delta_MV = 1.5

I_0 = 15

# Let the user select the number of macrophages. This was done for ease of use for the 
# teachers. Now they can call just one file "run.sh" to run all simulations and generate all plots.
parser = argparse.ArgumentParser(description="Run SIR simulation with a specified number of macrophages.")
parser.add_argument(
    "--macrophages", 
    type=int,        
    required=True,  
    help="Number of macrophages to simulate."
)

args = parser.parse_args() 

num_macrophages_fixed_param = args.macrophages


# Pre-calculate neighbors
neighbor_dict = {}
for r in range(nx): 
    for c in range(ny): 
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nx and 0 <= nc < ny:
                neighbors.append((nr, nc))
        neighbor_dict[(r, c)] = neighbors

runs = 20
sir_results_all_runs = {"S":[], "I": [], "R": []}

print(f"Starting simulations for {num_macrophages_fixed_param} macrophages...")
for run_idx in tqdm.tqdm(range(runs), desc=f"Run for {num_macrophages_fixed_param} Macrophages"):
    current_seed = run_idx

    mac_params = {
        'count': num_macrophages_fixed_param,
        'degradation_rate': delta_MV
    }

    S_series, I_series, R_series = run_simulation_core(nx, ny, dx, dy, D_v, mu_v, dt, steps, I_T, P_I, f_ij, P_R, I_0, current_seed, neighbor_dict, mac_params)

    sir_results_all_runs["S"].append(S_series)
    sir_results_all_runs["I"].append(I_series)
    sir_results_all_runs["R"].append(R_series)

# Save data
pickle_filename_save = f"sirdict_{num_macrophages_fixed_param}_macrophages.pkl"
with open(pickle_filename_save, "wb") as f:
    pickle.dump(sir_results_all_runs, f)
print("Simulations finished and data saved.")
































