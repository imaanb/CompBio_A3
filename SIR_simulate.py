import tqdm
import pickle
from SIR_ABM_macrophages import run_simulation_core # Import the core function
import argparse

nx_param = 50 
ny_param = 50 
dx = dy = 1.0
D_v = 1.0
mu_v = 0.1
dt = 1.0
steps = 100

infection_threshold = 0.2
infection_prob = 0.2
emission_rate = 1.0
removal_prob = 0.1
degradation_rate_param = 1.5

num_initial_infected_param = 15

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



# Pre-calculate neighbor_dict
neighbor_dict = {}
for r in range(nx_param): # Original i
    for c in range(ny_param): # Original j
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nx_param and 0 <= nc < ny_param:
                neighbors.append((nr, nc))
        neighbor_dict[(r, c)] = neighbors

runs = 20
sir_results_all_runs = {"S":[], "I": [], "R": []}

print(f"Starting simulations for {num_macrophages_fixed_param} macrophages...")
for run_idx in tqdm.tqdm(range(runs), desc=f"Run for {num_macrophages_fixed_param} Macrophages"):
    current_seed = run_idx

    mac_params = {
        'count': num_macrophages_fixed_param,
        'degradation_rate': degradation_rate_param
    }

    S_series, I_series, R_series = run_simulation_core(
        N_ROWS=nx_param, N_COLS=ny_param, dx=dx, dy=dy, D_v=D_v, mu_v=mu_v, dt=dt, steps=steps,
        infection_threshold=infection_threshold, infection_prob=infection_prob,
        emission_rate=emission_rate, removal_prob=removal_prob,
        num_initial_infected=num_initial_infected_param,
        seed_value=current_seed,
        neighbor_dict=neighbor_dict,
        macrophage_params=mac_params
    )
    sir_results_all_runs["S"].append(S_series)
    sir_results_all_runs["I"].append(I_series)
    sir_results_all_runs["R"].append(R_series)

pickle_filename_save = f"sirdict_{num_macrophages_fixed_param}_macrophages.pkl"
with open(pickle_filename_save, "wb") as f:
    pickle.dump(sir_results_all_runs, f)
print("Simulations finished and data saved.")
































