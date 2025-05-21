import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
from SIR_ABM_macrophages import run_simulation_core 

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

# Pre-calculate neighbor_dict (keys are (row, col))
neighbor_dict = {}
for r in range(nx_param): 
    for c in range(ny_param): 
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nx_param and 0 <= nc < ny_param:
                neighbors.append((nr, nc))
        neighbor_dict[(r, c)] = neighbors

runs = 20
macrophages_numbers_to_test = np.linspace(1, 201, 11).astype(int)

score_dict_S_counts = {}

print("Starting simulations")
for num_macrophages_current_run in macrophages_numbers_to_test:
    score_dict_S_counts[num_macrophages_current_run] = []
    for run_idx in tqdm.tqdm(range(runs), desc=f"Macrophages: {num_macrophages_current_run}"):
         current_seed = run_idx

         mac_params = {
             'count': int(num_macrophages_current_run),
             'degradation_rate': degradation_rate_param
         }

         S_series, _I_series, _R_series = run_simulation_core(
             nx=nx_param, ny=ny_param, dx=dx, dy=dy, D_v=D_v, mu_v=mu_v, dt=dt, steps=steps,
             I_T=infection_threshold, P_I=infection_prob,
             f_ij=emission_rate, P_R=removal_prob,
             I_0=num_initial_infected_param,
             seed_value=current_seed,
             neighbor_dict=neighbor_dict,
             macrophage_params=mac_params
         )
         score_dict_S_counts[num_macrophages_current_run].append(S_series)

with open("score_dict_S_vs_macrophages.pkl", "wb") as f:
    pickle.dump(score_dict_S_counts, f)
print("Simulations finished and data saved.")


