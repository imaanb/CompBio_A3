import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm

def run_simulation_core(
    N_ROWS, N_COLS, dx, dy, D_v, mu_v, dt, steps,
    infection_threshold, infection_prob, emission_rate, removal_prob,
    num_initial_infected,
    seed_value,
    neighbor_dict,
    macrophage_params=None, # Dictionary: {'count': int, 'degradation_rate': float} or None
    return_full_history=False 
):
    np.random.seed(seed_value)

    sir_grid = np.full((N_ROWS, N_COLS), 0) # 0=S, 1=I, 2=R

    infected_flat_indices = np.random.choice(N_ROWS * N_COLS, size=num_initial_infected, replace=False)
    for flat_idx in infected_flat_indices:
        r, c = divmod(flat_idx, N_COLS)
        sir_grid[r, c] = 1

    macrophage_positions = []
    degradation_rate = 0.0
    actual_num_macrophages = 0

    if macrophage_params and macrophage_params.get('count', 0) > 0 : # Check if count is positive
        actual_num_macrophages = int(macrophage_params['count'])
        degradation_rate = macrophage_params['degradation_rate']
        
        if actual_num_macrophages > 0:
            # Simple random placement, could be on infected cells
            macrophage_flat_indices = np.random.choice(N_ROWS * N_COLS, size=actual_num_macrophages, replace=False)
            for flat_idx in macrophage_flat_indices:
                r, c = divmod(flat_idx, N_COLS)
                macrophage_positions.append((r, c))
        else: 
            macrophage_params = None # Effectively disable macrophages

    mesh = Grid2D(dx=dx, dy=dy, nx=N_COLS, ny=N_ROWS)
    V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)
    eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)

    S_counts, I_counts, R_counts = [], [], []
    
    # For storing history if requested
    history_sir_grid = []
    history_V_field = []
    history_macrophage_pos = []

    for step_num in range(steps):

        # Store history for animation purposes
        if return_full_history:
            history_sir_grid.append(sir_grid.copy())
            history_V_field.append(V.value.copy().reshape((N_ROWS, N_COLS))) # Store reshaped
            history_macrophage_pos.append(list(macrophage_positions)) # Copy list of tuples

        S_counts.append(np.sum(sir_grid == 0))
        I_counts.append(np.sum(sir_grid == 1))
        R_counts.append(np.sum(sir_grid == 2))

        # Virus Emission
        for r in range(N_ROWS):
            for c in range(N_COLS):
                if sir_grid[r, c] == 1:
                    fipy_idx = c + r * N_COLS
                    V[fipy_idx] += emission_rate * dt

        # Virus Diffusion & Decay
        V.updateOld()
        eq.solve(var=V, dt=dt)
        V.setValue(np.maximum(0, V.value))

        # Infection & Death
        for r in range(N_ROWS):
            for c in range(N_COLS):
                fipy_idx = c + r * N_COLS
                current_virus_level = V[fipy_idx]

                if sir_grid[r, c] == 0 and current_virus_level > infection_threshold:
                    if np.random.rand() < infection_prob:
                        sir_grid[r, c] = 1
                elif sir_grid[r, c] == 1:
                    if np.random.rand() < removal_prob:
                        sir_grid[r, c] = 2

        # Macrophage Movement & Degradation
        if macrophage_params and actual_num_macrophages > 0 and macrophage_positions:
            new_macrophage_positions = []
            for r_macro, c_macro in macrophage_positions:
                best_r, best_c = r_macro, c_macro
                fipy_idx_current = c_macro + r_macro * N_COLS
                max_val = V[fipy_idx_current]

                for nr, nc in neighbor_dict.get((r_macro, c_macro), []): # Use .get for safety
                    fipy_idx_neighbor = nc + nr * N_COLS
                    if V[fipy_idx_neighbor] > max_val:
                        best_r, best_c = nr, nc
                        max_val = V[fipy_idx_neighbor]
                
                fipy_idx_degrade = best_c + best_r * N_COLS
                V[fipy_idx_degrade] = max(V[fipy_idx_degrade] - degradation_rate * dt, 0.0)
                new_macrophage_positions.append((best_r, best_c))
            macrophage_positions = new_macrophage_positions
            
    if return_full_history:
        return S_counts, I_counts, R_counts, history_sir_grid, history_V_field, history_macrophage_pos
    else:
        return S_counts, I_counts, R_counts







#
#
#
#
#
#import numpy as np
#from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
#
#def run_simulation_core(
#    N_ROWS, N_COLS, dx, dy, D_v, mu_v, dt, steps,
#    infection_threshold, infection_prob, emission_rate, removal_prob,
#    num_initial_infected,
#    seed_value,
#    neighbor_dict,
#    macrophage_params=None # Dictionary: {'count': int, 'degradation_rate': float} or None
#):
#    np.random.seed(seed_value)
#
#    sir_grid = np.full((N_ROWS, N_COLS), 0) # 0=S, 1=I, 2=R
#
#    infected_flat_indices = np.random.choice(N_ROWS * N_COLS, size=num_initial_infected, replace=False)
#    for flat_idx in infected_flat_indices:
#        r, c = divmod(flat_idx, N_COLS)
#        sir_grid[r, c] = 1
#
#    macrophage_positions = []
#    degradation_rate = 0.0
#    actual_num_macrophages = 0
#
#    if macrophage_params:
#        actual_num_macrophages = macrophage_params['count']
#        degradation_rate = macrophage_params['degradation_rate']
#        macrophage_flat_indices = np.random.choice(N_ROWS * N_COLS, size=int(actual_num_macrophages), replace=False)
#        for flat_idx in macrophage_flat_indices:
#            r, c = divmod(flat_idx, N_COLS)
#            macrophage_positions.append((r, c))
#
#    mesh = Grid2D(dx=dx, dy=dy, nx=N_COLS, ny=N_ROWS)
#    V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)
#    eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)
#
#    S_counts, I_counts, R_counts = [], [], []
#
#    for _ in range(steps):
#        S_counts.append(np.sum(sir_grid == 0))
#        I_counts.append(np.sum(sir_grid == 1))
#        R_counts.append(np.sum(sir_grid == 2))
#
#        for r in range(N_ROWS):
#            for c in range(N_COLS):
#                if sir_grid[r, c] == 1:
#                    fipy_idx = c + r * N_COLS
#                    V[fipy_idx] += emission_rate * dt
#
#        V.updateOld()
#        eq.solve(var=V, dt=dt)
#        V.setValue(np.maximum(0, V.value)) # Ensures V >= 0
#
#        for r in range(N_ROWS):
#            for c in range(N_COLS):
#                fipy_idx = c + r * N_COLS
#                current_virus_level = V[fipy_idx]
#
#                if sir_grid[r, c] == 0 and current_virus_level > infection_threshold:
#                    if np.random.rand() < infection_prob:
#                        sir_grid[r, c] = 1
#                elif sir_grid[r, c] == 1:
#                    if np.random.rand() < removal_prob:
#                        sir_grid[r, c] = 2
#
#        if macrophage_params and actual_num_macrophages > 0:
#            new_macrophage_positions = []
#            for r_macro, c_macro in macrophage_positions:
#                best_r, best_c = r_macro, c_macro
#                fipy_idx_current = c_macro + r_macro * N_COLS
#                max_val = V[fipy_idx_current]
#
#                for nr, nc in neighbor_dict[(r_macro, c_macro)]:
#                    fipy_idx_neighbor = nc + nr * N_COLS
#                    if V[fipy_idx_neighbor] > max_val:
#                        best_r, best_c = nr, nc
#                        max_val = V[fipy_idx_neighbor]
#
#                fipy_idx_degrade = best_c + best_r * N_COLS
#                V[fipy_idx_degrade] = max(V[fipy_idx_degrade] - degradation_rate * dt, 0.0)
#                new_macrophage_positions.append((best_r, best_c))
#            macrophage_positions = new_macrophage_positions
#
#    return S_counts, I_counts, R_counts
