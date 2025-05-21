import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from numba import njit, types
import numba.typed

@njit
def virus_emission(sir_grid, V_array, f_ij, dt, nx, ny):
    for r in range(nx):
        for c in range(ny):
            if sir_grid[r, c] == 1:
                index = c + r * ny
                V_array[index] += f_ij * dt

@njit
def enforce_non_negative(V_array):
    for i in range(V_array.shape[0]):
        if V_array[i] < 0:
            V_array[i] = 0.0

@njit
def infection_death(sir_grid, V_array, I_T, P_I, P_R, nx, ny):
    for r in range(nx):
        for c in range(ny):
            index = c + r * ny
            current_virus_level = V_array[index]

            if sir_grid[r, c] == 0 and current_virus_level > I_T:
                if np.random.rand() < P_I:
                    sir_grid[r, c] = 1
            elif sir_grid[r, c] == 1:
                if np.random.rand() < P_R:
                    sir_grid[r, c] = 2

@njit
def macrophage_update(
    current_macrophage_positions_typed, V_array, typed_neighbor_dict,
    delta_MV, dt, nx, ny
):
    _dummy_int64_tuple = (np.int64(0), np.int64(0)) 

    new_macrophage_positions_typed = numba.typed.List([_dummy_int64_tuple])
    new_macrophage_positions_typed.clear()

    default_empty_neighbor_list = numba.typed.List([_dummy_int64_tuple])
    default_empty_neighbor_list.clear()



    for r_macro, c_macro in current_macrophage_positions_typed:
        best_r, best_c = r_macro, c_macro
        current_index = c_macro + r_macro * ny
        max_val = V_array[current_index]

        neighbors = typed_neighbor_dict.get((r_macro, c_macro), default_empty_neighbor_list)
        for nr, nc in neighbors:
            index_neighbor = nc + nr * ny
            if V_array[index_neighbor] > max_val:
                best_r, best_c = nr, nc
                max_val = V_array[index_neighbor]
        
        index_degrade = best_c + best_r * ny
        V_array[index_degrade] = max(V_array[index_degrade] - delta_MV * dt, 0.0)
        new_macrophage_positions_typed.append((best_r, best_c))
    return new_macrophage_positions_typed

def run_simulation_core(
    nx, ny, dx, dy, D_v, mu_v, dt, steps, I_T, P_I, f_ij, P_R, I_0, seed_value, neighbor_dict,
    macrophage_params=None, return_full_history=False
):
    np.random.seed(seed_value)
    _ = np.random.rand()    
    sir_grid = np.full((nx, ny), 0)

    infected_flat_indices = np.random.choice(nx * ny, size=I_0, replace=False)
    for flat_idx in infected_flat_indices:
        r, c = divmod(flat_idx, ny)
        sir_grid[r, c] = 1

    macrophage_positions_typed = numba.typed.List.empty_list(types.UniTuple(types.int64, 2))
    typed_neighbor_dict = numba.typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.ListType(types.UniTuple(types.int64, 2))
    )
    delta_MV = 0.0
    actual_num_macrophages = 0
    
    active_macrophages = False
    if macrophage_params and macrophage_params.get('count', 0) > 0:
        actual_num_macrophages = int(macrophage_params['count'])
        if actual_num_macrophages > 0:
            active_macrophages = True
            delta_MV = macrophage_params['degradation_rate']
            
            macrophage_flat_indices = np.random.choice(nx * ny, size=actual_num_macrophages, replace=False)
            temp_py_list_mac_pos = []
            for flat_idx in macrophage_flat_indices:
                r, c = divmod(flat_idx, ny)
                temp_py_list_mac_pos.append((r,c))
            for r_val, c_val in temp_py_list_mac_pos:
                macrophage_positions_typed.append((r_val, c_val))

            for py_key, py_value_list in neighbor_dict.items():
                typed_value_list = numba.typed.List.empty_list(types.UniTuple(types.int64, 2))
                for item_tuple in py_value_list:
                    typed_value_list.append(item_tuple)
                typed_neighbor_dict[py_key] = typed_value_list
        else:
             macrophage_params = None # Effectively disable
    else:
        macrophage_params = None


    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)
    eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)

    S_counts, I_counts, R_counts = [], [], []
    
    history_sir_grid = []
    history_V_field = []
    history_macrophage_pos = []

    V_numpy_array = V.value

    for step_num in range(steps):

        if return_full_history:
            history_sir_grid.append(sir_grid.copy())
            history_V_field.append(V_numpy_array.copy().reshape((nx, ny)))
            if active_macrophages:
                py_list_for_hist = []
                for i in range(len(macrophage_positions_typed)):
                    py_list_for_hist.append(macrophage_positions_typed[i])
                history_macrophage_pos.append(py_list_for_hist)
            else:
                history_macrophage_pos.append([])


        S_counts.append(np.sum(sir_grid == 0))
        I_counts.append(np.sum(sir_grid == 1))
        R_counts.append(np.sum(sir_grid == 2))

        virus_emission(sir_grid, V_numpy_array, f_ij, dt, nx, ny)

        V.updateOld()
        eq.solve(var=V, dt=dt)
        enforce_non_negative(V_numpy_array)

        infection_death(sir_grid, V_numpy_array, I_T, P_I, P_R, nx, ny)

        if active_macrophages and len(macrophage_positions_typed) > 0:
            macrophage_positions_typed = macrophage_update(
                macrophage_positions_typed, V_numpy_array, typed_neighbor_dict,
                delta_MV, dt, nx, ny
            )
            
    if return_full_history:
        return S_counts, I_counts, R_counts, history_sir_grid, history_V_field, history_macrophage_pos
    else:
        return S_counts, I_counts, R_counts