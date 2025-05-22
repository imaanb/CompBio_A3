"""
Course: Computational Biology
Names: Lisa Pijpers, Petr Chalupský and Imaan Bijl
Student IDs: 15746704, 15719227 and 15395812

File description:
    This file contains the core simulation function for the SIR model with optional
    macrophages. It uses the FiPy library for solving partial differential equations.
"""

import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from numba import njit, types
import numba.typed

@njit
def _numba_virus_emission(sir_grid, V_array, emission_rate, dt, N_ROWS, N_COLS):
    """
    Function to update the virus concentration in the grid based on the infected cells.

    Args:
        sir_grid (array): 2D array representing the SIR grid (S=0, I=1, R=2).
        V_array (array): 1D array representing the virus concentration.
        emission_rate (float): Rate of virus emission from infected cells.
        dt (float): Time step for the simulation.
        N_ROWS (int): Number of rows in the grid.
        N_COLS (int): Number of columns in the grid.
    """
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if sir_grid[r, c] == 1: # Check if cell is infected
                fipy_idx = c + r * N_COLS # Calculate the index of V array
                V_array[fipy_idx] += emission_rate * dt # Update virus concentration

@njit
def _numba_enforce_non_negative(V_array):
    """
    Function to ensure that the virus concentration is non-negative.

    Args:
        V_array (array): 1D array representing the virus concentration.
    """
    for i in range(V_array.shape[0]):
        if V_array[i] < 0:
            V_array[i] = 0.0 # Set negative virus concentration to zero

@njit
def _numba_infection_death(sir_grid, V_array, infection_threshold, infection_prob, removal_prob, N_ROWS, N_COLS):
    """
    Function to update the SIR grid based on the virus concentration and infection parameters.
    
    Args:
        sir_grid (array): 2D array representing the SIR grid (S=0, I=1, R=2).
        V_array (array): 1D array representing the virus concentration.
        infection_threshold (float): Virus concentration threshold for infection.
        infection_prob (float): Probability of infection.
        removal_prob (float): Probability of removal.
        N_ROWS (int): Number of rows in the grid.
        N_COLS (int): Number of columns in the grid.
    """

    for r in range(N_ROWS):
        for c in range(N_COLS):
            fipy_idx = c + r * N_COLS # Calculate the index of V array
            current_virus_level = V_array[fipy_idx] # Get virus concentration

            if sir_grid[r, c] == 0 and current_virus_level > infection_threshold: # Check if cell is susceptible and if virus concentration is high enough
                if np.random.rand() < infection_prob: # Cell gets infected with given probablity
                    sir_grid[r, c] = 1
            elif sir_grid[r, c] == 1: # Check if cell is infected
                if np.random.rand() < removal_prob: # Cell gets removed with given probability
                    sir_grid[r, c] = 2

@njit
def _numba_macrophage_update(
    current_macrophage_positions_typed, V_array, typed_neighbor_dict,
    degradation_rate, dt, N_ROWS, N_COLS
):
    """
    Function to update the positions of macrophages based on the virus concentration.
    
    Args:
        current_macrophage_positions_typed (list): List of current macrophage positions.
        V_array (array): 1D array representing the virus concentration.
        typed_neighbor_dict (dictionary): Dictionary of neighboring cells for each macrophage.
        degradation_rate (float): Rate of virus degradation from macrophage.
        dt (float): Time step for the simulation.
        N_ROWS (int): Number of rows in the grid.
        N_COLS (int): Number of columns in the grid.
    
    Returns:
        new_macrophage_positions_typed (list): Updated list of macrophage positions.
    """
    _dummy_int64_tuple = (np.int64(0), np.int64(0))

    new_macrophage_positions_typed = numba.typed.List([_dummy_int64_tuple])
    new_macrophage_positions_typed.clear()

    default_empty_neighbor_list = numba.typed.List([_dummy_int64_tuple])
    default_empty_neighbor_list.clear()

    for r_macro, c_macro in current_macrophage_positions_typed:
        best_r, best_c = r_macro, c_macro
        fipy_idx_current = c_macro + r_macro * N_COLS # Convert to flat index of 1d array
        max_val = V_array[fipy_idx_current] # Set current virus concentration as max value

        neighbors = typed_neighbor_dict.get((r_macro, c_macro), default_empty_neighbor_list)
        for nr, nc in neighbors:
            fipy_idx_neighbor = nc + nr * N_COLS
            if V_array[fipy_idx_neighbor] > max_val: # Check if neighbor has higher virus concentration
                best_r, best_c = nr, nc
                max_val = V_array[fipy_idx_neighbor] # Update maximum virus value in neighborhood of macrophage
        
        fipy_idx_degrade = best_c + best_r * N_COLS # Find index of new position with highest virus concentration
        V_array[fipy_idx_degrade] = max(V_array[fipy_idx_degrade] - degradation_rate * dt, 0.0) # Decrease virus concentration at new position of macrophage
        new_macrophage_positions_typed.append((best_r, best_c)) # Save new position of macrophage
    return new_macrophage_positions_typed

def run_simulation_core(
    N_ROWS, N_COLS, dx, dy, D_v, mu_v, dt, steps,
    infection_threshold, infection_prob, emission_rate, removal_prob,
    num_initial_infected,
    seed_value,
    neighbor_dict,
    macrophage_params=None,
    return_full_history=False
):
    """
    Function to run the SIR model simulation with optional macrophage dynamics.
    
    Args:  
        N_ROWS (int): Number of rows in the grid.
        N_COLS (int): Number of columns in the grid.
        dx (float): Length step size x-direction.
        dy (float): Length step size y-direction.
        D_v (float): Diffusion coefficient for the virus.
        mu_v (float): Decay rate for the virus.
        dt (float): Time step for the simulation.
        steps (int): Number of simulation steps.
        infection_threshold (float): Virus concentration threshold for infection.
        infection_prob (float): Probability of infection.
        emission_rate (float): Rate of virus emission from infected cells.
        removal_prob (float): Probability of removal.
        num_initial_infected (int): Number of initially infected cells.
        seed_value (int): Random seed.
        neighbor_dict (dict): Dictionary of neighboring cells for each cell.
        macrophage_params (dict): Parameters for macrophages.
        return_full_history (bool): Whether to return full history of the simulation.
        
        Returns:
        S_counts (list): List of susceptible cell counts at each step.
        I_counts (list): List of infected cell counts at each step.
        R_counts (list): List of removed cell counts at each step.
        history_sir_grid (list): Full history of the SIR grid at each step.
        history_V_field (list): Full history of the virus concentration field at each step.
        history_macrophage_pos (list): Full history of macrophage positions at each step.
        """
    np.random.seed(seed_value)
    _ = np.random.rand()   
    sir_grid = np.full((N_ROWS, N_COLS), 0)

    # Initialize grid randomly with given number of initially infected cells
    infected_flat_indices = np.random.choice(N_ROWS * N_COLS, size=num_initial_infected, replace=False)
    for flat_idx in infected_flat_indices:
        r, c = divmod(flat_idx, N_COLS)
        sir_grid[r, c] = 1

    macrophage_positions_typed = numba.typed.List.empty_list(types.UniTuple(types.int64, 2))
    typed_neighbor_dict = numba.typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.ListType(types.UniTuple(types.int64, 2))
    )
    degradation_rate = 0.0
    actual_num_macrophages = 0
    
    active_macrophages = False
    # Check if macrophages are included in the simulation
    if macrophage_params and macrophage_params.get('count', 0) > 0:
        actual_num_macrophages = int(macrophage_params['count'])
        if actual_num_macrophages > 0:
            active_macrophages = True
            degradation_rate = macrophage_params['degradation_rate']
            
            # Initialize random positions for macrophages
            macrophage_flat_indices = np.random.choice(N_ROWS * N_COLS, size=actual_num_macrophages, replace=False)
            temp_py_list_mac_pos = []
            for flat_idx in macrophage_flat_indices:
                r, c = divmod(flat_idx, N_COLS)
                temp_py_list_mac_pos.append((r,c))
            for r_val, c_val in temp_py_list_mac_pos:
                macrophage_positions_typed.append((r_val, c_val))

            # Create neighbor dictionary for macrophages
            for py_key, py_value_list in neighbor_dict.items():
                typed_value_list = numba.typed.List.empty_list(types.UniTuple(types.int64, 2))
                for item_tuple in py_value_list:
                    typed_value_list.append(item_tuple)
                typed_neighbor_dict[py_key] = typed_value_list
        else:
             macrophage_params = None # Effectively disable
    else:
        macrophage_params = None

    # Create PDE for virus concentration using fipy
    mesh = Grid2D(dx=dx, dy=dy, nx=N_COLS, ny=N_ROWS)
    V = CellVariable(name="virus", mesh=mesh, value=0.0, hasOld=True)
    eq = TransientTerm() == DiffusionTerm(D_v) - ImplicitSourceTerm(mu_v)

    S_counts, I_counts, R_counts = [], [], []
    
    history_sir_grid = []
    history_V_field = []
    history_macrophage_pos = []

    # Set initial condition for PDE
    V_numpy_array = V.value

    # After initialization, update virus concentration PDE, SIR grid, macrophage positions
    # for each time step
    for step_num in range(steps):

        if return_full_history: # Check if history needs to be saved
            history_sir_grid.append(sir_grid.copy())
            history_V_field.append(V_numpy_array.copy().reshape((N_ROWS, N_COLS)))
            if active_macrophages: # Check if macrophages are included in the simulation
                py_list_for_hist = []
                for i in range(len(macrophage_positions_typed)):
                    py_list_for_hist.append(macrophage_positions_typed[i])
                history_macrophage_pos.append(py_list_for_hist)
            else:
                history_macrophage_pos.append([])

        # Count the state of cells at current time step and save the values
        S_counts.append(np.sum(sir_grid == 0))
        I_counts.append(np.sum(sir_grid == 1))
        R_counts.append(np.sum(sir_grid == 2))

        # Update virus concentration based on infected cells using function
        _numba_virus_emission(sir_grid, V_numpy_array, emission_rate, dt, N_ROWS, N_COLS)

        # Update virus concentration PDE using fipy
        V.updateOld()
        eq.solve(var=V, dt=dt)

        # Set negative values of virus concentratino to zero
        _numba_enforce_non_negative(V_numpy_array)

        # Update ABM
        _numba_infection_death(sir_grid, V_numpy_array, infection_threshold, infection_prob, removal_prob, N_ROWS, N_COLS)

        # Update macrophage dynamics (position and virus degradation)
        if active_macrophages and len(macrophage_positions_typed) > 0:
            macrophage_positions_typed = _numba_macrophage_update(
                macrophage_positions_typed, V_numpy_array, typed_neighbor_dict,
                degradation_rate, dt, N_ROWS, N_COLS
            )
            
    if return_full_history:
        return S_counts, I_counts, R_counts, history_sir_grid, history_V_field, history_macrophage_pos
    else:
        return S_counts, I_counts, R_counts