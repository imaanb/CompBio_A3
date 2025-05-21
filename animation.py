import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import argparse
from SIR_ABM_macrophages import run_simulation_core

parser = argparse.ArgumentParser(description="Run and animate SIR simulation with optional macrophages.")
parser.add_argument("--macrophages", type=int, default=50, help="Number of macrophages to simulate (0 for none).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps.")
parser.add_argument("--save_anim", action="store_true", help="Save the animation to a file (e.g., animation.mp4).")
parser.add_argument("--anim_file", type=str, default="sir_animation.mp4", help="Filename for the saved animation.")
parser.add_argument("--fps", type=int, default=10, help="FPS for the saved animation.")

args = parser.parse_args()

# Simulation parameters
nx = 50
ny = 50
dx = 1
dy = 1
D_v = 1
mu_v = 0.1
dt = 1 

I_T = 0.2
P_I = 0.2
f_ij = 1.0
P_R = 0.1
delta_MV = 1.5 

I_0 = 15
num_macrophages = args.macrophages
current_seed = args.seed
total_steps = args.steps

# Pre-calculate neighbor_dict
neighbor_dict = {}
for r in range(nx):
    for c in range(ny):
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nx and 0 <= nc < ny:
                neighbors.append((nr, nc))
        neighbor_dict[(r, c)] = neighbors

# Setup macrophage parameters for the simulation function
if num_macrophages > 0:
    mac_params_for_sim = {
        'count': num_macrophages,
        'degradation_rate': delta_MV
    }
else:
    mac_params_for_sim = None

# Simulate
print("Running simulation to gather history")
_S, _I, _R, history_sir, history_V, history_macro_pos = run_simulation_core(nx, ny, dx, dy, D_v, mu_v, dt, total_steps, I_T, P_I, f_ij, P_R, I_0, current_seed, neighbor_dict, mac_params_for_sim,return_full_history=True)

print("Simulation history gathered.")


# Animate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cmap_sir = ListedColormap(['lightgreen', 'orangered', 'black'])  # 0=S, 1=I, 2=R

# Initial state for plots
im1 = ax1.imshow(history_sir[0], cmap=cmap_sir, origin='lower', vmin=0, vmax=2)
ax1.set_title(f"ABM, t=0")
ax1.set_xticks([])
ax1.set_yticks([])


# The V history is already reshaped in run_simulation_core when return_full_history=True
initial_v_max = np.max(history_V) if len(history_V) > 0 and np.any(history_V) else 3.5
im2 = ax2.imshow(history_V[0], origin='lower', cmap='viridis', vmin=0, vmax=max(0.1, initial_v_max)) # Ensure vmax is not 0
ax2.set_title("Virus Field, t=0")
ax2.set_xticks([])
ax2.set_yticks([])
cbar = fig.colorbar(im2, ax=ax2, label='Virus concentration')

macrophage_plot, = ax1.plot([], [], 'wo', markersize=5, markeredgecolor='black')

# --- Animation Function ---
def update_plot(frame_num):
    current_time = frame_num * dt

    im1.set_data(history_sir[frame_num])
    ax1.set_title(f"ABM, t={current_time:.1f}")

    virus_data = history_V[frame_num]
    im2.set_data(virus_data)
    ax2.set_title(f"Virus Field, t={current_time:.1f}")
    current_v_max = np.max(virus_data)
    im2.set_clim(vmin=0, vmax=max(0.1, current_v_max)) 


    macro_coords = history_macro_pos[frame_num]
    if macro_coords:
        ys, xs = zip(*macro_coords) 
        macrophage_plot.set_data(xs, ys)
    else:
        macrophage_plot.set_data([], [])

    return im1, im2, macrophage_plot

#
num_frames = len(history_sir)

print(f"Creating animation with {num_frames} frames")
ani = FuncAnimation(fig, update_plot, frames=num_frames,
                    interval=1000/args.fps, blit=False, repeat=False) 

if args.save_anim:
    print(f"Saving animation to {args.anim_file} (this may take a while)...")
    try:
        ani.save(args.anim_file, writer='ffmpeg', fps=args.fps, dpi=150)
        print(f"Animation saved to {args.anim_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure ffmpeg is installed and in your system's PATH.")
        print("Alternatively, try saving as a GIF: ani.save('animation.gif', writer='imagemagick', fps=10)")
else:
    plt.tight_layout()
    plt.show()

print("Done.")
