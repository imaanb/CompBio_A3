import numpy as np
import matplotlib.pyplot as plt
import pickle

steps = 100
min_mac = 1 
max_mac = 201
print("Loading data and plotting")
try:
    with open("score_dict_S_vs_macrophages.pkl", "rb") as f:
        score_dict_S_counts = pickle.load(f)
except FileNotFoundError:
    print("Error: score_dict_S_vs_macrophages.pkl not found. Please run the simulation section first.")
    exit()

time_axis = np.arange(steps)
plt.figure(figsize=(10,6))
sorted_mac_numbers = sorted(score_dict_S_counts.keys())

for macrophage_number_plot in sorted_mac_numbers:
    S_runs_for_this_mac_count = np.array(score_dict_S_counts[macrophage_number_plot])
    mean_S = np.mean(S_runs_for_this_mac_count, axis=0)
    if max_mac == min_mac: color_norm = 0.5
    else: color_norm = (macrophage_number_plot - min_mac) / (max_mac - min_mac)
    plt.plot(time_axis, mean_S, label=f"{macrophage_number_plot}", color=plt.cm.coolwarm(color_norm))

plt.xlabel("Timesteps")
plt.ylabel("Number of Susceptible cells")
plt.title("Susceptible cells over time for different numbers of macrophages")
plt.legend(title="Number of Macrophages")
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("sir_plot_S_vs_macrophages.png", dpi=300)
plt.show()
print("Plotting finished.")
