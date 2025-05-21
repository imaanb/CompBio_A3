import numpy as np
import matplotlib.pyplot as plt
import pickle

steps = 100
runs = 20

sir_results_all_runs = {}
print("Loading data and plotting")
pickle_filename_load_1 = f"sirdict_0_macrophages.pkl"
pickle_filename_load_2 = f"sirdict_50_macrophages.pkl"

with open(pickle_filename_load_1, "rb") as f:
    sir_results_all_runs[0] = pickle.load(f)
with open(pickle_filename_load_2, "rb") as f:
    sir_results_all_runs[1] = pickle.load(f)

print(sir_results_all_runs)
time_axis = np.arange(steps)
fig, ax =plt.subplots(1,2, figsize=(12,6),sharey=True)

for j in range(2):
    for i in range(len(sir_results_all_runs[j]["S"])):
        ax[j].plot(time_axis, sir_results_all_runs[j]["S"][i], alpha=0.1, color="green")
        ax[j].plot(time_axis, sir_results_all_runs[j]["I"][i], alpha=0.1, color="red")
        ax[j].plot(time_axis, sir_results_all_runs[j]["R"][i], alpha=0.1, color="black")

    ax[j].plot(time_axis, np.mean(sir_results_all_runs[j]["S"], axis=0), label="Susceptible (Mean)", color="green", linewidth=2)
    ax[j].plot(time_axis, np.mean(sir_results_all_runs[j]["I"], axis=0), label="Infected (Mean)", color="red", linewidth=2)
    ax[j].plot(time_axis, np.mean(sir_results_all_runs[j]["R"], axis=0), label="Removed (Mean)", color="black", linewidth=2)

plt.legend()
ax[0].set_title('SIR with no macrophages')
ax[1].set_title('SIR with 50 macrophages')
ax[0].set_xlabel("Timesteps")
ax[1].set_xlabel("Timesteps")
ax[0].set_ylabel("Number of Cells")
ax[0].grid(True)
ax[1].grid(True)
plt.tight_layout()
plt.savefig(f"no_macrophages_vs_macrophages.png", dpi=300)
plt.show()
print("Plotting finished.")
