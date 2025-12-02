import matplotlib.pyplot as plt
import numpy as np

def plot(data = None, min_len = None, mean = None, std = None, label = "label", xlabel = "xlabel", ylabel = 'ylabel', title = 'title', filename = 'filename'):
    plt.figure(figsize=(8, 4))
    plt.plot(data, label=label, color='blue')
    plt.fill_between(np.arange(min_len),
                    mean - std,
                    mean + std,
                    alpha=0.3, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_importance_weights(rhos, noise, save_path):
    """
    Plot mean and std of importance sampling weights (rho) over steps.

    Args:
        rhos (list of list of list): Shape [seeds][episodes][steps] rho values.
        noise (float): Noise level for title/filename.
        save_path (str): Where to save the plot image.
    """
    # Find shortest episode length across all seeds and episodes
    min_rho_len = min(min(len(ep_rhos) for ep_rhos in seed_rhos) for seed_rhos in rhos)

    # Convert to numpy array for slicing: (seeds, episodes, steps)
    rhos_array = np.array([
        [ep_rhos[:min_rho_len] for ep_rhos in seed_rhos]
        for seed_rhos in rhos
    ])

    # Compute mean and std over seeds and episodes for each step
    mean_rho = np.mean(rhos_array, axis=(0, 1))
    std_rho = np.std(rhos_array, axis=(0, 1))

    steps = np.arange(min_rho_len)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, mean_rho, label='Mean Importance Sampling Weight (rho)')
    plt.fill_between(steps, mean_rho - std_rho, mean_rho + std_rho, alpha=0.3)
    plt.xlabel('Step')
    plt.ylabel('Importance Sampling Weight (rho)')
    plt.title(f'Importance Sampling Weights over Steps (Noise={noise})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ Saved importance sampling weights plot: {save_path}")
