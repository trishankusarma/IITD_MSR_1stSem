# Code for part C
import os
import json
import random
import numpy as np
import torch
from torch.distributions import Normal
from reinforce_baselines import PolicyNetwork
import matplotlib.pyplot as plt

os.makedirs("gradients", exist_ok=True)
os.makedirs("plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE_FOR_PARTC = 20
ITERATIONS = 10
SAMPLE_SIZE_FOR_PARTD = [30, 40, 50, 60, 70, 80, 90, 100]
HIDDEN_DIM = 64

base_titles = {
    "no_baseline": "No Baseline",
    "reward_to_go": "Reward-to-Go",
    "avg_reward": "Average Reward Baseline",
    "value_function": "Value Function Baseline"
}

# LOAD TRAJECTORIES
def load_trajectories(path):
    with open(path, "r") as f:
        return json.load(f)   # list of dicts: states, actions, rewards

# COMPUTE GRADIENT ESTIMATE FROM A SET OF TRAJECTORIES
def compute_gradient_estimate(policy, trajectories, gamma=0.99):
    policy.zero_grad()
    all_grads = []

    for traj in trajectories:
        states = torch.tensor(np.array(traj['states']), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(traj['actions']), dtype=torch.float32, device=device)
        rewards = traj['rewards']

        # compute returns-to-go
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # log-probs
        mean, std = policy(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        loss = -(log_probs * returns).mean()

        policy.zero_grad()
        loss.backward()

        # flatten concatenated gradients into a 1-D vector
        grad_vec = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
        all_grads.append(grad_vec.detach().cpu().numpy())

    return np.mean(np.stack(all_grads), axis=0)

# MAIN: 20-SAMPLE, 10 REPEATS
def gradient_estimation_for_samples(baseline_type, model_path, traj_path, samples = SAMPLE_SIZE_FOR_PARTC):
    print(f"\nGradient Estimation for {baseline_type}")

    # load policy
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = 4
    output_dim = 1
    policy = PolicyNetwork(input_dim, output_dim, hidden_dim = HIDDEN_DIM).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    # load 500 trajectories
    trajectories = load_trajectories(traj_path)

    # Perform 10 repetitions
    gradient_list = []

    for rep in range(ITERATIONS):
        sampled_trajs = random.sample(trajectories, samples)
        grad_est = compute_gradient_estimate(policy, sampled_trajs)
        gradient_list.append(grad_est)
        print(f" repetition {rep+1} done")

    return gradient_list

def runGradientEstimation_ForSampleSize(baselines, sample_size=SAMPLE_SIZE_FOR_PARTC):
    all_gradients = {}

    for b in baselines:
        model_path = f"models/reinforce_{b}.pt"
        traj_path = f"trajectories/reinforce_{b}.json"

        grads = gradient_estimation_for_samples(
            baseline_type=b,
            model_path=model_path,
            traj_path=traj_path,
            samples = sample_size
        )

        all_gradients[b] = grads

    # saving to json
    import json
    with open(f"gradients/grad_{sample_size}sample_10reps.json", "w") as f:
        json.dump({k: [g.tolist() for g in v] for k,v in all_gradients.items()}, f, indent=4)

    print(f"\nSaved gradient estimates for sample_size {sample_size}!")

# For part E
def load_gradients(sample_size):
    """Load gradient JSON file and return dict: baseline -> list of 10 vectors."""
    path = f"gradients/grad_{sample_size}sample_10reps.json"
    with open(path, "r") as f:
        return json.load(f)

def compute_magnitude(vec):
    """Compute L2 norm of a gradient vector."""
    v = np.array(vec)
    return np.linalg.norm(v)

def collect_stats(baselines):
    """Load all sample sizes and compute mean/std magnitudes."""
    stats = {b: {"means": [], "stds": []} for b in baselines}

    for N in SAMPLE_SIZE_FOR_PARTD:
        grad_data = load_gradients(N)

        for baseline in baselines:
            reps = grad_data[baseline]  # list of 10 gradient vectors

            magnitudes = [compute_magnitude(g) for g in reps]
            stats[baseline]["means"].append(np.mean(magnitudes))
            stats[baseline]["stds"].append(np.std(magnitudes))

    return stats

def plot_stats(stats, baselines):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, baseline in enumerate(baselines):
        ax = axes[idx]

        means = np.array(stats[baseline]["means"])
        stds = np.array(stats[baseline]["stds"])

        ax.plot(SAMPLE_SIZE_FOR_PARTD, means, marker='o', label="Mean Gradient Magnitude")
        ax.fill_between(
            SAMPLE_SIZE_FOR_PARTD, means - stds, means + stds,
            alpha=0.3, label="±1 Std Dev"
        )

        ax.set_title(base_titles[baseline])
        ax.set_xlabel("Sample Size (Number of Trajectories)")
        ax.set_ylabel("Gradient Magnitude")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/gradient_estimate_variance.png")
    plt.show()

    print(f"\nSaved gradient estimate variance plot for all baseline in one plot!")

if __name__ == "__main__":
    baselines = ["no_baseline", "reward_to_go", "avg_reward", "value_function"]

    # Question C
    runGradientEstimation_ForSampleSize(baselines, sample_size=SAMPLE_SIZE_FOR_PARTC)
    # Question D
    # Repeating the above experiment for [30, 40, 50, 60, 70, 80, 90, 100]
    for sample_size in SAMPLE_SIZE_FOR_PARTD:
        runGradientEstimation_ForSampleSize(baselines, sample_size=sample_size)
    
    stats = collect_stats(baselines)
    plot_stats(stats, baselines)