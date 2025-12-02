import matplotlib.pyplot as plt
import os
import numpy as np

def plot_graph(metaData, rewards, window = 1):
    title, label, filename = metaData["title"], metaData["label"], metaData["filename"]

    # make sure plots folder exists
    os.makedirs("plots", exist_ok=True)

    # Use moving average with window=2
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        x = np.arange(window, len(smoothed) + window)
        plt.figure(figsize=(10, 6))
        plt.plot(x, smoothed, label=f"{label} (Window={window})")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label=label)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    # Save before showing
    save_path = os.path.join("plots", filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved at: {save_path}")

    plt.show()


def plot_average_goal_visits(results, algorithms):
    """
    results: dict containing algorithm -> { safe_visits : safe_visits, risky_visits : risky_visits}
    """
    algorithms = list(results.keys())
    safe_visits = [results[algo]["safe_visits"] for algo in algorithms]
    risky_visits = [results[algo]["risky_visits"] for algo in algorithms]

    x = np.arange(len(algorithms))  # algorithm positions
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    safe_bars = ax.bar(x - width/2, safe_visits, width, label="Safe Goal")
    risky_bars = ax.bar(x + width/2, risky_visits, width, label="Risky Goal")

    # Add labels, title, legend
    ax.set_ylabel("Average Goal Visits")
    ax.set_title("Average Safe vs Risky Goal Visits Across Algorithms")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate bars with values
    for bars in [safe_bars, risky_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    # Save in plots folder
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", "average_goal_visits.png") # given file name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Grouped bar chart saved at: {save_path}")

    plt.show()

