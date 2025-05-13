# python graphs.py

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_training_reward(log_path="training_logO.csv"):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    df = pd.read_csv(log_path)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Episode"], df["TotalReward"], label="Raw Reward", color="grey", alpha=0.6)

    # Moving averages
    df["MA_10"] = df["TotalReward"].rolling(window=10).mean()
    df["MA_50"] = df["TotalReward"].rolling(window=50).mean()
    plt.plot(df["Episode"], df["MA_10"], label="10-Episode Avg", color="blue", linestyle="-")
    plt.plot(df["Episode"], df["MA_50"], label="50-Episode Avg", color="red", linestyle="-")

    # Benchmark line at 900
    plt.axhline(y=900, color="orange", linestyle="--", linewidth=1, label="Target Score (900)")

    # Highlight best performing episode
    max_idx = df["TotalReward"].idxmax()
    best_ep = df.loc[max_idx, "Episode"]
    best_reward = df.loc[max_idx, "TotalReward"]
    plt.scatter([best_ep], [best_reward], color="lime", s=60, label=f"Best: Ep {int(best_ep)} ({int(best_reward)})", edgecolors="green", zorder=5)

    plt.title("Training Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_plot.png")
    print("Saved enhanced reward plot as reward_plot.png")
    plt.show()

def plot_sample_trajectory(path_file="ep_100_path.npy"):
    if not os.path.exists(path_file):
        print(f"Trajectory file not found: {path_file}")
        return

    trajectory = np.load(path_file)
    if trajectory.shape[1] != 2:
        print("Invalid trajectory shape.")
        return

    x, y = trajectory[:, 0], trajectory[:, 1]
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', markersize=2, linewidth=1, color='blue')
    plt.title("Agent Trajectory - Sample Episode")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sample_trajectory.png")
    print("Saved trajectory plot as sample_trajectoryO.png")
    plt.show()

def plot_model_architecture_vertical_compact():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    def draw_block(y, label, shape="", desc="", color='lightgray', height=1.0):
        rect = patches.FancyBboxPatch((3, y), 4, height, boxstyle="round,pad=0.05", ec="black", fc=color)
        ax.add_patch(rect)
        lines = [label]
        if shape: lines.append(shape)
        if desc: lines.append(desc)
        for i, text in enumerate(lines):
            offset = 0.3 - i * 0.25
            ax.text(5, y + height / 2 + offset, text, ha='center', va='center', fontsize=9, wrap=True)

    layers = [
        ("Input", "3×96×96", "stack of grayscale frames", 'lightblue'),
        ("Conv2D (6, 7×7, stride 3)", "6×30×30", "extracts local features", 'lightgreen'),
        ("ReLU", "", "non-linear activation", 'white'),
        ("MaxPool (2×2)", "6×15×15", "downsamples spatial size", 'white'),
        ("Conv2D (12, 4×4)", "12×12×12", "detects higher-level patterns", 'lightgreen'),
        ("ReLU", "", "non-linear activation", 'white'),
        ("MaxPool (2×2)", "12×6×6", "final spatial reduction", 'white'),
        ("Flatten", "432", "flattens to vector", 'orange'),
        ("Dense (216)", "216", "learns abstract Q-state features", 'violet'),
        ("Dense (Q-values)", "5 (actions)", "outputs Q-values", 'salmon')
    ]

    y_pos = 12
    spacing = 1.2

    for label, shape, desc, color in layers:
        draw_block(y_pos, label, shape, desc, color)
        y_pos -= spacing

    for i in range(len(layers) - 1):
        y_start = 12 - i * spacing - 1.0
        y_end = y_start - 0.1
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle="->", color='black'))

    plt.title("DQN Model Architecture (Compact with Output Shapes & Descriptions)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_generalization_rewards():
    seeds = [5216, 9315, 3413, 7276, 6481, 7906, 2168, 8629, 3417, 9710]
    rewards = [917.70, 886.67, 928.00, 887.54, 554.61, 381.61, 868.75, 896.67, 891.49, 877.61]

    # Index of top 3 and worst
    top3Idx = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:3]
    worstIdx = [rewards.index(min(rewards))]

    colors = ['skyblue'] * len(seeds)
    for i in top3Idx:
        colors[i] = 'blue'
    for i in worstIdx:
        colors[i] = 'red'

    plt.figure(figsize=(10, 6))
    bars = plt.bar([f"Seed {s}" for s in seeds], rewards, color=colors)
    plt.axhline(900, color='orange', linestyle='--', linewidth=1.5, label='900 Reward Benchmark')
    plt.xticks(rotation=45)
    plt.ylabel("Total Reward")
    plt.title("Generalization Test: Reward per Random Seed")
    plt.legend()

    # Annotate bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        label = "Top" if i in top3Idx else ("Worst" if i in worstIdx else "")
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f"{yval:.1f}\n{label}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig("generalization_test_rewards.png")
    plt.show()


if __name__ == "__main__":
    plot_training_reward()
    #plot_sample_trajectory()
    plot_model_architecture_vertical_compact()
    plot_generalization_rewards()