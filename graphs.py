# python graphs.py

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.patches as mpatches #custom key colours

def plot_training_reward(log_path="trainLog3.csv"):
    # plot raw and smoothed reward curves from training log
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    df = pd.read_csv(log_path)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Episode"], df["TotalReward"], label="Raw Reward", color="grey", alpha=0.6)

    # calculate moving averages
    df["MA_10"] = df["TotalReward"].rolling(window=10).mean()
    df["MA_50"] = df["TotalReward"].rolling(window=50).mean()
    plt.plot(df["Episode"], df["MA_10"], label="10-Episode Avg", color="blue", linestyle="-")
    plt.plot(df["Episode"], df["MA_50"], label="50-Episode Avg", color="red", linestyle="-")

    # draw performance benchmark line
    plt.axhline(y=900, color="orange", linestyle="--", linewidth=1, label="Target Score (900)")

    # mark best episode
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
    # plot x/y path from stored trajectory array
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
    # draw vertical block diagram of dqn model
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # internal function to draw layer block
    def draw_block(y, label, shape="", desc="", color='lightgray', height=1.0):
        rect = patches.FancyBboxPatch((3, y), 4, height, boxstyle="round,pad=0.05", ec="black", fc=color)
        ax.add_patch(rect)
        lines = [label]
        if shape: lines.append(shape)
        if desc: lines.append(desc)
        for i, text in enumerate(lines):
            offset = 0.3 - i * 0.25
            ax.text(5, y + height / 2 + offset, text, ha='center', va='center', fontsize=9, wrap=True)

    # define layer list
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

    # draw blocks
    for label, shape, desc, color in layers:
        draw_block(y_pos, label, shape, desc, color)
        y_pos -= spacing

    # connect arrows between layers
    for i in range(len(layers) - 1):
        y_start = 12 - i * spacing - 1.0
        y_end = y_start - 0.1
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle="->", color='black'))

    plt.title("DQN Model Architecture (Compact with Output Shapes & Descriptions)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_generalization_rewards():
    # bar chart comparing test reward over 10 random seeds
    seeds = [5216, 9315, 3413, 7276, 6481, 7906, 2168, 8629, 3417, 9710]
    rewards = [917.70, 886.67, 928.00, 887.54, 554.61, 381.61, 868.75, 896.67, 891.49, 877.61]

    # get top 3 and lowest reward indices
    top3Idx = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:3]
    worstIdx = [rewards.index(min(rewards))]

    # set bar colors
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

    # add reward values and labels
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        label = "Top" if i in top3Idx else ("Worst" if i in worstIdx else "")
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f"{yval:.1f}\n{label}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig("generalization_test_rewards.png")
    plt.show()

def plot_training_logs_with_ma50(episode_limit=1000):
    """
    Plots raw and 50-episode average reward for preset training logs.
    Adds markers and annotations for max scores, and displays legend last.
    """

    # File paths
    LOG_PATHS = [
        "trainLogO2.csv",       # original
        "trainLog3.csv",        # improved
        "trainLog4.csv"         # experimental
    ]

    # Labels
    LABELS = [
        "Original (Orange)",
        "Improved (Indigo)",
        "Experimental (Emerald)"
    ]

    # Short codes for annotations
    SHORT_CODES = {
        "Original (Orange)": "org",
        "Improved (Indigo)": "imp",
        "Experimental (Emerald)": "exp"
    }

    # Color scheme: (raw, avg, point)
    COLORS = {
        "Original (Orange)": ("sandybrown", "orangered", "orangered"),
        "Improved (Indigo)": ("thistle", "indigo", "indigo"),
        "Experimental (Emerald)": ("mediumaquamarine", "seagreen", "seagreen")
    }

    all_logs = []
    reward_900_counts = {}

    for path, label in zip(LOG_PATHS, LABELS):
        try:
            df = pd.read_csv(path)
            df = df[df["Episode"] < episode_limit].copy()
            df["Model"] = label
            reward_900_counts[label] = (df["TotalReward"] > 900).sum()
            all_logs.append(df)
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
            continue

    if not all_logs:
        print("❌ No logs could be loaded.")
        return

    # Combine and compute 50-episode moving average
    combined = pd.concat(all_logs, ignore_index=True)
    combined["MA50"] = combined.groupby("Model")["TotalReward"].transform(
        lambda x: x.rolling(50, min_periods=1).mean()
    )

    # Determine Y-axis range
    min_val = combined["TotalReward"].min()
    max_val = combined["TotalReward"].max()

    # Start plotting
    plt.figure(figsize=(14, 7))

    # Plot raw lines
    for label in LABELS:
        group = combined[combined["Model"] == label]
        raw_color, _, _ = COLORS[label]
        plt.plot(group["Episode"], group["TotalReward"], color=raw_color, alpha=0.5, label=f"{label} - Raw")

    # Plot smoothed lines
    for label in LABELS:
        group = combined[combined["Model"] == label]
        _, avg_color, _ = COLORS[label]
        plt.plot(group["Episode"], group["MA50"], color=avg_color, linewidth=2, label=f"{label} - 50ep Avg")

    # Plot and annotate max points
    for label in LABELS:
        group = combined[combined["Model"] == label]
        _, _, point_color = COLORS[label]
        short = SHORT_CODES[label]

        max_idx = group["TotalReward"].idxmax()
        max_ep = group.loc[max_idx, "Episode"]
        max_val_group = group.loc[max_idx, "TotalReward"]

        plt.scatter(max_ep, max_val_group, color=point_color, s=90, zorder=5)
        plt.text(
            max_ep, max_val_group + 25,
            f"{short}.R$_{{max}}$\n{max_val_group:.2f}",
            ha="center", fontsize=10, fontweight="bold", color=point_color
        )

    # Target line
    plt.axhline(900, color="royalblue", linestyle="--", linewidth=2, label="Target Reward (900)")

    # Style
    plt.title(f"Training Reward Comparison (First {episode_limit} Episodes, with 50-Episode Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.ylim(min_val - 10, max_val + 100)
    plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training_logs_with_ma250(episode_limit=1000):
    """
    Plots raw and 250-episode average reward for preset training logs.
    Adds markers and annotations for max scores, and displays legend last.
    """

    # File paths
    LOG_PATHS = [
        "trainLogO2.csv",       # original
        "trainLog3.csv",        # improved
        "trainLog4.csv"         # experimental
    ]

    # Labels
    LABELS = [
        "Original (Orange)",
        "Improved (Indigo)",
        "Experimental (Emerald)"
    ]

    # Colors: (raw, avg, point)
    COLORS = {
        "Original (Orange)": ("sandybrown", "orangered", "orangered"),
        "Improved (Indigo)": ("thistle", "indigo", "indigo"),
        "Experimental (Emerald)": ("mediumaquamarine", "seagreen", "seagreen")
    }

    # Short codes for annotations
    SHORT_CODES = {
        "Original (Orange)": "org",
        "Improved (Indigo)": "imp",
        "Experimental (Emerald)": "exp"
    }

    all_logs = []
    reward_900_counts = {}

    for path, label in zip(LOG_PATHS, LABELS):
        try:
            df = pd.read_csv(path)
            df = df[df["Episode"] < episode_limit].copy()
            df["Model"] = label
            reward_900_counts[label] = (df["TotalReward"] > 900).sum()
            all_logs.append(df)
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
            continue

    if not all_logs:
        print("❌ No logs could be loaded.")
        return

    # Combine and compute 250-ep moving average
    combined = pd.concat(all_logs, ignore_index=True)
    combined["MA250"] = combined.groupby("Model")["TotalReward"].transform(
        lambda x: x.rolling(250, min_periods=1).mean()
    )

    # Determine Y-axis range
    min_val = combined["TotalReward"].min()
    max_val = combined["TotalReward"].max()

    # Start plotting
    plt.figure(figsize=(14, 7))

    # Plot raw lines
    for label in LABELS:
        group = combined[combined["Model"] == label]
        raw_color, _, _ = COLORS[label]
        plt.plot(group["Episode"], group["TotalReward"], color=raw_color, alpha=0.5, label=f"{label} - Raw")

    # Plot smoothed lines
    for label in LABELS:
        group = combined[combined["Model"] == label]
        _, avg_color, _ = COLORS[label]
        plt.plot(group["Episode"], group["MA250"], color=avg_color, linewidth=2, label=f"{label} - 250ep Avg")

    # Plot and annotate max points
    for label in LABELS:
        group = combined[combined["Model"] == label]
        _, _, point_color = COLORS[label]
        short = SHORT_CODES[label]

        max_idx = group["TotalReward"].idxmax()
        max_ep = group.loc[max_idx, "Episode"]
        max_val_group = group.loc[max_idx, "TotalReward"]

        plt.scatter(max_ep, max_val_group, color=point_color, s=90, zorder=5)
        plt.text(
            max_ep, max_val_group + 25,
            f"{short}.R$_{{max}}$\n{max_val_group:.2f}",
            ha="center", fontsize=10, fontweight="bold", color=point_color
        )

    # Target line
    plt.axhline(900, color="royalblue", linestyle="--", linewidth=2, label="Target Reward (900)")

    # Style
    plt.title(f"Training Reward Comparison (First {episode_limit} Episodes, with 250-Episode Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.ylim(min_val - 10, max_val + 100)
    plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_high_reward_distribution():
    """
    Generates a stacked bar chart showing the count of episodes with reward >900
    and those between 800–900 for Original, Improved, and Experimental models.
    """

    # File paths
    LOG_PATHS = {
        "Original (Orange)": "trainLogO2.csv",
        "Improved (Indigo)": "trainLog3.csv",
        "Experimental (Emerald)": "trainLog4.csv"
    }

    # Model-specific bar colors
    COLORS = {
        "Original (Orange)": ("orangered", "sandybrown"),
        "Improved (Indigo)": ("indigo", "thistle"),
        "Experimental (Emerald)": ("seagreen", "mediumaquamarine")
    }

    # Track result counts
    solved_counts = {}
    nearly_counts = {}

    for label, path in LOG_PATHS.items():
        try:
            df = pd.read_csv(path)
            solved_counts[label] = (df["TotalReward"] > 900).sum()
            nearly_counts[label] = ((df["TotalReward"] > 800) & (df["TotalReward"] <= 900)).sum()
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
            solved_counts[label] = 0
            nearly_counts[label] = 0

    labels = list(LOG_PATHS.keys())
    solved = [solved_counts[l] for l in labels]
    nearly = [nearly_counts[l] for l in labels]
    bar_x = range(len(labels))

    # Extract plot colors
    dark_colors = [COLORS[l][0] for l in labels]
    light_colors = [COLORS[l][1] for l in labels]

    # Plot
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(bar_x, solved, color=dark_colors)
    bars2 = plt.bar(bar_x, nearly, bottom=solved, color=light_colors)

    # Label counts on bars
    for i, (s, n) in enumerate(zip(solved, nearly)):
        total = s + n
        plt.text(i, s / 2, f"{s}", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        plt.text(i, s + n / 2, f"{n}", ha="center", va="center", fontsize=10, fontweight="bold", color="black")
        plt.text(i, s + n + 2, f"Total: {total}", ha="center", fontsize=10, fontweight="bold")

    # Axes and styling
    plt.xticks(bar_x, labels)
    plt.ylabel("Number of Episodes")
    plt.title("High Reward Score Episodes per Model")

    # Custom grey legend
    legend_nearly = mpatches.Patch(color='lightgray', label="Nearly Solved (800–900)")
    legend_solved = mpatches.Patch(color='dimgray', label="Solved (>900)")
    plt.legend(handles=[legend_solved, legend_nearly])

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def model_generalization_comparison():
    # Dataset mapping for display and title
    dataset_names = {
        1: "Cuban Primes",
        2: "Random Primes",
        3: "Balanced Primes",
        4: "Euclidian Primes",
        5: "Sophie Germain Primes"
    }

    print("Available Datasets:")
    for key, name in dataset_names.items():
        print(f"[{key}] {name}")

    try:
        dataset_choice = int(input("Enter the number of the dataset you wish to view (1–5): "))
        dataset_label = dataset_names.get(dataset_choice)
        if not dataset_label:
            raise ValueError
    except ValueError:
        print("❌ Invalid selection. Please enter a number between 1 and 5.")
        return

    # Dataset definitions
    if dataset_choice == 1:  # Cuban Primes
        seeds = [7, 1657, 2269, 3169, 4219, 5167, 6211, 7057, 8269, 9241]
        original_rewards = [605.33, 585.71, 651.70, 678.20, 486.87, 326.33, 933.70, 822.79, 835.58, 293.94]
        improved_rewards = [642.95, 896.43, 893.20, 866.17, 861.39, 859.25, 469.34, 525.00, 700.61, 809.09]
        experimental_rewards = [197.81, 878.57, 814.97, 869.92, 942.70, 642.95, 912.10, 833.82, 811.04, 318.18]

    elif dataset_choice == 2:  # Random Primes
        seeds = [173, 1741, 2857, 3187, 4157, 5387, 6883, 7547, 8089, 9547]
        original_rewards = [195.30, 938.20, 671.22, 383.75, 868.50, 9.15, 488.24, 671.08, 350.38, 871.12]
        improved_rewards = [876.51, 892.70, 885.24, 937.20, 943.70, 755.63, 620.59, 758.43, 892.37, 901.50]
        experimental_rewards = [889.93, 889.05, 888.93, 942.80, 888.19, 431.69, 826.47, 508.43, 804.58, 874.73]

    elif dataset_choice == 3:  # Balanced Primes
        seeds = [2, 1103, 2287, 3307, 4013, 5107, 6073, 7523, 8117, 9397]
        original_rewards = [380.60, 57.72, 588.31, 827.34, 82.80, 490.16, 839.72, 893.44, 101.34, 624.14]
        improved_rewards = [186.57, 524.16, 831.82, 868.86, 878.49, 450.82, 942.20, 306.56, 795.97, 861.69]
        experimental_rewards = [311.94, 507.38, 705.19, 872.32, 874.91, 933.40, 882.27, 739.34, 883.22, 237.16]

    elif dataset_choice == 4:  # Euclidian Primes
        seeds = [61, 1607, 2557, 3761, 4099, 5113, 6277, 7517, 8291, 9473]
        original_rewards = [592.07, 439.06, 305.23, 324.56, 96.55, 258.66, 495.24, 285.19, 695.05, 229.59]
        improved_rewards = [590.32, 884.37, 713.73, 875.44, 813.79, 282.98, 284.35, 888.89, 889.40, 873.78]
        experimental_rewards = [861.29, 888.28, 370.59, 285.96, 837.93, 599.09, 842.18, 344.44, 797.53, 881.27]

    elif dataset_choice == 5:  # Sophie Germain Primes
        seeds = [89, 1297, 2371, 3209, 4561, 5021, 6301, 7351, 8597, 9631]
        original_rewards = [588.74, 232.28, 146.03, 844.44, 825.37, 552.60, 464.19, 308.93, 650.00, 275.38]
        improved_rewards = [419.87, 862.03, 844.44, 912.00, 888.81, 874.03, 302.03, 501.37, 442.86, 192.31]
        experimental_rewards = [284.11, 858.86, 852.38, 485.19, 881.34, 883.77, 883.11, 875.95, 857.14, 875.38]

    # Plot
    x = np.arange(len(seeds))
    original_mean = np.mean(original_rewards)
    improved_mean = np.mean(improved_rewards)
    experimental_mean = np.mean(experimental_rewards)

    plt.figure(figsize=(12, 7))
    plt.plot(x, original_rewards, '-', label="Original (Ochre)", color="orangered")
    plt.plot(x, improved_rewards, '-', label="Improved (Indigo)", color="indigo")
    plt.plot(x, experimental_rewards, '-', label="Experimental (Emerald)", color="seagreen")

    plt.hlines(original_mean, xmin=-0.5, xmax=9.5, colors="sandybrown", linestyles='--', linewidth=2, label="Original Avg")
    plt.hlines(improved_mean, xmin=-0.5, xmax=9.5, colors="thistle", linestyles='--', linewidth=2, label="Improved Avg")
    plt.hlines(experimental_mean, xmin=-0.5, xmax=9.5, colors="mediumaquamarine", linestyles='--', linewidth=2, label="Experimental Avg")

    plt.axhline(900, color="royalblue", linestyle="dotted", linewidth=2, label="Solved Threshold (900)")
    plt.xticks(ticks=x, labels=[str(seed) for seed in seeds])
    plt.xlabel("Seed")
    plt.ylabel("Reward")
    plt.title(f"Generalisation Test Reward Scores Comparison by Seed - ({dataset_label})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_supergroup_comparison(option=2, selected_model=None):
    # Supergroup seeds and rewards
    seeds = [7, 89, 173, 1297, 1607, 1657, 1741, 2269, 2287, 2371, 2857, 3169, 3187, 3209, 3307, 3761, 4157, 4219, 4561,
             5021, 6073, 6211, 7523, 7517, 8269, 8291, 8597, 8089, 9473, 9547]

    original_rewards = [
        605.33, 588.74, 195.30, 232.28, 439.06, 585.71, 938.20, 651.70, 588.31, 146.03,
        671.22, 678.20, 383.75, 844.44, 827.34, 324.56, 868.50, 486.87, 825.37, 552.60,
        839.72, 933.70, 893.44, 285.19, 835.58, 695.05, 650.00, 350.38, 229.59, 871.12
    ]

    improved_rewards = [
        642.95, 419.87, 876.51, 862.03, 884.37, 896.43, 892.70, 893.20, 831.82, 844.44,
        885.24, 866.17, 937.20, 912.00, 868.86, 875.44, 943.70, 861.39, 888.81, 874.03,
        942.20, 469.34, 306.56, 888.89, 700.61, 889.40, 442.86, 892.37, 873.78, 901.50
    ]

    experimental_rewards = [
        197.81, 284.11, 889.93, 858.86, 888.28, 878.57, 889.05, 814.97, 705.19, 852.38,
        888.93, 869.92, 942.80, 485.19, 872.32, 285.96, 888.19, 942.70, 881.34, 883.77,
        882.27, 912.10, 739.34, 344.44, 811.04, 797.53, 857.14, 804.58, 881.27, 874.73
    ]

    # Prompt user for option
    option = input(
        "Choose an option:\n[1] Plot full supergroup\n[2] Plot single model sorted by reward\nEnter 1 or 2: ")

    if option == "1":
        # Supergroup plot
        x = np.arange(len(seeds))
        mean_orig = np.mean(original_rewards)
        mean_impr = np.mean(improved_rewards)
        mean_exper = np.mean(experimental_rewards)

        plt.figure(figsize=(14, 7))
        plt.plot(x, original_rewards, '-', label="Original (Ochre)", color="orangered")
        plt.plot(x, improved_rewards, '-', label="Improved (Indigo)", color="indigo")
        plt.plot(x, experimental_rewards, '-', label="Experimental (Emerald)", color="seagreen")

        plt.hlines(mean_orig, xmin=-0.5, xmax=len(seeds) - 0.5, colors="sandybrown", linestyles='--', linewidth=2,
                   label="Original Avg")
        plt.hlines(mean_impr, xmin=-0.5, xmax=len(seeds) - 0.5, colors="thistle", linestyles='--', linewidth=2,
                   label="Improved Avg")
        plt.hlines(mean_exper, xmin=-0.5, xmax=len(seeds) - 0.5, colors="mediumaquamarine", linestyles='--',
                   linewidth=2, label="Experimental Avg")

        plt.axhline(900, color="royalblue", linestyle="dotted", linewidth=2, label="Solved Threshold (900)")
        plt.xticks(ticks=x, labels=[str(seed) for seed in seeds], rotation=45)
        plt.xlabel("Seed")
        plt.ylabel("Reward")
        plt.title("Generalisation Test Reward Scores Comparison (Supergroup)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    elif option == "2":
        model_choice = input("Select model:\n[1] Original\n[2] Improved\n[3] Experimental\nEnter 1, 2, or 3: ")
        model_map = {
            "1": ("Original", original_rewards, "orangered"),
            "2": ("Improved", improved_rewards, "indigo"),
            "3": ("Experimental", experimental_rewards, "seagreen")
        }

        if model_choice in model_map:
            label, rewards, color = model_map[model_choice]
            sorted_indices = np.argsort(rewards)
            sorted_rewards = np.array(rewards)[sorted_indices]
            sorted_seeds = np.array(seeds)[sorted_indices]
            mean_val = np.mean(rewards)

            x = np.arange(len(sorted_rewards))
            y_min = sorted_rewards.min() - 20
            y_max = sorted_rewards.max() + 20

            plt.figure(figsize=(12, 7))
            plt.plot(x, sorted_rewards, '-', color=color, label=f"{label} Sorted")
            plt.hlines(mean_val, xmin=-0.5, xmax=len(sorted_rewards) - 0.5, colors=color, linestyles='--', linewidth=2,
                       label=f"{label} Avg")
            plt.axhline(900, color="royalblue", linestyle="dotted", linewidth=2, label="Solved Threshold (900)")
            plt.xticks(ticks=x, labels=[str(s) for s in sorted_seeds], rotation=45)
            plt.ylim(y_min, y_max)
            plt.xlabel("Seed (sorted by reward)")
            plt.ylabel("Reward")
            plt.title(f"{label} Model - Sorted Generalisation Rewards")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
        else:
            print("❌ Invalid model selection.")
    else:
        print("❌ Invalid option selected.")

def plot_top_supergroup_comparison():

    # Subgroup seeds
    subgroup_seeds = [1741, 4157, 3307, 6073, 9547]
    original_rewards_sub = [938.20, 868.50, 827.34, 839.72, 871.12]
    improved_rewards_sub = [892.70, 943.70, 868.86, 942.20, 901.50]
    experimental_rewards_sub = [889.05, 888.19, 872.32, 882.27, 874.73]

    # X-axis indices
    x = np.arange(len(subgroup_seeds))

    # Calculate means
    original_mean_sub = np.mean(original_rewards_sub)
    improved_mean_sub = np.mean(improved_rewards_sub)
    experimental_mean_sub = np.mean(experimental_rewards_sub)

    # Plot setup
    plt.figure(figsize=(12, 7))

    # Plot reward lines
    plt.plot(x, original_rewards_sub, '-', label="Original (Ochre)", color="orangered")
    plt.plot(x, improved_rewards_sub, '-', label="Improved (Indigo)", color="indigo")
    plt.plot(x, experimental_rewards_sub, '-', label="Experimental (Emerald)", color="seagreen")

    # Plot average lines
    plt.hlines(original_mean_sub, xmin=-0.5, xmax=len(subgroup_seeds) - 0.5, colors="sandybrown", linestyles='--',
               linewidth=2, label="Original Avg")
    plt.hlines(improved_mean_sub, xmin=-0.5, xmax=len(subgroup_seeds) - 0.5, colors="thistle", linestyles='--',
               linewidth=2, label="Improved Avg")
    plt.hlines(experimental_mean_sub, xmin=-0.5, xmax=len(subgroup_seeds) - 0.5, colors="mediumaquamarine",
               linestyles='--', linewidth=2, label="Experimental Avg")

    # Solved threshold line
    plt.axhline(900, color="royalblue", linestyle="dotted", linewidth=2, label="Solved Threshold (900)")

    # X-axis labels
    plt.xticks(ticks=x, labels=[str(seed) for seed in subgroup_seeds])
    plt.xlabel("Seed")
    plt.ylabel("Reward")
    plt.title("Generalisation Test Reward Scores Comparison (Subgroup)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    #plot_training_reward()                     # raw, 50avg, high point comparison of 1 datasets
    #plot_sample_trajectory()                   # x

    #plot_model_architecture_vertical_compact() # model architecture manual graphic

    #plot_training_logs_with_ma50()             # raw, 50avg, high point comparison of 3 datasets
    #plot_training_logs_with_ma250()            # "", 250avg, ""
    #plot_high_reward_distribution()            # plots frequency of episodes with >800 reward value in the datasets

    #plot_generalization_rewards()              # graph the reward values of a 10 random seed generalization test run
    #model_generalization_comparison()          # choose to plot 3 models preloaded generalisation comparison scores vs each other
    plot_supergroup_comparison()               # preset super generalisation plot
    #plot_top_supergroup_comparison()           # preset best generalisation plots


