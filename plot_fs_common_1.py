import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# Feature Selection Comparison Script
# -----------------------------
# This script compares feature selection methods across multiple datasets.
# It loads results, computes ranks, and visualizes performance and complexity.
# Usage: Run directly, ensure required pickle files are present in reaf_data/.

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

try:
    # Load experiment results
    data_0 = pd.read_pickle("reafs_data/exp0_settings_merged_results.pkl")
    data_1 = pd.read_pickle("reafs_data/exp5_settings_merged_results.pkl")
except FileNotFoundError:
    print("Error: Required pickle files not found. Please check reaf_data/ folder.")
    exit(1)

# Combine datasets
data = pd.concat((data_0, data_1))

# Remove columns where all values are the same
data = data.loc[:, ~data.eq(data.iloc[0]).all()]

# Copy for further processing
data_c = data.copy()

# Map the max number of features using the lookup list
# Lookup tables for dataset properties
lookup_max_features = {
    "cammarota2022": 117,
    "dotson2023_sel_pd": 508,
    "dotson2023_sel_rh": 508,
    "dotson2023_yield_pd": 508,
    "dotson2023_yield_rh": 508,
    "gallarati2022": 26,
    "gallarati2023": 119,
    "haas2022": 82,
    "lau2021": 22,
    "schoepfer2023_cc": 232,
    "schoepfer2023_cp": 232,
    "schoepfer2023_da_f": 232,
    "schoepfer2023_oa": 232,
    "souza2023": 70,
    "wang2023_adi": 44,
    "wang2023_ti": 44,
}

lookup_dataset_size = {
    "cammarota2022": 25,
    "dotson2023_sel_pd": 52,
    "dotson2023_sel_rh": 32,
    "dotson2023_yield_pd": 33,
    "dotson2023_yield_rh": 55,
    "gallarati2022": 101,
    "gallarati2023": 407,
    "haas2022": 51,
    "lau2021": 29,
    "schoepfer2023_cc": 29,
    "schoepfer2023_cp": 30,
    "schoepfer2023_da_f": 30,
    "schoepfer2023_oa": 19,
    "souza2023": 88,
    "wang2023_adi": 29,
    "wang2023_ti": 21,
}

# Add dataset size column for complexity calculation
data_c["dataset_size"] = data_c["dataset_name"].map(lookup_dataset_size)


# Calculate complexity: features per sample
data_c["complexity"] = data_c["n_features"] / data_c["dataset_size"]

# Drop the temporary 'dataset_size' column
data_c.drop(columns=["dataset_size"], inplace=True)

# Post correction of composite due to undefined cv_spearman values
data_c.loc[data_c["cv_spearman"] == -2, "composite"] = (data_c.loc[data_c["cv_spearman"] == -2, "composite"] * 3) / 2

# Treat undefined cv_spearman values as missing so they are ignored in rank/median
data_c.loc[data_c["cv_spearman"] == -2, "cv_spearman"] = np.nan



# Compute ranks for each method within each dataset and target
ranks_per_method = data_c.groupby(
    ["dataset_name", "dataset_i_target", "dataset_i_direction"]
)[
    [
        "cv_r2",
        "accuracy",
        "discovery",
        "cv_spearman",
        "composite",
        "n_features",
        "complexity",
    ]
].rank(
    ascending=False
)

# Extract method identifiers for grouping
fs_methods = data_c[["base_estimator", "feature_selector"]]

# Extract raw complexity for later use in plots
raw_complexity = data_c[["complexity"]].rename(columns={"complexity": "raw_complexity"})

# Combine method identifiers, ranks, and complexity into a single DataFrame for analysis
data_cc = (
    pd.concat((fs_methods, ranks_per_method, raw_complexity), axis=1)
    .groupby(["base_estimator", "feature_selector"])
    .median()
)

order = [8, 7, 3, 0, 4, 5, 6, 1, 2]

metric = "composite"

x_labels = [
    "All Features",
    "Sequential",
    "ARD+BRR",
    "ARD+ARD",
    "LASSO",
    "KBest F",
    "KBest MI",
    "Beam Collin.",
    "RFE",
]

x_positions = np.arange(len(x_labels))

# -----------------------------
# Plotting the median ranks for each metric
# -----------------------------

bar_width = 0.8
fig, ax = plt.subplots(figsize=(6, 4))

means = data_cc[metric].iloc[order]

ax.bar(
    x_positions - 0.3,
    data_cc["composite"].iloc[order] - 9,
    0.2,
    bottom=9,
    color="#007480",
    hatch="",
    zorder=1,
    label="$S_{\mathrm{ReaFS}}$",
)
ax.bar(
    x_positions - 0.1,
    data_cc["accuracy"].iloc[order] - 9,
    0.2,
    bottom=9,
    color="#b51f1f",
    hatch="//",
    zorder=1,
    label="$R$",
)
ax.bar(
    x_positions + 0.1,
    data_cc["cv_r2"].iloc[order] - 9,
    0.2,
    bottom=9,
    color="#f39869",
    hatch="..",
    zorder=1,
    label="$Q^2_t$",
)
ax.bar(
    x_positions + 0.3,
    data_cc["cv_spearman"].iloc[order] - 9,
    0.2,
    bottom=9,
    color="#c2ddb0",
    hatch="\\\\",
    zorder=1,
    label="$r_{st}$",
)

ax.invert_yaxis()

# Customize the plot
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("Median Rank")
ax.set_ylim(None, 1)
# ax.set_title("Scores by Feature Selector", fontweight='bold')
ax.legend(loc="upper right", bbox_to_anchor=(0.85, 1))
# plt.tight_layout()
plt.savefig("results/fs_bench_full_1_0.svg")
plt.show()

# -----------------------------
# Plotting the Pareto front for each metric bwtween complexity and performance
# -----------------------------

metrics = ["composite", "accuracy", "cv_r2", "cv_spearman"]
labels = ["$S_{\mathrm{ReaFS}}$", "$R$", "$Q^2_t$", "$r_{st}$"]

epfl_colors_c = [
    "#00A79F",
    "#FF0000",
    "#F39869",
    "#C2DDB0",
    "#5C2483",
    "#5B3428",
    "#ED6E9C",
    "#CAC7C7",
    "#C8D300",
]

tab10_new_c =[
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC949",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

tab20_custom = [
    "#4E79A7",
    "#A5BCD5",
    "#F28E2B",
    "#F9CC9F",
    "#E15759",
    "#F4C2C3",
    "#76B7B2",
    "#C1E1DF",
    "#59A14F",
    "#A8D2A3",
    "#EDC949",
    "#F8EAB9",
    "#B07AA1",
    "#DFC9D9"

]

color_order = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]

markers = ["o", "X", "s", "^", "D", "v", "P", "*", "<", ">", "p", "h", "d", "H", "8", "."]

fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)
axes = axes.flatten()
for ax, metric, label in zip(axes, metrics, labels):
    # Extract complexity (mean number of features) and scores (mean metric)
    complexity = data_cc["raw_complexity"].iloc[order]
    scores = data_cc[metric].iloc[order]

    # Create the scatter plot point-by-point because matplotlib only accepts
    # one marker style per scatter call.
    for i, (x, y) in enumerate(zip(complexity, scores)):
        ax.scatter(
            x,
            y,
            color=tab10_new_c[color_order[i]],
            marker=markers[i],
            s=60,
            ec="#505050",
        )

    sorted_indices = np.argsort(complexity)
    pareto_front = np.minimum.accumulate(scores.iloc[sorted_indices])

    ax.plot(
        complexity.iloc[sorted_indices],
        pareto_front,
        linestyle="--",
        color="gray",
        label="Pareto Front",
    )

    # Customize the plot
    ax.set_title(label, fontweight="bold")

    ax.set_ylim(9.5, 0.5)
    ax.set_yticks(np.arange(1, 10))

axes[0].set_ylabel("Median rank score", fontweight="bold")
axes[2].set_ylabel("Median rank score", fontweight="bold")

axes[2].set_xlabel("Median complexity", fontweight="bold")
axes[3].set_xlabel("Median complexity", fontweight="bold")

# plt.tight_layout()
plt.savefig("results/fs_bench_pareto_1_0.svg")
plt.show()
