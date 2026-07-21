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
data_c.loc[data_c["cv_spearman"] == -2, "cv_spearman"] = 0

# ###
print(data_c)

data_c_groups = data_c.sort_values(["feature_selector","dataset_name","dataset_i_target","dataset_i_direction"]).groupby(["dataset_name", "dataset_i_target", "dataset_i_direction"])

composites_for_violin = []
recalls_for_violin = []
qt2s_for_violin = []
rsts_for_violin = []

just_to_make_sure_label = []

for name, group in data_c_groups:
    composites_for_violin.append(group["composite"].values)
    recalls_for_violin.append(group["accuracy"].values)
    qt2s_for_violin.append(group["cv_r2"].values)
    rsts_for_violin.append(group["cv_spearman"].values)
    print(name)

dataset_names_lut = [
    "CHF-T",
    "CHF-B",
    "HH-S",
    "HF-S",
    "HH-Y",
    "HF-Y",
    "CDA-G",
    "CDA-T",
    "PS",
    "AC",
    "CC-O",
    "CC-BL",
    "CP-BL",
    "DA-BL",
    "OA-BL",
    "CHI-G",
    "CHI-S",
    "CHI-A",
    "CC-R",
    "RCC"
]

print("Labels for violin plots:", group["feature_selector"].values)

composites_for_violin = np.asarray(composites_for_violin)
recalls_for_violin = np.asarray(recalls_for_violin)
qt2s_for_violin = np.asarray(qt2s_for_violin)
rsts_for_violin = np.asarray(rsts_for_violin)

# ### Dataset distrubution

fig, axes = plt.subplots(4, figsize=(10, 16))

for ax, data, title, main_color in zip(
    axes,
    [composites_for_violin, recalls_for_violin, qt2s_for_violin, rsts_for_violin],
    ["$S_{\mathrm{ReaFS}}$", "$R$", "$Q^2_t$", "$r_{st}$"],
    ["#007480", "#b51f1f", "#f39869", "#c2ddb0"],
):
    violin = ax.violinplot(data.T, showmeans=True, showmedians=True)
    violin["cmeans"].set_color(main_color)
    violin["cmedians"].set_color("#000000")
    violin["cbars"].set_color(main_color)
    violin["cmins"].set_color(main_color)
    violin["cmaxes"].set_color(main_color)

    for body in violin["bodies"]:
        body.set_facecolor(main_color)

    ax.set_ylabel(title, fontweight="bold")
    ax.set_xticks([])

    if title == "$r_{st}$":
        ax.set_xlabel("Dataset Index", fontweight="bold")
        ax.set_xticks(np.arange(1, data.shape[0] + 1))
        # ax.set_xticklabels([f"{i+1}" for i in range(data.shape[0])])
        ax.set_xticklabels(dataset_names_lut, rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.savefig("results/fs_bench_data_dist_SI_1_0.svg")
plt.show()

# ### Methods distrubution

order = [8, 6, 7, 2, 3, 4, 5, 0, 1]

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

fig, axes = plt.subplots(4, figsize=(10, 16))

for ax, data, title, main_color in zip(
    axes,
    [composites_for_violin.T, recalls_for_violin.T, qt2s_for_violin.T, rsts_for_violin.T],
    ["$S_{\mathrm{ReaFS}}$", "$R$", "$Q^2_t$", "$r_{st}$"],
    ["#007480", "#b51f1f", "#f39869", "#c2ddb0"],
):
    print(data.shape)
    print(data.T.shape)
    
    violin = ax.violinplot(data[order].T, showmeans=True, showmedians=True)
    violin["cmeans"].set_color(main_color)
    violin["cmedians"].set_color("#000000")
    violin["cbars"].set_color(main_color)
    violin["cmins"].set_color(main_color)
    violin["cmaxes"].set_color(main_color)

    for body in violin["bodies"]:
        body.set_facecolor(main_color)

    ax.set_ylabel(title, fontweight="bold")

    ax.set_xticks([])

    if title == "$r_{st}$":
        ax.set_xlabel("FS methods", fontweight="bold")
        ax.set_xticks(x_positions + 1)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.savefig("results/fs_bench_method_dist_SI_1_0.svg")
plt.show()

