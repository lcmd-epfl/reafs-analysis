import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

data = pd.read_pickle("reafs_data/exp6_1_settings_merged_results.pkl")

data = data.loc[:, ~data.eq(data.iloc[0]).all()]
dataf = data.copy()

dataf_c = dataf.copy()

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

# Add dataset size column for complexity calculation
dataf_c["dataset_size"] = dataf_c["dataset_name"].map(lookup_dataset_size)


# Calculate complexity: features per sample
dataf_c["complexity"] = dataf_c["n_features"] / dataf_c["dataset_size"]

# Drop the temporary 'dataset_size' column
dataf_c.drop(columns=["dataset_size"], inplace=True)

ranks_per_method = dataf_c.groupby(['dataset_name', 'dataset_i_target', 'dataset_i_direction'])[['cv_r2','accuracy','discovery','cv_spearman','composite','n_features','complexity']].rank(ascending=False)
fs_methods = dataf_c[['feature_selector']]
raw_complexity = dataf_c[['complexity']].rename(columns={"complexity": "raw_complexity"})

dataf_cc = pd.concat((fs_methods,ranks_per_method,raw_complexity),axis=1).groupby(['feature_selector'],as_index=False).median()

dataf_cc.to_csv("results/fs_bench_seqs_1_0.csv", index=False)

metrics = ["composite", "accuracy", "cv_r2", "cv_spearman"]
labels = ["$S_{\\mathrm{ReaFS}}$", "$R$", "$Q^2_t$", "$r_{st}$"]


fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)
axes = axes.flatten()
for ax, metric, label in zip(axes, metrics, labels):
    # Extract complexity (mean number of features) and scores (mean metric)
    complexity = dataf_cc["raw_complexity"]
    scores = dataf_cc[metric]

    # Create the scatter plot
    scatter = ax.scatter(complexity, scores, color=tab10_new_c[3], label="Variants")
    vanilla = ax.scatter(complexity.iloc[102], scores.iloc[102], color=tab10_new_c[2], label="Base Seq.", s=90, marker='X', ec="#505050")
    best_com = ax.scatter(complexity.iloc[282], scores.iloc[282], color=tab10_new_c[1], label="Best $S_{\\mathrm{ReaFS}}$", s=90, marker='P', ec="#505050")
    best_sum = ax.scatter(complexity.iloc[135], scores.iloc[135], color=tab10_new_c[0], label="Best Rank Sum", s=50, marker='s', ec="#505050")
    best_rec = ax.scatter(complexity.iloc[365], scores.iloc[365], color=tab10_new_c[4], label="Best $R$", s=50, marker='v', ec="#505050")
    best_r2t = ax.scatter(complexity.iloc[371], scores.iloc[371], color=tab10_new_c[5], label="Best $Q^2_t$", s=50, marker='^', ec="#505050")
    best_rst = ax.scatter(complexity.iloc[195], scores.iloc[195], color=tab10_new_c[6], label="Best $r_{st}$", s=50, marker='<', ec="#505050")


    # Customize the plot
    ax.set_title(label, fontweight="bold")

    ax.set_ylim(360, 30)
    ax.set_xlim(-0.04, .72)

axes[0].set_ylabel("Median rank score", fontweight="bold")
axes[2].set_ylabel("Median rank score", fontweight="bold")

axes[2].set_xlabel("Median complexity", fontweight="bold")
axes[3].set_xlabel("Median complexity", fontweight="bold")

# axes[1].legend(loc='upper right', bbox_to_anchor=(1.0,1.5), ncols=4, frameon=True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, 
              bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=10)


# plt.tight_layout()

plt.savefig("results/fs_bench_seqs_1_0.svg")

plt.show()












