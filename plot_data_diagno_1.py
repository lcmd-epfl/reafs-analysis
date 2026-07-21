import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from reafs.datasets import full_datasets, from_dataset
from reafs.utils import local_outlier_factor, coeff_var_gaps, quantile_tail_density

from sklearn.neighbors import LocalOutlierFactor

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

def gen_dataset(datasets):
    for dataset in datasets:
        products = list(product([dataset[0]], dataset[1], dataset[2]))
        for p in products:
            yield p
rows = []

for dataset in gen_dataset(full_datasets):
    data_ds = from_dataset(*dataset)
    y_vals_before = np.sort(data_ds.y, stable=True)[:-5]  # remove top 5 values
    y_vals_after = np.sort(data_ds.y, stable=True)

    if dataset[0] == "lau2021":
        print(y_vals_after)


    if dataset[0] == "schoepfer2023_cc":
        print(y_vals_after)

    if dataset[0] == "wang2023_adi":
        print(y_vals_after)

    # clf_ds = LocalOutlierFactor()
    # _ = clf_ds.fit_predict(y_vals_before.reshape(-1, 1))
    # nlof_vals_before = clf_ds.negative_outlier_factor_

    # _ = clf_ds.fit_predict(y_vals_after.reshape(-1, 1))
    # nlof_vals_after = clf_ds.negative_outlier_factor_

    nlof_vals_before = local_outlier_factor(y_vals_before)
    nlof_vals_after = local_outlier_factor(y_vals_after)

    top_y_nlof_before = nlof_vals_before[-5:].mean()
    top_y_nlof_after = nlof_vals_after[-5:].mean()

    top_1_nlof_before = nlof_vals_before[-1]
    top_1_nlof_after = nlof_vals_after[-1]

    cvg_before = coeff_var_gaps(y_vals_before)
    cvg_after = coeff_var_gaps(y_vals_after)

    qtd_before = quantile_tail_density(y_vals_before, q=0.8)
    qtd_after = quantile_tail_density(y_vals_after, q=0.8)

    rows.append(
        {
            "dataset_name": dataset[0],
            "dataset_i_target": dataset[1],
            "dataset_i_direction": dataset[2],
            "size": len(data_ds.y),
            "nLOF_full_before": nlof_vals_before.mean(),
            "nLOF_top5_before": top_y_nlof_before,
            "nLOF_top1_before": top_1_nlof_before,
            "nLOF_full_after": nlof_vals_after.mean(),
            "nLOF_top5_after": top_y_nlof_after,
            "nLOF_top1_after": top_1_nlof_after,
            "nLOF_top5_diff": top_y_nlof_before - top_y_nlof_after,
            "cvg_before": cvg_before,
            "cvg_after": cvg_after,
            "qtd_before": qtd_before,
            "qtd_after": qtd_after,
            "qtd_90_before": quantile_tail_density(y_vals_before, q=0.9),
            "qtd_90_after": quantile_tail_density(y_vals_after, q=0.9),
            "qtd_66_before": quantile_tail_density(y_vals_before, q=0.666),
            "qtd_66_after": quantile_tail_density(y_vals_after, q=0.666),
        }
    )

df_nlof_top5 = pd.DataFrame(rows)

data = pd.read_pickle("reafs_data/exp6_1_settings_merged_results.pkl")

data = data.loc[:, ~data.eq(data.iloc[0]).all()]
dataf = data.copy()

dataf_c = dataf.copy()

# Map the max number of features using the lookup list
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
    "wang2023_ti": 44
}

lookup_dataset_size = {
    "cammarota2022": 25, # CHF
    "dotson2023_sel_pd": 52, # HH
    "dotson2023_sel_rh": 32, # HF
    "dotson2023_yield_pd": 33, # HH
    "dotson2023_yield_rh": 55, # HF
    "gallarati2022": 101, # CDA
    "gallarati2023": 407, # PS
    "haas2022": 51, # CA
    "lau2021": 29, # CC
    "schoepfer2023_cc": 29, # CC
    "schoepfer2023_cp": 30, # CP
    "schoepfer2023_da_f": 30, # DA
    "schoepfer2023_oa": 19, # OA
    "souza2023": 88, # CHI
    "wang2023_adi": 29, # CC
    "wang2023_ti": 21, # RCC
}

# Map the max number of features using the lookup list
dataf_c["dataset_size"] = dataf_c["dataset_name"].map(lookup_dataset_size)

# Recalculate complexity using the lookup list
dataf_c["complexity"] = dataf_c["n_features"] / dataf_c["dataset_size"]

# Drop the temporary 'dataset_size' column if not needed
dataf_c.drop(columns=["dataset_size"], inplace=True)


# Display the updated DataFrame
dataf_c

ranks_per_method = dataf_c.groupby(['dataset_name', 'dataset_i_target', 'dataset_i_direction'])[['cv_r2','accuracy','discovery','cv_spearman','composite','n_features','complexity']].rank(ascending=False)
fs_methods = dataf_c[['feature_selector']]
raw_complexity = dataf_c[['complexity']].rename(columns={"complexity": "raw_complexity"})


dataf_cc = pd.concat((fs_methods,ranks_per_method,raw_complexity),axis=1).groupby(['feature_selector'],as_index=False).median()
dataf_cc

df_merged = dataf_c.merge(
    df_nlof_top5,
    on=["dataset_name", "dataset_i_target", "dataset_i_direction"],
    how="left",
)

selected_idx = [102, 282, 365, 371, 195]

selected_feature_selectors = dataf_cc.loc[selected_idx, "feature_selector"]
df_merged = df_merged[df_merged["feature_selector"].isin(selected_feature_selectors)]

df_merged_mean = df_merged[
    [
        "dataset_name",
        "dataset_i_target",
        "dataset_i_direction",
        "composite",
        "nLOF_full_before",
        "nLOF_top5_before",
        "nLOF_top1_before",
        "nLOF_full_after",
        "nLOF_top5_after",
        "nLOF_top1_after",
        "cvg_before",
        "cvg_after",
        "qtd_before",
        "qtd_after",
        "qtd_90_before",
        "qtd_90_after",
        "qtd_66_before",
        "qtd_66_after",
    ]
].groupby(["dataset_name", "dataset_i_target", "dataset_i_direction"], as_index=False).mean()

plot_df = df_merged_mean.copy()
markers = ["o", "X", "s", "^", "D", "v", "P", "*", "<", ">", "p", "h", "d", "H", "8", "."]

reaction_lookup = {
    "cammarota2022": "CHF",
    "dotson2023_sel_pd": "HH",
    "dotson2023_sel_rh": "HF",
    "dotson2023_yield_pd": "HH",
    "dotson2023_yield_rh": "HF",
    "gallarati2022": "CDA",
    "gallarati2023": "PS",
    "haas2022": "AC",
    "lau2021": "CC",
    "schoepfer2023_cc": "CC",
    "schoepfer2023_cp": "CP",
    "schoepfer2023_da_f": "DA",
    "schoepfer2023_oa": "OA",
    "souza2023": "CHI",
    "wang2023_adi": "CC",
    "wang2023_ti": "RCC",
}

plot_df["reaction"] = plot_df["dataset_name"].map(reaction_lookup)

missing_reactions = sorted(plot_df.loc[plot_df["reaction"].isna(), "dataset_name"].unique())
if missing_reactions:
    raise ValueError(f"Missing reaction mapping for: {missing_reactions}")

reaction_order = ["CHF", "HH", "HF", "CDA", "PS", "AC", "CC", "CP", "DA", "OA", "CHI", "RCC"]
tab20_colors = list(plt.get_cmap("tab20").colors)
reaction_colors = {reaction: tab20_colors[i] for i, reaction in enumerate(reaction_order)}

fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=True)

for i, reaction in enumerate(reaction_order):
    subset = plot_df[plot_df["reaction"] == reaction]
    if subset.empty:
        continue
    axes[0].scatter(
        subset["cvg_before"],
        subset["composite"],
        label=reaction,
        s=100,
        alpha=0.85,
        color=reaction_colors[reaction],
        marker=markers[i % len(markers)],
        ec="#505050",
        linewidth=0.5,
    )

axes[0].set_ylabel("$S_\mathrm{ReaFS}$")
axes[0].set_xlabel("CVG")
axes[0].set_title("")
# axes[0].grid(True, alpha=0.3)

for i, reaction in enumerate(reaction_order):
    subset = plot_df[plot_df["reaction"] == reaction]
    if subset.empty:
        continue
    axes[1].scatter(
        subset["qtd_before"],
        subset["composite"],
        label=reaction,
        s=100,
        alpha=0.85,
        color=reaction_colors[reaction],
        marker=markers[i % len(markers)],
        ec="#505050",
        linewidth=0.5,
    )

# axes[1].set_ylabel("$S_\mathrm{ReaFS}$")
axes[1].set_xlabel("80th PTD")
axes[1].set_title("")
# axes[1].grid(True, alpha=0.3)

for i, reaction in enumerate(reaction_order):
    subset = plot_df[plot_df["reaction"] == reaction]
    if subset.empty:
        continue
    axes[2].scatter(
        subset["nLOF_top1_before"],
        subset["composite"],
        label=reaction,
        s=100,
        alpha=0.85,
        color=reaction_colors[reaction],
        marker=markers[i % len(markers)],
        ec="#505050",
        linewidth=0.5,
    )

# axes[2].set_ylabel("$S_\mathrm{ReaFS}$")
axes[2].set_xlabel("nLOF")
axes[2].set_title("")
# axes[2].grid(True, alpha=0.3)

# axes[0].legend(loc="upper left", ncols=3, frameon=False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, 
              bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=10)

# plt.tight_layout()
plt.savefig("results/data_diagno_0_1.svg")
plt.show()


### grid SI plot with comprehensive metrics
fig_si, axes_si = plt.subplots(4, 3, figsize=(12, 16))

# Define metric pairs: (column_before, column_after, y_label)
metrics = [
    ("nLOF_top5_before", "nLOF_top5_after", "nLOF Top 5"),
    ("nLOF_top1_before", "nLOF_top1_after", "nLOF Top 1"),
    ("nLOF_full_before", "nLOF_full_after", "nLOF Full"),
    ("cvg_before", "cvg_after", "CVG"),
    ("qtd_before", "qtd_after", "PTD (80%)"),
    ("qtd_90_before", "qtd_90_after", "PTD (90%)"),
]

# Create 4x3 layout: grouped by metric pairs
# Row 0: M0 before, M1 before, M2 before
# Row 1: M0 after, M1 after, M2 after
# Row 2: M3 before, M4 before, M5 before
# Row 3: M3 after, M4 after, M5 after
plot_sequence = [
    (0, "before"), (1, "before"), (2, "before"),
    (0, "after"), (1, "after"), (2, "after"),
    (3, "before"), (4, "before"), (5, "before"),
    (3, "after"), (4, "after"), (5, "after"),
]

for plot_position, (metric_idx, before_after) in enumerate(plot_sequence):
    row_idx = plot_position // 3
    col_idx = plot_position % 3
    ax = axes_si[row_idx, col_idx]
    
    col_before, col_after, metric_label = metrics[metric_idx]
    col = col_before if before_after == "before" else col_after
    label_suffix = "(Top 5 Removed)" if before_after == "before" else "(All Data)"
    
    for i, reaction in enumerate(reaction_order):
        subset = plot_df[plot_df["reaction"] == reaction]
        if subset.empty:
            continue
        ax.scatter(
            subset["composite"],
            subset[col],
            label=reaction,
            s=100,
            alpha=0.85,
            color=reaction_colors[reaction],
            marker=markers[i % len(markers)],
            ec="#505050",
            linewidth=0.5,
        )
    ax.set_xlabel("$S_\mathrm{ReaFS}$")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} {label_suffix}", fontweight="bold")
    ax.grid(True, alpha=0.2)

# Add a single legend at the top
handles, labels = axes_si[0, 0].get_legend_handles_labels()
fig_si.legend(handles, labels, loc="lower center", ncol=6, 
              bbox_to_anchor=(0.5, 0.03), frameon=False, fontsize=10)

# Manual spacing control instead of tight_layout
plt.subplots_adjust(left=0.1, right=1, top=1, bottom=0.1, wspace=.3, hspace=0.3)
plt.savefig("results/data_diagno_SI_grid_1.svg", dpi=300, bbox_inches=None)
# plt.savefig("results/data_diagno_SI_grid.png", dpi=300, bbox_inches='tight')
plt.show()



