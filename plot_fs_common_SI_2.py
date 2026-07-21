import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

LOOKUP_DATASET_SIZE = {
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

METHOD_ORDER = [8, 7, 3, 0, 4, 5, 6, 1, 2]
X_LABELS = [
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
X_POSITIONS = np.arange(len(X_LABELS))
METHOD_LABELS = {
    "bounded": "bounded $S_{\\mathrm{ReaFS}}$",
    "unbounded": "Unbounded $uS_{\\mathrm{ReaFS}}$",
}
METHOD_COLORS = {
    "bounded": "#007480",
    "unbounded": "#f39869",
}

DATASET_LABELS = [
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
    "RCC",
]
DATASET_POSITIONS = np.arange(len(DATASET_LABELS))


def load_results(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_pickle(path) for path in paths]
    data = pd.concat(frames)
    data = data.loc[:, ~data.eq(data.iloc[0]).all()].copy()

    undefined_spearman = data["cv_spearman"] == -2
    data["composite_bar"] = data["composite"].mask(undefined_spearman, np.nan)
    data["composite_violin"] = data["composite"].mask(undefined_spearman, 0.0)
    data["cv_spearman"] = data["cv_spearman"].mask(undefined_spearman, np.nan)

    data["dataset_size"] = data["dataset_name"].map(LOOKUP_DATASET_SIZE)
    data["complexity"] = data["n_features"] / data["dataset_size"]
    data.drop(columns=["dataset_size"], inplace=True)

    return data


def summarize_composite_ranks(data: pd.DataFrame) -> pd.DataFrame:
    ranks_per_method = data.groupby(
        ["dataset_name", "dataset_i_target", "dataset_i_direction"]
    )[["composite_bar"]].rank(ascending=False)
    fs_methods = data[["base_estimator", "feature_selector"]]
    return (
        pd.concat((fs_methods, ranks_per_method), axis=1)
        .groupby(["base_estimator", "feature_selector"])
        .median()
    )


def summarize_composite_values(data: pd.DataFrame) -> pd.Series:
    return data.groupby(["base_estimator", "feature_selector"])["composite_violin"].apply(
        lambda series: series.to_numpy()
    )


def summarize_dataset_composite_values(data: pd.DataFrame) -> list[np.ndarray]:
    grouped = data.sort_values(
        ["feature_selector", "dataset_name", "dataset_i_target", "dataset_i_direction"]
    ).groupby(["dataset_name", "dataset_i_target", "dataset_i_direction"])
    return [group["composite_violin"].values for _, group in grouped]


def plot_bar_comparison(bounded_summary: pd.DataFrame, unbounded_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 0.35

    bounded_values = bounded_summary["composite_bar"].iloc[METHOD_ORDER]
    unbounded_values = unbounded_summary["composite_bar"].iloc[METHOD_ORDER]

    ax.bar(
        X_POSITIONS - bar_width / 2,
        bounded_values - 9,
        bottom=9,
        width=bar_width,
        color=METHOD_COLORS["bounded"],
        label=METHOD_LABELS["bounded"],
    )
    ax.bar(
        X_POSITIONS + bar_width / 2,
        unbounded_values - 9,
        bottom=9,
        hatch="..",
        width=bar_width,
        color=METHOD_COLORS["unbounded"],
        label=METHOD_LABELS["unbounded"],
    )

    ax.invert_yaxis()
    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(X_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Median Rank")
    ax.set_ylim(9.5, 0.5)
    ax.legend(loc="upper right")
    # plt.tight_layout()
    plt.savefig("results/fs_bench_composite_bar_bounded_vs_unbounded.svg")
    plt.show()


def plot_violin_comparison(bounded_values: pd.Series, unbounded_values: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    violin_width = 0.32

    bounded_data = bounded_values.iloc[METHOD_ORDER].tolist()
    unbounded_data = unbounded_values.iloc[METHOD_ORDER].tolist()

    bounded_plot = ax.violinplot(
        bounded_data,
        positions=X_POSITIONS - violin_width / 2,
        widths=violin_width,
        showmeans=True,
        showmedians=True,
    )
    unbounded_plot = ax.violinplot(
        unbounded_data,
        positions=X_POSITIONS + violin_width / 2,
        widths=violin_width,
        showmeans=True,
        showmedians=True,
    )

    for body in bounded_plot["bodies"]:
        body.set_facecolor(METHOD_COLORS["bounded"])
        body.set_alpha(0.75)
    for body in unbounded_plot["bodies"]:
        body.set_facecolor(METHOD_COLORS["unbounded"])
        body.set_alpha(0.75)

    for violin in (bounded_plot, unbounded_plot):
        violin["cmeans"].set_color("#FF0000")
        violin["cmedians"].set_color("#000000")
        violin["cbars"].set_color("#444444")
        violin["cmins"].set_color("#444444")
        violin["cmaxes"].set_color("#444444")

    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(X_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Composite")
    # ax.legend(
    #     handles=[
    #         plt.Line2D([0], [0], color=METHOD_COLORS["bounded"], lw=8),
    #         plt.Line2D([0], [0], color=METHOD_COLORS["unbounded"], lw=8),
    #     ],
    #     labels=[METHOD_LABELS["bounded"], METHOD_LABELS["unbounded"]],
    #     loc="upper right",
    # )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/fs_bench_composite_violin_bounded_vs_unbounded.svg")
    plt.show()


def plot_dataset_violin_comparison(
    bounded_dataset_values: list[np.ndarray], unbounded_dataset_values: list[np.ndarray]
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    violin_width = 0.32
    bounded_data = bounded_dataset_values
    unbounded_data = unbounded_dataset_values

    bounded_plot = ax.violinplot(
        bounded_data,
        positions=DATASET_POSITIONS - violin_width / 2,
        widths=violin_width,
        showmeans=True,
        showmedians=True,
    )
    unbounded_plot = ax.violinplot(
        unbounded_data,
        positions=DATASET_POSITIONS + violin_width / 2,
        widths=violin_width,
        showmeans=True,
        showmedians=True,
    )

    for body in bounded_plot["bodies"]:
        body.set_facecolor(METHOD_COLORS["bounded"])
        body.set_alpha(0.75)
    for body in unbounded_plot["bodies"]:
        body.set_facecolor(METHOD_COLORS["unbounded"])
        body.set_alpha(0.75)

    for violin in (bounded_plot, unbounded_plot):
        violin["cmeans"].set_color("#FF0000")
        violin["cmedians"].set_color("#000000")
        violin["cbars"].set_color("#444444")
        violin["cmins"].set_color("#444444")
        violin["cmaxes"].set_color("#444444")

    ax.set_xticks(DATASET_POSITIONS)
    ax.set_xticklabels(DATASET_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Composite")
    ax.set_xlabel("Dataset Index")
    # ax.legend(
    #     handles=[
    #         plt.Line2D([0], [0], color=METHOD_COLORS["bounded"], lw=8),
    #         plt.Line2D([0], [0], color=METHOD_COLORS["unbounded"], lw=8),
    #     ],
    #     labels=[METHOD_LABELS["bounded"], METHOD_LABELS["unbounded"]],
    #     loc="upper right",
    # )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/fs_bench_composite_dataset_violin_bounded_vs_unbounded.svg")
    plt.show()


try:
    bounded_data = load_results(
        [
            "reafs_data/exp0_settings_merged_results.pkl",
            "reafs_data/exp5_settings_merged_results.pkl",
        ]
    )
    unbounded_data = load_results(
        [
            "reafs_data/exp7_settings_merged_results.pkl",
            "reafs_data/exp8_settings_merged_results.pkl",
        ]
    )
except FileNotFoundError:
    print("Error: Required pickle files not found. Please check reaf_data/ folder.")
    raise SystemExit(1)

bounded_summary = summarize_composite_ranks(bounded_data)
unbounded_summary = summarize_composite_ranks(unbounded_data)

bounded_values = summarize_composite_values(bounded_data)
unbounded_values = summarize_composite_values(unbounded_data)

bounded_dataset_values = summarize_dataset_composite_values(bounded_data)
unbounded_dataset_values = summarize_dataset_composite_values(unbounded_data)

plot_bar_comparison(bounded_summary, unbounded_summary)
plot_dataset_violin_comparison(bounded_dataset_values, unbounded_dataset_values)
plot_violin_comparison(bounded_values, unbounded_values)
