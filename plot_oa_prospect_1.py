import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from reafs.datasets import from_dataset
from reafs.utils import (
    cov_det,
    custom_correlation_matrix,
    highest_absolute_off_diagonal,
    laplacian_score,
)
from reafs.metrics import (
    TestScore,
    TrainScore,
    DiffScore,
    YRandomization,
    ncrps_score,
    noise_resilience,
    jackknife
)
from reafs.cross_validation import LeaveOneMoreOut, LeaveTopPOut
from reafs.feature_selection import BeamSequential
from reafs.settings import reafs_base_evaluator

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import LeavePOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
from sklearn.base import clone
from sklearn.utils import check_X_y

from scipy.stats import spearmanr

from src.fs_selector import fs_selector

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

epfl_colors_c = [
    "#00A79F",
    "#F39869",
    "#C2DDB0",
    "#FF0000",
    "#5C2483",
    "#5B3428",
    "#ED6E9C",
    "#CAC7C7",
    "#C8D300",
    "#4F8FCC",
    "#EC6608",
    "#FBEE66",
    "#B51F1F",
    "#007480",
]

cmap = ListedColormap(epfl_colors_c)

data = from_dataset("schoepfer2023_oa", i_target=0, direction=1)

mlr_base = make_pipeline(StandardScaler(), BayesianRidge())

evaluation_splitter = LeaveTopPOut(5)

splits = evaluation_splitter.split(
    data.X, data.y
)

split_train, split_test = next(splits)

print(split_train)
print(split_test)

fs_1 = BeamSequential(
    estimator=clone(mlr_base),
    n_features_to_select="auto",
    tol=1e-3,
    scoring=TestScore(cv=LeaveOneMoreOut(n_repeats=5,n_min=1),scoring="neg_mean_squared_error"),
    n_candidates=10
)

fs_5 = BeamSequential(
    estimator=clone(mlr_base),
    n_features_to_select="auto",
    tol=1e-3,
    scoring=TestScore(cv=LeavePOut(1),scoring="neg_mean_squared_error"),
    n_candidates=1
)


def reafs_fs_selector(fs, X, y, split_train):

    X, y = check_X_y(X, y)

    fs_selector = make_pipeline(
        StandardScaler(),
        fs,
    )

    fs_selector.fit(X[split_train], y[split_train])

    return fs_selector[-1].get_support(indices=True)

def stepwise_fs_selector(X, y, split_train):

    X_train, y_train = X.iloc[split_train], y.iloc[split_train]

    data = pd.concat([X_train, y_train], axis=1)

    res_sel = fs_selector(data, "ddg")

    selected_features_names = res_sel["Model"].iloc[0]
    selected_features_idx = [X.columns.get_loc(col) for col in selected_features_names]

    return selected_features_idx
import sys
if len(sys.argv) > 1:
    exp_i = int(sys.argv[1])
else:
    exp_i = 2

if exp_i == 0:
    selected_features = stepwise_fs_selector(data.X, data.y, split_train)
    fancy_names = ["$\\mu_{f}$", "$\\hat{Q}^{4}_{Bur.}$", "$\\Sigma NBO$"]
elif exp_i == 1:
    selected_features = reafs_fs_selector(fs_1, data.X, data.y, split_train)
    fancy_names = ["$O^{3}_{nH}$", "$\\hat{Z}$", "$\\hat{Q}^{4}_{nH}$", "$\\hat{H}^{4}_{Bur.}$", "$\\hat{H}^{6}_{Bur.}$"]
elif exp_i == 2:
    selected_features = reafs_fs_selector(fs_5, data.X, data.y, split_train)
    fancy_names = ["$\\Theta$", "$O^{2}_{Bur.}$", "$O^{5}_{Bur.}$", "$^{2}\\hat{\\kappa}$", "$\\hat{Q}^{3}$"]

selected_feature_names = data.X.columns[selected_features].tolist()
display_feature_names = (
    fancy_names if len(fancy_names) == len(selected_feature_names) else selected_feature_names
)

results = {}

X, y = check_X_y(data.X, data.y)

X_train, y_train = X[split_train][:, selected_features], y[split_train]
X_test, y_test = X[split_test][:, selected_features], y[split_test]

results["n_train_samples"] = X_train.shape[0]
results["n_test_samples"] = X_test.shape[0]
results["n_samples"] = X.shape[0]
results["n_features"] = X.shape[1]
results["n_selected_features"] = len(selected_features)

mlr_base.fit(X_train, y_train)

y_train_preds = mlr_base.predict(X_train)

coefs = mlr_base[-1].coef_
intercept = mlr_base[-1].intercept_

results["r2_train"] = r2_score(y_train, y_train_preds)
results["mae_train"] = mean_absolute_error(y_train, y_train_preds)
results["rmse_train"] = np.sqrt(mean_squared_error(y_train, y_train_preds))

y_test_preds = mlr_base.predict(X_test)

results["r2_test"] = r2_score(y_test, y_test_preds)
results["mae_test"] = mean_absolute_error(y_test, y_test_preds)
results["rmse_test"] = np.sqrt(mean_squared_error(y_test, y_test_preds))

y_loo_preds = cross_val_predict(mlr_base, X_train, y_train, cv=LeavePOut(1))

results["r2_loo_train"] = r2_score(y_train, y_loo_preds)
results["mae_loo_train"] = mean_absolute_error(y_train, y_loo_preds)
results["rmse_loo_train"] = np.sqrt(mean_squared_error(y_train, y_loo_preds))

reafs_results = reafs_base_evaluator()(mlr_base, X_train, y_train, X_test, y_test)

j_scores, j_coefs = jackknife(mlr_base, X_train, y_train, scoring="neg_mean_squared_error")

results["jackknife_score"] = np.sqrt(np.abs(j_scores))
results["jackknife_coefs"] = j_coefs

# parity plot

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(y_train, y_train_preds, color="#007480", s=60, marker="D", label="Train")
ax.scatter(y_test, y_test_preds, color="#B51F1F", s=60, marker="o", label="Test")
ax.plot(
    [data.y.min(), data.y.max()],
    [data.y.min(), data.y.max()],
    ls="--",
    lw=2,
    color="#CAC7C7",
)
ax.set_xlabel("True $\Delta \Delta G^{\ddagger}$ (kcal/mol)")
ax.set_ylabel("Predicted $\Delta \Delta G^{\ddagger}$ (kcal/mol)")

ax.text(
    0.04,
    0.96,
    f"""n={results['n_samples']}
$R^2={results['r2_train']:.3f}$
$R^{{2}}_{{loo}}={results['r2_loo_train']:.3f}$
$MAE={results['mae_train']:.3f}$
$MAE_{{loo}}={results['mae_loo_train']:.3f}$
""",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
)

ax.legend()

plt.savefig(f"results/model_prospect_parity_plot_{exp_i}.svg", bbox_inches=None)

plt.show()

# PCA projection plot (inspired by plot_model_cards.py)
def two_component_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2-dimensional")
    if vectors.shape[1] >= 2:
        return vectors[:, :2]
    return np.column_stack([vectors[:, 0], np.zeros(vectors.shape[0])])


def compute_arrow_scale(scores, vectors, target_fraction=0.35):
    score_span_x = np.ptp(scores[:, 0]) if scores.shape[1] > 0 else 0.0
    score_span_y = np.ptp(scores[:, 1]) if scores.shape[1] > 1 else 0.0
    score_span = max(score_span_x, score_span_y)

    vector_extent = np.max(np.abs(vectors))
    if np.isclose(vector_extent, 0.0):
        return 1.0
    if np.isclose(score_span, 0.0):
        score_span = 1.0

    return target_fraction * score_span / vector_extent


def plot_feature_arrows(ax, vectors, feature_names, *, scale=1.0, color="black"):
    for i, feature_name in enumerate(feature_names):
        dx = vectors[i, 0] * scale
        dy = vectors[i, 1] * scale
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", lw=1.6, color=color),
        )
        ax.text(
            dx * 1.08,
            dy * 1.08,
            feature_name,
            color=color,
            fontsize=10,
            ha="center",
            va="center",
        )


def fit_axes_to_scores_and_arrows(
    ax, scores, vectors, *, scale=1.0, margin_fraction=0.15
):
    arrow_endpoints = vectors * scale
    x_values = np.concatenate([scores[:, 0], arrow_endpoints[:, 0]])
    y_values = np.concatenate([scores[:, 1], arrow_endpoints[:, 1]])

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    x_span = x_max - x_min
    y_span = y_max - y_min

    if np.isclose(x_span, 0.0):
        x_span = 1.0
    if np.isclose(y_span, 0.0):
        y_span = 1.0

    x_margin = x_span * margin_fraction
    y_margin = y_span * margin_fraction

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)


def make_pca_projection(X, n_components=2):
    n_samples, n_features = X.shape
    n_components_feasible = max(1, min(n_components, n_features, n_samples - 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components_feasible)
    x_scores = pca.fit_transform(X_scaled)
    return x_scores, pca, scaler

# Fit PCA on train, project test
X_train_pca, pca_model, pca_scaler = make_pca_projection(X_train, n_components=2)
X_test_scaled = pca_scaler.transform(X_test)
X_test_pca = pca_model.transform(X_test_scaled)

pc_var = pca_model.explained_variance_ratio_ * 100
pc1_var = pc_var[0]
pc2_var = pc_var[1] if pc_var.shape[0] > 1 else 0.0


fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    color="#007480",
    s=60,
    marker="D",
    label="Train",
)
ax.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    color="#B51F1F",
    s=60,
    marker="o",
    label="Test",
)

all_scores = np.vstack([X_train_pca[:, :2], X_test_pca[:, :2]])
loadings = two_component_vectors(pca_model.components_.T)
arrow_scale = compute_arrow_scale(all_scores, loadings)

plot_feature_arrows(
    ax,
    loadings,
    display_feature_names,
    scale=arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(ax, all_scores, loadings, scale=arrow_scale)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(f"PC 1 ({pc1_var:.1f}%)")
ax.set_ylabel(f"PC 2 ({pc2_var:.1f}%)")
ax.legend(loc="best")
plt.savefig(f"results/model_prospect_pca_projection_{exp_i}.svg", bbox_inches=None)
plt.show()

def pairwise_distance_calc(X, y, *, metric="euclidean", standardize=True):
    """Return pairwise distances for X and y and the corresponding index pairs."""
    X_array = np.asarray(X)
    y_array = np.asarray(y).reshape(-1, 1)

    if standardize:
        X_mean = X_array.mean(axis=0)
        X_std = X_array.std(axis=0)
        X_std = np.where(X_std == 0, 1.0, X_std)
        X_array = (X_array - X_mean) / X_std

        y_mean = y_array.mean(axis=0)
        y_std = y_array.std(axis=0)
        y_std = np.where(y_std == 0, 1.0, y_std)
        y_array = (y_array - y_mean) / y_std

    diff_X = X_array[:, None, :] - X_array[None, :, :]
    dist_X = np.linalg.norm(diff_X, axis=-1)

    diff_y = y_array[:, None, :] - y_array[None, :, :]
    dist_y = np.linalg.norm(diff_y, axis=-1)

    upper_idx = np.triu_indices(dist_X.shape[0], k=1)
    return dist_X[upper_idx], dist_y[upper_idx], upper_idx

def compute_vif(X):
    """Compute variance inflation factor (VIF) for each column in X.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix with shape (n_samples, n_features).

    Returns
    -------
    dict
        Mapping from feature name (or index for ndarray input) to VIF value.
    """
    if hasattr(X, "columns"):
        feature_names = list(X.columns)
        X_array = X.to_numpy(dtype=float)
    else:
        X_array = np.asarray(X, dtype=float)
        feature_names = list(range(X_array.shape[1]))

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")

    n_samples, n_features = X_array.shape
    if n_features < 2:
        raise ValueError("VIF is only defined for at least two features")
    if n_samples <= n_features:
        raise ValueError("Need more samples than features to compute stable VIF values")

    vifs = {}
    for feature_idx, feature_name in enumerate(feature_names):
        y_target = X_array[:, feature_idx]
        X_others = np.delete(X_array, feature_idx, axis=1)

        reg = LinearRegression()
        reg.fit(X_others, y_target)
        r2 = reg.score(X_others, y_target)

        if np.isclose(1.0 - r2, 0.0):
            vif_value = np.inf
        else:
            vif_value = 1.0 / (1.0 - r2)

        vifs[feature_name] = float(vif_value)

    return vifs


vif_values = compute_vif(X_train)

results["vif"] = vif_values

scorings = [
    TestScore(LeavePOut(1), "neg_mean_squared_error"),
    TestScore(LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_mean_squared_error"),
    DiffScore(LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_mean_squared_error"),
    YRandomization(random_state=42),
]

scoring_names = [
    "mse_loo_2",
    "mse_lomo",
    "diff_lomo",
    "y_randomization_mse",
]

for i, scoring in enumerate(scorings):
    results[scoring_names[i]] = scoring(mlr_base, X_train, y_train)

###
print("debug")
print(TestScore(LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_root_mean_squared_error")(mlr_base, X_train, y_train))
print(np.sqrt(np.abs(results["mse_lomo"])))

corr = custom_correlation_matrix(X_train)

results["correlation"] = corr

results["max_abs_corr"] = highest_absolute_off_diagonal(corr)
results["corr_det"] = np.linalg.det(corr)

dist_X, dist_y, idx_pairs = pairwise_distance_calc(X_train, y_train)

linreg = LinearRegression()
linreg.fit(dist_X.reshape(-1, 1), dist_y)

results["distance_correlation"] = r2_score(dist_y, linreg.predict(dist_X.reshape(-1, 1)))

# plot VIF values as text in a table-like format

fig, ax = plt.subplots(figsize=(6, 2))
rows = 1
cols = len(selected_features) + 1

ax.set_ylim(0, rows)
ax.set_xlim(0, cols + 0.5)

vif_data = results["vif"]

x_pos = range(1, len(selected_features) + 1)

for x, (k, v) in zip(x_pos, vif_data.items()):
    ax.text(
        x=x + 0.75,
        y=rows - 0.38,
        s=f"{v:.3f}",
        va="center",
        ha="right",
        weight="normal",
    )

ax.text(x=0.9, y=rows - 0.38, s="VIF $\\downarrow$", va="center", ha="right", weight="bold", fontsize=12)

for x, v in zip(x_pos, display_feature_names):
    ax.text(x=x, y=rows - 0.2, s=f"{v}", va="center", ha="left", weight="bold", fontsize=12)

ax.plot([1, cols], [rows - 0.28, rows - 0.28], lw=".5", c="black")
ax.axis('off')

plt.savefig(f"results/model_prospect_vif_table_{exp_i}.svg", bbox_inches=None)
plt.show()

# plot model validation scores as table

metrics_to_plot = [
    ("$RMSE_{LOMO} \\downarrow$", f"{np.sqrt(np.abs(results['mse_lomo'])):.3f}"),
    ("$\\Delta RMSE_{LOMO} \\downarrow$", f"{np.sqrt(np.abs(results['diff_lomo'])):.3f}"),
    ("Jack. score $\\downarrow$", f"{results['jackknife_score']:.3f}"),
    ("Jack. coef. $\\downarrow$", f"{results['jackknife_coefs']:.3f}"),
    ("Y-Rand. $\\uparrow$", f"{np.sqrt(np.abs(results['y_randomization_mse'])):.3f}"),
    ("Dist. $R^2 \\uparrow$", f"{results['distance_correlation']:.3f}"),
]

rows = len(metrics_to_plot) - 1
cols = 2

fig, ax = plt.subplots(figsize=(4, rows * .5))

ax.set_ylim(-1, rows + 1)
ax.set_xlim(0, cols + 0.5)

for idx, (label, value) in enumerate(metrics_to_plot):

    ax.text(x=1, y=rows-idx, s=label, va="center", ha="left", weight="bold", fontsize=12)
    ax.text(x=2, y=rows-idx, s=f"{value}", va="center", ha="left", weight="normal", fontsize=12)

ax.axis("off")

plt.savefig(f"results/model_prospect_validation_table_{exp_i}.svg", bbox_inches=None)
plt.show()

# reafs table

reafs_u = (reafs_results['accuracy'] + reafs_results['cv_r2'] + reafs_results['cv_spearman']) / 3

reafs_metrics_to_plot = [
    ("$S_{\\text{ReaFS}} \\uparrow$", f"{reafs_results['composite']:.3f}"),
    ("$uS_{\\text{ReaFS}} \\uparrow$", f"{reafs_u:.3f}"),
    ("R $\\uparrow$", f"{reafs_results['accuracy']:.3f}"),
    ("$Q^2_t \\uparrow$", f"{reafs_results['cv_r2']:.3f}"),
    ("$r_{st} \\uparrow$", f"{reafs_results['cv_spearman']:.3f}"),
    ("Max $\\rho \\downarrow$", f"{results['max_abs_corr']:.3f}"),
    ("D $\\uparrow$", f"{results['corr_det']:.3f}"),
]

rows = len(reafs_metrics_to_plot) - 1
cols = 2

fig, ax = plt.subplots(figsize=(4, rows * .5))

ax.set_ylim(-1, rows + 1)
ax.set_xlim(0, cols + 0.5)

for idx, (label, value) in enumerate(reafs_metrics_to_plot):

    ax.text(x=1, y=rows-idx, s=label, va="center", ha="left", weight="bold", fontsize=12)
    ax.text(x=2, y=rows-idx, s=f"{value}", va="center", ha="left", weight="normal", fontsize=12)

ax.axis("off")

plt.savefig(f"results/model_prospect_reafs_table_{exp_i}.svg", bbox_inches=None)
plt.show()

# plot normalized linear equation

fig, ax = plt.subplots(figsize=(6, 2))

# Build equation string with intercept first and readable coefficient terms.
equation_parts = ["$\\Delta \\Delta G^{\\ddagger} = "]
intercept_sign = "" if intercept >= 0 else "-"
equation_parts.append(f"{intercept_sign}{abs(intercept):.3f}$\n$")

for idx, (coef, name) in enumerate(zip(coefs, display_feature_names), start=1):
    sign = "+" if coef >= 0 else "-"
    equation_parts.append(
        f"{sign} {abs(coef):.3f} \\ {str(name).replace('$', '')}"
    )
    if idx % 3 == 0 and idx < len(coefs):
        equation_parts.append("$\n$")

equation_parts.append("$")
equation_str = "".join(equation_parts)

print(equation_str)

ax.text(0.02, 0.5, equation_str, ha="left", va="center", fontsize=12, transform=ax.transAxes)
ax.axis("off")

plt.savefig(f"results/model_prospect_equation_{exp_i}.svg", bbox_inches=None)
plt.show()

print(results)
print(reafs_results)


