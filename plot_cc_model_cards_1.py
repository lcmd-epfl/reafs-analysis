import numpy as np

from matplotlib import pyplot as plt

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
from reafs.cross_validation import LeaveOneMoreOut

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import LeavePOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
from sklearn.utils import check_X_y, check_array

from scipy.stats import spearmanr

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class PreFittedRegression(BaseEstimator, RegressorMixin):
    """Regressor with pre-set coefficients (no fitting required)."""
    
    def __init__(self, babu, rajan, func_ops=None):
        if rajan is None:
            rajan = 0.0
        babu = np.asarray(babu, dtype=float)
        if babu.ndim != 1:
            raise ValueError("Coefficients must be a 1D array-like")
        
        if func_ops is None:
            func_ops = [lambda x: x] * len(babu)

        if len(babu) != len(func_ops):
            raise ValueError("Length of coefficients must match length of func_ops")
        self.babu = babu
        self.rajan = rajan

        self.coef_ = babu
        self.intercept_ = rajan
        self.func_ops = func_ops
    
    def fit(self, X, y=None):
        """No-op fit (coefficients already set)."""
        # Check that X, y, coef, and intercept have compatible shapes
        X,y = check_X_y(X, y)
        n_features = X.shape[1]
        if len(self.coef_) != n_features:
            raise ValueError(f"Number of coefficients ({len(self.coef_)}) does not match number of features ({n_features})")
        if not np.isscalar(self.intercept_):
            raise ValueError("Intercept must be a scalar value")
        self.coef_ = self.coef_
        self.intercept_ = self.intercept_
        return self
    
    def predict(self, X):
        """Predict using pre-set coefficients."""
        X = check_array(X)
        pred = X @ self.coef_ + self.intercept_
        return pred

data_lau = from_dataset("lau2021", i_target=0, direction=1)
data_wang = from_dataset("wang2023_adi", i_target=0, direction=1)
data_schoepfer = from_dataset("schoepfer2023_cc", i_target=0, direction=1)

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

ligand_classes_lau = {
    "non-BnBiOx": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "BnBiOx": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "BiIm": [20, 21, 22, 23, 24, 25, 26, 27, 28],
}

ligand_classes_wang = {
    "non-BnBiOx": [1, 10, 2, 3, 4, 0, 11, 7, 5],
    "BnBiOx": [6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "BiIm": [9, 22, 8, 23, 24, 25, 26, 27, 28],
}

ligand_classes_schoepfer = {
    "non-BnBiOx": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "BnBiOx": [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "BiIm": [20, 21, 22, 23, 24, 25, 26, 27, 28],
}

selected_features_lau = ["Pol", "NBON1", "NBOC4"]
selected_features_wang = ["NPA_22", "Distance(5, 22)", "Distance(22, 24)"]
selected_features_schoepfer = ["nbo-", "fbonhq3", "f1chi"]
# selected_features_schoepfer = ["f0chi", "bpa"] # s1
# selected_features_schoepfer = ["cnbo1/2", "f1ka"] # s2
# selected_features_schoepfer = ["cnbo1/2", "f3chiv", "fbonhh2"] # s3
# selected_features_schoepfer = ["cnbo1/2", "f1k"] # s4
# selected_features_schoepfer = ["f1k", "f1chi", "bpa"] # s5
# selected_features_schoepfer = ["f1kb", "f0chiv", "f2chiv"] # s6
# selected_features_schoepfer = ['nbo-', 'f0chiv', 'bonhh3', 'fbonhh6'] # a1
# selected_features_schoepfer = ['s1chi', 'k_ups', 'onh_t', 'k_phial', 'b_vol'] # b1

fancy_names_lau = ["Pol", "$NBO_{N1}$", "$NBO_{C4}$"]
fancy_names_wang = ["$q_{Ni}$", "$d_{N-Ni}$", "$d_{Ni-I}$"]
fancy_names_schoepfer = ["$\\Delta NBO$", "$\\hat{Q}^{3}_{Bur.}$", "$^{1}\\hat{\\chi}$"]
# fancy_names_schoepfer = ["$^{0}\\hat{\\chi}$", "$\\text{BPA}$"]
# fancy_names_schoepfer = ["$q_{1}/q_{2}$", "$^{1}\\hat{\\kappa}_{a}$"]
# fancy_names_schoepfer = ["$q_{1}/q_{2}$", "$^{3}\\hat{\\chi}_{v}$", "$\\hat{H}^{2}_{nH}$"]
# fancy_names_schoepfer = ["$q_{1}/q_{2}$", "$^{1}\\hat{\\kappa}$"]
# fancy_names_schoepfer = ["$^{1}\\hat{\\kappa}$", "$^{1}\\hat{\\chi}$", "BPA"]
# fancy_names_schoepfer = ["$^{1}\\hat{\\kappa}_{b}$", "$^{0}\\hat{\\chi}_{v}$", "$^{2}\\hat{\\chi}_{v}$"]
# fancy_names_schoepfer = ["$\\Delta NBO$", "$^{0}\\hat{\\chi}_{v}$", "$H^{3}_{Bur.}$", "$^{6}\\hat{H}^{6}_{nH}$"] # a1
# fancy_names_schoepfer = ['s1chi', 'k\_ups', 'onh\_t', 'k\_phial', 'b\_vol'] # b1

mlr_lau = make_pipeline(StandardScaler(), LinearRegression())
mlr_wang = make_pipeline(StandardScaler(), LinearRegression())
mlr_schoepfer = make_pipeline(StandardScaler(), BayesianRidge())
# mlr_schoepfer = make_pipeline(LinearRegression())
# mlr_schoepfer = make_pipeline(PreFittedRegression([1.0,1.0], 0.0))

import sys
if len(sys.argv) > 1:
    exp = sys.argv[1]
else:
    exp = "lau"  # "lau", "wang", or "schoepfer"

additional_name = ""

data = eval(f"data_{exp}")
ligand_classes = eval(f"ligand_classes_{exp}")
selected_features = eval(f"selected_features_{exp}")
fancy_names = eval(f"fancy_names_{exp}")
mlr = eval(f"mlr_{exp}")

results = {}

################

results["n_samples"] = data.X.shape[0]
results["n_features"] = data.X.shape[1]
results["n_selected_features"] = len(selected_features)

mlr.fit(data.X[selected_features], data.y)

y_train_preds = mlr.predict(data.X[selected_features])

coefs = mlr[-1].coef_
intercept = mlr[-1].intercept_

results["r2"] = r2_score(data.y, y_train_preds)
results["mae"] = mean_absolute_error(data.y, y_train_preds)
results["mse"] = mean_squared_error(data.y, y_train_preds)
results["rmse"] = np.sqrt(results["mse"])

y_loo_preds = cross_val_predict(
    mlr, data.X[selected_features], data.y, cv=LeavePOut(p=1)
)

# cv_test = LeaveOneMoreOut(n_repeats=5, n_min=1)
# rmses = np.empty((cv_test.get_n_splits(data.X[selected_features], data.y),), dtype=float)
# i = 0
# for train_idx, test_idx in cv_test.split(data.X[selected_features], data.y):
#     X_tr, X_ts = data.X[selected_features].iloc[train_idx], data.X[selected_features].iloc[test_idx]
#     y_tr, y_ts = data.y.iloc[train_idx], data.y.iloc[test_idx]
# 
# 
#     mlr.fit(X_tr, y_tr).predict(X_ts)
#     print("2", np.asarray(X_ts), np.array([1.0, 1.0]))    
# 
#     y_ts_pred = np.asarray(X_ts) @ np.array([1.0, 1.0]) + 0.0
#     rmses[i] = np.sqrt(np.average((y_ts - y_ts_pred) ** 2))
#     i += 1
# 
# print(f"LOMO RMSEs: {rmses}", np.mean(rmses))
# exit(0)

results["rmse_lomo"] = TestScore(LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_root_mean_squared_error")(mlr, data.X[selected_features], data.y)

results["r2_loo"] = r2_score(data.y, y_loo_preds)
results["mae_loo"] = mean_absolute_error(data.y, y_loo_preds)
results["mse_loo"] = mean_squared_error(data.y, y_loo_preds)
results["rmse_loo"] = np.sqrt(results["mse_loo"])

j_scores, j_coefs = jackknife(mlr, data.X[selected_features], data.y, scoring="neg_mean_squared_error")

results["jackknife_score"] = np.sqrt(np.abs(j_scores))
results["jackknife_coefs"] = j_coefs


# parity plot

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(data.y, y_train_preds, color="#007480", s=60, marker="D", label="Train")
ax.scatter(data.y, y_loo_preds, color="#B51F1F", s=60, marker="o", label="LOO")
ax.plot(
    [data.y.min(), data.y.max()],
    [data.y.min(), data.y.max()],
    ls="--",
    lw=2,
    color="#CAC7C7",
)
ax.set_xlabel("True $\Delta \Delta G^{\ddagger}$ (kcal/mol)")
ax.set_ylabel("Predicted $\Delta \Delta G^{\ddagger}$ (kcal/mol)")

# put text or annotation with number of samples and R² on the plot
ax.text(
    0.04,
    0.96,
    f"""n={results['n_samples']}
$R^2={results['r2']:.3f}$
$R^{{2}}_{{loo}}={results['r2_loo']:.3f}$
$MAE={results['mae']:.3f}$
$MAE_{{loo}}={results['mae_loo']:.3f}$
""",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
)

ax.legend(loc="lower right")
# plt.tight_layout()

plt.savefig(f"results/model_card_parity_plot_{exp}_{additional_name}.svg", bbox_inches=None)

plt.show()

###


def make_pls_projection(X, y, n_components=2):
    n_samples, n_features = X.shape
    n_components_feasible = max(1, min(n_components, n_features, n_samples - 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pls = PLSRegression(n_components=n_components_feasible)
    x_scores, _ = pls.fit_transform(X_scaled, y)
    return x_scores, pls


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


def scale_point_sizes_from_target(y, min_size=30.0, max_size=220.0):
    y_array = np.asarray(y, dtype=float).ravel()
    y_min = np.min(y_array)
    y_max = np.max(y_array)

    if np.isclose(y_max - y_min, 0.0):
        return np.full_like(y_array, (min_size + max_size) / 2.0, dtype=float)

    y_norm = (y_array - y_min) / (y_max - y_min)
    return min_size + y_norm * (max_size - min_size)


pls_scores, pls_model = make_pls_projection(
    data.X[selected_features], data.y, n_components=2
)

pls_vectors = two_component_vectors(pls_model.x_weights_)
pls_arrow_scale = compute_arrow_scale(pls_scores, pls_vectors)

colors = [
    next(i for i, (k, v) in enumerate(ligand_classes.items()) if j in v)
    for j in data.X.index
]

sizes = scale_point_sizes_from_target(data.y)

# PLS scores plot

fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(
    pls_scores[:, 0],
    pls_scores[:, 1],
    c=[epfl_colors_c[i] for i in colors],
    s=sizes,
    marker="o",
)

plot_feature_arrows(
    ax,
    pls_vectors,
    fancy_names,
    scale=pls_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(ax, pls_scores, pls_vectors, scale=pls_arrow_scale)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("LS 1")
ax.set_ylabel("LS 2")

from matplotlib.lines import Line2D

legend_labels = list(ligand_classes.keys())
legend_colors = [epfl_colors_c[i] for i in range(len(legend_labels))]

handles = [
    Line2D([0], [0], marker='o', color='w', label=label,
           markerfacecolor=color, markersize=10)
    for label, color in zip(legend_labels, legend_colors)
]

ax.legend(handles=handles, loc="best")

# plt.tight_layout()

plt.savefig(f"results/model_card_pls_projection_{exp}_{additional_name}.svg", bbox_inches=None)

plt.show()


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


vif_values = compute_vif(data.X[selected_features])

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
    "mse_diff_lomo",
    "y_randomization_mse",
]

for i, scoring in enumerate(scorings):
    results[scoring_names[i]] = scoring(mlr, data.X[selected_features], data.y)

corr = custom_correlation_matrix(data.X[selected_features])

results["correlation"] = corr

# plot correlation matrix as heatmap with values on the cells

fig, ax = plt.subplots(figsize=(2, 2))

custom_cmap = LinearSegmentedColormap.from_list(
    "custom_gradient", ["#007480", "#CAC7C7", "#B51F1F"]
)

ax.matshow(corr, cmap=custom_cmap, vmin=-1, vmax=1)
# print the correlation values on the heatmap
for (i, j), val in np.ndenumerate(corr):
    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)

ax.xaxis.set_ticks_position("bottom")
ax.xaxis.set_label_position("bottom")

ax.set_xticks(range(len(selected_features)))
ax.set_xticklabels(fancy_names, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(selected_features)))
ax.set_yticklabels(fancy_names, fontsize=10)
# plt.tight_layout()

plt.savefig(f"results/model_card_correlation_matrix_{exp}_{additional_name}.svg", bbox_inches=None)

plt.show()

# plot VIF values as text in a table-like format

fig, ax = plt.subplots(figsize=(6, 2))
rows = 1
cols = 4

ax.set_ylim(0, rows)
ax.set_xlim(0, cols + 0.5)

vif_data = results["vif"]

# x_pos = [1, 2, 3]
x_pos = np.arange(1, len(vif_data) + 1)

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

for x, v in zip(x_pos, fancy_names):
    ax.text(x=x, y=rows - 0.2, s=f"{v}", va="center", ha="left", weight="bold", fontsize=12)

ax.plot([1, cols], [rows - 0.28, rows - 0.28], lw=".5", c="black")
ax.axis('off')

plt.savefig(f"results/model_card_vif_table_{exp}_{additional_name}.svg", bbox_inches=None)

plt.show()

# plot model validation scores as table

fig, ax = plt.subplots(figsize=(4, 2))
rows = 4
cols = 2

ax.set_ylim(-1, rows + 1)
ax.set_xlim(0, cols + 0.5)

metrics_to_plot = [
    ("$RMSE_{LOMO} \\downarrow$", f"{np.sqrt(np.abs(results['mse_lomo'])):.3f}"),
    ("$\Delta RMSE_{LOMO} \\downarrow$", f"{np.sqrt(np.abs(results['mse_diff_lomo'])):.3f}"),
    ("Jack. score $\\downarrow$", f"{results['jackknife_score']:.3f}"),
    ("Jack. coef. $\\downarrow$", f"{results['jackknife_coefs']:.3f}"),
    ("Y-Rand. $\\uparrow$", f"{np.sqrt(np.abs(results['y_randomization_mse'])):.3f}"),
]

for idx, (label, value) in enumerate(metrics_to_plot):

    ax.text(x=1, y=rows-idx, s=label, va="center", ha="left", weight="bold", fontsize=12)
    ax.text(x=2, y=rows-idx, s=f"{value}", va="center", ha="left", weight="normal", fontsize=12)

ax.axis("off")
# plt.tight_layout()

plt.savefig(f"results/model_card_validation_table_{exp}_{additional_name}.svg", bbox_inches=None)

plt.show()

# plot normalized linear equation

fig, ax = plt.subplots(figsize=(6, 2))

# Build equation string
equation_parts = [f"$\\Delta \\Delta G^{{\\ddagger}} = "]
intercept_sign = "" if intercept >= 0 else "-"
equation_parts.append(f" {intercept_sign} {abs(intercept):.3f}$\n$")
for i, (coef, name) in enumerate(zip(coefs, fancy_names)):
    sign = "+" if coef >= 0 else "-"
    first_sign = "+" if coef >= 0 else "-"
    if i == 0:
        equation_parts.append(f"{first_sign} {abs(coef):.3f} \\ {name.replace('$', '')}")
    else:
        equation_parts.append(f" {sign} {abs(coef):.3f} \\ {name.replace('$', '')}")

intercept_sign = "+" if intercept >= 0 else "-"
# equation_parts.append(f" {intercept_sign} {abs(intercept):.3f}$")
equation_parts.append(f"$")

equation_str = "".join(equation_parts)

print(equation_str)

ax.text(0.5, 0.5, equation_str, ha="left", va="center", fontsize=12, transform=ax.transAxes)
ax.axis("off")

plt.savefig(f"results/model_card_equation_{exp}_{additional_name}.svg", bbox_inches=None)

plt.show()


print(results)
