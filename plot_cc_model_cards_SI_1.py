import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from reafs.datasets import from_dataset
from reafs.utils import (
    cov_det,
    custom_correlation_matrix,
    highest_absolute_off_diagonal,
    laplacian_score,
    jackknife,
)
from reafs.metrics import (
    TestScore,
    TrainScore,
    DiffScore,
    YRandomization,
    ncrps_score,
    noise_resilience,
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

from scipy.stats import mode, spearmanr

import sys

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

data_lau = from_dataset("lau2021", i_target=0, direction=1)
data_wang = from_dataset("wang2023_adi", i_target=0, direction=1)
data_schoepfer = from_dataset("schoepfer2023_cc", i_target=0, direction=1)

print("LAU2021:", data_lau.X.shape, data_lau.y.shape)
print("WANG2023_ADI:", data_wang.X.shape, data_wang.y.shape)
print("SCHOEPFER2023_CC:", data_schoepfer.X.shape, data_schoepfer.y.shape)

ligand_classes = {
    "non-BnBiOx": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "BnBiOx": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "BiIm": [20, 21, 22, 23, 24, 25, 26, 27, 28],
}

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

# make cmap out of epfl_colors_c

epfl_cmap = ListedColormap(epfl_colors_c)

lig_class_cmap = ListedColormap(epfl_colors_c[0:3])

if len(sys.argv) < 2:
    raise ValueError("Error: MODE argument required. Usage: python plot_cc_model_cards_SI_1.py [ms|mi]")

MODE = sys.argv[1]
if MODE not in ["ms", "mi"]:
    raise ValueError(f"Error: MODE must be 'ms' or 'mi', got '{MODE}'. Usage: python plot_cc_model_cards_SI_1.py [ms|mi]")

if MODE == "ms":
    selected_features_lau = ["Pol", "NBON1", "NBOC4"]
    selected_features_wang = ["NPA_22", "Distance(5, 22)", "Distance(22, 24)"]
    selected_features_schoepfer = ["nbo-", "fbonhq3", "f1chi"]
elif MODE == "mi":

    fs_method = make_pipeline(
        StandardScaler(), SelectKBest(score_func=mutual_info_regression, k=3)
    )

    selected_features_idx_lau = fs_method.fit(data_lau.X, data_lau.y)[-1].get_support(
        indices=True
    )
    selected_features_idx_wang = fs_method.fit(data_wang.X, data_wang.y)[
        -1
    ].get_support(indices=True)
    selected_features_idx_schoepfer = fs_method.fit(data_schoepfer.X, data_schoepfer.y)[
        -1
    ].get_support(indices=True)

    selected_features_lau = data_lau.X.columns[selected_features_idx_lau].tolist()
    selected_features_wang = data_wang.X.columns[selected_features_idx_wang].tolist()
    selected_features_schoepfer = data_schoepfer.X.columns[
        selected_features_idx_schoepfer
    ].tolist()


mlr_lau = make_pipeline(StandardScaler(), LinearRegression())
mlr_lau.fit(data_lau.X[selected_features_lau], data_lau.y)

mlr_wang = make_pipeline(StandardScaler(), LinearRegression())
mlr_wang.fit(data_wang.X[selected_features_wang], data_wang.y)

mlr_schoepfer = make_pipeline(StandardScaler(), BayesianRidge())
mlr_schoepfer.fit(data_schoepfer.X[selected_features_schoepfer], data_schoepfer.y)

# LOO CV R^2
loo_preds_lau = cross_val_predict(
    mlr_lau, data_lau.X[selected_features_lau], data_lau.y, cv=LeavePOut(p=1)
)
loo_preds_wang = cross_val_predict(
    mlr_wang, data_wang.X[selected_features_wang], data_wang.y, cv=LeavePOut(p=1)
)
loo_preds_schoepfer = cross_val_predict(
    mlr_schoepfer,
    data_schoepfer.X[selected_features_schoepfer],
    data_schoepfer.y,
    cv=LeavePOut(p=1),
)

pca_lau = make_pipeline(StandardScaler(), PLSRegression())
pca_lau.fit(data_lau.X[selected_features_lau], data_lau.y)

pca_wang = make_pipeline(StandardScaler(), PLSRegression())
pca_wang.fit(data_wang.X[selected_features_wang], data_wang.y)

pca_schoepfer = make_pipeline(StandardScaler(), PLSRegression())
pca_schoepfer.fit(data_schoepfer.X[selected_features_schoepfer], data_schoepfer.y)


def make_pca_projection(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X_scaled)
    return X_proj, pca.explained_variance_ratio_, pca


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
            arrowprops=dict(arrowstyle="->", lw=1.6, color=color, alpha=0.85),
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


def compute_pls_r2_curve(X, y, cv, max_components=None):
    n_samples, n_features = X.shape
    n_components_max_feasible = min(n_features, n_samples - 1)
    if max_components is None:
        max_components = n_components_max_feasible
    max_components = max(1, min(max_components, n_components_max_feasible))

    component_counts = np.arange(1, max_components + 1)
    train_r2_scores = []
    cv_r2_scores = []

    for n_comp in component_counts:
        pls_model = make_pipeline(StandardScaler(), PLSRegression(n_components=n_comp))
        pls_model.fit(X, y)
        train_r2_scores.append(pls_model.score(X, y))

        y_pred_cv = cross_val_predict(pls_model, X, y, cv=cv)
        cv_r2_scores.append(r2_score(y, y_pred_cv))

    return component_counts, np.array(train_r2_scores), np.array(cv_r2_scores)


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

pca_lau_transformed, pca_lau_expl_var, pca_lau_model = make_pca_projection(
    data_lau.X[selected_features_lau], n_components=2
)
colors_lau = [
    next(i for i, (k, v) in enumerate(ligand_classes.items()) if j in v)
    for j in data_lau.X.index
]
sizes_lau = scale_point_sizes_from_target(data_lau.y)
axes[0].scatter(
    pca_lau_transformed[:, 0],
    pca_lau_transformed[:, 1],
    c=colors_lau,
    cmap=lig_class_cmap,
    s=sizes_lau,
)
pca_lau_vectors = two_component_vectors(pca_lau_model.components_.T)
lau_pca_arrow_scale = compute_arrow_scale(pca_lau_transformed, pca_lau_vectors)
plot_feature_arrows(
    axes[0],
    pca_lau_vectors,
    selected_features_lau,
    scale=lau_pca_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(
    axes[0], pca_lau_transformed, pca_lau_vectors, scale=lau_pca_arrow_scale
)
axes[0].set_xlabel(f"PC1 ({pca_lau_expl_var[0] * 100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({pca_lau_expl_var[1] * 100:.1f}% var)")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[0].set_title(
    f"Lau2021 PCA \n(2 PCs: {np.sum(pca_lau_expl_var) * 100:.1f}% var)",
    fontweight="bold",
)

pca_wang_transformed, pca_wang_expl_var, pca_wang_model = make_pca_projection(
    data_wang.X[selected_features_wang], n_components=2
)
colors_wang = [
    next(i for i, (k, v) in enumerate(ligand_classes.items()) if j in v)
    for j in data_wang.X.index
]
sizes_wang = scale_point_sizes_from_target(data_wang.y)
axes[1].scatter(
    pca_wang_transformed[:, 0],
    pca_wang_transformed[:, 1],
    c=colors_wang,
    cmap=lig_class_cmap,
    s=sizes_wang,
)
pca_wang_vectors = two_component_vectors(pca_wang_model.components_.T)
wang_pca_arrow_scale = compute_arrow_scale(pca_wang_transformed, pca_wang_vectors)
plot_feature_arrows(
    axes[1],
    pca_wang_vectors,
    selected_features_wang,
    scale=wang_pca_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(
    axes[1], pca_wang_transformed, pca_wang_vectors, scale=wang_pca_arrow_scale
)
axes[1].set_xlabel(f"PC1 ({pca_wang_expl_var[0] * 100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca_wang_expl_var[1] * 100:.1f}% var)")
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title(
    f"Wang2023 PCA \n(2 PCs: {np.sum(pca_wang_expl_var) * 100:.1f}% var)",
    fontweight="bold",
)

pca_schoepfer_transformed, pca_schoepfer_expl_var, pca_schoepfer_model = (
    make_pca_projection(data_schoepfer.X[selected_features_schoepfer], n_components=2)
)
colors_schoepfer = [
    next(i for i, (k, v) in enumerate(ligand_classes.items()) if j in v)
    for j in data_schoepfer.X.index
]
sizes_schoepfer = scale_point_sizes_from_target(data_schoepfer.y)
axes[2].scatter(
    pca_schoepfer_transformed[:, 0],
    pca_schoepfer_transformed[:, 1],
    c=colors_schoepfer,
    cmap=lig_class_cmap,
    s=sizes_schoepfer,
)
pca_schoepfer_vectors = two_component_vectors(pca_schoepfer_model.components_.T)
schoepfer_pca_arrow_scale = compute_arrow_scale(
    pca_schoepfer_transformed, pca_schoepfer_vectors
)
plot_feature_arrows(
    axes[2],
    pca_schoepfer_vectors,
    selected_features_schoepfer,
    scale=schoepfer_pca_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(
    axes[2],
    pca_schoepfer_transformed,
    pca_schoepfer_vectors,
    scale=schoepfer_pca_arrow_scale,
)
axes[2].set_xlabel(f"PC1 ({pca_schoepfer_expl_var[0] * 100:.1f}% var)")
axes[2].set_ylabel(f"PC2 ({pca_schoepfer_expl_var[1] * 100:.1f}% var)")
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_title(
    f"Schoepfer2023 PCA \n(2 PCs: {np.sum(pca_schoepfer_expl_var) * 100:.1f}% var)",
    fontweight="bold",
)

legend_labels = ligand_classes.keys()
legend_colors = [lig_class_cmap(i) for i in range(len(ligand_classes))]

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=label,
        markerfacecolor=color,
        markersize=10,
    )
    for label, color in zip(legend_labels, legend_colors)
]

if MODE == "ms":
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.87),
        frameon=True,
        fontsize=10,
    )

if MODE == "mi":
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.72),
        frameon=True,
        fontsize=10,
    )

# plt.tight_layout()
plt.savefig(f"results/model_comp_SI_pca_{MODE}_1.svg")

plt.show()

pls_lau_scores, pls_lau_model = make_pls_projection(
    data_lau.X[selected_features_lau],
    data_lau.y,
    n_components=2,
)
pls_wang_scores, pls_wang_model = make_pls_projection(
    data_wang.X[selected_features_wang],
    data_wang.y,
    n_components=2,
)
pls_schoepfer_scores, pls_schoepfer_model = make_pls_projection(
    data_schoepfer.X[selected_features_schoepfer],
    data_schoepfer.y,
    n_components=2,
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(
    pls_lau_scores[:, 0],
    pls_lau_scores[:, 1],
    c=colors_lau,
    cmap=lig_class_cmap,
    s=sizes_lau,
    marker="^",
)
pls_lau_vectors = two_component_vectors(pls_lau_model.x_weights_)
lau_pls_arrow_scale = compute_arrow_scale(pls_lau_scores, pls_lau_vectors)
plot_feature_arrows(
    axes[0],
    pls_lau_vectors,
    selected_features_lau,
    scale=lau_pls_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(
    axes[0], pls_lau_scores, pls_lau_vectors, scale=lau_pls_arrow_scale
)
axes[0].set_xlabel("PLS LV1")
axes[0].set_ylabel("PLS LV2")
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title(
    f"Lau2021 PLS (R²={pls_lau_model.score(StandardScaler().fit_transform(data_lau.X[selected_features_lau]), data_lau.y):.3f})",
    fontweight="bold",
)

axes[1].scatter(
    pls_wang_scores[:, 0],
    pls_wang_scores[:, 1],
    c=colors_wang,
    cmap=lig_class_cmap,
    s=sizes_wang,
    marker="^",
)
pls_wang_vectors = two_component_vectors(pls_wang_model.x_weights_)
wang_pls_arrow_scale = compute_arrow_scale(pls_wang_scores, pls_wang_vectors)
plot_feature_arrows(
    axes[1],
    pls_wang_vectors,
    selected_features_wang,
    scale=wang_pls_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(
    axes[1], pls_wang_scores, pls_wang_vectors, scale=wang_pls_arrow_scale
)
axes[1].set_xlabel("PLS LV1")
axes[1].set_ylabel("PLS LV2")
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title(
    f"Wang2023 PLS (R²={pls_wang_model.score(StandardScaler().fit_transform(data_wang.X[selected_features_wang]), data_wang.y):.3f})",
    fontweight="bold",
)

axes[2].scatter(
    pls_schoepfer_scores[:, 0],
    pls_schoepfer_scores[:, 1],
    c=colors_schoepfer,
    cmap=lig_class_cmap,
    s=sizes_schoepfer,
    marker="^",
)
pls_schoepfer_vectors = two_component_vectors(pls_schoepfer_model.x_weights_)
schoepfer_pls_arrow_scale = compute_arrow_scale(
    pls_schoepfer_scores, pls_schoepfer_vectors
)
plot_feature_arrows(
    axes[2],
    pls_schoepfer_vectors,
    selected_features_schoepfer,
    scale=schoepfer_pls_arrow_scale,
    color="#453A4C",
)
fit_axes_to_scores_and_arrows(
    axes[2],
    pls_schoepfer_scores,
    pls_schoepfer_vectors,
    scale=schoepfer_pls_arrow_scale,
)
axes[2].set_xlabel("PLS LV1")
axes[2].set_ylabel("PLS LV2")
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_title(
    f"Schoepfer2023 PLS (R²={pls_schoepfer_model.score(StandardScaler().fit_transform(data_schoepfer.X[selected_features_schoepfer]), data_schoepfer.y):.3f})",
    fontweight="bold",
)

legend_labels = ligand_classes.keys()
legend_colors = [lig_class_cmap(i) for i in range(len(ligand_classes))]

handles = [
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        label=label,
        markerfacecolor=color,
        markersize=10,
    )
    for label, color in zip(legend_labels, legend_colors)
]

if MODE == "ms":
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.87),
        frameon=True,
        fontsize=10,
    )
if MODE == "mi":
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.72),
        frameon=True,
        fontsize=10,
    )


# plt.tight_layout()
plt.savefig(f"results/model_comp_SI_pls_{MODE}_1.svg")

plt.show()

cv_strategy = LeavePOut(p=1)
lau_components, lau_train_r2, lau_cv_r2 = compute_pls_r2_curve(
    data_lau.X[selected_features_lau],
    data_lau.y,
    cv=cv_strategy,
)
wang_components, wang_train_r2, wang_cv_r2 = compute_pls_r2_curve(
    data_wang.X[selected_features_wang],
    data_wang.y,
    cv=cv_strategy,
)
schoepfer_components, schoepfer_train_r2, schoepfer_cv_r2 = compute_pls_r2_curve(
    data_schoepfer.X[selected_features_schoepfer],
    data_schoepfer.y,
    cv=cv_strategy,
)

print("PLS R² by number of components (train / LOOCV)")
for i, n_comp in enumerate(lau_components):
    print(
        f"Lau2021 n_comp={n_comp}: train={lau_train_r2[i]:.4f}, LOOCV={lau_cv_r2[i]:.4f}"
    )
for i, n_comp in enumerate(wang_components):
    print(
        f"Wang2023 n_comp={n_comp}: train={wang_train_r2[i]:.4f}, LOOCV={wang_cv_r2[i]:.4f}"
    )
for i, n_comp in enumerate(schoepfer_components):
    print(
        f"Schoepfer2023 n_comp={n_comp}: train={schoepfer_train_r2[i]:.4f}, LOOCV={schoepfer_cv_r2[i]:.4f}"
    )

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

axes[0].plot(
    lau_components,
    lau_train_r2,
    marker="s",
    linewidth=2,
    color="#F39869",
    label="Train R²",
)
axes[0].plot(
    lau_components,
    lau_cv_r2,
    marker="D",
    linewidth=2,
    color="#FF0000",
    label="LOOCV R²",
)
axes[0].set_xlabel("Number of PLS latent variables")
axes[0].set_ylabel("R²")
axes[0].set_title("Lau2021 PLS", fontweight="bold")
axes[0].grid(alpha=0.3, linestyle="--")
axes[0].legend(loc="best")

axes[1].plot(
    wang_components,
    wang_train_r2,
    marker="s",
    linewidth=2,
    color="#F39869",
    label="Train R²",
)
axes[1].plot(
    wang_components,
    wang_cv_r2,
    marker="D",
    linewidth=2,
    color="#FF0000",
    label="LOOCV R²",
)
axes[1].set_xlabel("Number of PLS latent variables")
axes[1].set_ylabel("R²")
axes[1].set_title("Wang2023 PLS", fontweight="bold")
axes[1].grid(alpha=0.3, linestyle="--")
axes[1].legend(loc="best")

axes[2].plot(
    schoepfer_components,
    schoepfer_train_r2,
    marker="s",
    linewidth=2,
    color="#F39869",
    label="Train R²",
)
axes[2].plot(
    schoepfer_components,
    schoepfer_cv_r2,
    marker="D",
    linewidth=2,
    color="#FF0000",
    label="LOOCV R²",
)
axes[2].set_xlabel("Number of PLS latent variables")
axes[2].set_ylabel("R²")
axes[2].set_title("Schoepfer2023 PLS", fontweight="bold")
axes[2].grid(alpha=0.3, linestyle="--")
axes[2].legend(loc="best")

# plt.tight_layout()
plt.savefig(f"results/model_comp_SI_pls_r2_curve_{MODE}_1.svg")

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


dist_X_wang, dist_y_wang, _ = pairwise_distance_calc(
    data_wang.X[selected_features_wang],
    data_wang.y,
)

dist_X_lau, dist_y_lau, _ = pairwise_distance_calc(
    data_lau.X[selected_features_lau],
    data_lau.y,
)

dist_X_schoepfer, dist_y_schoepfer, _ = pairwise_distance_calc(
    data_schoepfer.X[selected_features_schoepfer],
    data_schoepfer.y,
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

axes[0].scatter(dist_X_lau, dist_y_lau, s=12, alpha=0.6)
linreg_lau = LinearRegression().fit(dist_X_lau.reshape(-1, 1), dist_y_lau)
y_pred_lau = linreg_lau.predict(dist_X_lau.reshape(-1, 1))
r2_lau = r2_score(dist_y_lau, y_pred_lau)
sorted_lau = np.argsort(dist_X_lau)
axes[0].plot(
    dist_X_lau[sorted_lau],
    y_pred_lau[sorted_lau],
    color="black",
    linewidth=1.8,
    label="Linear fit",
)
ratio_lau = np.divide(
    dist_y_lau, dist_X_lau, out=np.full_like(dist_y_lau, np.nan), where=dist_X_lau > 0
)
ratio_std_lau = np.nanstd(ratio_lau)
axes[0].text(
    0.03,
    0.97,
    f"R²={r2_lau:.3f}\nSlope={linreg_lau.coef_[0]:.3f}\nRatio SD={ratio_std_lau:.3f}",
    transform=axes[0].transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)
axes[0].set_xlabel("Pairwise distance in X")
axes[0].set_ylabel("Pairwise distance in y")
axes[0].set_title("Lau2021 X vs y distances", fontweight="bold")
axes[0].legend(loc="lower right")

axes[1].scatter(dist_X_wang, dist_y_wang, s=12, alpha=0.6)
linreg_wang = LinearRegression().fit(dist_X_wang.reshape(-1, 1), dist_y_wang)
y_pred_wang = linreg_wang.predict(dist_X_wang.reshape(-1, 1))
r2_wang = r2_score(dist_y_wang, y_pred_wang)
sorted_wang = np.argsort(dist_X_wang)
axes[1].plot(
    dist_X_wang[sorted_wang],
    y_pred_wang[sorted_wang],
    color="black",
    linewidth=1.8,
    label="Linear fit",
)
ratio_wang = np.divide(
    dist_y_wang,
    dist_X_wang,
    out=np.full_like(dist_y_wang, np.nan),
    where=dist_X_wang > 0,
)
ratio_std_wang = np.nanstd(ratio_wang)
axes[1].text(
    0.03,
    0.97,
    f"R²={r2_wang:.3f}\nSlope={linreg_wang.coef_[0]:.3f}\nRatio SD={ratio_std_wang:.3f}",
    transform=axes[1].transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)
axes[1].set_xlabel("Pairwise distance in X")
axes[1].set_ylabel("Pairwise distance in y")
axes[1].set_title("Wang2023 X vs y distances", fontweight="bold")
axes[1].legend(loc="lower right")

axes[2].scatter(dist_X_schoepfer, dist_y_schoepfer, s=12, alpha=0.6)
linreg_schoepfer = LinearRegression().fit(
    dist_X_schoepfer.reshape(-1, 1), dist_y_schoepfer
)
y_pred_schoepfer = linreg_schoepfer.predict(dist_X_schoepfer.reshape(-1, 1))
r2_schoepfer = r2_score(dist_y_schoepfer, y_pred_schoepfer)
sorted_schoepfer = np.argsort(dist_X_schoepfer)
axes[2].plot(
    dist_X_schoepfer[sorted_schoepfer],
    y_pred_schoepfer[sorted_schoepfer],
    color="black",
    linewidth=1.8,
    label="Linear fit",
)
ratio_schoepfer = np.divide(
    dist_y_schoepfer,
    dist_X_schoepfer,
    out=np.full_like(dist_y_schoepfer, np.nan),
    where=dist_X_schoepfer > 0,
)
ratio_std_schoepfer = np.nanstd(ratio_schoepfer)
axes[2].text(
    0.03,
    0.97,
    f"R²={r2_schoepfer:.3f}\nSlope={linreg_schoepfer.coef_[0]:.3f}\nRatio SD={ratio_std_schoepfer:.3f}",
    transform=axes[2].transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)
axes[2].set_xlabel("Pairwise distance in X")
axes[2].set_ylabel("Pairwise distance in y")
axes[2].set_title("Schoepfer2023 X vs y distances", fontweight="bold")
axes[2].legend(loc="lower right")

# plt.tight_layout()
plt.savefig(f"results/model_comp_SI_pairwise_distances_{MODE}_1.svg")

plt.show()


###
# Variance Inflation Factor (VIF) for selected features in each dataset
###


vif_lau = compute_vif(data_lau.X[selected_features_lau])
vif_wang = compute_vif(data_wang.X[selected_features_wang])
vif_schoepfer = compute_vif(data_schoepfer.X[selected_features_schoepfer])

print("Lau2021 VIF:", {k: round(v, 3) if isinstance(v, float) else v for k, v in vif_lau.items()})
print("Wang2023 VIF:", {k: round(v, 3) if isinstance(v, float) else v for k, v in vif_wang.items()})
print("Schoepfer2023 VIF:", {k: round(v, 3) if isinstance(v, float) else v for k, v in vif_schoepfer.items()})

###
# Correlation matrices
###

corr_lau = custom_correlation_matrix(data_lau.X[selected_features_lau])
corr_wang = custom_correlation_matrix(data_wang.X[selected_features_wang])
corr_schoepfer = custom_correlation_matrix(
    data_schoepfer.X[selected_features_schoepfer]
)

# Max absolute off-diagonal correlation values
print("LAU2021 Max Correlation Matrix:\n", highest_absolute_off_diagonal(corr_lau))
print(
    "WANG2023_ADI Max Correlation Matrix:\n", highest_absolute_off_diagonal(corr_wang)
)
print(
    "SCHOEPFER2023_CC Max Correlation Matrix:\n",
    highest_absolute_off_diagonal(corr_schoepfer),
)

# Determinant of Correlation matrices
print("LAU2021 Correlation Matrix Determinant:", np.linalg.det(corr_lau))
print("WANG2023_ADI Correlation Matrix Determinant:", np.linalg.det(corr_wang))
print("SCHOEPFER2023_CC Correlation Matrix Determinant:", np.linalg.det(corr_schoepfer))


###
# Regression metrics summary (train, LOOCV, LOMO, jackknife)
###


def compute_regression_metrics(model, X, y):
    """Return a dict with common regression metrics for a fitted model.

    Computes train predictions, LOOCV predictions, LOMO RMSE (TestScore), and jackknife.
    """
    metrics = {}

    # Train predictions
    y_pred_train = model.predict(X)
    metrics["r2_train"] = r2_score(y, y_pred_train)
    metrics["mae_train"] = mean_absolute_error(y, y_pred_train)
    metrics["mse_train"] = mean_squared_error(y, y_pred_train)
    metrics["rmse_train"] = np.sqrt(metrics["mse_train"])

    # LOOCV predictions
    loo_preds = cross_val_predict(model, X, y, cv=LeavePOut(p=1))
    metrics["r2_loo"] = r2_score(y, loo_preds)
    metrics["mae_loo"] = mean_absolute_error(y, loo_preds)
    metrics["mse_loo"] = mean_squared_error(y, loo_preds)
    metrics["rmse_loo"] = np.sqrt(metrics["mse_loo"])

    # LOMO RMSE (uses TestScore wrapper returning negative RMSE by convention)
    try:
        metrics["rmse_lomo"] = abs(
            TestScore(
                LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_root_mean_squared_error"
            )(model, X, y)
        )
    except Exception:
        metrics["rmse_lomo"] = np.nan

    # LOMO RMSE (uses DiffScore wrapper returning negative RMSE by convention)
    try:
        metrics["rmse_diff_lomo"] = abs(
            DiffScore(
                LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_root_mean_squared_error"
            )(model, X, y)
        )
    except Exception:
        metrics["rmse_diff_lomo"] = np.nan

    # Y-randomiztaion
    try:
        metrics["mae_y_random"] = YRandomization(
            scoring="neg_mean_absolute_error", random_state=42
        )(model, X, y)
        metrics["rmse_y_random"] = abs(
            YRandomization(scoring="neg_root_mean_squared_error", random_state=42)(
                model, X, y
            )
        )
    except Exception:
        metrics["mae_y_random"] = np.nan
        metrics["rmse_y_random"] = np.nan

    # Jackknife (prefer RMSE-based jackknife when available)
    try:
        jk_score, jk_coefs = jackknife(
            model, X, y, scoring="neg_root_mean_squared_error"
        )
        metrics["jackknife_score"] = jk_score
        metrics["jackknife_coefs"] = jk_coefs
    except Exception:
        metrics["jackknife_score"] = np.nan
        metrics["jackknife_coefs"] = None

    # Noise resilience
    try:
        metrics["noise_resilience_score"] = noise_resilience(to_compute="scores")(
            model, X, y
        )
        metrics["noise_resilience_coefs"] = noise_resilience(to_compute="coefs")(
            model, X, y
        )
    except Exception:
        metrics["noise_resilience_score"] = np.nan
        metrics["noise_resilience_coefs"] = None

    return metrics


# Compute metrics for each dataset
metrics_lau = compute_regression_metrics(
    mlr_lau, data_lau.X[selected_features_lau], data_lau.y
)
metrics_wang = compute_regression_metrics(
    mlr_wang, data_wang.X[selected_features_wang], data_wang.y
)
metrics_schoepfer = compute_regression_metrics(
    mlr_schoepfer, data_schoepfer.X[selected_features_schoepfer], data_schoepfer.y
)

print("LAU2021 metrics:", metrics_lau)
print("WANG2023 metrics:", metrics_wang)
print("SCHOEPFER2023 metrics:", metrics_schoepfer)

# Small summary table (RMSE / RMSE_LOO / RMSE_LOMO)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")

cols = ["LAU2021", "WANG2023", "SCHOEPFER2023"]
rows = [
    "r2_train",
    "r2_loo",
    "mae_train",
    "mae_loo",
    "rmse_lomo",
    "rmse_diff_lomo",
    "mae_y_random",
    "rmse_y_random",
    "jackknife_score",
    "jackknife_coefs",
    "noise_resilience_score",
    "noise_resilience_coefs",
]
values = [
    [
        metrics_lau.get(r, np.nan),
        metrics_wang.get(r, np.nan),
        metrics_schoepfer.get(r, np.nan),
    ]
    for r in rows
]

# Draw text table
x0 = 0.05
y0 = 0.85
dy = 0.08
ax.text(x0, y0 + dy, " ", fontsize=1)
ax.text(x0 + 0.12, y0, "  ", fontsize=1)
ax.text(x0, y0, "Metric", weight="bold")
for i, c in enumerate(cols):
    ax.text(x0 + 0.33 + i * 0.22, y0, c, weight="bold")

for i, row in enumerate(rows):
    y = y0 - (i + 1) * dy
    ax.text(x0, y, row.replace("_", " "), weight="bold")
    for j in range(len(cols)):
        val = values[i][j]
        if isinstance(val, float) and np.isfinite(val):
            s = f"{val:.3f}"
        else:
            s = str(val)
        ax.text(x0 + 0.33 + j * 0.22, y, s)

plt.savefig(f"results/model_comp_SI_metrics_{MODE}_1.svg")
plt.show()

###
# Feature--target correlations
###

# Compute f_regression mutual info and laplacian score for each feature in each dataset
f_reg_lau, _ = f_regression(data_lau.X, data_lau.y)
mi_lau = mutual_info_regression(data_lau.X, data_lau.y)
laplacian_lau = laplacian_score(data_lau.X)

f_reg_wang, _ = f_regression(data_wang.X, data_wang.y)
mi_wang = mutual_info_regression(data_wang.X, data_wang.y)
laplacian_wang = laplacian_score(data_wang.X)

f_reg_schoepfer, _ = f_regression(data_schoepfer.X, data_schoepfer.y)
mi_schoepfer = mutual_info_regression(data_schoepfer.X, data_schoepfer.y)
laplacian_schoepfer = laplacian_score(data_schoepfer.X)

# Check the values for the selected features of each dataset and compute their relative ranking within the full feature set of each dataset
print(
    f"LAU2021 MAX: f_reg={np.max(f_reg_lau):.3f}, mi={np.max(mi_lau):.3f}, laplacian={np.max(laplacian_lau):.3f}"
)
for feature in selected_features_lau:
    idx = data_lau.X.columns.get_loc(feature)
    print(
        f"LAU2021 - {feature}: f_reg={f_reg_lau[idx]:.3f}, mi={mi_lau[idx]:.3f}, laplacian={laplacian_lau[idx]:.3f}"
    )

print(
    f"WANG2023_ADI MAX: f_reg={np.max(f_reg_wang):.3f}, mi={np.max(mi_wang):.3f}, laplacian={np.max(laplacian_wang):.3f}"
)
for feature in selected_features_wang:
    idx = data_wang.X.columns.get_loc(feature)
    print(
        f"WANG2023_ADI - {feature}: f_reg={f_reg_wang[idx]:.3f}, mi={mi_wang[idx]:.3f}, laplacian={laplacian_wang[idx]:.3f}"
    )

print(
    f"SCHOEPFER2023_CC MAX: f_reg={np.max(f_reg_schoepfer):.3f}, mi={np.max(mi_schoepfer):.3f}, laplacian={np.max(laplacian_schoepfer):.3f}"
)
for feature in selected_features_schoepfer:
    idx = data_schoepfer.X.columns.get_loc(feature)
    print(
        f"SCHOEPFER2023_CC - {feature}: f_reg={f_reg_schoepfer[idx]:.3f}, mi={mi_schoepfer[idx]:.3f}, laplacian={laplacian_schoepfer[idx]:.3f}"
    )

