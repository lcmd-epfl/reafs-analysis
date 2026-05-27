import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from reafs.datasets import from_dataset
from reafs.utils import (
	custom_correlation_matrix,
	highest_absolute_off_diagonal,
	laplacian_score,
)
from reafs.metrics import (
	TestScore,
	DiffScore,
	YRandomization,
	noise_resilience,
	jackknife,
)
from reafs.cross_validation import LeaveOneMoreOut, LeaveTopPOut
from reafs.feature_selection import BeamSequential
from reafs.settings import reafs_base_evaluator

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeavePOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y

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
custom_cmap = LinearSegmentedColormap.from_list(
	"custom_gradient", ["#007480", "#CAC7C7", "#B51F1F"]
)


def reafs_fs_selector(fs, X, y, split_train):
	X, y = check_X_y(X, y)
	fs_pipe = make_pipeline(StandardScaler(), fs)
	fs_pipe.fit(X[split_train], y[split_train])
	return fs_pipe[-1].get_support(indices=True)


def stepwise_fs_selector(X, y, split_train):
	X_train, y_train = X.iloc[split_train], y.iloc[split_train]
	train_data = pd.concat([X_train, y_train], axis=1)
	res_sel = fs_selector(train_data, "ddg")
	selected_feature_names = res_sel["Model"].iloc[0]
	selected_feature_idx = [X.columns.get_loc(col) for col in selected_feature_names]
	return selected_feature_idx


def make_pca_projection(X, n_components=2):
	n_samples, n_features = X.shape
	n_components_feasible = max(1, min(n_components, n_features, n_samples - 1))
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	pca = PCA(n_components=n_components_feasible)
	x_scores = pca.fit_transform(X_scaled)
	return x_scores, pca, scaler


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


def compute_vif(X):
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
		vifs[feature_name] = np.inf if np.isclose(1.0 - r2, 0.0) else 1.0 / (1.0 - r2)
	return {k: float(v) for k, v in vifs.items()}


def top_corr_pair(corr):
	corr = np.asarray(corr, dtype=float)
	corr_abs = np.abs(corr.copy())
	np.fill_diagonal(corr_abs, -np.inf)
	i, j = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
	return i, j, float(corr[i, j])


def correlation_with_top_removed(X_df):
	corr_full = custom_correlation_matrix(X_df)
	i, j, top_corr = top_corr_pair(corr_full)
	removed_feature = X_df.columns[j]
	reduced_df = X_df.drop(columns=[removed_feature])
	corr_reduced = custom_correlation_matrix(reduced_df)
	return {
		"corr_full": corr_full,
		"corr_reduced": corr_reduced,
		"top_pair": (X_df.columns[i], X_df.columns[j]),
		"top_corr_value": top_corr,
		"removed_feature": removed_feature,
	}


def collect_feature_scores(X_train_df, y_train, selected_feature_names):
	f_vals, _ = f_regression(X_train_df, y_train)
	mi_vals = mutual_info_regression(X_train_df, y_train, random_state=0)
	lap_vals = laplacian_score(X_train_df)

	f_series = pd.Series(f_vals, index=X_train_df.columns)
	mi_series = pd.Series(mi_vals, index=X_train_df.columns)
	lap_series = pd.Series(lap_vals, index=X_train_df.columns)

	rows = []
	for name in selected_feature_names:
		rows.append(
			{
				"feature": name,
				"f_test": float(f_series[name]),
				"f_rank": int(f_series.rank(ascending=False, method="min")[name]),
				"mutual_info": float(mi_series[name]),
				"mi_rank": int(mi_series.rank(ascending=False, method="min")[name]),
				"laplacian": float(lap_series[name]),
				"lap_rank": int(lap_series.rank(ascending=False, method="min")[name]),
			}
		)
	return pd.DataFrame(rows)


data = from_dataset("schoepfer2023_oa", i_target=0, direction=1)

mlr_base = make_pipeline(StandardScaler(), BayesianRidge())
evaluation_splitter = LeaveTopPOut(5)
split_train, split_test = next(evaluation_splitter.split(data.X, data.y))

# Keep these definitions identical to plot_reafs_appli_prospect_2.py.
fs_1 = BeamSequential(
	estimator=clone(mlr_base),
	n_features_to_select="auto",
	tol=1e-3,
	scoring=TestScore(
		cv=LeaveOneMoreOut(n_repeats=5, n_min=1),
		scoring="neg_mean_squared_error",
	),
	n_candidates=10,
)

fs_5 = BeamSequential(
	estimator=clone(mlr_base),
	n_features_to_select="auto",
	tol=1e-3,
	scoring=TestScore(cv=LeavePOut(1), scoring="neg_mean_squared_error"),
	n_candidates=1,
)

experiments = [
	{
		"key": "stepwise",
		"label": "Stepwise",
		"selector": lambda: stepwise_fs_selector(data.X, data.y, split_train),
	},
	{
		"key": "reafs_lomo",
		"label": "Test LOMO",
		"selector": lambda: reafs_fs_selector(
			fs_1,
			data.X,
			data.y,
			split_train,
		),
	},
	{
		"key": "reafs_loo",
		"label": "Base Sequential",
		"selector": lambda: reafs_fs_selector(
			fs_5,
			data.X,
			data.y,
			split_train,
		),
	},
]

X_all, y_all = check_X_y(data.X, data.y)
X_train_all, y_train = X_all[split_train], y_all[split_train]
X_test_all, y_test = X_all[split_test], y_all[split_test]
X_train_df_full = data.X.iloc[split_train].copy()

results = []
feature_tables = []

for exp in experiments:
	selected_features = exp["selector"]()
	selected_feature_names = data.X.columns[selected_features].tolist()

	X_train = X_train_all[:, selected_features]
	X_test = X_test_all[:, selected_features]
	X_train_df = data.X.iloc[split_train, selected_features].copy()

	model = clone(mlr_base)
	model.fit(X_train, y_train)

	y_train_preds = model.predict(X_train)
	y_test_preds = model.predict(X_test)
	y_loo_preds = cross_val_predict(model, X_train, y_train, cv=LeavePOut(1))

	# Keep ReaFS evaluation on a fresh fitted model to avoid state changes from
	# other diagnostics (jackknife, noise resilience, etc.).
	model_reafs = clone(mlr_base)
	model_reafs.fit(X_train, y_train)
	reafs_results = reafs_base_evaluator()(model_reafs, X_train, y_train, X_test, y_test)

	scorings = {
		"rmse_lomo": TestScore(LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_root_mean_squared_error"),
		"delta_rmse_lomo": DiffScore(LeaveOneMoreOut(n_repeats=5, n_min=1), "neg_root_mean_squared_error"),
		"y_randomization_mae": YRandomization(scoring="neg_mean_absolute_error"),
		"y_randomization_rmse": YRandomization(scoring="neg_root_mean_squared_error"),
	}

	corr_train = custom_correlation_matrix(X_train)
	corr_top_removed = correlation_with_top_removed(X_train_df)
	vif_values = compute_vif(X_train_df)

	j_scores, j_coefs = jackknife(model, X_train, y_train, scoring="neg_mean_squared_error")
	nr_score = noise_resilience(to_compute="scores")(model, X_train, y_train)
	nr_coef = noise_resilience(to_compute="coefs")(model, X_train, y_train)
	feature_scores = collect_feature_scores(
		X_train_df_full,
		y_train,
		selected_feature_names,
	)
	feature_scores.insert(0, "model", exp["label"])
	feature_tables.append(feature_scores)

	pca_train_scores, pca_model, pca_scaler = make_pca_projection(X_train, n_components=2)
	pca_test_scores = pca_model.transform(pca_scaler.transform(X_test))

	result = {
		"key": exp["key"],
		"label": exp["label"],
		"selected_features": selected_features,
		"selected_feature_names": selected_feature_names,
		"n_train_samples": int(X_train.shape[0]),
		"n_test_samples": int(X_test.shape[0]),
		"n_features": int(X_all.shape[1]),
		"n_selected_features": int(len(selected_features)),
		"r2_train": float(r2_score(y_train, y_train_preds)),
		"r2_loo_train": float(r2_score(y_train, y_loo_preds)),
		"mae_train": float(mean_absolute_error(y_train, y_train_preds)),
		"mae_loo_train": float(mean_absolute_error(y_train, y_loo_preds)),
		"r2_test": float(r2_score(y_test, y_test_preds)),
		"mae_test": float(mean_absolute_error(y_test, y_test_preds)),
		"rmse_lomo": float(scorings["rmse_lomo"](model, X_train, y_train)),
		"delta_rmse_lomo": float(scorings["delta_rmse_lomo"](model, X_train, y_train)),
		"y_randomization_mae": float(scorings["y_randomization_mae"](model, X_train, y_train)),
		"y_randomization_rmse": float(scorings["y_randomization_rmse"](model, X_train, y_train)),
		"jackknife_score": float(j_scores),
		"jackknife_coef": float(j_coefs),
		"noise_resilience_score": float(nr_score),
		"noise_resilience_coef": float(nr_coef),
		"max_abs_corr": float(highest_absolute_off_diagonal(corr_train)),
		"corr_det": float(np.linalg.det(corr_train)),
		"vif": vif_values,
		"corr_train": corr_train,
		"corr_top_removed": corr_top_removed,
		"legacy_like": {
			"rmse_train": float(np.sqrt(np.mean((y_train - y_train_preds) ** 2))),
			"rmse_test": float(np.sqrt(np.mean((y_test - y_test_preds) ** 2))),
			"rmse_loo_train": float(np.sqrt(np.mean((y_train - y_loo_preds) ** 2))),
		},
		"reafs": {
			"S_ReaFS": float(reafs_results["composite"]),
			"R": float(reafs_results["accuracy"]),
			"Q2_t": float(reafs_results["cv_r2"]),
			"r_st": float(reafs_results["cv_spearman"]),
		},
		"pca": {
			"train_scores": pca_train_scores,
			"test_scores": pca_test_scores,
			"loadings": two_component_vectors(pca_model.components_.T),
			"pc_var": pca_model.explained_variance_ratio_ * 100,
		},
	}
	results.append(result)


summary_rows = []
for res in results:
	summary_rows.append(
		{
			"Model": res["label"],
			"n_selected": res["n_selected_features"],
			"R2_train": res["r2_train"],
			"R2_test": res["r2_test"],
			"R2_LOO_train": res["r2_loo_train"],
			"MAE_train": res["mae_train"],
			"MAE_test": res["mae_test"],
			"MAE_LOO_train": res["mae_loo_train"],
			"RMSE_train": res["legacy_like"]["rmse_train"],
			"RMSE_test": res["legacy_like"]["rmse_test"],
			"RMSE_LOO_train": res["legacy_like"]["rmse_loo_train"],
			"RMSE_LOMO": res["rmse_lomo"],
			"Delta_RMSE_LOMO": res["delta_rmse_lomo"],
			"Y_rand_MAE": res["y_randomization_mae"],
			"Y_rand_RMSE": res["y_randomization_rmse"],
			"Jackknife_score": res["jackknife_score"],
			"Jackknife_coef": res["jackknife_coef"],
			"Noise_res_score": res["noise_resilience_score"],
			"Noise_res_coef": res["noise_resilience_coef"],
			"Max_abs_corr": res["max_abs_corr"],
			"Det_corr": res["corr_det"],
			"S_ReaFS_test": res["reafs"]["S_ReaFS"],
			"R_test": res["reafs"]["R"],
			"Q2_t_test": res["reafs"]["Q2_t"],
			"r_st_test": res["reafs"]["r_st"],
			"Features": ", ".join(res["selected_feature_names"]),
		}
	)

summary_df = pd.DataFrame(summary_rows)
feature_scores_df = pd.concat(feature_tables, axis=0, ignore_index=True)

print("\n=== Summary diagnostics (prospective SI) ===")
print(summary_df.round(3))
print("\n=== Feature-level F-test / MI / Laplacian ===")
print(feature_scores_df.round(3))

summary_df.to_csv("results/model_prospect_SI_summary_metrics.csv", index=False)
feature_scores_df.to_csv("results/model_prospect_SI_feature_scores.csv", index=False)


# Parity: 3 panels side-by-side
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
global_y_min = np.min(y_all)
global_y_max = np.max(y_all)

for ax, res in zip(axes, results):
	selected = res["selected_features"]
	model = clone(mlr_base)
	model.fit(X_train_all[:, selected], y_train)
	y_train_pred = model.predict(X_train_all[:, selected])
	y_test_pred = model.predict(X_test_all[:, selected])
	y_loo_pred = cross_val_predict(model, X_train_all[:, selected], y_train, cv=LeavePOut(1))

	ax.scatter(y_train, y_train_pred, color="#007480", s=45, marker="D", label="Train")
	ax.scatter(y_test, y_test_pred, color="#B51F1F", s=60, marker="o", label="Test")
	ax.plot([global_y_min, global_y_max], [global_y_min, global_y_max], "--", lw=2, color="#CAC7C7")

	ax.text(
		0.03,
		0.97,
		(
			f"$R^2$={res['r2_train']:.3f}\n"
			f"$R^2_{{LOO}}$={res['r2_loo_train']:.3f}\n"
			f"MAE={res['mae_train']:.3f}\n"
			f"MAE$_{{LOO}}$={res['mae_loo_train']:.3f}"
		),
		transform=ax.transAxes,
		va="top",
		fontsize=10,
		bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
	)
	ax.set_title(res["label"], fontweight="bold", fontsize=12)
	ax.set_xlabel("True $\\Delta \\Delta G^{\\ddagger}$")
	ax.set_ylabel("Predicted $\\Delta \\Delta G^{\\ddagger}$")
	ax.legend(loc="lower left", fontsize=10)

# plt.tight_layout()
plt.savefig("results/model_prospect_SI_parity_3panel.svg")
plt.close(fig)


# PCA projection: 3 panels side-by-side
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, res in zip(axes, results):
	train_scores = res["pca"]["train_scores"]
	test_scores = res["pca"]["test_scores"]
	loadings = res["pca"]["loadings"]
	pc_var = res["pca"]["pc_var"]

	ax.scatter(train_scores[:, 0], train_scores[:, 1], color="#007480", s=45, marker="D", label="Train")
	ax.scatter(test_scores[:, 0], test_scores[:, 1], color="#B51F1F", s=60, marker="o", label="Test")

	combined_scores = np.vstack([train_scores[:, :2], test_scores[:, :2]])
	arrow_scale = compute_arrow_scale(combined_scores, loadings)
	plot_feature_arrows(
		ax,
		loadings,
		res["selected_feature_names"],
		scale=arrow_scale,
		color="#453A4C",
	)
	fit_axes_to_scores_and_arrows(ax, combined_scores, loadings, scale=arrow_scale)

	pc1 = pc_var[0]
	pc2 = pc_var[1] if pc_var.shape[0] > 1 else 0.0
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
	ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
	ax.set_title(res["label"], fontweight="bold", fontsize=12)
	ax.legend(loc="best", fontsize=10)

# plt.tight_layout()
plt.savefig("results/model_prospect_SI_pca_3panel.svg")
plt.close(fig)


# Correlation matrices with top-correlation feature removed: 3 panels side-by-side
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, res in zip(axes, results):
	corr_reduced = res["corr_top_removed"]["corr_reduced"]
	removed_feature = res["corr_top_removed"]["removed_feature"]
	top_pair = res["corr_top_removed"]["top_pair"]
	top_corr = res["corr_top_removed"]["top_corr_value"]

	im = ax.matshow(corr_reduced, cmap=custom_cmap, vmin=-1, vmax=1)
	for (i, j), val in np.ndenumerate(corr_reduced):
		ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)

	reduced_cols = [c for c in res["selected_feature_names"] if c != removed_feature]
	ax.xaxis.set_ticks_position("bottom")
	ax.set_xticks(range(len(reduced_cols)))
	ax.set_xticklabels(reduced_cols, rotation=45, ha="right", fontsize=10)
	ax.set_yticks(range(len(reduced_cols)))
	ax.set_yticklabels(reduced_cols, fontsize=10)
	ax.set_title(
		(
			f"{res['label']}\n"
			f"Top pair: {top_pair[0]} vs {top_pair[1]} ({top_corr:.2f})\n"
			f"Removed: {removed_feature}"
		),
		fontsize=10,
		fontweight="bold",
	)

fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
fig.subplots_adjust(left=0.05, right=0.92, bottom=0.2, top=0.82, wspace=0.35)
plt.savefig("results/model_prospect_SI_corr_top_removed_3panel.svg")
plt.close(fig)


# Train diagnostics table (3 columns, one per model)
train_metrics_labels = [
	"$R^2$",
	"$R^2_{LOO}$",
	"MAE",
	"MAE$_{LOO}$",
	"RMSE$_{LOMO}$",
	"$\\Delta$RMSE$_{LOMO}$",
	"Y-rand MAE",
	"Y-rand RMSE",
	"Jack. score",
	"Jack. coef",
	"Noise-res score",
	"Noise-res coef",
	"Max $\\rho$",
	"Det(corr)",
]

train_metric_keys = [
	"r2_train",
	"r2_loo_train",
	"mae_train",
	"mae_loo_train",
	"rmse_lomo",
	"delta_rmse_lomo",
	"y_randomization_mae",
	"y_randomization_rmse",
	"jackknife_score",
	"jackknife_coef",
	"noise_resilience_score",
	"noise_resilience_coef",
	"max_abs_corr",
	"corr_det",
]

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis("off")

table_values = []
for label, key in zip(train_metrics_labels, train_metric_keys):
	row = [label]
	for res in results:
		row.append(f"{res[key]:.3f}")
	table_values.append(row)

col_labels = ["Metric"] + [res["label"] for res in results]
table = ax.table(cellText=table_values, colLabels=col_labels, loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.4)

# plt.tight_layout()
plt.savefig("results/model_prospect_SI_train_metrics_table.svg")
plt.close(fig)


# Prospective test ReaFS metrics table
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis("off")

reafs_labels = ["$S_{ReaFS}$", "R", "$Q^2_t$", "$r_{st}$"]
reafs_keys = ["S_ReaFS", "R", "Q2_t", "r_st"]

reafs_table_values = []
for label, key in zip(reafs_labels, reafs_keys):
	row = [label]
	for res in results:
		row.append(f"{res['reafs'][key]:.3f}")
	reafs_table_values.append(row)

table = ax.table(
	cellText=reafs_table_values,
	colLabels=["Test metric"] + [res["label"] for res in results],
	loc="center",
	cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# plt.tight_layout()
plt.savefig("results/model_prospect_SI_test_reafs_table.svg")
plt.close(fig)


# VIF table for each model
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
for ax, res in zip(axes, results):
	ax.axis("off")
	vif_items = list(res["vif"].items())
	row_text = [[k, f"{v:.3f}"] for k, v in vif_items]
	table = ax.table(
		cellText=row_text,
		colLabels=["Feature", "VIF"],
		loc="center",
		cellLoc="left",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1, 1.4)
	ax.set_title(res["label"], fontsize=12, fontweight="bold")

# plt.tight_layout()
plt.savefig("results/model_prospect_SI_vif_3panel.svg")
plt.close(fig)

