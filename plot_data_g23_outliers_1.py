import numpy as np

from reafs.visualization import target_distribution_plot
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from reafs.datasets import full_datasets, from_dataset

dataset_name = "gallarati2023"
dataset_i_target = 0
dataset_i_direction = 1

data = from_dataset(dataset_name, dataset_i_target, dataset_i_direction)

y = np.asarray(data.y)
y = np.sort(y)
clf = LocalOutlierFactor()
_ = clf.fit_predict(y.reshape(-1, 1))
X_scores = clf.negative_outlier_factor_

# Scale LOF scores to match the range of KDE values
kde = gaussian_kde(y)
y_range = np.linspace(y.min(), y.max(), 64)
kde_values = kde(y_range)

fig, ax1 = plt.subplots(figsize=(6, 3))

# Kernel density plot using gaussian_kde
ax1.plot(y_range, kde_values, c="#007480", lw=3, label="KDE Density", zorder=1)
ax1.set_xlabel(r"$\Delta \Delta G^{\ddagger}$ (kcal/mol)")
ax1.set_ylabel("Density", color="#007480")
ax1.tick_params(axis='y', labelcolor="#007480")

# Create a second y-axis for the scaled outlier scores
ax2 = ax1.twinx()
ax2.scatter(y[:-5], X_scores[:-5], c="#B51F1F", alpha=.7, s=100, label="Outlier Scores", zorder=2)
ax2.scatter(y[-5:], X_scores[-5:], c="#f39869", alpha=.7, s=100, marker="*", label="Outlier Scores (test)", zorder=3)
ax2.set_ylabel("nLOF", color="#B51F1F")
ax2.tick_params(axis='y', labelcolor="#B51F1F")

# Calculate standard deviation of the negative outlier factors
std_dev_outlier_factors = np.std(X_scores[:-5])

# Add a text box with the standard deviation
textstr = f"$\\sigma={std_dev_outlier_factors:.2f}$"
ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            color="#B51F1F")

# Add legends
fig.tight_layout()

plt.savefig("results/g23_outliers_0_1.svg")