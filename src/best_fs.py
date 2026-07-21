

from reafs.feature_selection import BeamSequential
from reafs.settings import reafs_base_estimator
from reafs.metrics import TestScore, YRandomization, DiffScore
from reafs.cross_validation import LeaveOneMoreOut

from sklearn.model_selection import LeavePOut

fs_1 = BeamSequential(
    estimator=reafs_base_estimator(0),
    n_features_to_select="auto",
    tol=1e-3,
    scoring=TestScore(cv=LeaveOneMoreOut(n_repeats=5,n_min=1),scoring="neg_mean_squared_error"),
    n_candidates=10
)

fs_2 = BeamSequential(
    estimator=reafs_base_estimator(0),
    n_features_to_select="auto",
    tol=1e-9,
    scoring=YRandomization(scoring="neg_mean_absolute_error", agg="mean", random_state=42),
    n_candidates=10
)

fs_3 = BeamSequential(
    estimator=reafs_base_estimator(0),
    n_features_to_select="auto",
    tol=1e-9,
    scoring=YRandomization(scoring="neg_mean_squared_error", agg="mean", random_state=42),
    n_candidates=10
)

fs_4 = BeamSequential(
    estimator=reafs_base_estimator(0),
    n_features_to_select="auto",
    tol=1e-3,
    scoring=DiffScore(cv=LeaveOneMoreOut(n_repeats=5,n_min=1),scoring="neg_mean_absolute_error"),
    n_candidates=10
)

fs_5 = BeamSequential(
    estimator=reafs_base_estimator(0),
    n_features_to_select="auto",
    tol=1e-3,
    scoring=TestScore(cv=LeavePOut(1),scoring="neg_mean_squared_error"),
    n_candidates=1
)