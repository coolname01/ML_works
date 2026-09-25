#!/usr/bin/env python3
"""Predict whether two anonymous observations share a latent identity.

Pipeline: load data -> feature engineering fitted on train.csv -> tree and
tree-ensemble candidates fitted on train.csv and compared on validation.csv ->
selection of the best configuration by validation ROC-AUC -> test predictions
-> submission.csv.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 20260916
ID_COL = "row_id"
TARGET_COL = "target"
EXPECTED_SHAPES = {"train": (2520, 514), "validation": (560, 514), "test": (1116, 513)}


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.shape != EXPECTED_SHAPES[name]:
            raise ValueError(f"{name}.csv has shape {frame.shape}, expected {EXPECTED_SHAPES[name]}")
        if frame.isna().any().any():
            raise ValueError(f"{name}.csv contains missing values")
    if list(sample_submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample_submission.csv must have columns row_id,target")
    if not np.array_equal(test[ID_COL].to_numpy(), sample_submission[ID_COL].to_numpy()):
        raise ValueError("row_id order differs between test.csv and sample_submission.csv")

    feature_columns = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    for frame in (validation, test):
        if [c for c in frame.columns if c not in (ID_COL, TARGET_COL)] != feature_columns:
            raise ValueError("feature columns differ between the provided files")
    return train, validation, test, sample_submission, feature_columns


class PairDistanceFeatures:
    """Row-level features for a vector of per-dimension distances between two observations.

    Every raw feature is non-negative and, taken alone, is larger for pairs with
    different identities, so the 512 columns behave like per-dimension distances.
    The unsupervised part summarises the distance profile of each row. The
    supervised part ranks dimensions by how well they separate the classes on the
    training rows and aggregates the most and least discriminative dimensions
    separately, which lets shallow trees approximate a weighted distance.
    """

    def __init__(self, group_sizes=(32, 64, 128, 256), weight_powers=(2, 4)):
        self.group_sizes = group_sizes
        self.weight_powers = weight_powers

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairDistanceFeatures":
        aucs = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        self.order_ = np.argsort(aucs)
        self.weights_ = np.clip(0.5 - aucs, 0.0, None)
        return self

    def transform(self, X: np.ndarray, raw_names: list[str]) -> pd.DataFrame:
        sorted_rows = np.sort(X, axis=1)
        features = {name: X[:, j] for j, name in enumerate(raw_names)}
        features.update(
            {
                "agg_l1": X.sum(axis=1),
                "agg_l2": np.sqrt((X**2).sum(axis=1)),
                "agg_std": X.std(axis=1),
                "agg_max": X.max(axis=1),
                "agg_q25": np.quantile(X, 0.25, axis=1),
                "agg_median": np.median(X, axis=1),
                "agg_q75": np.quantile(X, 0.75, axis=1),
                "agg_q90": np.quantile(X, 0.90, axis=1),
                "agg_n_below_0.1": (X < 0.1).sum(axis=1),
                "agg_n_below_0.5": (X < 0.5).sum(axis=1),
                "agg_n_above_1": (X > 1.0).sum(axis=1),
                "agg_n_above_2": (X > 2.0).sum(axis=1),
                "agg_log_sum": np.log(X + 1e-3).sum(axis=1),
                "agg_sqrt_sum": np.sqrt(X).sum(axis=1),
                "agg_top16_mean": sorted_rows[:, -16:].mean(axis=1),
                "agg_bottom16_mean": sorted_rows[:, :16].mean(axis=1),
            }
        )
        for k in self.group_sizes:
            top = X[:, self.order_[:k]].sum(axis=1)
            bottom = X[:, self.order_[-k:]].sum(axis=1)
            features[f"grp_top{k}_sum"] = top
            features[f"grp_bottom{k}_sum"] = bottom
            features[f"grp_top{k}_ratio"] = top / (bottom + 1e-6)
        for p in self.weight_powers:
            features[f"grp_weighted_sum_p{p}"] = (X * self.weights_**p).sum(axis=1)
        return pd.DataFrame(features)


@dataclass
class Candidate:
    name: str
    family: str
    make: Callable[[], object]
    feature_set: str = "engineered"
    selectable: bool = True


@dataclass
class Result:
    name: str
    family: str
    feature_set: str
    validation_auc: float
    fit_seconds: float
    selectable: bool
    members: list[str] = field(default_factory=list)


def lgbm(**params):
    base = dict(
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=20,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )
    return lambda: lgb.LGBMClassifier(**{**base, **params})


def xgboost(**params):
    base = dict(
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=5,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return lambda: xgb.XGBClassifier(**{**base, **params})


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            "DecisionTree depth=3 leaf=20 (notebook baseline)",
            "DecisionTree",
            lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED),
            feature_set="raw",
        ),
        Candidate(
            "DecisionTree depth=3 leaf=20",
            "DecisionTree",
            lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED),
        ),
        Candidate(
            "DecisionTree depth=6 leaf=20",
            "DecisionTree",
            lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
        ),
        Candidate(
            "RandomForest 800 trees max_features=sqrt",
            "RandomForest",
            lambda: RandomForestClassifier(n_estimators=800, n_jobs=-1, random_state=SEED),
        ),
        Candidate(
            "RandomForest 800 trees max_features=0.2",
            "RandomForest",
            lambda: RandomForestClassifier(n_estimators=800, max_features=0.2, n_jobs=-1, random_state=SEED),
        ),
        Candidate(
            "ExtraTrees 800 trees max_features=sqrt",
            "ExtraTrees",
            lambda: ExtraTreesClassifier(n_estimators=800, n_jobs=-1, random_state=SEED),
        ),
        Candidate(
            "ExtraTrees 800 trees max_features=0.2",
            "ExtraTrees",
            lambda: ExtraTreesClassifier(n_estimators=800, max_features=0.2, n_jobs=-1, random_state=SEED),
        ),
        Candidate(
            "HistGradientBoosting lr=0.05 iter=300 leaves=7",
            "HistGradientBoosting",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=7, random_state=SEED
            ),
        ),
        Candidate(
            "HistGradientBoosting lr=0.03 iter=800 leaves=15",
            "HistGradientBoosting",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.03, max_iter=800, max_leaf_nodes=15, random_state=SEED
            ),
        ),
        Candidate(
            "HistGradientBoosting lr=0.05 iter=400 leaves=31",
            "HistGradientBoosting",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=31, random_state=SEED
            ),
        ),
        Candidate(
            "LightGBM lr=0.03 n=600 leaves=7 colsample=0.3",
            "LightGBM",
            lgbm(learning_rate=0.03, n_estimators=600, num_leaves=7, colsample_bytree=0.3),
        ),
        Candidate(
            "LightGBM lr=0.03 n=600 leaves=15 colsample=0.5",
            "LightGBM",
            lgbm(learning_rate=0.03, n_estimators=600, num_leaves=15, colsample_bytree=0.5),
        ),
        Candidate(
            "LightGBM lr=0.05 n=400 leaves=31 colsample=0.5",
            "LightGBM",
            lgbm(learning_rate=0.05, n_estimators=400, num_leaves=31, colsample_bytree=0.5),
        ),
        Candidate(
            "LightGBM lr=0.02 n=1000 leaves=15 colsample=0.3",
            "LightGBM",
            lgbm(learning_rate=0.02, n_estimators=1000, num_leaves=15, colsample_bytree=0.3),
        ),
        Candidate(
            "LightGBM lr=0.03 n=800 leaves=15 colsample=0.2",
            "LightGBM",
            lgbm(learning_rate=0.03, n_estimators=800, num_leaves=15, colsample_bytree=0.2),
        ),
        Candidate(
            "XGBoost depth=3 lr=0.03 n=600",
            "XGBoost",
            xgboost(max_depth=3, learning_rate=0.03, n_estimators=600),
        ),
        Candidate(
            "XGBoost depth=5 lr=0.03 n=600",
            "XGBoost",
            xgboost(max_depth=5, learning_rate=0.03, n_estimators=600),
        ),
        Candidate(
            "CatBoost depth=6 iter=1000 (defaults)",
            "CatBoost",
            lambda: CatBoostClassifier(
                iterations=1000, random_seed=SEED, verbose=0, thread_count=-1, allow_writing_files=False
            ),
        ),
        Candidate(
            "CatBoost depth=4 lr=0.03 iter=1500",
            "CatBoost",
            lambda: CatBoostClassifier(
                iterations=1500,
                depth=4,
                learning_rate=0.03,
                random_seed=SEED,
                verbose=0,
                thread_count=-1,
                allow_writing_files=False,
            ),
        ),
        Candidate(
            "LogisticRegression C=0.01 (reference only)",
            "LogisticRegression",
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)),
            selectable=False,
        ),
    ]


BLENDS = [
    ["LightGBM", "HistGradientBoosting"],
    ["LightGBM", "HistGradientBoosting", "CatBoost"],
    ["LightGBM", "HistGradientBoosting", "CatBoost", "ExtraTrees"],
    ["LightGBM", "HistGradientBoosting", "XGBoost", "CatBoost", "ExtraTrees", "RandomForest"],
]


def evaluate_candidates(candidates, matrices, y_train, y_validation):
    results, models, predictions = [], {}, {}
    for candidate in candidates:
        X_train, X_validation = matrices[candidate.feature_set]
        started = time.perf_counter()
        model = candidate.make().fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        proba = model.predict_proba(X_validation)[:, 1]
        auc = roc_auc_score(y_validation, proba)
        results.append(
            Result(candidate.name, candidate.family, candidate.feature_set, auc, fit_seconds, candidate.selectable)
        )
        models[candidate.name] = model
        predictions[candidate.name] = proba
        print(f"  {auc:.4f}  {fit_seconds:6.1f}s  {candidate.name}", flush=True)
    return results, models, predictions


def add_blends(results, predictions, y_validation):
    best_per_family = {}
    for result in results:
        if result.selectable and (
            result.family not in best_per_family or result.validation_auc > best_per_family[result.family].validation_auc
        ):
            best_per_family[result.family] = result

    for families in BLENDS:
        members = [best_per_family[family].name for family in families]
        proba = np.mean([predictions[name] for name in members], axis=0)
        name = "Blend(" + " + ".join(families) + ")"
        auc = roc_auc_score(y_validation, proba)
        fit_seconds = sum(best_per_family[family].fit_seconds for family in families)
        results.append(Result(name, "Blend", "engineered", auc, fit_seconds, True, members))
        predictions[name] = proba
        print(f"  {auc:.4f}  {fit_seconds:6.1f}s  {name}", flush=True)
    return results


def results_table(results) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "model": [r.name for r in results],
            "family": [r.family for r in results],
            "features": [r.feature_set for r in results],
            "validation_auc": [round(r.validation_auc, 4) for r in results],
            "fit_seconds": [round(r.fit_seconds, 1) for r in results],
            "selectable": [r.selectable for r in results],
        }
    )
    return table.sort_values("validation_auc", ascending=False).reset_index(drop=True)


def build_submission(sample_submission, test, proba, path: Path):
    submission = sample_submission.copy()
    submission[TARGET_COL] = proba
    assert list(submission.columns) == [ID_COL, TARGET_COL]
    assert len(submission) == len(test) == EXPECTED_SHAPES["test"][0]
    assert np.array_equal(submission[ID_COL].to_numpy(), test[ID_COL].to_numpy())
    assert submission[ID_COL].is_unique
    assert not submission[TARGET_COL].isna().any()
    assert submission[TARGET_COL].between(0.0, 1.0).all()
    submission.to_csv(path, index=False)
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--results", type=Path, default=Path("validation_results.csv"))
    parser.add_argument(
        "--refit-on-train-val",
        action="store_true",
        help="After selecting on validation, refit the chosen configuration on train+validation "
        "before predicting test (off by default: the submitted model is fitted on train.csv only).",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    started = time.perf_counter()

    train, validation, test, sample_submission, feature_columns = load_data(args.data_dir)
    X_train_raw = train[feature_columns].to_numpy(dtype=np.float64)
    X_validation_raw = validation[feature_columns].to_numpy(dtype=np.float64)
    X_test_raw = test[feature_columns].to_numpy(dtype=np.float64)
    y_train = train[TARGET_COL].to_numpy()
    y_validation = validation[TARGET_COL].to_numpy()
    print(f"train {train.shape}, validation {validation.shape}, test {test.shape}")
    print(f"positive rate: train {y_train.mean():.3f}, validation {y_validation.mean():.3f}")

    engineer = PairDistanceFeatures().fit(X_train_raw, y_train)
    X_train = engineer.transform(X_train_raw, feature_columns)
    X_validation = engineer.transform(X_validation_raw, feature_columns)
    print(f"engineered feature matrix: {X_train.shape[1]} columns ({len(feature_columns)} raw)")
    matrices = {
        "raw": (X_train_raw, X_validation_raw),
        "engineered": (X_train.to_numpy(), X_validation.to_numpy()),
    }

    print("\nvalidation ROC-AUC (all models fitted on train.csv only):")
    candidates = build_candidates()
    results, models, predictions = evaluate_candidates(candidates, matrices, y_train, y_validation)
    results = add_blends(results, predictions, y_validation)

    table = results_table(results)
    table.to_csv(args.results, index=False)
    print("\n" + table.to_string(index=False))

    selected = max((r for r in results if r.selectable), key=lambda r: r.validation_auc)
    members = selected.members or [selected.name]
    print(f"\nselected: {selected.name} (validation AUC {selected.validation_auc:.4f})")

    if args.refit_on_train_val:
        X_full_raw = np.vstack([X_train_raw, X_validation_raw])
        y_full = np.concatenate([y_train, y_validation])
        engineer = PairDistanceFeatures().fit(X_full_raw, y_full)
        X_full = engineer.transform(X_full_raw, feature_columns).to_numpy()
        by_name = {c.name: c for c in candidates}
        models = {name: by_name[name].make().fit(X_full, y_full) for name in members}
        print("final models refitted on train+validation")

    X_test = engineer.transform(X_test_raw, feature_columns).to_numpy()
    test_proba = np.mean([models[name].predict_proba(X_test)[:, 1] for name in members], axis=0)
    submission = build_submission(sample_submission, test, test_proba, args.submission)
    print(f"\nwrote {args.submission} ({len(submission)} rows); mean predicted probability {test_proba.mean():.3f}")
    print(f"total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
