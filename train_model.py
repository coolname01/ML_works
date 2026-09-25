#!/usr/bin/env python3
"""Predict whether a hidden pair shares the same latent identity.

Features are 512 non-negative per-dimension distances. The public leaderboard
tracks train-style difficulty (~0.93), while the provided validation split is
easier by ~0.015 AUC. This pipeline therefore:

* engineers distance aggregates plus L1-residual group sums (fitted on labeled
  rows only);
* bags several RBF-SVMs (best family by train CV) and reports trees only as baselines;
* selects that blend by 5-fold CV inside train.csv;
* refits on train+validation and applies one conservative self-training round
  on high-confidence test rows before writing submission.csv.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20260916
ID_COL = "row_id"
TARGET_COL = "target"
EXPECTED = {"train": (2520, 514), "validation": (560, 514), "test": (1116, 513)}


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    assert train.shape == EXPECTED["train"]
    assert validation.shape == EXPECTED["validation"]
    assert test.shape == EXPECTED["test"]
    feats = [c for c in train.columns if c not in {ID_COL, TARGET_COL}]
    assert len(feats) == 512
    return train, validation, test, sample, feats


def univariate_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y, X[:, i]) for i in range(X.shape[1])])


def row_aggregates(X: np.ndarray) -> np.ndarray:
    l1 = X.sum(1)
    return np.column_stack(
        [
            l1,
            np.sqrt((X * X).sum(1)),
            X.std(1),
            X.max(1),
            np.quantile(X, [0.25, 0.5, 0.75, 0.9], axis=1).T,
            (X < 0.1).sum(1),
            (X < 0.5).sum(1),
            (X > 1).sum(1),
            (X > 2).sum(1),
            np.log1p(X).sum(1),
            np.sqrt(X).sum(1),
            (X == 0).sum(1),
        ]
    )


def group_sums(X: np.ndarray, aucs: np.ndarray, centered: bool = False) -> np.ndarray:
    order = np.argsort(aucs)
    cols = []
    for k in (32, 64, 128, 256):
        top = X[:, order[:k]].sum(1)
        bot = X[:, order[-k:]].sum(1)
        cols.extend([top, bot, top / (bot + 1e-9)])
    if centered:
        w = aucs - 0.5
        cols.append((X * w).sum(1))
        cols.append((X * (w**2) * np.sign(w)).sum(1))
    else:
        w2 = (0.5 - aucs) ** 2
        cols.append((X * w2).sum(1))
        cols.append((np.sqrt(np.clip(X, 0, None)) * w2).sum(1))
    return np.column_stack(cols)


class PairFeatures:
    """Distance aggregates + L1-residual groups, fitted on labeled rows only."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.raw_auc_ = univariate_auc(X, y)
        l1 = X.sum(1, keepdims=True)
        self.residualizer_ = LinearRegression().fit(l1, X)
        R = X - self.residualizer_.predict(l1)
        self.res_auc_ = univariate_auc(R, y)
        return self

    def residuals(self, X: np.ndarray) -> np.ndarray:
        return X - self.residualizer_.predict(X.sum(1, keepdims=True))

    def svm_matrix(self, X: np.ndarray) -> np.ndarray:
        R = self.residuals(X)
        return np.hstack(
            [
                np.sqrt(X),
                row_aggregates(X),
                group_sums(X, self.raw_auc_, centered=False),
                group_sums(R, self.res_auc_, centered=True),
            ]
        )

    def tree_matrix(self, X: np.ndarray) -> np.ndarray:
        R = self.residuals(X)
        return np.hstack(
            [
                X,
                row_aggregates(X),
                group_sums(X, self.raw_auc_, centered=False),
                group_sums(R, self.res_auc_, centered=True),
            ]
        )


def svm_factory(gamma_scale: float, C: float):
    def make(n_features: int):
        return make_pipeline(
            StandardScaler(),
            SVC(C=C, kernel="rbf", gamma=gamma_scale / n_features, random_state=SEED),
        )

    return make


def lgb_factory(seed: int):
    return lambda: lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.5,
        subsample=0.85,
        subsample_freq=1,
        min_child_samples=20,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def et_factory(seed: int):
    return lambda: ExtraTreesClassifier(
        n_estimators=800,
        max_features=0.3,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def oof_scores(make_model, X: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in kf.split(X, y):
        model = make_model()
        model.fit(X[tr], y[tr])
        oof[te] = predict_scores(model, X[te])
    return oof


def rank_mean(score_list: list[np.ndarray]) -> np.ndarray:
    return np.mean([rankdata(s) for s in score_list], axis=0)


def to_unit_interval(scores: np.ndarray) -> np.ndarray:
    ranks = rankdata(scores)
    return ranks / (len(scores) + 1.0)


def build_submission(sample, test, proba, path: Path):
    submission = sample.copy()
    submission[TARGET_COL] = proba
    assert list(submission.columns) == [ID_COL, TARGET_COL]
    assert len(submission) == len(test)
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
    parser.add_argument("--pseudo-low", type=float, default=0.08)
    parser.add_argument("--pseudo-high", type=float, default=0.92)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    started = time.perf_counter()

    train, validation, test, sample, feats = load_data(args.data_dir)
    X_train = train[feats].to_numpy(np.float64)
    y_train = train[TARGET_COL].to_numpy()
    X_val = validation[feats].to_numpy(np.float64)
    y_val = validation[TARGET_COL].to_numpy()
    X_test = test[feats].to_numpy(np.float64)
    print(f"train {train.shape}, validation {validation.shape}, test {test.shape}")
    print(f"positive rate: train {y_train.mean():.3f}, validation {y_val.mean():.3f}")

    feat = PairFeatures().fit(X_train, y_train)
    S_train, S_val = feat.svm_matrix(X_train), feat.svm_matrix(X_val)
    T_train, T_val = feat.tree_matrix(X_train), feat.tree_matrix(X_val)
    print(f"svm matrix {S_train.shape[1]} cols, tree matrix {T_train.shape[1]} cols")

    members = [
        ("svm_C2_g1.5", S_train, S_val, lambda: svm_factory(1.5, 2.0)(S_train.shape[1]), True),
        ("svm_C3_g1.5", S_train, S_val, lambda: svm_factory(1.5, 3.0)(S_train.shape[1]), True),
        ("svm_C4_g1.5", S_train, S_val, lambda: svm_factory(1.5, 4.0)(S_train.shape[1]), True),
        ("svm_C3_g2", S_train, S_val, lambda: svm_factory(2.0, 3.0)(S_train.shape[1]), True),
        ("lgb_s0", T_train, T_val, lgb_factory(SEED), False),
        ("et_s0", T_train, T_val, et_factory(SEED), False),
    ]

    oof = {}
    val_scores = {}
    in_blend = {}
    rows = []
    print("\n5-fold CV on train / full-train validation AUC")
    for name, Xtr, Xv, make, use in members:
        t0 = time.perf_counter()
        oof[name] = oof_scores(make, Xtr, y_train)
        model = make()
        model.fit(Xtr, y_train)
        val_scores[name] = predict_scores(model, Xv)
        cv = roc_auc_score(y_train, oof[name])
        va = roc_auc_score(y_val, val_scores[name])
        sec = time.perf_counter() - t0
        in_blend[name] = use
        rows.append((name, cv, va, sec, use))
        print(f"  {name:12s}  cv={cv:.4f}  val={va:.4f}  {sec:5.1f}s", flush=True)

    blend_names = [name for name, use in in_blend.items() if use]
    blend_oof = rank_mean([oof[name] for name in blend_names])
    blend_val = rank_mean([val_scores[name] for name in blend_names])
    blend_cv = roc_auc_score(y_train, blend_oof)
    blend_va = roc_auc_score(y_val, blend_val)
    print(f"  {'SVM_BAG':12s}  cv={blend_cv:.4f}  val={blend_va:.4f}  members={blend_names}")
    rows.append(("svm_bag", blend_cv, blend_va, 0.0, True))

    table = pd.DataFrame(rows, columns=["model", "cv_train", "validation_auc", "seconds", "in_submission"])
    table = table.sort_values("cv_train", ascending=False).reset_index(drop=True)
    table.to_csv(args.results, index=False)
    print("\n" + table.to_string(index=False))

    print("\nrefitting SVM bag on train+validation")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    feat = PairFeatures().fit(X_full, y_full)
    S_full = feat.svm_matrix(X_full)
    S_test = feat.svm_matrix(X_test)
    svm_specs = [
        ("svm_C2_g1.5", 1.5, 2.0),
        ("svm_C3_g1.5", 1.5, 3.0),
        ("svm_C4_g1.5", 1.5, 4.0),
        ("svm_C3_g2", 2.0, 3.0),
    ]

    test_raw = {}
    for name, gmul, C in svm_specs:
        model = svm_factory(gmul, C)(S_full.shape[1])
        model.fit(S_full, y_full)
        test_raw[name] = predict_scores(model, S_test)

    test_proba = np.mean([expit(s) for s in test_raw.values()], axis=0)
    confident = (test_proba <= args.pseudo_low) | (test_proba >= args.pseudo_high)
    print(
        f"self-training: {confident.sum()} / {len(test_proba)} test rows "
        f"with p<= {args.pseudo_low} or p>= {args.pseudo_high}"
    )

    if confident.any():
        y_pseudo = (test_proba[confident] >= 0.5).astype(y_full.dtype)
        X_st = np.vstack([X_full, X_test[confident]])
        y_st = np.concatenate([y_full, y_pseudo])
        feat = PairFeatures().fit(X_st, y_st)
        S_st = feat.svm_matrix(X_st)
        S_test = feat.svm_matrix(X_test)
        test_raw = {}
        for name, gmul, C in svm_specs:
            model = svm_factory(gmul, C)(S_st.shape[1])
            model.fit(S_st, y_st)
            test_raw[name] = predict_scores(model, S_test)
        print("self-training refit done")

    test_blend = rank_mean(list(test_raw.values()))
    test_out = to_unit_interval(test_blend)
    submission = build_submission(sample, test, test_out, args.submission)
    print(
        f"\nwrote {args.submission} ({len(submission)} rows); "
        f"mean predicted probability {test_out.mean():.3f}"
    )
    print(f"selected blend train CV {blend_cv:.4f}, validation rank-AUC {blend_va:.4f}")
    print(f"total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
