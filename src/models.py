from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONFIGURATIONS = {
    "A": ("f1_", "f2_", "f3_"),
    "B": ("h1_",),
    "C": ("f1_", "f2_", "f3_", "h1_"),
    "D": ("f1_", "f2_", "f3_", "h2_"),
    "E": ("f1_", "f2_", "f3_", "h3_"),
    "F": ("f1_", "f2_", "f3_", "h4_"),
}
ALERT_BUDGETS = (0.05, 0.10, 0.20)


def feature_columns(frame: pd.DataFrame, configuration: str) -> list[str]:
    prefixes = CONFIGURATIONS[configuration]
    columns = [column for column in frame if column.startswith(prefixes)]
    if not columns or any("carbon" in column.lower() for column in columns):
        raise ValueError("Missing feature family or carbon leakage")
    return columns


def make_estimator(columns: list[str], family: str, training_size: int, regularization: float = 0.1):
    spectral = [column for column in columns if column.startswith(("h1_", "h4_"))]
    tabular = [column for column in columns if column not in spectral]
    transformers = []
    if tabular:
        transformers.append(("tabular", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), tabular))
    if spectral:
        transformers.append(("spectrum", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(10, len(spectral), training_size - 1), random_state=813)),
        ]), spectral))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    if family == "lr":
        classifier = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000, solver="lbfgs")
    elif family == "rf":
        classifier = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=5,
                                            class_weight="balanced_subsample", random_state=813, n_jobs=1)
    else:
        raise ValueError(f"Unknown model family {family}")
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def _inner_splits(target: np.ndarray, groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    positive_groups = len(np.unique(groups[target == 1]))
    negative_groups = len(np.unique(groups[target == 0]))
    fold_count = min(3, positive_groups, negative_groups)
    if fold_count < 2:
        raise ValueError("Inner grouped CV needs two positive and negative spatial groups")
    for trial in range(20):
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed + trial)
        splits = list(splitter.split(np.zeros(len(target)), target, groups))
        if all(len(np.unique(target[train])) == 2 and len(np.unique(target[test])) == 2 for train, test in splits):
            return splits
    raise ValueError("Cannot form class-complete inner grouped folds")


def _inner_scores(frame: pd.DataFrame, target: np.ndarray, columns: list[str], family: str,
                  splits: list[tuple[np.ndarray, np.ndarray]], regularization: float) -> np.ndarray:
    scores = np.full(len(target), np.nan)
    for train, test in splits:
        if frame.iloc[train][columns].notna().sum().eq(0).any():
            raise ValueError("A model feature is wholly missing in an inner training fold")
        estimator = make_estimator(columns, family, len(train), regularization)
        estimator.fit(frame.iloc[train][columns], target[train])
        scores[test] = estimator.predict_proba(frame.iloc[test][columns])[:, 1]
    return scores


def train_spatial_oof(frame: pd.DataFrame, target: np.ndarray, spatial: dict, configuration: str,
                      family: str, model_dir: Path, plan_sha256: str) -> tuple[pd.DataFrame, list]:
    columns = feature_columns(frame, configuration)
    model_dir.mkdir(parents=True, exist_ok=True)
    results = []
    models = []
    for fold_index, (train, test) in enumerate(spatial["splits"]):
        training_frame = frame.iloc[train].reset_index(drop=True)
        training_target = target[train]
        inner = _inner_splits(training_target, spatial["groups"][train], 813 + fold_index)
        if family == "lr":
            candidates = (0.01, 0.1, 1.0)
            assessed = []
            for regularization in candidates:
                scores = _inner_scores(training_frame, training_target, columns, family, inner, regularization)
                assessed.append(average_precision_score(training_target, scores))
            chosen = candidates[int(np.argmax(assessed))]
        else:
            chosen = 0.1
        inner_scores = _inner_scores(training_frame, training_target, columns, family, inner, chosen)
        thresholds = {budget: float(np.quantile(inner_scores, 1 - budget)) for budget in ALERT_BUDGETS}
        clipped_inner = np.clip(inner_scores, 1e-5, 1 - 1e-5)
        inner_logit = np.log(clipped_inner / (1 - clipped_inner)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(inner_logit, training_target)
        model = make_estimator(columns, family, len(train), chosen)
        if frame.iloc[train][columns].notna().sum().eq(0).any():
            raise ValueError("A model feature is wholly missing in an outer training fold")
        model.fit(frame.iloc[train][columns], training_target)
        predictions = model.predict_proba(frame.iloc[test][columns])[:, 1]
        clipped_test = np.clip(predictions, 1e-5, 1 - 1e-5)
        calibrated = calibrator.predict_proba(np.log(clipped_test / (1 - clipped_test)).reshape(-1, 1))[:, 1]
        result = pd.DataFrame({"sample_index": test, "fold": fold_index, "model": f"{family}_{configuration}",
                               "risk": predictions, "calibrated_risk": calibrated,
                               "target": target[test], "analysis_plan_sha256": plan_sha256})
        for budget, threshold in thresholds.items():
            result[f"alert_{int(budget * 100)}"] = predictions >= threshold
            result[f"threshold_{int(budget * 100)}"] = threshold
        results.append(result)
        model_path = model_dir / f"{family}_{configuration}_fold{fold_index}.joblib"
        joblib.dump({"model": model, "calibrator": calibrator, "features": columns, "regularization": chosen,
                     "analysis_plan_sha256": plan_sha256, "train_indices": train, "test_indices": test,
                     "inner_thresholds": thresholds}, model_path)
        models.append(model)
    return pd.concat(results, ignore_index=True).sort_values("sample_index"), models
