from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .evaluation import paired_block_bootstrap, prediction_metrics
from .models import train_spatial_oof
from .spatial_cv import assign_spatial_folds, projected_xy


def original_fold_splits(frame: pd.DataFrame, buffer_m: float = 60) -> dict:
    folds = frame.spatial_fold.to_numpy(dtype=int)
    groups = frame.spatial_block.to_numpy(dtype=int)
    xy = projected_xy(frame.lon.to_numpy(), frame.lat.to_numpy())
    target = frame.target_binary.to_numpy(dtype=int)
    splits = []
    for fold in sorted(np.unique(folds)):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        distance, _ = cKDTree(xy[test]).query(xy[train])
        train = train[distance >= buffer_m]
        if target[train].sum() == 0 or target[test].sum() == 0:
            raise ValueError("Sensitivity fold missing positive cases")
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Spatial groups cross sensitivity folds")
        splits.append((train, test))
    return {"groups": groups, "folds": folds, "splits": splits,
            "block_size_m": 1000, "buffer_m": buffer_m}


def fit_subset(frame: pd.DataFrame, spatial: dict, name: str, analysis_type: str,
               configurations: tuple[str, ...], families: tuple[str, ...],
               output_dir: Path, plan_sha256: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = frame.target_binary.to_numpy(dtype=int)
    if not (0 < target.sum() < len(target)):
        raise ValueError("Subset lacks two outcome classes")
    predictions = []
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for family in families:
        for configuration in configurations:
            oof, _ = train_spatial_oof(frame, target, spatial, configuration, family,
                                       output_dir / "model_metadata" / name, plan_sha256)
            oof["analysis_type"] = analysis_type
            oof["analysis_name"] = name
            predictions.append(oof)
            metric = prediction_metrics(target, oof.sort_values("sample_index").risk.to_numpy())
            rows.append({"analysis_name": name, "model": f"{family}_{configuration}",
                         "positive_patches": int(frame.loc[frame.target_binary == 1, "positive_patch_id"].nunique()),
                         **metric, "brier_calibrated": float(brier_score_loss(target, oof.sort_values("sample_index").calibrated_risk)),
                         "analysis_type": analysis_type, "analysis_plan_sha256": plan_sha256})
            print(f"{name}: {family}_{configuration} complete", flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_parquet(output_dir / f"{name}_oof.parquet", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / f"{name}_metrics.csv", index=False)
    indexed = {model: group.sort_values("sample_index") for model, group in combined.groupby("model")}
    comparisons = []
    for family in families:
        pairs = (("C", "A"), ("C", "D"), ("E", "A")) if name == "dieback_only" else (("C", "A"),)
        for first, second in pairs:
            if f"{family}_{first}" not in indexed or f"{family}_{second}" not in indexed:
                continue
            boot = paired_block_bootstrap(target, indexed[f"{family}_{first}"].risk.to_numpy(),
                                          indexed[f"{family}_{second}"].risk.to_numpy(), spatial["groups"])
            comparisons.append({"analysis_name": name, "family": family, "comparison": f"{first}-{second}",
                                "n": len(frame), "positives": int(target.sum()),
                                "positive_patches": int(frame.loc[frame.target_binary == 1, "positive_patch_id"].nunique()),
                                **boot, "analysis_type": analysis_type, "analysis_plan_sha256": plan_sha256})
    differences = pd.DataFrame(comparisons)
    differences.to_csv(output_dir / f"{name}_paired.csv", index=False)
    return combined, metrics, differences


def full_rf_inference(selected: pd.DataFrame, oof: pd.DataFrame, plan_sha256: str) -> pd.DataFrame:
    target = (selected.target == "Persistent Degradation").to_numpy(dtype=int)
    indexed = {model: group.sort_values("sample_index") for model, group in oof.groupby("model")}
    rows = []
    for first, second in (("C", "A"), ("C", "D"), ("E", "A")):
        first_oof = indexed[f"rf_{first}"]
        second_oof = indexed[f"rf_{second}"]
        boot = paired_block_bootstrap(target, first_oof.risk.to_numpy(), second_oof.risk.to_numpy(),
                                      selected.spatial_block.to_numpy())
        rows.append({"family": "rf", "comparison": f"{first}-{second}",
                     "pr_auc_first": average_precision_score(target, first_oof.risk),
                     "pr_auc_second": average_precision_score(target, second_oof.risk),
                     "roc_auc_first": roc_auc_score(target, first_oof.risk),
                     "roc_auc_second": roc_auc_score(target, second_oof.risk),
                     "brier_first": brier_score_loss(target, first_oof.risk),
                     "brier_second": brier_score_loss(target, second_oof.risk),
                     "brier_calibrated_first": brier_score_loss(target, first_oof.calibrated_risk),
                     "brier_calibrated_second": brier_score_loss(target, second_oof.calibrated_risk),
                     **boot, "analysis_type": "PRESPECIFIED_SECONDARY", "analysis_plan_sha256": plan_sha256})
    return pd.DataFrame(rows)


def individual_score_ci(target: np.ndarray, risk: np.ndarray, groups: np.ndarray,
                        repeats: int = 500, seed: int = 813) -> dict:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    pr_scores = []
    roc_scores = []
    for _ in range(repeats):
        sampled = generator.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        if len(np.unique(target[indices])) < 2:
            continue
        pr_scores.append(average_precision_score(target[indices], risk[indices]))
        roc_scores.append(roc_auc_score(target[indices], risk[indices]))
    return {"pr_auc_ci_low": float(np.quantile(pr_scores, .025)),
            "pr_auc_ci_high": float(np.quantile(pr_scores, .975)),
            "roc_auc_ci_low": float(np.quantile(roc_scores, .025)),
            "roc_auc_ci_high": float(np.quantile(roc_scores, .975)),
            "valid_bootstraps": len(pr_scores)}


def all_model_metrics(selected: pd.DataFrame, oof: pd.DataFrame, h4: pd.DataFrame,
                      cnn: pd.DataFrame, plan_sha256: str) -> pd.DataFrame:
    from sklearn.calibration import calibration_curve

    groups = selected.spatial_block.to_numpy()
    frames = [oof, h4]
    rows = []
    for frame in frames:
        for model, group in frame.groupby("model"):
            ordered = group.sort_values("sample_index")
            target = ordered.target.to_numpy(dtype=int)
            risk = ordered.risk.to_numpy(dtype=float)
            calibrated = ordered.calibrated_risk.to_numpy(dtype=float)
            rows.append({"model": model, "seed": np.nan,
                         "pr_auc": average_precision_score(target, risk),
                         "roc_auc": roc_auc_score(target, risk),
                         "brier": brier_score_loss(target, risk),
                         "brier_calibrated": brier_score_loss(target, calibrated),
                         **individual_score_ci(target, risk, groups),
                         "analysis_type": "PRESPECIFIED_SECONDARY", "analysis_plan_sha256": plan_sha256})
    for (model, seed), group in cnn.groupby(["model", "seed"]):
        ordered = group.sort_values("sample_index")
        target = ordered.target.to_numpy(dtype=int)
        risk = ordered.risk.to_numpy(dtype=float)
        rows.append({"model": model, "seed": seed,
                     "pr_auc": average_precision_score(target, risk), "roc_auc": roc_auc_score(target, risk),
                     "brier": brier_score_loss(target, risk), "brier_calibrated": np.nan,
                     **individual_score_ci(target, risk, groups, repeats=300, seed=int(seed)),
                     "analysis_type": "PRESPECIFIED_SECONDARY", "analysis_plan_sha256": plan_sha256})
    return pd.DataFrame(rows)
