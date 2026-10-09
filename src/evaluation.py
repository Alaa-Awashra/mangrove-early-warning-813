from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, roc_curve)
from sklearn.calibration import calibration_curve


def prediction_metrics(target: np.ndarray, risk: np.ndarray, alert: np.ndarray | None = None) -> dict:
    target = np.asarray(target, dtype=int)
    risk = np.asarray(risk, dtype=float)
    result = {"n": len(target), "positives": int(target.sum()), "prevalence": float(target.mean()),
              "pr_auc": float(average_precision_score(target, risk)) if target.sum() else np.nan,
              "roc_auc": float(roc_auc_score(target, risk)) if len(np.unique(target)) == 2 else np.nan,
              "brier": float(brier_score_loss(target, risk))}
    if alert is not None:
        alert = np.asarray(alert, dtype=bool)
        matrix = confusion_matrix(target, alert, labels=[0, 1])
        result.update({"balanced_accuracy": float(balanced_accuracy_score(target, alert)),
                       "sensitivity": float(recall_score(target, alert, zero_division=0)),
                       "specificity": float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else np.nan,
                       "precision": float(precision_score(target, alert, zero_division=0)),
                       "f1": float(f1_score(target, alert, zero_division=0)),
                       "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
                       "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])})
    return result


def paired_block_bootstrap(target: np.ndarray, first: np.ndarray, second: np.ndarray,
                           groups: np.ndarray, repeats: int = 1000, seed: int = 813) -> dict:
    target = np.asarray(target, dtype=int)
    difference = float(average_precision_score(target, first) - average_precision_score(target, second))
    unique_groups = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        sampled_groups = generator.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled_groups])
        if target[indices].sum() == 0 or target[indices].sum() == len(indices):
            continue
        deltas.append(average_precision_score(target[indices], first[indices])
                      - average_precision_score(target[indices], second[indices]))
    return {"pr_auc_difference": difference, "ci_low": float(np.quantile(deltas, 0.025)) if deltas else np.nan,
            "ci_high": float(np.quantile(deltas, 0.975)) if deltas else np.nan,
            "valid_bootstraps": len(deltas), "requested_bootstraps": repeats}


def metrics_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, data in oof.groupby("model"):
        for scope, subset in [("pooled", data)] + [(f"fold_{fold}", group) for fold, group in data.groupby("fold")]:
            for budget in (5, 10, 20):
                metric = prediction_metrics(subset["target"].to_numpy(), subset["risk"].to_numpy(),
                                            subset[f"alert_{budget}"].to_numpy())
                if "calibrated_risk" in subset:
                    metric["brier_calibrated"] = float(brier_score_loss(subset["target"], subset["calibrated_risk"]))
                rows.append({"model": model, "scope": scope, "budget_percent": budget, **metric})
    return pd.DataFrame(rows)


def save_core_figures(oof: pd.DataFrame, destination) -> None:
    from pathlib import Path

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    selected = [name for name in ("lr_A", "lr_C", "lr_D", "lr_E", "rf_A", "rf_C") if name in set(oof["model"])]
    for kind in ("pr", "roc"):
        figure, axis = plt.subplots(figsize=(7, 6))
        for name in selected:
            group = oof[oof["model"] == name]
            target = group["target"].to_numpy()
            risk = group["risk"].to_numpy()
            if kind == "pr":
                precision, recall, _ = precision_recall_curve(target, risk)
                axis.plot(recall, precision, label=f"{name} ({average_precision_score(target, risk):.3f})")
                axis.set(xlabel="Recall", ylabel="Precision", title="Spatial OOF precision-recall")
            else:
                false_rate, true_rate, _ = roc_curve(target, risk)
                axis.plot(false_rate, true_rate, label=f"{name} ({roc_auc_score(target, risk):.3f})")
                axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Spatial OOF ROC")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(destination / ("pr_curves.png" if kind == "pr" else "roc_curves.png"), dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 6))
    for name in ("lr_A", "lr_C"):
        group = oof[oof["model"] == name]
        observed, predicted = calibration_curve(group["target"], group["calibrated_risk"], n_bins=8,
                                                strategy="quantile")
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="Mean calibrated predicted risk", ylabel="Observed event fraction",
             title="Outer OOF calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "calibration.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, name in zip(axes, ("lr_A", "lr_C")):
        group = oof[oof["model"] == name]
        matrix = confusion_matrix(group["target"], group["alert_10"], labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(xlabel="Alert", ylabel="Future label", title=f"{name}: 10% budget")
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def save_label_examples(dates: list[str], observations: dict, cohort: pd.DataFrame,
                        t0: str, destination) -> None:
    from pathlib import Path

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    positives = cohort[cohort["target"] == "Persistent Degradation"].head(3)
    stable = cohort[cohort["target"] == "Stable"].head(2)
    examples = pd.concat((positives, stable), ignore_index=False)
    if examples.empty:
        return
    time = pd.to_datetime(dates, utc=True)
    figure, axes = plt.subplots(len(examples), 1, figsize=(10, 2.4 * len(examples)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (index, row) in zip(axes, examples.iterrows()):
        axis.plot(time, observations["ndvi"][:, index], marker=".", markersize=3, linewidth=0.7)
        axis.axvline(pd.Timestamp(t0), color="black", linestyle="--", linewidth=1)
        if pd.notna(row["T_onset"]):
            axis.axvline(pd.Timestamp(row["T_onset"]), color="orange", linewidth=1)
        if pd.notna(row["T_detect"]):
            axis.axvline(pd.Timestamp(row["T_detect"]), color="red", linewidth=1)
        axis.set(ylabel="NDVI", title=f"{row['target']}: native support {row['native_support_id']}")
    axes[-1].set_xlabel("Acquisition date; dashed=t0, orange=onset, red=detection")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_lead_times(incremental: pd.DataFrame, destination) -> None:
    from pathlib import Path

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    for kind in ("HSI_added", "baseline_only"):
        subset = incremental[(incremental["budget_percent"] == 10) & (incremental["type"] == kind)]
        if not subset.empty:
            axis.hist(subset["lead_days"], bins=min(10, max(1, len(subset))), alpha=0.6, label=kind)
    axis.set(xlabel="T_detect - t0 (days)", ylabel="True future alerts", title="Incremental 10% alerts")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
