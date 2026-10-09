from __future__ import annotations

import numpy as np
import pandas as pd


def early_warning_table(oof: pd.DataFrame, cohort: pd.DataFrame, t0: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_time = pd.Timestamp(t0)
    merged = oof.merge(cohort[["sample_index", "T_onset", "T_detect", "positive_patch_id", "subtype"]],
                       on="sample_index", validate="many_to_one")
    merged["lead_days"] = (pd.to_datetime(merged["T_detect"], utc=True) - prediction_time).dt.total_seconds() / 86400
    merged["onset_days"] = (pd.to_datetime(merged["T_onset"], utc=True) - prediction_time).dt.total_seconds() / 86400
    rows = []
    for model, group in merged.groupby("model"):
        for budget in (5, 10, 20):
            alerted = group[group[f"alert_{budget}"]]
            true = alerted[(alerted["target"] == 1) & (alerted["lead_days"] > 0)]
            rows.append({"model": model, "budget_percent": budget, "alerts": len(alerted),
                         "true_future_alerts": len(true), "false_alerts": len(alerted) - len(true),
                         "case_recall": len(true) / max(1, int(group["target"].sum())),
                         "positive_patches_alerted": true["positive_patch_id"].nunique(),
                         "median_lead_days": true["lead_days"].median(),
                         "lead_q25_days": true["lead_days"].quantile(0.25),
                         "lead_q75_days": true["lead_days"].quantile(0.75),
                         "median_onset_days": true["onset_days"].median()})
    baseline = merged[merged["model"] == "lr_A"].set_index("sample_index")
    enhanced = merged[merged["model"] == "lr_C"].set_index("sample_index")
    extra = []
    for budget in (5, 10, 20):
        cases = enhanced[(enhanced["target"] == 1) & (enhanced["lead_days"] > 0)]
        for sample_index, case in cases.iterrows():
            added = bool(case[f"alert_{budget}"] and not baseline.loc[sample_index, f"alert_{budget}"])
            baseline_only = bool(baseline.loc[sample_index, f"alert_{budget}"] and not case[f"alert_{budget}"])
            if added or baseline_only:
                extra.append({"sample_index": sample_index, "budget_percent": budget,
                              "type": "HSI_added" if added else "baseline_only",
                              "lead_days": case["lead_days"], "positive_patch_id": case["positive_patch_id"],
                              "subtype": case["subtype"]})
    columns = ("sample_index", "budget_percent", "type", "lead_days", "positive_patch_id", "subtype")
    return pd.DataFrame(rows), pd.DataFrame(extra, columns=columns)
