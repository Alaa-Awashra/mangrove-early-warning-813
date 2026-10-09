from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import binomtest

from .models import _inner_scores, _inner_splits


def reconcile(cohort: pd.DataFrame, plan_sha256: str) -> pd.DataFrame:
    excluded = cohort["already_detected_pre_t0"].to_numpy(dtype=bool)
    at_risk = cohort.loc[~excluded]
    counts = [
        ("original_supports", len(cohort)),
        ("already_detected_pre_t0", int(excluded.sum())),
        ("Persistent Degradation", int((at_risk.target == "Persistent Degradation").sum())),
        ("  canopy_dieback_like", int((at_risk.subtype == "canopy_dieback_like").sum())),
        ("  loss_to_water", int((at_risk.subtype == "loss_to_water").sum())),
        ("Transient Stress", int((at_risk.target == "Transient Stress").sum())),
        ("Stable", int((at_risk.target == "Stable").sum())),
        ("Unresolved", int((at_risk.target == "Unresolved").sum())),
        ("other_exclusion", int((~at_risk.target.isin(("Persistent Degradation", "Transient Stress", "Stable", "Unresolved"))).sum())),
        ("final_primary_modeled", int(at_risk.target.isin(("Persistent Degradation", "Stable")).sum())),
    ]
    table = pd.DataFrame(counts, columns=["cohort_step", "count"])
    table["analysis_type"] = "POSTHOC_SENSITIVITY"
    table["analysis_plan_sha256"] = plan_sha256
    assert counts[1][1] + counts[2][1] + counts[5][1] + counts[6][1] + counts[7][1] + counts[8][1] == counts[0][1]
    assert counts[3][1] + counts[4][1] == counts[2][1]
    assert counts[2][1] + counts[6][1] == counts[9][1]
    return table


def post_detection_followup(cohort: pd.DataFrame, labels: list[dict], dates: list[str],
                            observations: dict, t0: str, plan_sha256: str) -> pd.DataFrame:
    prediction_time = pd.Timestamp(t0)
    horizon = prediction_time + relativedelta(months=24)
    times = pd.to_datetime(dates, utc=True)
    rows = []
    positives = cohort[(~cohort.already_detected_pre_t0) & (cohort.target == "Persistent Degradation")]
    for row in positives.itertuples(index=False):
        position = int(row.sample_index)
        label = labels[position]
        detection = pd.Timestamp(row.T_detect)
        water = observations["water_fraction"][:, position]
        coverage = observations["coverage"][:, position]
        ndvi = observations["ndvi"][:, position]
        clear = (coverage >= 0.5) & np.isfinite(water)
        structural = (water >= 0.5) & ((water - label["water_baseline"]) >= 0.4)
        canopy = np.isfinite(ndvi) & (water <= label["water_anomaly_threshold"])
        valid = clear & (structural | canopy)
        after = (times > detection) & (times <= horizon)
        days = (horizon - detection).total_seconds() / 86400
        rows.append({"sample_id": row.sample_id, "native_support_id": row.native_support_id,
                     "patch_id": row.positive_patch_id, "subtype": row.subtype,
                     "T_onset": row.T_onset, "T_detect": row.T_detect,
                     "onset_days_from_t0": (pd.Timestamp(row.T_onset) - prediction_time).total_seconds() / 86400,
                     "detect_days_from_t0": (detection - prediction_time).total_seconds() / 86400,
                     "days_from_Tdetect_to_horizon_end": days,
                     "number_valid_observations_after_Tdetect": int((valid & after).sum()),
                     "analysis_type": "POSTHOC_SENSITIVITY", "analysis_plan_sha256": plan_sha256})
    result = pd.DataFrame(rows)
    assert len(result) == 172 and result.sample_id.is_unique
    return result


def audit_alerts(selected: pd.DataFrame, oof: pd.DataFrame, followup: pd.DataFrame,
                 metadata_dir: Path, plan_sha256: str, t0: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    checked = {}
    target = (selected.target == "Persistent Degradation").to_numpy(dtype=int)
    groups = selected.spatial_block.to_numpy()
    for model in ("lr_A", "lr_C"):
        predictions = oof[oof.model == model].set_index("sample_index").sort_index()
        assert len(predictions) == len(selected) and predictions.index.is_unique
        for fold in range(5):
            saved = joblib.load(metadata_dir / f"{model}_fold{fold}.joblib")
            train = np.asarray(saved["train_indices"], dtype=int)
            test = np.asarray(saved["test_indices"], dtype=int)
            assert not np.intersect1d(train, test).size
            assert np.array_equal(np.sort(test), np.flatnonzero(selected.spatial_fold.to_numpy() == fold))
            assert saved["analysis_plan_sha256"] == plan_sha256
            training = selected.iloc[train].reset_index(drop=True)
            inner = _inner_splits(target[train], groups[train], 813 + fold)
            scores = _inner_scores(training, target[train], saved["features"], "lr", inner,
                                   saved["regularization"])
            for budget in (5, 10, 20):
                threshold = float(np.quantile(scores, 1 - budget / 100))
                assert np.isclose(threshold, saved["inner_thresholds"][budget / 100], atol=1e-10)
                assert np.allclose(predictions.loc[test, f"threshold_{budget}"], threshold, atol=1e-10)
                assert np.array_equal(predictions.loc[test, f"alert_{budget}"].to_numpy(dtype=bool),
                                      predictions.loc[test, "risk"].to_numpy() >= threshold)
            assert np.allclose(predictions.loc[test, "risk"],
                               saved["model"].predict_proba(selected.iloc[test][saved["features"]])[:, 1], atol=1e-10)
        checked[model] = predictions
    assert selected.native_support_id.is_unique
    positives = selected[selected.target == "Persistent Degradation"].copy()
    assert not positives.already_detected_pre_t0.any()
    assert positives.positive_patch_id.notna().all()
    followup_index = followup.set_index("sample_id")
    rows = []
    for budget in (5, 10, 20):
        for row in positives.itertuples(index=False):
            index = row.sample_index
            first = checked["lr_A"].loc[index]
            second = checked["lr_C"].loc[index]
            assert first.fold == second.fold == row.spatial_fold
            detection = pd.Timestamp(row.T_detect)
            assert detection > pd.Timestamp(t0)
            follow = followup_index.loc[row.sample_id]
            rows.append({"sample_id": row.sample_id, "patch_id": int(row.positive_patch_id),
                         "subtype": row.subtype, "fold": int(first.fold), "budget_percent": budget,
                         "risk_A": float(first.risk), "risk_C": float(second.risk),
                         "threshold_A": float(first[f"threshold_{budget}"]),
                         "threshold_C": float(second[f"threshold_{budget}"]),
                         "alert_A": bool(first[f"alert_{budget}"]), "alert_C": bool(second[f"alert_{budget}"]),
                         "T_onset": row.T_onset, "T_detect": row.T_detect,
                         "lead_time_days": float(follow.detect_days_from_t0),
                         "days_after_Tdetect": float(follow.days_from_Tdetect_to_horizon_end),
                         "valid_obs_after_Tdetect": int(follow.number_valid_observations_after_Tdetect),
                         "carbon_stock": row.carbon_stock_MgC if "carbon_stock_MgC" in selected else np.nan,
                         "analysis_type": "PRESPECIFIED_SECONDARY", "analysis_plan_sha256": plan_sha256})
    audit = pd.DataFrame(rows)
    paired = []
    for budget, group in audit.groupby("budget_percent"):
        both = int((group.alert_A & group.alert_C).sum())
        c_only = int((~group.alert_A & group.alert_C).sum())
        a_only = int((group.alert_A & ~group.alert_C).sum())
        neither = int((~group.alert_A & ~group.alert_C).sum())
        p_value = binomtest(min(c_only, a_only), c_only + a_only, 0.5).pvalue if c_only + a_only else 1.0
        paired.append({"budget_percent": budget, "C_only": c_only, "A_only": a_only,
                       "both": both, "neither": neither, "exact_mcnemar_p": p_value,
                       "analysis_type": "PRESPECIFIED_SECONDARY", "analysis_plan_sha256": plan_sha256})
    return audit, pd.DataFrame(paired)


def verify_detection_dates(cohort: pd.DataFrame, labels: list[dict], dates: list[str],
                           observations: dict, t0: str, plan_sha256: str) -> pd.DataFrame:
    prediction = pd.Timestamp(t0)
    reference_start = prediction - relativedelta(months=24)
    reference_end = prediction - relativedelta(months=6)
    horizon = prediction + relativedelta(months=24)
    times = pd.to_datetime(dates, utc=True)
    rows = []
    positives = cohort[(~cohort.already_detected_pre_t0) & (cohort.target == "Persistent Degradation")]
    for row in positives.itertuples(index=False):
        position = int(row.sample_index)
        label = labels[position]
        ndvi = observations["ndvi"][:, position]
        water = observations["water_fraction"][:, position]
        coverage = observations["coverage"][:, position]
        clear = (coverage >= .5) & np.isfinite(water)
        water_anomaly = water > label["water_anomaly_threshold"]
        reference = (times >= reference_start) & (times <= reference_end) & clear & np.isfinite(ndvi) & ~water_anomaly
        references = np.flatnonzero(reference)
        future = np.flatnonzero((times > prediction) & (times <= horizon))
        valid = np.zeros(len(future), dtype=bool)
        degraded = np.zeros(len(future), dtype=bool)
        for future_position, index in enumerate(future):
            if not clear[index]:
                continue
            structural = bool(water[index] >= .5 and water[index] - label["water_baseline"] >= .4)
            canopy = bool(np.isfinite(ndvi[index]) and not water_anomaly[index])
            valid[future_position] = canopy or structural
            if not valid[future_position]:
                continue
            if canopy:
                year_day = times[index].dayofyear
                seasonal = [reference_index for reference_index in references
                            if min(abs(times[reference_index].dayofyear - year_day),
                                   365 - abs(times[reference_index].dayofyear - year_day)) <= 45]
                if len(seasonal) < 3:
                    seasonal = references
                baseline = ndvi[seasonal]
                median = float(np.median(baseline))
                mad = float(np.median(np.abs(baseline - median)))
                degraded[future_position] = ndvi[index] - median <= -max(.10, 2.5 * 1.4826 * mad)
            degraded[future_position] |= structural
        actual_onset = pd.Timestamp(row.T_onset)
        actual_detection = pd.Timestamp(row.T_detect)
        onset_matches = np.flatnonzero(times[future] == actual_onset)
        detect_matches = np.flatnonzero(times[future] == actual_detection)
        onset_valid_degraded = bool(len(onset_matches) and valid[onset_matches[0]] and degraded[onset_matches[0]])
        detect_valid_degraded = bool(len(detect_matches) and valid[detect_matches[0]] and degraded[detect_matches[0]])
        first_real_time_detection = None
        for confirmation in np.flatnonzero(valid & degraded):
            candidates = [onset for onset in np.flatnonzero(valid & degraded & (np.arange(len(valid)) < confirmation))
                          if (times[future[confirmation]] - times[future[onset]]).days >= 30]
            if any(degraded[np.flatnonzero(valid & (np.arange(len(valid)) >= onset) &
                                             (np.arange(len(valid)) <= confirmation))].mean() >= .60
                   for onset in candidates):
                first_real_time_detection = times[future[confirmation]]
                break
        first_iso = first_real_time_detection.isoformat() if first_real_time_detection is not None else None
        rows.append({"sample_id": row.sample_id, "patch_id": row.positive_patch_id,
                     "stored_T_onset": row.T_onset, "stored_T_detect": row.T_detect,
                     "first_real_time_T_detect": first_iso,
                     "onset_is_valid_degraded": onset_valid_degraded,
                     "detect_is_valid_degraded": detect_valid_degraded,
                     "detect_matches_first_real_time": bool(first_real_time_detection == actual_detection),
                     "analysis_type": "POSTHOC_SENSITIVITY", "analysis_plan_sha256": plan_sha256})
    return pd.DataFrame(rows)
