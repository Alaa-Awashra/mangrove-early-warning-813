from __future__ import annotations

from datetime import datetime

import numpy as np
from dateutil.relativedelta import relativedelta


def _as_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _mad(values: np.ndarray) -> float:
    center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def _day_distance(first: datetime, second: datetime) -> int:
    difference = abs(first.timetuple().tm_yday - second.timetuple().tm_yday)
    return min(difference, 365 - difference)


def _persistent_episode(dates: list[datetime], valid: np.ndarray, degraded: np.ndarray) -> dict | None:
    degraded_indices = np.flatnonzero(valid & degraded)
    for onset_index in degraded_indices:
        later_valid = np.flatnonzero(valid & (np.arange(len(valid)) >= onset_index))
        if len(later_valid) < 2 or degraded[later_valid].mean() < 0.60:
            continue
        confirmations = [index for index in degraded_indices if index > onset_index and (dates[index] - dates[onset_index]).days >= 30]
        if not confirmations:
            continue
        detect_index = None
        for index in confirmations:
            observed = later_valid[later_valid <= index]
            if degraded[observed].mean() >= 0.60:
                detect_index = index
                break
        if detect_index is None:
            continue
        prior_ok = np.flatnonzero(valid & ~degraded & (np.arange(len(valid)) < onset_index))
        return {
            "onset_index": int(onset_index),
            "detect_index": int(detect_index),
            "last_ok_index": int(prior_ok[-1]) if len(prior_ok) else None,
        }
    return None


def _detected_by_now(dates: list[datetime], valid: np.ndarray, degraded: np.ndarray) -> bool:
    degraded_indices = np.flatnonzero(valid & degraded)
    for onset_index in degraded_indices:
        for confirmation_index in degraded_indices:
            if confirmation_index <= onset_index or (dates[confirmation_index] - dates[onset_index]).days < 30:
                continue
            observed = np.flatnonzero(valid & (np.arange(len(valid)) >= onset_index)
                                          & (np.arange(len(valid)) <= confirmation_index))
            if len(observed) >= 2 and degraded[observed].mean() >= 0.60:
                return True
    return False


def _first_realtime_detection(dates: list[datetime], valid: np.ndarray, degraded: np.ndarray) -> int | None:
    # Amendment 1: earliest confirmation over any onset, using only observations available by that date.
    degraded_indices = np.flatnonzero(valid & degraded)
    for confirmation_index in degraded_indices:
        for onset_index in degraded_indices:
            if onset_index >= confirmation_index or (dates[confirmation_index] - dates[onset_index]).days < 30:
                continue
            observed = np.flatnonzero(valid & (np.arange(len(valid)) >= onset_index)
                                      & (np.arange(len(valid)) <= confirmation_index))
            if len(observed) >= 2 and degraded[observed].mean() >= 0.60:
                return int(confirmation_index)
    return None


def classify_trajectory(
    dates: list[str | datetime],
    ndvi: np.ndarray,
    water_fraction: np.ndarray,
    coverage: np.ndarray,
    t0: str | datetime,
    water_floor: float = 0.25,
    mndwi: np.ndarray | None = None,
) -> dict:
    observation_dates = [_as_utc(date) for date in dates]
    t0 = _as_utc(t0)
    ndvi = np.asarray(ndvi, dtype=float)
    water_fraction = np.asarray(water_fraction, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    mndwi = np.full(len(ndvi), np.nan) if mndwi is None else np.asarray(mndwi, dtype=float)
    if len(observation_dates) != len(ndvi) or len(ndvi) != len(water_fraction) or len(ndvi) != len(coverage) or len(ndvi) != len(mndwi):
        raise ValueError("Trajectory arrays must have equal length")
    if any(left >= right for left, right in zip(observation_dates, observation_dates[1:])):
        raise ValueError("Observation dates must be strictly increasing")
    reference_start = t0 - relativedelta(months=24)
    reference_end = t0 - relativedelta(months=6)
    monitoring_start = reference_end
    horizon_end = t0 + relativedelta(months=24)
    clear = (coverage >= 0.5) & np.isfinite(water_fraction)
    pre_water = np.array([(date >= reference_start and date <= t0) for date in observation_dates]) & clear
    result = {
        "target": "Unresolved",
        "subtype": None,
        "T_onset": None,
        "T_detect": None,
        "T_detect_realtime": None,
        "T_last_ok": None,
        "already_detected_pre_t0": False,
        "reference_count": 0,
        "future_valid_count": 0,
        "seasonal_fallback_count": 0,
        "water_baseline": np.nan,
        "water_anomaly_threshold": np.nan,
        "pre_t0_mndwi_median": np.nan,
        "pre_t0_mndwi_mad": np.nan,
    }
    if pre_water.sum() < 6:
        return result
    water_baseline = float(np.median(water_fraction[pre_water]))
    water_mad = _mad(water_fraction[pre_water])
    anomaly_threshold = water_baseline + max(water_floor, 3 * 1.4826 * water_mad)
    water_anomaly = clear & (water_fraction > anomaly_threshold)
    result["water_baseline"] = water_baseline
    result["water_anomaly_threshold"] = anomaly_threshold
    available_mndwi = mndwi[pre_water & np.isfinite(mndwi)]
    if len(available_mndwi):
        result["pre_t0_mndwi_median"] = float(np.median(available_mndwi))
        result["pre_t0_mndwi_mad"] = _mad(available_mndwi)
    reference = np.array([(reference_start <= date <= reference_end) for date in observation_dates])
    reference_valid = reference & clear & np.isfinite(ndvi) & ~water_anomaly
    reference_indices = np.flatnonzero(reference_valid)
    result["reference_count"] = int(len(reference_indices))
    if len(reference_indices) < 6 or (observation_dates[reference_indices[-1]] - observation_dates[reference_indices[0]]).days < 274:
        return result
    monitoring = np.array([(monitoring_start < date <= t0) for date in observation_dates])
    future = np.array([(t0 < date <= horizon_end) for date in observation_dates])
    evaluate = monitoring | future
    valid = np.zeros(len(observation_dates), dtype=bool)
    degraded = np.zeros(len(observation_dates), dtype=bool)
    structural = np.zeros(len(observation_dates), dtype=bool)
    for index in np.flatnonzero(evaluate):
        if not clear[index]:
            continue
        structural[index] = water_fraction[index] >= 0.50 and water_fraction[index] - water_baseline >= 0.40
        canopy_valid = np.isfinite(ndvi[index]) and not water_anomaly[index]
        valid[index] = canopy_valid or structural[index]
        if not valid[index]:
            continue
        if canopy_valid:
            seasonal = [reference_index for reference_index in reference_indices if _day_distance(observation_dates[reference_index], observation_dates[index]) <= 45]
            if len(seasonal) < 3:
                seasonal = reference_indices
                result["seasonal_fallback_count"] += 1
            values = ndvi[seasonal]
            threshold = max(0.10, 2.5 * 1.4826 * _mad(values))
            degraded[index] = ndvi[index] - np.median(values) <= -threshold
        degraded[index] = degraded[index] or structural[index]
    monitoring_indices = np.flatnonzero(monitoring)
    monitoring_detected = _detected_by_now(
        [observation_dates[index] for index in monitoring_indices],
        valid[monitoring_indices],
        degraded[monitoring_indices],
    )
    result["already_detected_pre_t0"] = monitoring_detected
    future_indices = np.flatnonzero(future)
    future_dates = [observation_dates[index] for index in future_indices]
    future_valid = valid[future_indices]
    future_degraded = degraded[future_indices]
    result["future_valid_count"] = int(future_valid.sum())
    realtime_index = _first_realtime_detection(future_dates, future_valid, future_degraded)
    if realtime_index is not None:
        result["T_detect_realtime"] = future_dates[realtime_index].isoformat()
    episode = _persistent_episode(future_dates, future_valid, future_degraded)
    if episode is not None:
        result["target"] = "Persistent Degradation"
        result["T_onset"] = future_dates[episode["onset_index"]].isoformat()
        result["T_detect"] = future_dates[episode["detect_index"]].isoformat()
        if episode["last_ok_index"] is not None:
            result["T_last_ok"] = future_dates[episode["last_ok_index"]].isoformat()
        else:
            earlier_ok = np.flatnonzero(monitoring & valid & ~degraded)
            if len(earlier_ok):
                result["T_last_ok"] = observation_dates[earlier_ok[-1]].isoformat()
        first = episode["onset_index"]
        structural_after_onset = np.flatnonzero(structural[future_indices] & (np.arange(len(future_indices)) >= first))
        structural_confirmed = any((future_dates[index] - future_dates[structural_after_onset[0]]).days >= 30 for index in structural_after_onset[1:]) if len(structural_after_onset) >= 2 else False
        result["subtype"] = "loss_to_water" if structural_confirmed else "canopy_dieback_like"
        return result
    valid_indices = np.flatnonzero(future_valid)
    if len(valid_indices) < 6 or (future_dates[valid_indices[-1]] - future_dates[valid_indices[0]]).days < 274:
        return result
    result["target"] = "Stable" if future_degraded[future_valid].sum() <= 1 else "Transient Stress"
    return result
