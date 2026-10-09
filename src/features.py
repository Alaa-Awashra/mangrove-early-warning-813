from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.ndimage import distance_transform_edt


def _time(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))


def require_pre_t0(dates: list[str | datetime], t0: str | datetime) -> None:
    prediction_time = _time(t0)
    if any(_time(date) > prediction_time for date in dates):
        raise ValueError("Post-t0 observation supplied to predictor builder")


def _safe_index(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (first - second) / (first + second)


def build_conventional_features(
    dates: list[str], observations: dict[str, np.ndarray], supports: dict, t0: str,
    near_t0_bands: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    prediction_time = _time(t0)
    selected = [index for index, date in enumerate(dates) if _time(date) <= prediction_time]
    pre_dates = [dates[index] for index in selected]
    require_pre_t0(pre_dates, t0)
    sample_count = len(supports["native_id"])
    result: dict[str, np.ndarray] = {}
    for name in ("ndvi", "ndmi", "mndwi", "water_fraction"):
        values = np.asarray(observations[name][selected], dtype=float)
        latest = np.full(sample_count, np.nan)
        fallback = np.zeros(sample_count, dtype=int)
        day_age = np.full(sample_count, np.nan)
        for sample_index in range(sample_count):
            for window, flag in ((30, 0), (60, 1)):
                eligible = [index for index, date in enumerate(pre_dates)
                            if 0 <= (prediction_time - _time(date)).days <= window and np.isfinite(values[index, sample_index])]
                if eligible:
                    latest[sample_index] = values[eligible[-1], sample_index]
                    fallback[sample_index] = flag
                    day_age[sample_index] = (prediction_time - _time(pre_dates[eligible[-1]])).total_seconds() / 86400
                    break
        result[f"f1_{name}"] = latest
        result[f"f1_{name}_fallback_60d"] = fallback.astype(float)
        result[f"f1_{name}_age_days"] = day_age
    history_start = prediction_time - relativedelta(months=24)
    history_indices = [index for index, date in enumerate(pre_dates) if _time(date) >= history_start]
    history_dates = [pre_dates[index] for index in history_indices]
    for name in ("ndvi", "ndmi"):
        values = np.asarray(observations[name][selected], dtype=float)[history_indices]
        count = np.isfinite(values).sum(axis=0)
        result[f"f2_{name}_valid_count"] = count.astype(float)
        result[f"f2_{name}_median"] = np.nanmedian(values, axis=0)
        result[f"f2_{name}_p10"] = np.nanpercentile(values, 10, axis=0)
        median = result[f"f2_{name}_median"]
        result[f"f2_{name}_mad"] = np.nanmedian(np.abs(values - median), axis=0)
        result[f"f2_{name}_amplitude"] = np.nanpercentile(values, 90, axis=0) - result[f"f2_{name}_p10"]
        years = np.array([(_time(date) - prediction_time).total_seconds() / (365.25 * 86400) for date in history_dates])
        slopes = np.full(sample_count, np.nan)
        for sample_index in range(sample_count):
            finite = np.isfinite(values[:, sample_index])
            if finite.sum() >= 3:
                slopes[sample_index] = np.polyfit(years[finite], values[finite, sample_index], 1)[0]
        result[f"f2_{name}_slope_year"] = slopes
    result["f3_mangrove_fraction"] = np.asarray(supports["mangrove_fraction"], dtype=float)
    grid = np.asarray(supports["mangrove_fraction_grid"]) >= 0.75
    edge_distance = distance_transform_edt(grid) * 60
    result["f3_mangrove_edge_distance_m"] = edge_distance[np.asarray(supports["row"]), np.asarray(supports["col"])]
    if near_t0_bands:
        require_pre_t0(near_t0_bands["dates"], t0)
        for band, values in near_t0_bands.items():
            if band != "dates":
                result[f"f1_{band.lower()}"] = np.asarray(values, dtype=float)
        bands = near_t0_bands
        for name, numerator, denominator in (("ndvi_bands", "B08", "B04"), ("ndmi_bands", "B08", "B11"),
                                             ("ndre", "B08", "B05"), ("mndwi_bands", "B03", "B11")):
            if numerator in bands and denominator in bands:
                result[f"f1_{name}"] = _safe_index(np.asarray(bands[numerator]), np.asarray(bands[denominator]))
        if "B07" in bands and "B05" in bands:
            with np.errstate(divide="ignore", invalid="ignore"):
                result["f1_cire"] = np.asarray(bands["B07"]) / np.asarray(bands["B05"]) - 1
    return pd.DataFrame(result)


def broadband_control_features(bands: dict[str, np.ndarray]) -> pd.DataFrame:
    output = {f"h2_{band.lower()}": np.asarray(values, dtype=float) for band, values in bands.items()}
    for name, first, second in (("ndvi", "B08", "B04"), ("ndmi", "B08", "B11"),
                                ("ndre", "B08", "B05"), ("mndwi", "B03", "B11")):
        output[f"h2_{name}"] = _safe_index(np.asarray(bands[first]), np.asarray(bands[second]))
    with np.errstate(divide="ignore", invalid="ignore"):
        output["h2_cire"] = np.asarray(bands["B07"]) / np.asarray(bands["B05"]) - 1
    return pd.DataFrame(output)
