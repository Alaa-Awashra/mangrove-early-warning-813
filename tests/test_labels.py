from datetime import datetime, timezone

import numpy as np
from dateutil.relativedelta import relativedelta

from src.labels import classify_trajectory


T0 = datetime(2023, 7, 27, tzinfo=timezone.utc)


def trajectory(future_ndvi, future_water=None, future_coverage=None):
    reference = [T0 - relativedelta(months=24) + relativedelta(months=2 * index) for index in range(10)]
    monitoring = [T0 - relativedelta(months=5) + relativedelta(months=index) for index in range(6)]
    future = [T0 + relativedelta(months=index) for index in range(1, len(future_ndvi) + 1)]
    dates = reference + monitoring + future
    ndvi = np.asarray([0.7] * 16 + future_ndvi, dtype=float)
    water = np.asarray([0.05] * 16 + (future_water or [0.05] * len(future)), dtype=float)
    coverage = np.asarray([1] * 16 + (future_coverage or [1] * len(future)), dtype=float)
    return dates, ndvi, water, coverage


def classify(future_ndvi, future_water=None, future_coverage=None):
    return classify_trajectory(*trajectory(future_ndvi, future_water, future_coverage), T0)


def test_stable_requires_no_more_than_one_drop():
    assert classify([0.7] * 12)["target"] == "Stable"


def test_persistent_detects_at_confirmation_not_onset():
    result = classify([0.3] * 12)
    assert result["target"] == "Persistent Degradation"
    assert result["T_onset"] != result["T_detect"]
    assert result["T_detect"].startswith("2023-09-27")


def test_recovery_is_transient():
    assert classify([0.3, 0.3] + [0.7] * 10)["target"] == "Transient Stress"


def test_insufficient_future_is_unresolved():
    assert classify([0.7] * 3)["target"] == "Unresolved"


def test_persistent_loss_to_water_subtype():
    result = classify([np.nan] * 12, [0.8] * 12)
    assert result["target"] == "Persistent Degradation"
    assert result["subtype"] == "loss_to_water"


def test_cloudy_future_does_not_become_stable():
    result = classify([0.7] * 12, future_coverage=[0] * 10 + [1, 1])
    assert result["target"] == "Unresolved"


def test_pre_t0_reference_requirement():
    dates, ndvi, water, coverage = trajectory([0.7] * 12)
    coverage[:10] = 0
    assert classify_trajectory(dates, ndvi, water, coverage, T0)["target"] == "Unresolved"


def test_pre_t0_persistent_detection_flagged():
    dates, ndvi, water, coverage = trajectory([0.7] * 12)
    ndvi[10:16] = 0.3
    result = classify_trajectory(dates, ndvi, water, coverage, T0)
    assert result["already_detected_pre_t0"] is True


def test_prior_realtime_detection_stays_excluded_after_pret0_recovery():
    dates, ndvi, water, coverage = trajectory([0.7] * 12)
    ndvi[[10, 12]] = 0.3
    result = classify_trajectory(dates, ndvi, water, coverage, T0)
    assert result["already_detected_pre_t0"] is True


def test_t_last_ok_uses_pre_t0_monitoring_when_onset_is_first_future_date():
    result = classify([0.3] * 12)
    assert result["T_last_ok"] is not None
    assert datetime.fromisoformat(result["T_last_ok"]) <= T0


def test_anomalous_tidal_water_does_not_create_canopy_degradation():
    result = classify([0.3] * 12, future_water=[0.35] * 12)
    assert result["target"] == "Unresolved"


def test_detection_waits_until_realtime_persistence_fraction_is_met():
    result = classify([0.3, 0.7, 0.7, 0.3] + [0.3] * 8)
    assert result["target"] == "Persistent Degradation"
    assert result["T_detect"].startswith("2023-12-27")


def test_realtime_detection_precedes_episode_detection():
    result = classify([0.3, 0.3] + [0.7] * 6 + [0.3] * 6)
    assert result["target"] == "Persistent Degradation"
    assert result["T_detect_realtime"].startswith("2023-09-27")
    assert result["T_detect"].startswith("2024-05-27")
    assert classify([0.3] * 12)["T_detect_realtime"] == classify([0.3] * 12)["T_detect"]
