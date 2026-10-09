from datetime import datetime, timezone

import numpy as np
import pytest

from src.features import build_conventional_features, require_pre_t0


T0 = "2023-07-27T10:02:31+00:00"


def test_post_t0_predictor_date_rejected():
    with pytest.raises(ValueError, match="Post-t0"):
        require_pre_t0(["2023-07-28T06:00:00+00:00"], T0)


def test_t0_predictor_date_allowed():
    require_pre_t0([T0], T0)


def test_future_observation_does_not_affect_conventional_features():
    dates = ["2022-01-01T06:00:00+00:00", "2023-07-20T06:00:00+00:00", "2023-08-20T06:00:00+00:00"]
    observations = {name: np.array([[0.5], [0.6], [0.1]]) for name in ("ndvi", "ndmi", "mndwi", "water_fraction")}
    supports = {"native_id": np.array([1]), "mangrove_fraction": np.array([0.9]),
                "mangrove_fraction_grid": np.ones((3, 3)), "row": np.array([1]), "col": np.array([1])}
    first = build_conventional_features(dates, observations, supports, T0)
    observations["ndvi"][-1, 0] = 0.99
    second = build_conventional_features(dates, observations, supports, T0)
    assert first.equals(second)


def test_future_near_t0_band_date_rejected():
    dates = ["2023-07-20T06:00:00+00:00"]
    observations = {name: np.array([[0.5]]) for name in ("ndvi", "ndmi", "mndwi", "water_fraction")}
    supports = {"native_id": np.array([1]), "mangrove_fraction": np.array([0.9]),
                "mangrove_fraction_grid": np.ones((3, 3)), "row": np.array([1]), "col": np.array([1])}
    with pytest.raises(ValueError, match="Post-t0"):
        build_conventional_features(dates, observations, supports, T0,
                                    {"dates": ["2023-07-28T06:00:00+00:00"], "B02": np.array([0.1])})
