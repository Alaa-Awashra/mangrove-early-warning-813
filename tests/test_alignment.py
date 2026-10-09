import numpy as np
from affine import Affine

from src.sentinel import _support_windows


def test_sentinel_window_covers_hyperspectral_support():
    supports = {"lon": np.array([46.0003]), "lat": np.array([-15.0003]),
                "transform": Affine(0.0006, 0, 46, 0, -0.0006, -15)}
    sentinel_transform = Affine(0.0002, 0, 46, 0, -0.0002, -15)
    windows = _support_windows(supports, "EPSG:4326", sentinel_transform, (10, 10))
    assert windows["row_min"][0] == 0
    assert windows["row_max"][0] == 3
    assert windows["col_min"][0] == 0
    assert 3 <= windows["col_max"][0] <= 4
    assert len(windows["groups"]) == 1


def test_support_outside_sentinel_tile_is_not_aggregated():
    supports = {"lon": np.array([46.1]), "lat": np.array([-15.0]),
                "transform": Affine(0.0006, 0, 46, 0, -0.0006, -15)}
    windows = _support_windows(supports, "EPSG:4326", Affine(0.0002, 0, 46, 0, -0.0002, -15), (10, 10))
    assert windows["groups"] == {}
