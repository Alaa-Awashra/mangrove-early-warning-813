import numpy as np
import pytest
from affine import Affine

from src.carbon import support_area_hectares, targeting_table


def test_targeting_uses_future_case_carbon_in_numerator():
    target = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    carbon = np.array([10, 100, 20, 1, 1, 1, 1, 1, 1, 1])
    conventional = np.array([0.1] * 10)
    enhanced = np.array([0.9, 0.01, 0.8] + [0.1] * 7)
    table = targeting_table(target, carbon, conventional, enhanced, quantitative=True, random_draws=20)
    assert table[(table.budget_percent == 20) & (table.ranking == "HSI_risk_x_carbon")].iloc[0]["capture"] == 1.0


def test_targeting_rejects_missing_carbon():
    with pytest.raises(ValueError):
        targeting_table(np.array([1, 0]), np.array([np.nan, 1]), np.array([0.5, 0.5]),
                        np.array([0.5, 0.5]), quantitative=False)


def test_native_support_area_uses_latitude_and_hectares():
    supports = {"transform": Affine(0.00054, 0, 0, 0, -0.00054, 1),
                "lat": np.array([0.0, 30.0])}
    area = support_area_hectares(supports)
    assert 0.35 < area[0] < 0.37
    assert area[1] < area[0]
