import pandas as pd
import pytest

from run import PLAN, plan_hash
from src.models import feature_columns


def test_carbon_is_not_a_degradation_feature():
    frame = pd.DataFrame({"f1_ndvi": [0.5], "h1_000": [0.3], "carbon_stock_mg": [8.0]})
    assert "carbon_stock_mg" not in feature_columns(frame, "A")
    assert "carbon_stock_mg" not in feature_columns(frame, "C")


def test_missing_feature_family_fails():
    with pytest.raises(ValueError):
        feature_columns(pd.DataFrame({"carbon_stock_mg": [8.0]}), "A")


def test_analysis_plan_exists_before_modeling():
    assert PLAN.is_file()
    assert len(plan_hash()) == 64
