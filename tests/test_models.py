import numpy as np
import pandas as pd

from src.models import train_spatial_oof


def synthetic_spatial_problem():
    generator = np.random.default_rng(813)
    groups = np.repeat(np.arange(6), 12)
    target = np.tile(np.array([1, 1] + [0] * 10), 6)
    frame = pd.DataFrame({"f1_ndvi": generator.normal(size=len(target)),
                          "f2_ndvi_slope": generator.normal(size=len(target)),
                          "f3_mangrove_fraction": generator.uniform(0.75, 1, size=len(target)),
                          "h1_000": generator.normal(size=len(target)),
                          "h1_001": generator.normal(size=len(target)),
                          "carbon_stock_mg": np.full(len(target), 100)})
    splits = []
    folds = np.full(len(target), -1)
    for fold in range(3):
        test = np.flatnonzero(np.isin(groups, (2 * fold, 2 * fold + 1)))
        train = np.flatnonzero(~np.isin(groups, (2 * fold, 2 * fold + 1)))
        splits.append((train, test))
        folds[test] = fold
    return frame, target, {"groups": groups, "folds": folds, "splits": splits}


def test_outer_oof_covers_every_sample_once_and_records_plan_hash(tmp_path):
    frame, target, spatial = synthetic_spatial_problem()
    oof, _ = train_spatial_oof(frame, target, spatial, "A", "lr", tmp_path, "a" * 64)
    assert len(oof) == len(frame)
    assert set(oof["sample_index"]) == set(range(len(frame)))
    assert set(oof["analysis_plan_sha256"]) == {"a" * 64}
    assert (oof["threshold_10"] > 0).all()


def test_outer_test_features_cannot_change_inner_alert_threshold(tmp_path):
    frame, target, spatial = synthetic_spatial_problem()
    original, _ = train_spatial_oof(frame, target, spatial, "A", "lr", tmp_path / "first", "a" * 64)
    changed = frame.copy()
    changed.loc[spatial["splits"][0][1], "f1_ndvi"] = 1000
    modified, _ = train_spatial_oof(changed, target, spatial, "A", "lr", tmp_path / "second", "a" * 64)
    first = original[original["fold"] == 0]
    second = modified[modified["fold"] == 0]
    assert first["threshold_10"].iloc[0] == second["threshold_10"].iloc[0]
