import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit import reconcile
from src.carbon import spatial_capture_bootstrap


ROOT = Path(__file__).resolve().parents[1]


def frozen_plan_digest(plan: Path) -> str:
    """SHA-256 of the frozen plan: the file content before the first post-audit amendment."""
    content = plan.read_bytes()
    marker = "\n2026-10-08 UTC — Amendment 1".encode()
    return hashlib.sha256(content[:content.index(marker)] if marker in content else content).hexdigest()


def test_cohort_reconciliation_excludes_pre_t0_once():
    cohort = pd.DataFrame({"already_detected_pre_t0": [True, False, False, False, False, False],
                           "target": ["Persistent Degradation", "Persistent Degradation",
                                      "Persistent Degradation", "Transient Stress", "Stable", "Unresolved"],
                           "subtype": ["canopy_dieback_like", "canopy_dieback_like", "loss_to_water",
                                       None, None, None]})
    result = reconcile(cohort, "test").set_index("cohort_step")["count"]
    assert result["original_supports"] == 6
    assert result["already_detected_pre_t0"] == 1
    assert result["Persistent Degradation"] == 2
    assert result["final_primary_modeled"] == 3


def test_saved_paired_warning_audit_reconciles_positive_cases():
    path = ROOT / "results/tables/hsi_added_warning_audit.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for budget, group in frame.groupby("budget_percent"):
        assert budget in (5, 10, 20)
        assert len(group) == 172
        assert group.sample_id.is_unique
        assert (group.alert_A == (group.risk_A >= group.threshold_A)).all()
        assert (group.alert_C == (group.risk_C >= group.threshold_C)).all()
        assert (pd.to_datetime(group.T_detect, utc=True) > pd.Timestamp("2023-12-27T09:28:43Z")).all()
    assert set(frame.analysis_type) == {"PRESPECIFIED_SECONDARY"}


def test_saved_followup_guard_is_conservative():
    path = ROOT / "results/tables/post_detection_followup_diagnostic.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    assert len(frame) == 172
    assert frame.sample_id.is_unique
    assert (frame.days_from_Tdetect_to_horizon_end >= 0).all()
    assert (frame.number_valid_observations_after_Tdetect >= 0).all()
    assert ((frame.days_from_Tdetect_to_horizon_end < 90) |
            (frame.number_valid_observations_after_Tdetect < 3)).sum() == 81


def test_independent_detection_audit_dates_are_real_and_no_later_than_stored():
    path = ROOT / "results/tables/label_detection_audit.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    assert len(frame) == 172
    assert frame.onset_is_valid_degraded.all()
    assert frame.detect_is_valid_degraded.all()
    assert (pd.to_datetime(frame.first_real_time_T_detect, utc=True) <=
            pd.to_datetime(frame.stored_T_detect, utc=True)).all()


def test_spatial_carbon_bootstrap_is_deterministic_and_paired():
    target = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    carbon = np.array([5, 2, 4, 3, 5, 2, 4, 3], dtype=float)
    groups = np.repeat(np.arange(4), 2)
    first = spatial_capture_bootstrap(target, carbon, np.linspace(.8, .1, 8),
                                      np.linspace(.1, .8, 8), groups, repeats=80)
    second = spatial_capture_bootstrap(target, carbon, np.linspace(.8, .1, 8),
                                       np.linspace(.1, .8, 8), groups, repeats=80)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 15
    assert (first.ci_low <= first.ci_high).all()


def test_new_result_metadata_uses_frozen_plan_hash():
    plan = ROOT / "results/feasibility/analysis_plan.md"
    digest = frozen_plan_digest(plan)
    for name in ("cohort_reconciliation.csv", "hsi_added_warning_audit.csv",
                 "dieback_only_paired.csv", "rf_inference.csv", "carbon_targeting_uncertainty.csv"):
        path = ROOT / "results/tables" / name
        if path.exists():
            result = pd.read_csv(path)
            assert set(result.analysis_plan_sha256) == {digest}
            assert result.analysis_type.nunique() == 1
