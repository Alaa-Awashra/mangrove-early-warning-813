import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/feasibility/analysis_plan.md"


def frozen_plan_digest(plan: Path) -> str:
    """SHA-256 of the frozen plan: the file content before the first post-audit amendment."""
    content = plan.read_bytes()
    marker = "\n2026-10-08 UTC â€” Amendment 1".encode()
    return hashlib.sha256(content[:content.index(marker)] if marker in content else content).hexdigest()


def test_analysis_plan_precedes_any_research_model_output():
    outputs = list((ROOT / "results/models/model_metadata").glob("*.joblib"))
    if not outputs:
        pytest.skip("No research model has been fitted")
    assert PLAN.is_file()
    saved = (ROOT / "results/feasibility/analysis_plan_sha256.txt").read_text().strip()
    assert saved == frozen_plan_digest(PLAN)


def test_required_tier1_outputs_after_completed_run():
    if not (ROOT / "results/dataset/cohort.parquet").exists():
        pytest.skip("Tier 1 not completed")
    required = ["results/dataset/cohort.parquet", "results/dataset/hsi_bands.csv",
                "results/models/oof_predictions.parquet", "results/tables/primary_comparison.csv",
                "results/tables/early_warning.csv", "results/tables/carbon_targeting.csv",
                "results/maps/alerts.geojson"]
    assert all((ROOT / path).is_file() for path in required)
    rasters = ["results/maps/conventional_risk.tif", "results/maps/hsi_risk.tif", "results/maps/carbon_priority.tif"]
    if not all((ROOT / path).is_file() for path in rasters):
        pytest.skip("Map rasters are regenerable binaries, not committed (run run.py --stage maps)")
