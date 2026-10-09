"""Reproduce the headline Madagascar result from committed sample data.

No credentials, network access, saved model weights, or full satellite scenes are required.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "data" / "sample_input"
EXPECTED = ROOT / "reproducibility" / "expected_headline.json"
OUTPUT = ROOT / "reproducibility" / "output"

TOLERANCE = 0.002


def compare(rows: list, name: str, observed: float, expected: float) -> None:
    difference = abs(float(observed) - float(expected))
    rows.append(
        {
            "quantity": name,
            "reproduced": round(float(observed), 4),
            "frozen": round(float(expected), 4),
            "abs_difference": difference,
            "status": "PASS" if difference <= TOLERANCE else "FAIL",
        }
    )


def integrity() -> list:
    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "notebooks" / "02_main_analysis.ipynb",
        SAMPLE_DATA / "model_cohort.parquet",
        SAMPLE_DATA / "example_emit_clip.tif",
        EXPECTED,
    ]

    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]

    return [
        {
            "quantity": "required public reproduction files present",
            "reproduced": len(required) - len(missing),
            "frozen": len(required),
            "abs_difference": len(missing),
            "status": "PASS" if not missing else "FAIL",
        }
    ]


def level1() -> list:
    from src.evaluation import paired_block_bootstrap
    from src.models import train_spatial_oof
    from src.secondary import original_fold_splits

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["primary"]

    selected = pd.read_parquet(SAMPLE_DATA / "model_cohort.parquet")
    target = (selected.target == "Persistent Degradation").to_numpy(dtype=int)

    spatial = original_fold_splits(
        selected.assign(target_binary=target),
        buffer_m=60,
    )

    risks = {}
    rows = []

    with tempfile.TemporaryDirectory() as directory:
        for configuration in ("A", "C", "D"):
            predictions, _ = train_spatial_oof(
                selected,
                target,
                spatial,
                configuration,
                "lr",
                Path(directory),
                "public_reproduction",
            )

            risks[configuration] = (
                predictions
                .sort_values("sample_index")
                .risk
                .to_numpy()
            )

            print(f"Refitted Madagascar LR {configuration}", flush=True)

    for configuration in ("A", "C", "D"):
        observed = average_precision_score(target, risks[configuration])
        compare(
            rows,
            f"Madagascar LR {configuration} PR-AUC",
            observed,
            expected[f"lr_{configuration}_pr_auc"],
        )

    for first, second in (("C", "A"), ("C", "D")):
        boot = paired_block_bootstrap(
            target,
            risks[first],
            risks[second],
            spatial["groups"],
        )

        key = f"{first}_minus_{second}"

        compare(
            rows,
            f"Madagascar {first}-{second}",
            boot["pr_auc_difference"],
            expected[key],
        )
        compare(
            rows,
            f"Madagascar {first}-{second} CI low",
            boot["ci_low"],
            expected[f"{key}_ci_low"],
        )
        compare(
            rows,
            f"Madagascar {first}-{second} CI high",
            boot["ci_high"],
            expected[f"{key}_ci_high"],
        )

    return rows


def main() -> None:
    start = time.time()

    rows = integrity() + level1()
    table = pd.DataFrame(rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "headline_check.csv", index=False)

    with pd.option_context("display.width", 180, "display.max_rows", 100):
        print(table.to_string(index=False))

    passed = bool((table.status == "PASS").all())

    print(
        f"\nHEADLINE CHECK: {'PASS' if passed else 'FAIL'} "
        f"({(table.status == 'PASS').sum()}/{len(table)} checks, "
        f"{time.time() - start:.0f} s)"
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
