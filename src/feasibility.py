from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected_components

from .labels import classify_trajectory


OBSERVATION_NAMES = ("ndvi", "ndmi", "mndwi", "water_fraction", "coverage")


def assemble_observations(items: list[dict], observation_dir: Path, support_count: int) -> tuple[list[str], dict[str, np.ndarray]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        path = observation_dir / f"{item['id']}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Sentinel observation cache: {path}")
        with np.load(path) as data:
            if any(len(data[name]) != support_count for name in OBSERVATION_NAMES):
                raise ValueError(f"Wrong support count in {path}")
            groups[item["properties"]["datetime"][:19]].append({name: data[name] for name in OBSERVATION_NAMES})
    dates = sorted(groups)
    observations = {name: np.full((len(dates), support_count), np.nan, dtype=np.float32) for name in OBSERVATION_NAMES}
    for date_index, date in enumerate(dates):
        records = groups[date]
        if len(records) == 1:
            for name in OBSERVATION_NAMES:
                observations[name][date_index] = records[0][name]
            continue
        ndvi = np.stack([record["ndvi"] for record in records])
        water = np.stack([record["water_fraction"] for record in records])
        coverage = np.stack([record["coverage"] for record in records])
        score = 3 * np.isfinite(ndvi) + 2 * np.isfinite(water) + np.nan_to_num(coverage, nan=-1)
        chosen = np.argmax(score, axis=0)
        columns = np.arange(support_count)
        for name in OBSERVATION_NAMES:
            stacked = np.stack([record[name] for record in records])
            observations[name][date_index] = stacked[chosen, columns]
    return [date + "+00:00" for date in dates], observations


def label_candidate(dates: list[str], observations: dict[str, np.ndarray], supports: dict, t0: str,
                    scene_id: str = "EMIT_L2A_RFL_001_20230727T100231_2320807_026") -> tuple[list[dict], dict]:
    count = len(supports["native_id"])
    labels = [
        classify_trajectory(
            dates,
            observations["ndvi"][:, index],
            observations["water_fraction"][:, index],
            observations["coverage"][:, index],
            t0,
            mndwi=observations["mndwi"][:, index],
        )
        for index in range(count)
    ]
    at_risk = np.array([not item["already_detected_pre_t0"] for item in labels])
    positive = np.array([item["target"] == "Persistent Degradation" for item in labels]) & at_risk
    stable = np.array([item["target"] == "Stable" for item in labels]) & at_risk
    unresolved = np.array([item["target"] == "Unresolved" for item in labels]) & at_risk
    transient = np.array([item["target"] == "Transient Stress" for item in labels]) & at_risk
    grid = np.zeros(tuple(supports["ortho_shape"]), dtype=bool)
    grid[supports["row"][positive], supports["col"][positive]] = True
    patches, patch_count = connected_components(grid, structure=np.ones((3, 3), dtype=int))
    patch_ids = patches[supports["row"], supports["col"]]
    for index, item in enumerate(labels):
        item["positive_patch_id"] = int(patch_ids[index]) if positive[index] else None
    summary = {
        "sensor": "EMIT",
        "scene_id": scene_id,
        "t0": t0,
        "native_supports_after_hsi_qa": count,
        "already_detected_pre_t0": int((~at_risk).sum()),
        "at_risk": int(at_risk.sum()),
        "persistent_positives": int(positive.sum()),
        "stable": int(stable.sum()),
        "unresolved": int(unresolved.sum()),
        "transient": int(transient.sum()),
        "positive_patches": int(patch_count),
        "loss_to_water": int(sum(item["subtype"] == "loss_to_water" for item, is_positive in zip(labels, positive) if is_positive)),
        "canopy_dieback_like": int(sum(item["subtype"] == "canopy_dieback_like" for item, is_positive in zip(labels, positive) if is_positive)),
        "patch_sizes": sorted(np.bincount(patch_ids[positive])[1:].tolist(), reverse=True),
    }
    return labels, summary


def save_feasibility(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
