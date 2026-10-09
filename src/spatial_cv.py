from __future__ import annotations

import numpy as np
from rasterio.warp import transform
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold


def projected_xy(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    zone = int(np.floor((float(np.nanmedian(longitude)) + 180) / 6)) + 1
    hemisphere = 32600 if float(np.nanmedian(latitude)) >= 0 else 32700
    east, north = transform("EPSG:4326", f"EPSG:{hemisphere + zone}", longitude.tolist(), latitude.tolist())
    return np.column_stack((east, north))


def estimate_block_size(xy: np.ndarray, predictor: np.ndarray, seed: int = 813) -> int:
    finite = np.isfinite(predictor)
    if finite.sum() < 100:
        return 1000
    generator = np.random.default_rng(seed)
    candidates = np.flatnonzero(finite)
    first = generator.choice(candidates, 10000, replace=True)
    second = generator.choice(candidates, 10000, replace=True)
    distances = np.linalg.norm(xy[first] - xy[second], axis=1)
    semivariance = 0.5 * (predictor[first] - predictor[second]) ** 2
    bins = np.arange(0, 3100, 250)
    means = np.array([np.mean(semivariance[(distances >= left) & (distances < right)])
                      if np.any((distances >= left) & (distances < right)) else np.nan
                      for left, right in zip(bins[:-1], bins[1:])])
    finite_bins = np.flatnonzero(np.isfinite(means))
    if len(finite_bins) < 4:
        return 1000
    sill = float(np.nanmedian(means[finite_bins[-3:]]))
    reached = np.flatnonzero(means >= 0.95 * sill)
    correlation_range = int(bins[reached[0] + 1]) if len(reached) else 1000
    return max(1000, int(np.ceil(correlation_range / 1000) * 1000))


def assign_spatial_folds(
    xy: np.ndarray, target: np.ndarray, patch_ids: np.ndarray, predictor: np.ndarray,
    seed: int = 813, buffer_m: float = 60,
) -> dict:
    block_size = estimate_block_size(xy, predictor)
    cell = np.floor(xy / block_size).astype(int)
    keys = [f"{east}_{north}" for east, north in cell]
    _, groups = np.unique(keys, return_inverse=True)
    parents = np.arange(len(np.unique(groups)))

    def root(block: int) -> int:
        while parents[block] != block:
            parents[block] = parents[parents[block]]
            block = parents[block]
        return int(block)

    for patch_id in np.unique(patch_ids[patch_ids > 0]):
        patch_blocks = np.unique(groups[patch_ids == patch_id])
        first_root = root(int(patch_blocks[0]))
        for block in patch_blocks[1:]:
            parents[root(int(block))] = first_root
    _, groups = np.unique([root(int(block)) for block in groups], return_inverse=True)
    if len(np.unique(groups)) < 3:
        raise ValueError("Fewer than three spatial blocks")
    for fold_count in (5, 3):
        for trial in range(20):
            split = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed + trial)
            folds = np.full(len(target), -1, dtype=int)
            for fold_index, (_, test_indices) in enumerate(split.split(xy, target, groups)):
                folds[test_indices] = fold_index
            if all(np.any(target[folds == index] == 1) and np.any(target[folds == index] == 0)
                   and len(np.unique(patch_ids[(folds == index) & (target == 1)])) >= 1
                   for index in range(fold_count)):
                break
        else:
            continue
        break
    else:
        raise ValueError("Positive events cannot be distributed across three spatial folds")
    splits = []
    for fold_index in range(fold_count):
        test_indices = np.flatnonzero(folds == fold_index)
        train_indices = np.flatnonzero(folds != fold_index)
        if buffer_m:
            distance, _ = cKDTree(xy[test_indices]).query(xy[train_indices])
            train_indices = train_indices[distance >= buffer_m]
        if len(np.unique(target[train_indices])) < 2:
            raise ValueError("Buffered training fold lost a class")
        splits.append((train_indices, test_indices))
    return {"folds": folds, "groups": groups, "block_size_m": block_size,
            "buffer_m": buffer_m, "splits": splits, "seed": seed + trial}
