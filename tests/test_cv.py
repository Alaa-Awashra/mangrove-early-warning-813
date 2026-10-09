import numpy as np

from src.spatial_cv import assign_spatial_folds


def test_blocks_never_cross_train_and_test():
    east = np.repeat(np.arange(8) * 1200, 20)
    north = np.tile(np.arange(20) * 10, 8)
    xy = np.column_stack((east, north)).astype(float)
    target = np.zeros(len(xy), dtype=int)
    target[::20] = 1
    patches = np.zeros(len(xy), dtype=int)
    patches[::20] = np.arange(1, 9)
    predictor = np.sin(east / 3000)
    spatial = assign_spatial_folds(xy, target, patches, predictor, buffer_m=0)
    for train, test in spatial["splits"]:
        assert not set(spatial["groups"][train]) & set(spatial["groups"][test])


def test_spatial_buffer_removes_near_test_supports():
    east = np.repeat(np.arange(8) * 1200, 20)
    north = np.tile(np.arange(20) * 10, 8)
    xy = np.column_stack((east, north)).astype(float)
    target = np.zeros(len(xy), dtype=int)
    target[::20] = 1
    patches = np.zeros(len(xy), dtype=int)
    patches[::20] = np.arange(1, 9)
    spatial = assign_spatial_folds(xy, target, patches, np.sin(east / 3000), buffer_m=60)
    for train, test in spatial["splits"]:
        distances = np.linalg.norm(xy[train, None, :] - xy[None, test, :], axis=2)
        assert distances.min() >= 60


def test_connected_positive_patch_merges_adjacent_blocks():
    east = np.repeat(np.arange(8) * 1200, 20)
    north = np.tile(np.arange(20) * 10, 8)
    xy = np.column_stack((east, north)).astype(float)
    target = np.zeros(len(xy), dtype=int)
    target[::20] = 1
    patches = np.zeros(len(xy), dtype=int)
    patches[::20] = np.arange(1, 9)
    patches[20] = 1
    spatial = assign_spatial_folds(xy, target, patches, np.sin(east / 3000), buffer_m=0)
    assert spatial["groups"][0] == spatial["groups"][20]
    assert spatial["folds"][0] == spatial["folds"][20]
