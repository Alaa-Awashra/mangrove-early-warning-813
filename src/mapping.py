from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def support_map(values: np.ndarray, supports: dict, tif_path: Path, title: str,
                colorbar_label: str, cmap: str = "viridis") -> None:
    values = np.asarray(values, dtype=float)
    if len(values) != len(supports["row"]):
        raise ValueError("Map values must align to native supports")
    shape = tuple(supports["ortho_shape"])
    grid = np.full(shape, np.nan, dtype=np.float32)
    grid[np.asarray(supports["row"]), np.asarray(supports["col"])] = values.astype(np.float32)
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=supports["transform"],
                       nodata=np.nan, compress="deflate") as destination:
        destination.write(grid, 1)
    figure, axis = plt.subplots(figsize=(9, 6))
    visible = np.ma.masked_invalid(grid)
    picture = axis.imshow(visible, cmap=cmap, interpolation="nearest")
    axis.set_title(title)
    axis.set_axis_off()
    figure.colorbar(picture, ax=axis, label=colorbar_label, shrink=0.8)
    figure.tight_layout()
    figure.savefig(tif_path.with_suffix(".png"), dpi=160)
    plt.close(figure)


def alert_geojson(frame: pd.DataFrame, destination: Path, longitude: str = "lon", latitude: str = "lat") -> None:
    features = []
    for _, row in frame.iterrows():
        properties = {key: value for key, value in row.items() if key not in (longitude, latitude)}
        for key, value in properties.items():
            if pd.isna(value):
                properties[key] = None
            elif isinstance(value, np.generic):
                properties[key] = value.item()
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(row[longitude]), float(row[latitude])]},
                         "properties": properties})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
