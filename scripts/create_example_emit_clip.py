"""Extract a small orthorectified Mahajanga EMIT reflectance example.

This requires the locally downloaded raw scene; the generated GeoTIFF is the
small committed example and does not require raw data to read.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SCENE = "EMIT_L2A_RFL_001_20231227T092843_2336106_021"
SOURCE = ROOT / "data/raw/emit" / SCENE / f"{SCENE}.nc"
OUTPUT = ROOT / "data/sample_input/example_emit_clip.tif"
SIZE = 64


def main() -> None:
    cohort = pd.read_parquet(ROOT / "results/dataset/model_cohort.parquet", columns=["ortho_row", "ortho_col"])
    center_row = int(cohort.ortho_row.median())
    center_col = int(cohort.ortho_col.median())
    row0, col0 = center_row - SIZE // 2, center_col - SIZE // 2
    band_table = pd.read_csv(ROOT / "results/dataset/hsi_bands.csv")
    band_indices = band_table.loc[band_table.retained, "sensor_band_index"].to_numpy(dtype=int)
    wavelengths = band_table.loc[band_table.retained, "wavelength_nm"].to_numpy()

    with Dataset(SOURCE) as scene:
        native_x = np.asarray(scene.groups["location"].variables["glt_x"][row0:row0 + SIZE, col0:col0 + SIZE], dtype=int) - 1
        native_y = np.asarray(scene.groups["location"].variables["glt_y"][row0:row0 + SIZE, col0:col0 + SIZE], dtype=int) - 1
        valid = (native_x >= 0) & (native_y >= 0)
        clip = np.full((len(band_indices), SIZE, SIZE), np.nan, dtype=np.float32)
        for native_row in np.unique(native_y[valid]):
            positions = np.where(valid & (native_y == native_row))
            low = int(native_x[positions].min())
            high = int(native_x[positions].max()) + 1
            raw = scene.variables["reflectance"][int(native_row), low:high, :]
            raw = np.ma.filled(raw, np.nan) if np.ma.isMaskedArray(raw) else np.asarray(raw)
            clip[:, positions[0], positions[1]] = raw[native_x[positions] - low][:, band_indices].T
        clip[(clip < 0) | (clip > 1)] = np.nan
        transform = Affine.from_gdal(*map(float, scene.getncattr("geotransform"))) * Affine.translation(col0, row0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUTPUT, "w", driver="GTiff", height=SIZE, width=SIZE,
                       count=len(band_indices), dtype="float32", crs="EPSG:4326",
                       transform=transform, nodata=np.nan, compress="deflate", predictor=3) as dst:
        dst.write(clip)
        dst.update_tags(source_scene=SCENE, source_product="EMIT L2A RFL V001",
                        band_selection="retained bands from results/dataset/hsi_bands.csv")
        for band, wavelength in enumerate(wavelengths, start=1):
            dst.set_band_description(band, f"{wavelength:.1f} nm")
    print(f"{OUTPUT.relative_to(ROOT)}: {len(band_indices)} bands, {SIZE}x{SIZE} pixels, {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
