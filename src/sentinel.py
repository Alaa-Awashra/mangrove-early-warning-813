from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform
from rasterio.windows import Window


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
OBSERVATION_BANDS = ("B03", "B04", "B08", "B11")
PREDICTOR_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


def boa_add_offset(processing_baseline: str | float) -> int:
    baseline = float(processing_baseline)
    if 3 <= baseline < 4:
        return 0
    if 4 <= baseline < 6:
        return -1000
    raise ValueError(f"Unverified Sentinel-2 processing baseline: {baseline}")


def search_s2(bbox: list[float], periods: list[str], cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    items = {}
    for period in periods:
        payload = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": period, "limit": 500}
        response = requests.post(CATALOG_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        for item in result["features"]:
            items[item["id"]] = item
        for link in result.get("links", []):
            if link.get("rel") != "next":
                continue
            next_url = link["href"]
            next_method = link.get("method", "GET")
            while next_url:
                page = requests.request(next_method, next_url, json=link.get("body"), timeout=60)
                page.raise_for_status()
                next_result = page.json()
                for item in next_result["features"]:
                    items[item["id"]] = item
                next_link = next((entry for entry in next_result.get("links", []) if entry.get("rel") == "next"), None)
                next_url = next_link["href"] if next_link else None
                if next_link:
                    next_method = next_link.get("method", "GET")
                    link = next_link
    sorted_items = sorted(items.values(), key=lambda item: (item["properties"]["datetime"], item["id"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted_items), encoding="utf-8")
    return sorted_items


def _support_windows(supports: dict, crs, transform_affine, shape: tuple[int, int]) -> dict:
    longitude = np.asarray(supports["lon"])
    latitude = np.asarray(supports["lat"])
    half_width = abs(supports["transform"].a) / 2
    half_height = abs(supports["transform"].e) / 2
    west, south = transform("EPSG:4326", crs, (longitude - half_width).tolist(), (latitude - half_height).tolist())
    east, north = transform("EPSG:4326", crs, (longitude + half_width).tolist(), (latitude + half_height).tolist())
    west, south, east, north = map(np.asarray, (west, south, east, north))
    x_min, x_max = np.minimum(west, east), np.maximum(west, east)
    y_min, y_max = np.minimum(south, north), np.maximum(south, north)
    col_min = np.floor((x_min - transform_affine.c) / transform_affine.a).astype(int)
    col_max = np.ceil((x_max - transform_affine.c) / transform_affine.a).astype(int)
    row_min = np.floor((transform_affine.f - y_max) / abs(transform_affine.e)).astype(int)
    row_max = np.ceil((transform_affine.f - y_min) / abs(transform_affine.e)).astype(int)
    inside = (col_min >= 0) & (row_min >= 0) & (col_max <= shape[1]) & (row_max <= shape[0])
    chip_x = np.floor(np.asarray((x_min + x_max) / 2) / 3000).astype(int)
    chip_y = np.floor(np.asarray((y_min + y_max) / 2) / 3000).astype(int)
    groups = {}
    for index in np.flatnonzero(inside):
        groups.setdefault((chip_x[index], chip_y[index]), []).append(index)
    return {"row_min": row_min, "row_max": row_max, "col_min": col_min, "col_max": col_max, "groups": groups}


def _signed(href: str, token: str) -> str:
    return href + ("&" if "?" in href else "?") + token.lstrip("?")


def process_s2_item(item: dict, supports: dict, token: str, destination: Path) -> Path:
    output = destination / f"{item['id']}.npz"
    if output.exists():
        return output
    count = len(supports["native_id"])
    arrays = {name: np.full(count, np.nan, dtype=np.float32) for name in ("ndvi", "ndmi", "mndwi", "water_fraction", "coverage")}
    scl_url = _signed(item["assets"]["SCL"]["href"], token)
    band_urls = {band: _signed(item["assets"][band]["href"], token) for band in OBSERVATION_BANDS}
    boa_offset = boa_add_offset(item["properties"]["s2:processing_baseline"])
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1"):
        with rasterio.open(scl_url) as scl_source:
            windows = _support_windows(supports, scl_source.crs, scl_source.transform, scl_source.shape)
            if not windows["groups"]:
                np.savez_compressed(output, **arrays)
                return output
            sources = {band: rasterio.open(url) for band, url in band_urls.items()}
            virtual = {
                band: WarpedVRT(source, crs=scl_source.crs, transform=scl_source.transform, width=scl_source.width, height=scl_source.height, resampling=Resampling.average, src_nodata=0, nodata=0)
                for band, source in sources.items()
            }
            try:
                for indices in windows["groups"].values():
                    indices = np.asarray(indices, dtype=int)
                    row_start = int(windows["row_min"][indices].min())
                    row_stop = int(windows["row_max"][indices].max())
                    col_start = int(windows["col_min"][indices].min())
                    col_stop = int(windows["col_max"][indices].max())
                    chip_window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                    scl_chip = scl_source.read(1, window=chip_window)
                    bands = {band: vrt.read(1, window=chip_window).astype(np.float32) for band, vrt in virtual.items()}
                    for band in bands:
                        bands[band][bands[band] == 0] = np.nan
                        bands[band] = (bands[band] + boa_offset) / 10000
                        bands[band][(bands[band] < 0) | (bands[band] > 1)] = np.nan
                    heights = windows["row_max"][indices] - windows["row_min"][indices]
                    widths = windows["col_max"][indices] - windows["col_min"][indices]
                    if not len(indices) or heights.max() == 0 or widths.max() == 0:
                        continue
                    row_offsets = np.arange(int(heights.max()))
                    col_offsets = np.arange(int(widths.max()))
                    local_rows = windows["row_min"][indices, None, None] - row_start + row_offsets[None, :, None]
                    local_cols = windows["col_min"][indices, None, None] - col_start + col_offsets[None, None, :]
                    footprint = (row_offsets[None, :, None] < heights[:, None, None]) & (col_offsets[None, None, :] < widths[:, None, None])
                    local_rows = np.clip(local_rows, 0, scl_chip.shape[0] - 1)
                    local_cols = np.clip(local_cols, 0, scl_chip.shape[1] - 1)
                    scl = scl_chip[local_rows, local_cols]
                    clear = np.isin(scl, (4, 5, 6, 7)) & footprint
                    clear_count = clear.sum(axis=(1, 2))
                    coverage = clear_count / footprint.sum(axis=(1, 2))
                    arrays["coverage"][indices] = coverage
                    eligible = coverage >= 0.5
                    if not eligible.any():
                        continue
                    green, red, near_ir, swir = (bands[band][local_rows, local_cols] for band in OBSERVATION_BANDS)
                    land = np.isin(scl, (4, 5, 7)) & footprint
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pixel_mndwi = (green - swir) / (green + swir)
                        pixel_ndvi = (near_ir - red) / (near_ir + red)
                        pixel_ndmi = (near_ir - swir) / (near_ir + swir)
                    water = ((scl == 6) | (pixel_mndwi > 0)) & clear
                    arrays["water_fraction"][indices[eligible]] = water.sum(axis=(1, 2))[eligible] / clear_count[eligible]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        for name, pixels, mask in (("mndwi", pixel_mndwi, clear),
                                                   ("ndvi", pixel_ndvi, land),
                                                   ("ndmi", pixel_ndmi, land)):
                            values = np.where(mask & np.isfinite(pixels), pixels, np.nan).reshape(len(indices), -1)
                            median = np.nanmedian(values, axis=1)
                            arrays[name][indices[eligible]] = median[eligible]
            finally:
                for vrt in virtual.values():
                    vrt.close()
                for source in sources.values():
                    source.close()
    destination.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    return output


def process_s2_catalog(items: list[dict], supports: dict, destination: Path, workers: int = 6) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    if all((destination / f"{item['id']}.npz").exists() for item in items):
        return [destination / f"{item['id']}.npz" for item in items]
    for attempt in range(6):
        try:
            response = requests.get(TOKEN_URL, timeout=30)
            response.raise_for_status()
            token = response.json()["token"]
            break
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))

    def process_with_retry(item: dict) -> Path:
        for attempt in range(3):
            try:
                return process_s2_item(item, supports, token, destination)
            except (rasterio.errors.RasterioIOError, requests.RequestException, OSError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Unreachable retry state")

    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_with_retry, item): item["id"] for item in items}
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def process_near_t0_bands(
    items: list[dict], supports: dict, t0: str, observation_dir: Path, destination: Path,
) -> dict[str, np.ndarray | list[str]]:
    from datetime import datetime, timedelta

    prediction_time = datetime.fromisoformat(t0.replace("Z", "+00:00"))
    selected = [item for item in items if prediction_time - timedelta(days=60) <= datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")) <= prediction_time]
    if not selected:
        raise ValueError("No near-t0 Sentinel-2 scenes")
    destination.mkdir(parents=True, exist_ok=True)
    token = ""
    if not all((destination / f"{item['id']}.npz").exists() for item in selected):
        for attempt in range(6):
            try:
                response = requests.get(TOKEN_URL, timeout=30)
                response.raise_for_status()
                token = response.json()["token"]
                break
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
    count = len(supports["native_id"])
    output = {band: np.full(count, np.nan, dtype=np.float32) for band in PREDICTOR_BANDS}
    output_dates = []
    for item in sorted(selected, key=lambda entry: entry["properties"]["datetime"]):
        output_dates.append(item["properties"]["datetime"])
        cached = destination / f"{item['id']}.npz"
        if not cached.exists():
            arrays = {band: np.full(count, np.nan, dtype=np.float32) for band in PREDICTOR_BANDS}
            offset = boa_add_offset(item["properties"]["s2:processing_baseline"])
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1"):
                with rasterio.open(_signed(item["assets"]["SCL"]["href"], token)) as scl_source:
                    windows = _support_windows(supports, scl_source.crs, scl_source.transform, scl_source.shape)
                    if windows["groups"]:
                        sources = {band: rasterio.open(_signed(item["assets"][band]["href"], token)) for band in PREDICTOR_BANDS}
                        virtual = {band: WarpedVRT(source, crs=scl_source.crs, transform=scl_source.transform,
                                                   width=scl_source.width, height=scl_source.height,
                                                   resampling=Resampling.average, src_nodata=0, nodata=0)
                                   for band, source in sources.items()}
                        try:
                            for indices in windows["groups"].values():
                                indices = np.asarray(indices, dtype=int)
                                row_start = int(windows["row_min"][indices].min())
                                row_stop = int(windows["row_max"][indices].max())
                                col_start = int(windows["col_min"][indices].min())
                                col_stop = int(windows["col_max"][indices].max())
                                chip_window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                                scl_chip = scl_source.read(1, window=chip_window)
                                chips = {band: vrt.read(1, window=chip_window).astype(np.float32) for band, vrt in virtual.items()}
                                for band, chip in chips.items():
                                    chip[chip == 0] = np.nan
                                    chip[:] = (chip + offset) / 10000
                                    chip[(chip < 0) | (chip > 1)] = np.nan
                                heights = windows["row_max"][indices] - windows["row_min"][indices]
                                widths = windows["col_max"][indices] - windows["col_min"][indices]
                                if not len(indices) or heights.max() == 0 or widths.max() == 0:
                                    continue
                                row_offsets = np.arange(int(heights.max()))
                                col_offsets = np.arange(int(widths.max()))
                                local_rows = windows["row_min"][indices, None, None] - row_start + row_offsets[None, :, None]
                                local_cols = windows["col_min"][indices, None, None] - col_start + col_offsets[None, None, :]
                                footprint = (row_offsets[None, :, None] < heights[:, None, None]) & (col_offsets[None, None, :] < widths[:, None, None])
                                local_rows = np.clip(local_rows, 0, scl_chip.shape[0] - 1)
                                local_cols = np.clip(local_cols, 0, scl_chip.shape[1] - 1)
                                scl = scl_chip[local_rows, local_cols]
                                clear = np.isin(scl, (4, 5, 6, 7)) & footprint
                                coverage = clear.sum(axis=(1, 2)) / footprint.sum(axis=(1, 2))
                                eligible = coverage >= 0.5
                                land = np.isin(scl, (4, 5, 7)) & footprint
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", RuntimeWarning)
                                    for band, chip in chips.items():
                                        values = np.where(land, chip[local_rows, local_cols], np.nan).reshape(len(indices), -1)
                                        median = np.nanmedian(values, axis=1)
                                        arrays[band][indices[eligible]] = median[eligible]
                        finally:
                            for vrt in virtual.values():
                                vrt.close()
                            for source in sources.values():
                                source.close()
            temporary = cached.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(cached)
        with np.load(cached) as data, np.load(observation_dir / f"{item['id']}.npz") as quality:
            valid = np.isfinite(quality["ndvi"])
            for band in PREDICTOR_BANDS:
                replace = valid & np.isfinite(data[band])
                output[band][replace] = data[band][replace]
    output["dates"] = output_dates
    return output
