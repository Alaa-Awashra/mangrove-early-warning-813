from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from netCDF4 import Dataset
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject


def mangrove_fraction(gmw_dir: Path, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    right, bottom = transform * (shape[1], shape[0])
    left, top = transform * (0, 0)
    paths = sorted(gmw_dir.glob("GMW_*_2020_v3.tif"))
    sources = [rasterio.open(path) for path in paths]
    if not sources:
        raise FileNotFoundError(f"No GMW 2020 tiles in {gmw_dir}")
    try:
        mosaic, source_transform = merge(sources, bounds=(left, bottom, right, top), nodata=0)
    finally:
        for source in sources:
            source.close()
    fraction = np.zeros(shape, dtype=np.float32)
    reproject(
        source=mosaic[0].astype(np.float32),
        destination=fraction,
        src_transform=source_transform,
        src_crs="EPSG:4326",
        dst_transform=transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.average,
    )
    return np.clip(fraction, 0, 1)


def emit_native_candidates(reflectance_path: Path, mask_path: Path, gmw_dir: Path) -> dict[str, np.ndarray | Affine]:
    with Dataset(reflectance_path) as reflectance, Dataset(mask_path) as quality:
        if reflectance.dimensions["downtrack"].size != quality.dimensions["downtrack"].size:
            raise ValueError("RFL and MASK native geometry differ")
        if reflectance.dimensions["crosstrack"].size != quality.dimensions["crosstrack"].size:
            raise ValueError("RFL and MASK native geometry differ")
        transform = Affine.from_gdal(*map(float, reflectance.getncattr("geotransform")))
        glt_x = np.asarray(reflectance.groups["location"].variables["glt_x"][:]).astype(np.int32)
        glt_y = np.asarray(reflectance.groups["location"].variables["glt_y"][:]).astype(np.int32)
        fraction = mangrove_fraction(gmw_dir, glt_x.shape, transform)
        candidates = (fraction >= 0.75) & (glt_x > 0) & (glt_y > 0)
        orthorectified_index = np.flatnonzero(candidates)
        native_y = glt_y.flat[orthorectified_index] - 1
        native_x = glt_x.flat[orthorectified_index] - 1
        valid_index = (native_y < reflectance.dimensions["downtrack"].size) & (native_x < reflectance.dimensions["crosstrack"].size)
        orthorectified_index = orthorectified_index[valid_index]
        native_y = native_y[valid_index]
        native_x = native_x[valid_index]
        native_id = native_y * reflectance.dimensions["crosstrack"].size + native_x
        _, first = np.unique(native_id, return_index=True)
        first.sort()
        orthorectified_index = orthorectified_index[first]
        native_y = native_y[first]
        native_x = native_x[first]
        native_id = native_id[first]
        flags = quality.variables["mask"][:]
        flags = np.ma.filled(flags, np.nan) if np.ma.isMaskedArray(flags) else np.asarray(flags)
        selected_flags = flags[native_y, native_x]
        good_qa = np.all(np.isfinite(selected_flags[:, [0, 1, 3, 4]]), axis=1) & np.all(selected_flags[:, [0, 1, 3, 4]] == 0, axis=1)
        rows, cols = np.unravel_index(orthorectified_index[good_qa], glt_x.shape)
        return {
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "native_y": native_y[good_qa].astype(np.int32),
            "native_x": native_x[good_qa].astype(np.int32),
            "native_id": native_id[good_qa].astype(np.int64),
            "mangrove_fraction": fraction[rows, cols],
            "mangrove_fraction_grid": fraction,
            "lon": (transform.c + (cols + 0.5) * transform.a).astype(np.float64),
            "lat": (transform.f + (rows + 0.5) * transform.e).astype(np.float64),
            "transform": transform,
            "ortho_shape": np.array(glt_x.shape),
            "n_masked_cells": np.array([int(candidates.sum())]),
            "n_unique_before_qa": np.array([len(first)]),
            "n_cloud_or_artifact": np.array([int((~good_qa).sum())]),
        }


def extract_emit_spectra(reflectance_path: Path, mask_path: Path, supports: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_y = supports["native_y"]
    native_x = supports["native_x"]
    with Dataset(reflectance_path) as reflectance, Dataset(mask_path) as quality:
        parameters = reflectance.groups["sensor_band_parameters"]
        wavelengths = np.asarray(parameters.variables["wavelengths"][:], dtype=np.float64)
        good_wavelengths = np.asarray(parameters.variables["good_wavelengths"][:], dtype=np.uint8)
        spectrum = np.full((len(native_y), len(wavelengths)), np.nan, dtype=np.float32)
        band_invalid = np.zeros_like(spectrum, dtype=bool)
        for row in np.unique(native_y):
            selected = np.flatnonzero(native_y == row)
            low, high = int(native_x[selected].min()), int(native_x[selected].max()) + 1
            values = reflectance.variables["reflectance"][int(row), low:high, :]
            values = np.ma.filled(values, np.nan) if np.ma.isMaskedArray(values) else np.asarray(values)
            spectrum[selected] = values[native_x[selected] - low]
            packed = quality.variables["band_mask"][int(row), low:high, :]
            packed = np.ma.filled(packed, 255) if np.ma.isMaskedArray(packed) else np.asarray(packed)
            band_invalid[selected] = np.unpackbits(packed[native_x[selected] - low].astype(np.uint8), axis=-1)[:, : len(wavelengths)].astype(bool)
    spectrum[band_invalid | (spectrum < 0) | (spectrum > 1)] = np.nan
    official = good_wavelengths == 1
    physical = (wavelengths >= 420) & (wavelengths <= 2450)
    atmospheric = ~(((wavelengths >= 1331) & (wavelengths <= 1460)) | ((wavelengths >= 1796) & (wavelengths <= 1938)))
    eligible = official & physical & atmospheric
    missingness = np.mean(~np.isfinite(spectrum), axis=0) if len(spectrum) else np.ones(len(wavelengths))
    retained = eligible & (missingness <= 0.05)
    return spectrum, wavelengths, retained, missingness


S2_CONTROL_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


def load_s2a_srf(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook["Spectral Responses (S2A)"]
    rows = list(worksheet.values)
    headers = [str(value) if value is not None else "" for value in rows[0]]
    grid = np.asarray([float(row[0]) for row in rows[1:] if isinstance(row[0], (int, float))], dtype=float)
    responses = {}
    for band in S2_CONTROL_BANDS:
        expected = f"S2A_SR_AV_{band[0]}{int(band[1:3]) if band != 'B8A' else '8A'}"
        column = headers.index(expected)
        responses[band] = np.asarray([float(row[column] or 0) for row in rows[1:] if isinstance(row[0], (int, float))], dtype=float)
    return grid, responses


def convolve_s2_srf(spectra: np.ndarray, wavelengths: np.ndarray, srf_grid: np.ndarray, responses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    spectra = np.asarray(spectra, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if spectra.shape[1] != len(wavelengths) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Spectra and strictly increasing wavelengths required")
    valid_pairs = np.isfinite(spectra[:, :-1]) & np.isfinite(spectra[:, 1:])
    average_spectra = np.nan_to_num((spectra[:, :-1] + spectra[:, 1:]) / 2, nan=0)
    output = {}
    for band, response in responses.items():
        if len(response) != len(srf_grid) or np.any(response < 0):
            raise ValueError(f"Invalid official SRF for {band}")
        sampled = np.interp(wavelengths, srf_grid, response, left=0, right=0)
        weights = np.diff(wavelengths) * (sampled[:-1] + sampled[1:]) / 2
        official_area = np.sum((response[:-1] + response[1:]) * np.diff(srf_grid) / 2)
        coverage = valid_pairs @ weights / official_area
        denominator = valid_pairs @ weights
        numerator = average_spectra @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            values = numerator / denominator
        values[(coverage < 0.95) | (denominator <= 0)] = np.nan
        output[band] = values.astype(np.float32)
    return output


def physiological_features(spectra: np.ndarray, wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    spectra = np.asarray(spectra, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    def reflectance(target: float) -> np.ndarray:
        nearest = int(np.argmin(np.abs(wavelengths - target)))
        if abs(wavelengths[nearest] - target) > 10:
            return np.full(len(spectra), np.nan)
        return spectra[:, nearest]

    reflectances = {target: reflectance(target) for target in (510, 531, 550, 570, 670, 700, 720, 750, 800, 900, 970, 1050, 1100, 1200, 1300)}
    with np.errstate(divide="ignore", invalid="ignore"):
        pri = (reflectances[531] - reflectances[570]) / (reflectances[531] + reflectances[570])
        cri1 = 1 / reflectances[510] - 1 / reflectances[550]
        ari = 1 / reflectances[550] - 1 / reflectances[700]
        osavi = 1.16 * (reflectances[800] - reflectances[670]) / (reflectances[800] + reflectances[670] + 0.16)
        mcari = ((reflectances[700] - reflectances[670]) - 0.2 * (reflectances[700] - reflectances[550])) * (reflectances[700] / reflectances[670])
        continuum_970 = reflectances[900] + (reflectances[1050] - reflectances[900]) * (970 - 900) / (1050 - 900)
        continuum_1200 = reflectances[1100] + (reflectances[1300] - reflectances[1100]) * (1200 - 1100) / (1300 - 1100)
        depth_970 = 1 - reflectances[970] / continuum_970
        depth_1200 = 1 - reflectances[1200] / continuum_1200
    edge = (wavelengths >= 680) & (wavelengths <= 760)
    if edge.sum() >= 3:
        slopes = np.gradient(spectra[:, edge], wavelengths[edge], axis=1)
        finite = np.isfinite(slopes).any(axis=1)
        red_edge_position = np.full(len(spectra), np.nan)
        red_edge_slope = np.full(len(spectra), np.nan)
        if finite.any():
            maxima = np.nanargmax(slopes[finite], axis=1)
            red_edge_position[finite] = wavelengths[edge][maxima]
            red_edge_slope[finite] = slopes[finite, maxima]
    else:
        red_edge_position = np.full(len(spectra), np.nan)
        red_edge_slope = np.full(len(spectra), np.nan)
    features = {
        "h3_pri": pri,
        "h3_cri1": cri1,
        "h3_ari": ari,
        "h3_osavi": osavi,
        "h3_mcari": mcari,
        "h3_red_edge_position_nm": red_edge_position,
        "h3_red_edge_slope": red_edge_slope,
        "h3_water_depth_970": depth_970,
        "h3_water_depth_1200": depth_1200,
    }
    for name, values in features.items():
        values[~np.isfinite(values)] = np.nan
        features[name] = values.astype(np.float32)
    return features
