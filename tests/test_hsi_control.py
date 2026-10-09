import numpy as np
import pytest
from pathlib import Path

from src.hsi import convolve_s2_srf, load_s2a_srf


def test_flat_spectrum_convolution_is_flat():
    wavelengths = np.arange(400, 901, 5, dtype=float)
    grid = wavelengths.copy()
    response = np.zeros(len(grid))
    response[(grid >= 500) & (grid <= 600)] = 1
    result = convolve_s2_srf(np.full((2, len(grid)), 0.35), wavelengths, grid, {"B03": response})
    assert np.allclose(result["B03"], 0.35)


def test_hsi_control_rejects_incomplete_band_coverage():
    wavelengths = np.arange(400, 901, 5, dtype=float)
    response = np.zeros(len(wavelengths))
    response[(wavelengths >= 500) & (wavelengths <= 600)] = 1
    spectra = np.full((1, len(wavelengths)), 0.35)
    spectra[:, (wavelengths >= 550) & (wavelengths <= 600)] = np.nan
    result = convolve_s2_srf(spectra, wavelengths, wavelengths, {"B03": response})
    assert np.isnan(result["B03"][0])


def test_hsi_control_rejects_unsorted_wavelengths():
    with pytest.raises(ValueError):
        convolve_s2_srf(np.ones((1, 2)), np.array([600, 500]), np.array([500, 600]), {"B03": np.array([1, 1])})


def test_official_s2a_srf_flat_spectrum_preserves_reflectance():
    path = Path("data/raw/s2_srf/Sentinel-2_SRF_2024_v4.0.xlsx")
    if not path.exists():
        pytest.skip("Official SRF workbook not prepared")
    grid, responses = load_s2a_srf(path)
    wavelengths = np.arange(400, 2501, 1, dtype=float)
    result = convolve_s2_srf(np.full((1, len(wavelengths)), 0.4), wavelengths, grid, responses)
    assert all(np.isclose(values[0], 0.4) for values in result.values())
