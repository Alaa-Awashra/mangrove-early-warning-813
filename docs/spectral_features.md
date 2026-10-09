# Hyperspectral feature definitions

The primary H2 spectral-resolution control uses the official ESA/Copernicus **Sentinel-2 Spectral Response Functions 2024 v4.0** workbook, [SentiWiki document library](https://sentiwiki.copernicus.eu/web/s2-documents), downloaded as `data/raw/s2_srf/Sentinel-2_SRF_2024_v4.0.xlsx` (SHA-256 `1a9edc27d692a570911a460d589f188da0fc3e27f0b0bd1ad322059c380519b0`). It uses the average **Sentinel-2A** response for B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12. Each EMIT reflectance spectrum is integrated against the measured SRF with wavelength-weighted trapezoids. At least 95% of the official SRF area must be covered by nonmissing adjacent HSI bands; otherwise that broadband value is missing. This holds acquisition time and native support fixed but does not reproduce Sentinel-2 spatial response, noise or calibration.

H3 is a compact set of **candidate physiological indicators**, interpreted cautiously because these indices were generally developed for vegetation and are not specific diagnoses of mangrove health:

| Feature | Formula or method | Scientific source |
| --- | --- | --- |
| PRI | `(R531 − R570)/(R531 + R570)` | [Gamon et al. index formulation](https://www.sciencedirect.com/science/article/pii/S0034425701002243) |
| CRI1 | `1/R510 − 1/R550` | [Gitelson et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/11950093/) |
| ARI | `1/R550 − 1/R700` | [Gitelson et al. (2001)](https://pubmed.ncbi.nlm.nih.gov/11460535/) |
| OSAVI | `1.16(R800 − R670)/(R800 + R670 + 0.16)` | [Rondeaux et al. (1996), standard form](https://www.slu.se/globalassets/ew/org/inst/vpe/vpe-report/vpe-report-no-1-2006.pdf) |
| MCARI | `[(R700 − R670) − 0.2(R700 − R550)](R700/R670)` | [Daughtry et al. (2000), reproduced here](https://pmc.ncbi.nlm.nih.gov/articles/PMC7937976/) |
| Red-edge position/slope | Wavelength and maximum first derivative over 680–760 nm | [Red-edge method](https://www.sciencedirect.com/science/article/pii/S0034425704001701) |
| 970/1200 nm water absorption depth | `1 − R_feature/R_linear_continuum`; shoulders 900/1050 and 1100/1300 nm respectively | Continuum-removal physical diagnostic; not a calibrated leaf-water-content retrieval |

Target wavelengths are represented by the nearest retained EMIT band within 10 nm; otherwise the feature is missing. Feature construction uses no labels. Scaling and imputation are fit inside outer training folds.
