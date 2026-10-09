from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
import io
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import joblib

from src.data import (download_emit, ensure_gmw_tiles, ensure_s2_srf,
                      ensure_sanderman_soil_carbon_zip, SANDERMAN_SOC_ZIP_URL)
from src.feasibility import assemble_observations, label_candidate
from src.features import broadband_control_features, build_conventional_features
from src.hsi import (convolve_s2_srf, emit_native_candidates, extract_emit_spectra,
                     load_s2a_srf, physiological_features)
from src.sentinel import process_near_t0_bands, process_s2_catalog, search_s2
from src.spatial_cv import assign_spatial_folds, projected_xy
from src.models import make_estimator, feature_columns, train_spatial_oof
from src.evaluation import (metrics_table, paired_block_bootstrap, save_core_figures,
                            save_label_examples, save_lead_times)
from src.early_warning import early_warning_table
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from src.mapping import support_map, alert_geojson
from src.carbon import align_soil_carbon_stock, plot_targeting, support_area_hectares, targeting_table


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "results/feasibility/analysis_plan.md"


def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["candidate"]


def plan_hash() -> str:
    if not PLAN.exists():
        raise FileNotFoundError("Preregistered analysis plan must exist before any model output")
    return hashlib.sha256(PLAN.read_bytes()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scene_paths(settings: dict) -> tuple[Path, Path, Path]:
    scene = settings["scene_id"]
    directory = ROOT / "data/raw/emit" / scene
    reflectance = directory / f"{scene}.nc"
    mask = directory / f"EMIT_L2A_MASK_001_{scene.removeprefix('EMIT_L2A_RFL_001_')}.nc"
    mangroves = ROOT / "data/raw/gmw_mahajanga_2020"
    return reflectance, mask, mangroves


def supports_and_items(settings: dict) -> tuple[dict, list[dict]]:
    reflectance, mask, mangroves = scene_paths(settings)
    if not reflectance.exists() or not mask.exists():
        raise FileNotFoundError("EMIT raw inputs missing; run python run.py --prepare-data")
    supports = emit_native_candidates(reflectance, mask, mangroves)
    t0 = datetime.fromisoformat(settings["t0"].replace("Z", "+00:00"))
    from dateutil.relativedelta import relativedelta

    periods = []
    for start in (t0 - relativedelta(months=24), t0 - relativedelta(months=12), t0,
                  t0 + relativedelta(months=12)):
        end = start + relativedelta(months=12)
        periods.append(f"{start.date().isoformat()}/{end.date().isoformat()}")
    items = search_s2(settings["bbox"], periods, ROOT / "data/cache/s2_items_mahajanga.json")
    unavailable = set(settings.get("excluded_s2_items", ()))
    items = [item for item in items if item["properties"].get("s2:mgrs_tile") in settings["s2_tiles"]
             and item["id"] not in unavailable]
    return supports, items


def prepare_data(settings: dict) -> None:
    ensure_gmw_tiles(ROOT / "data/raw/gmw_mahajanga_2020", ROOT / "data/cache/gmw_v3_2020_gtiff.zip",
                     tuple(settings["gmw_tiles"]))
    ensure_s2_srf(ROOT / "data/raw/s2_srf/Sentinel-2_SRF_2024_v4.0.xlsx")
    download_emit(settings["scene_id"], tuple(settings["bbox"]), settings["t0"][:10], ROOT / "data/raw/emit")
    supports, items = supports_and_items(settings)
    cache = ROOT / "data/cache/s2_observations_mahajanga"
    for start in range(0, len(items), 100):
        process_s2_catalog(items[start:start + 100], supports, cache, workers=12)
        print(f"Sentinel-2 cached: {min(start + 100, len(items))}/{len(items)}", flush=True)


def feasibility(settings: dict) -> tuple[dict, list[dict], dict, list[str], dict, list[dict]]:
    supports, items = supports_and_items(settings)
    dates, observations = assemble_observations(items, ROOT / "data/cache/s2_observations_mahajanga",
                                                len(supports["native_id"]))
    destination = ROOT / "results/feasibility/mahajanga_summary.json"
    interim = ROOT / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)
    label_cache = interim / "mahajanga_labels.json"
    if destination.exists() and label_cache.exists():
        cached = json.loads(label_cache.read_text(encoding="utf-8"))
        summary = json.loads(destination.read_text(encoding="utf-8"))
        if cached.get("analysis_plan_sha256") == plan_hash() and len(cached["labels"]) == len(supports["native_id"]):
            return supports, items, observations, dates, summary, cached["labels"]
    labels, summary = label_candidate(dates, observations, supports, settings["t0"], settings["scene_id"])
    summary["aoi"] = settings["aoi"]
    summary["s2_catalog_items"] = len(items)
    summary["s2_unique_dates"] = len(dates)
    summary["analysis_plan_sha256"] = plan_hash()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    label_cache.write_text(json.dumps({"analysis_plan_sha256": plan_hash(), "labels": labels}), encoding="utf-8")
    return supports, items, observations, dates, summary, labels


def quick(settings: dict) -> None:
    plan_hash()
    reflectance, mask, mangroves = scene_paths(settings)
    if not all(path.exists() for path in (reflectance, mask, mangroves)):
        raise FileNotFoundError("Prepared raw inputs missing; run --prepare-data")
    supports = emit_native_candidates(reflectance, mask, mangroves)
    if len(supports["native_id"]) < 300:
        raise RuntimeError("Predictor-side native-support minimum not met")
    print(f"Quick data check passed: {len(supports['native_id'])} unique QA-valid native supports")


def dataset_stage(settings: dict, supports: dict, items: list[dict], observations: dict,
                  dates: list[str], summary: dict, labels: list[dict]) -> pd.DataFrame:
    if not settings["selected"]:
        raise RuntimeError("Candidate must be selected by frozen feasibility before dataset/modeling")
    (ROOT / "results/feasibility/analysis_plan_sha256.txt").write_text(plan_hash() + "\n", encoding="utf-8")
    reflectance, mask, _ = scene_paths(settings)
    spectrum, wavelengths, retained, missingness = extract_emit_spectra(reflectance, mask, supports)
    with Dataset(reflectance) as product:
        official_good = np.asarray(product.groups["sensor_band_parameters"].variables["good_wavelengths"][:]) == 1
    reason = []
    for wavelength, good, missing, keep in zip(wavelengths, official_good, missingness, retained):
        if keep:
            reason.append("retained")
        elif not good:
            reason.append("official_bad")
        elif wavelength < 420 or wavelength > 2450:
            reason.append("physical_limit")
        elif 1331 <= wavelength <= 1460 or 1796 <= wavelength <= 1938:
            reason.append("atmospheric_absorption")
        elif missing > 0.05:
            reason.append("predictor_missingness_gt_5_percent")
        else:
            reason.append("excluded_other")
    dataset_dir = ROOT / "results/dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sensor_band_index": np.arange(len(wavelengths)), "wavelength_nm": wavelengths,
                  "retained": retained, "reason": reason, "missingness": missingness,
                  "analysis_plan_sha256": plan_hash()}).to_csv(dataset_dir / "hsi_bands.csv", index=False)
    near_cache = ROOT / "data/processed/near_t0_bands.npz"
    near_cache.parent.mkdir(parents=True, exist_ok=True)
    if near_cache.exists():
        with np.load(near_cache) as saved:
            near = {name: saved[name] for name in saved.files if name != "dates"}
            near["dates"] = saved["dates"].tolist()
    else:
        near = process_near_t0_bands(items, supports, settings["t0"],
                                     ROOT / "data/cache/s2_observations_mahajanga",
                                     ROOT / "data/cache/s2_near_t0_mahajanga")
        np.savez_compressed(near_cache, **near)
    conventional = build_conventional_features(dates, observations, supports, settings["t0"], near)
    retained_spectrum = spectrum[:, retained]
    srf_grid, responses = load_s2a_srf(ROOT / "data/raw/s2_srf/Sentinel-2_SRF_2024_v4.0.xlsx")
    broadband = convolve_s2_srf(retained_spectrum, wavelengths[retained], srf_grid, responses)
    control = broadband_control_features(broadband)
    physiological = pd.DataFrame(physiological_features(retained_spectrum, wavelengths[retained]))
    spectral = pd.DataFrame(retained_spectrum, columns=[f"h1_{index:03d}" for index in range(retained.sum())])
    ready = (wavelengths[retained] >= 400) & (wavelengths[retained] <= 1700)
    spectral_813 = pd.DataFrame(retained_spectrum[:, ready], columns=[f"h4_{index:03d}" for index in range(ready.sum())])
    metadata = pd.DataFrame({"sample_index": np.arange(len(labels)),
                             "sample_id": [f"{settings['scene_id']}:{native}" for native in supports["native_id"]],
                             "native_support_id": supports["native_id"], "lon": supports["lon"],
                             "lat": supports["lat"], "ortho_row": supports["row"],
                             "ortho_col": supports["col"], "mangrove_fraction": supports["mangrove_fraction"],
                             "target": [label["target"] for label in labels],
                             "subtype": [label["subtype"] for label in labels],
                             "T_onset": [label["T_onset"] for label in labels],
                             "T_detect": [label["T_detect"] for label in labels],
                             "T_last_ok": [label["T_last_ok"] for label in labels],
                             "already_detected_pre_t0": [label["already_detected_pre_t0"] for label in labels],
                             "positive_patch_id": [label["positive_patch_id"] for label in labels],
                             "reference_count": [label["reference_count"] for label in labels],
                             "future_valid_count": [label["future_valid_count"] for label in labels],
                             "seasonal_fallback_count": [label["seasonal_fallback_count"] for label in labels],
                             "analysis_plan_sha256": plan_hash()})
    metadata["onset_bracket_days"] = (
        pd.to_datetime(metadata["T_onset"], utc=True) - pd.to_datetime(metadata["T_last_ok"], utc=True)
    ).dt.total_seconds() / 86400
    cohort = pd.concat([metadata, conventional, spectral, control, physiological, spectral_813], axis=1)
    if cohort["native_support_id"].duplicated().any():
        raise ValueError("Native detector support duplicated")
    cohort.to_parquet(dataset_dir / "cohort.parquet", index=False)
    pd.DataFrame([{"step": "GMW >=75% orthorectified", "count": int(supports["n_masked_cells"][0])},
                  {"step": "Unique native supports", "count": int(supports["n_unique_before_qa"][0])},
                  {"step": "HSI QA-valid", "count": len(labels)},
                  {"step": "Already detected pre-t0 excluded", "count": summary["already_detected_pre_t0"]},
                  {"step": "Persistent positive", "count": summary["persistent_positives"]},
                  {"step": "Stable negative", "count": summary["stable"]},
                  {"step": "Transient excluded", "count": summary["transient"]},
                  {"step": "Unresolved excluded", "count": summary["unresolved"]}]).to_csv(dataset_dir / "cohort_flow.csv", index=False)
    pd.DataFrame([{"class": key, "count": summary[key]} for key in ("persistent_positives", "stable", "transient", "unresolved")]).to_csv(dataset_dir / "label_summary.csv", index=False)
    cohort[cohort["positive_patch_id"].notna()].groupby("positive_patch_id").agg(
        native_supports=("sample_id", "count"), subtype=("subtype", "first"),
        centroid_lon=("lon", "mean"), centroid_lat=("lat", "mean")
    ).reset_index().to_csv(dataset_dir / "positive_patches.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(cohort["lon"], cohort["lat"], c=cohort["mangrove_fraction"], s=2, cmap="Greens")
    axis.set(xlabel="Longitude", ylabel="Latitude", title=f"{settings['aoi']} — EMIT native supports")
    figure.tight_layout()
    figure.savefig(ROOT / "results/feasibility/selected_aoi_map.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(cohort["lon"], cohort["lat"], c="lightgray", s=2)
    excluded = cohort[cohort["already_detected_pre_t0"]]
    axis.scatter(excluded["lon"], excluded["lat"], c="crimson", s=5)
    axis.set(xlabel="Longitude", ylabel="Latitude", title=f"Already detected by t0: {len(excluded)} supports")
    figure.tight_layout()
    figure.savefig(ROOT / "results/feasibility/already_detected_map.png", dpi=160)
    plt.close(figure)
    save_label_examples(dates, observations, cohort, settings["t0"],
                        ROOT / "results/figures/label_timeseries_examples.png")
    write_manifest(settings, items)
    return cohort


def write_manifest(settings: dict, items: list[dict]) -> None:
    reflectance, mask, mangroves = scene_paths(settings)
    sources = []
    for path, role in ((reflectance, "predictor"), (mask, "predictor_QA")):
        sources.append({"source": "EMIT L2A V001", "provider": "NASA LP DAAC",
                        "scene_id": settings["scene_id"], "reference_date": settings["t0"],
                        "api_identifier": f"CMR:EMITL2ARFL:{settings['scene_id']}",
                        "local_path": str(path.relative_to(ROOT)), "sha256": file_hash(path),
                        "access": "Earthdata authentication required", "role": role})
    for name in settings["gmw_tiles"]:
        path = mangroves / name
        sources.append({"source": "Global Mangrove Watch v3 2020", "provider": "JAXA/Aberystwyth/Wetlands International",
                        "scene_id": name, "reference_date": "2020", "url": "https://zenodo.org/records/6894273",
                        "local_path": str(path.relative_to(ROOT)), "sha256": file_hash(path),
                        "access": "Open; see Zenodo license", "role": "pre_t0_mask"})
    sources.append({"source": "Sentinel-2 L2A", "provider": "Copernicus via Microsoft Planetary Computer",
                    "scene_ids": [item["id"] for item in items],
                    "reference_date": f"{items[0]['properties']['datetime']} to {items[-1]['properties']['datetime']}",
                    "api_identifier": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a",
                    "local_path": "data/cache/s2_observations_mahajanga/", "access": "Open STAC, signed COG URLs",
                    "role": "pre_t0_predictor_and_post_t0_label"})
    srf = ROOT / "data/raw/s2_srf/Sentinel-2_SRF_2024_v4.0.xlsx"
    sources.append({"source": "Sentinel-2A measured SRF 2024 v4.0", "provider": "ESA/Copernicus",
                    "scene_id": "COPE-GSEG-EOPG-TN-15-0007-v4", "reference_date": "2024-06-19",
                    "url": "https://sentiwiki.copernicus.eu/web/s2-documents",
                    "local_path": str(srf.relative_to(ROOT)), "sha256": file_hash(srf),
                    "access": "Open", "role": "spectral_resolution_control"})
    (ROOT / "data_manifest.json").write_text(json.dumps({"analysis_plan_sha256": plan_hash(),
                                                            "sources": sources}, indent=2), encoding="utf-8")


def model_stage(settings: dict, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selected = cohort[(cohort["target"].isin(("Persistent Degradation", "Stable"))) &
                      (~cohort["already_detected_pre_t0"])].copy().reset_index(drop=True)
    selected["sample_index"] = np.arange(len(selected))
    target = (selected["target"] == "Persistent Degradation").to_numpy(dtype=int)
    xy = projected_xy(selected["lon"].to_numpy(), selected["lat"].to_numpy())
    patches = selected["positive_patch_id"].fillna(0).to_numpy(dtype=int)
    spatial = assign_spatial_folds(xy, target, patches, selected["f2_ndvi_median"].to_numpy())
    selected["spatial_fold"] = spatial["folds"]
    selected["spatial_block"] = spatial["groups"]
    dataset_dir = ROOT / "results/dataset"
    selected.to_parquet(dataset_dir / "model_cohort.parquet", index=False)
    model_dir = ROOT / "results/models"
    metadata_dir = model_dir / "model_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fold_metadata = {"analysis_plan_sha256": plan_hash(), "block_size_m": spatial["block_size_m"],
                     "buffer_m": spatial["buffer_m"], "assignment_seed": spatial["seed"],
                     "blocks": int(len(np.unique(spatial["groups"]))), "folds": []}
    for fold_index, (_, test) in enumerate(spatial["splits"]):
        positives = target[test] == 1
        fold_metadata["folds"].append({"fold": fold_index, "samples": len(test),
                                       "positive_samples": int(positives.sum()),
                                       "positive_patches": int(len(np.unique(patches[test][positives])))})
    (metadata_dir / "spatial_cv.json").write_text(json.dumps(fold_metadata, indent=2), encoding="utf-8")
    results = []
    fold_models = {}
    for family, configurations in (("lr", ("A", "B", "C", "D", "E")),
                                   ("rf", ("A", "C", "D", "E"))):
        for configuration in configurations:
            output, models = train_spatial_oof(selected, target, spatial, configuration, family,
                                               metadata_dir, plan_hash())
            results.append(output)
            fold_models[f"{family}_{configuration}"] = models
            print(f"Spatial OOF complete: {family}_{configuration}", flush=True)
    oof = pd.concat(results, ignore_index=True)
    oof.to_parquet(model_dir / "oof_predictions.parquet", index=False)
    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics = metrics_table(oof)
    metrics["analysis_plan_sha256"] = plan_hash()
    metrics.to_csv(tables / "metrics_all.csv", index=False)
    metrics[metrics["scope"] == "pooled"].to_csv(tables / "metrics_summary.csv", index=False)
    indexed = {name: oof[oof["model"] == name].set_index("sample_index").sort_index()
               for name in ("lr_A", "lr_C", "lr_D")}
    primary = paired_block_bootstrap(target, indexed["lr_C"]["risk"].to_numpy(),
                                     indexed["lr_A"]["risk"].to_numpy(), spatial["groups"])
    primary.update({"model_A_pr_auc": float(metrics[(metrics.model == "lr_A") & (metrics.scope == "pooled") &
                                                      (metrics.budget_percent == 10)].iloc[0]["pr_auc"]),
                    "model_C_pr_auc": float(metrics[(metrics.model == "lr_C") & (metrics.scope == "pooled") &
                                                      (metrics.budget_percent == 10)].iloc[0]["pr_auc"]),
                    "analysis_plan_sha256": plan_hash()})
    pd.DataFrame([primary]).to_csv(tables / "primary_comparison.csv", index=False)
    spectral = paired_block_bootstrap(target, indexed["lr_C"]["risk"].to_numpy(),
                                      indexed["lr_D"]["risk"].to_numpy(), spatial["groups"])
    spectral["comparison"] = "full_HSI_C_minus_official_SRF_broadband_D"
    spectral["analysis_plan_sha256"] = plan_hash()
    pd.DataFrame([spectral]).to_csv(tables / "spectral_control.csv", index=False)
    pd.DataFrame([{"comparison": "C_minus_A", **primary}, {"comparison": "C_minus_D", **spectral}]).to_csv(
        tables / "paired_differences.csv", index=False)
    subtype_rows = []
    for subtype in ("canopy_dieback_like", "loss_to_water"):
        subset = selected[(selected["target"] == "Stable") | (selected["subtype"] == subtype)]
        indices = subset["sample_index"].to_numpy(dtype=int)
        subtype_target = (subset["subtype"] == subtype).to_numpy(dtype=int)
        for name in ("lr_A", "lr_C", "lr_D"):
            score = indexed[name].loc[indices, "risk"].to_numpy()
            subtype_rows.append({"subtype_vs_stable": subtype, "model": name, "n": len(indices),
                                 "positive_cases": int(subtype_target.sum()),
                                 "pr_auc": float(average_precision_score(subtype_target, score)),
                                 "analysis_plan_sha256": plan_hash()})
    pd.DataFrame(subtype_rows).to_csv(tables / "subtype_metrics.csv", index=False)
    early, incremental = early_warning_table(oof, selected, settings["t0"])
    early["analysis_plan_sha256"] = plan_hash()
    incremental["analysis_plan_sha256"] = plan_hash()
    early.to_csv(tables / "early_warning.csv", index=False)
    incremental.to_csv(tables / "incremental_warnings.csv", index=False)
    positive_indices = np.flatnonzero(target == 1)
    paired_rows = []
    for budget in (5, 10, 20):
        baseline_alert = indexed["lr_A"].loc[positive_indices, f"alert_{budget}"].to_numpy(dtype=bool)
        enhanced_alert = indexed["lr_C"].loc[positive_indices, f"alert_{budget}"].to_numpy(dtype=bool)
        paired_rows.append({"budget_percent": budget, "both": int((baseline_alert & enhanced_alert).sum()),
                            "hsi_only": int((~baseline_alert & enhanced_alert).sum()),
                            "baseline_only": int((baseline_alert & ~enhanced_alert).sum()),
                            "neither": int((~baseline_alert & ~enhanced_alert).sum()),
                            "analysis_plan_sha256": plan_hash()})
    pd.DataFrame(paired_rows).to_csv(tables / "paired_alert_contingency.csv", index=False)
    save_core_figures(oof, ROOT / "results/figures")
    save_lead_times(incremental, ROOT / "results/figures/lead_time.png")
    final_dir = model_dir / "final_visualization_models"
    final_dir.mkdir(parents=True, exist_ok=True)
    for configuration in ("A", "C", "D", "E"):
        columns = feature_columns(selected, configuration)
        model = make_estimator(columns, "lr", len(selected), 0.1)
        model.fit(selected[columns], target)
        joblib.dump({"model": model, "features": columns, "analysis_plan_sha256": plan_hash(),
                     "purpose": "full-cohort visualization only; evaluation uses outer OOF"},
                    final_dir / f"lr_{configuration}.joblib")
    coefficient_rows = []
    for configuration in ("A", "E"):
        saved = joblib.load(final_dir / f"lr_{configuration}.joblib")
        coefficients = saved["model"].named_steps["classifier"].coef_[0]
        coefficient_rows.extend({"model": f"lr_{configuration}", "feature": name,
                                 "standardized_coefficient": float(value), "analysis_plan_sha256": plan_hash()}
                                for name, value in zip(saved["features"], coefficients))
    coefficient_table = pd.DataFrame(coefficient_rows)
    coefficient_table.to_csv(tables / "coefficients.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, configuration in zip(axes, ("A", "E")):
        subset = coefficient_table[coefficient_table["model"] == f"lr_{configuration}"].copy()
        subset = subset.iloc[np.argsort(np.abs(subset["standardized_coefficient"]))[-15:]]
        axis.barh(subset["feature"], subset["standardized_coefficient"])
        axis.set_title(f"Full-cohort descriptive LR {configuration}")
    figure.tight_layout()
    figure.savefig(ROOT / "results/figures/coefficients.png", dpi=160)
    plt.close(figure)
    combined = joblib.load(final_dir / "lr_C.joblib")
    combined_model = combined["model"]
    spectral_pipeline = combined_model.named_steps["preprocess"].named_transformers_["spectrum"]
    spectral_columns = [name for name in combined["features"] if name.startswith("h1_")]
    tabular_count = len(combined["features"]) - len(spectral_columns)
    contribution = combined_model.named_steps["classifier"].coef_[0][tabular_count:] @ spectral_pipeline.named_steps["pca"].components_
    band_table = pd.read_csv(dataset_dir / "hsi_bands.csv")
    wavelength = band_table.loc[band_table["retained"], "wavelength_nm"].to_numpy()
    pd.DataFrame({"wavelength_nm": wavelength, "standardized_contribution": contribution,
                  "analysis_plan_sha256": plan_hash()}).to_csv(tables / "wavelength_importance.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(wavelength, contribution)
    axis.set(xlabel="Wavelength (nm)", ylabel="Back-projected standardized LR coefficient",
             title="Full-cohort descriptive spectral contribution")
    figure.tight_layout()
    figure.savefig(ROOT / "results/figures/wavelength_importance.png", dpi=160)
    plt.close(figure)
    permutation_rows = []
    for configuration in ("A", "E"):
        columns = feature_columns(selected, configuration)
        for fold_index, (_, test) in enumerate(spatial["splits"]):
            importance = permutation_importance(fold_models[f"rf_{configuration}"][fold_index],
                                                selected.iloc[test][columns], target[test],
                                                scoring="average_precision", n_repeats=3,
                                                random_state=813 + fold_index, n_jobs=1)
            permutation_rows.extend({"model": f"rf_{configuration}", "fold": fold_index,
                                     "feature": name, "mean_pr_auc_decrease": float(mean),
                                     "analysis_plan_sha256": plan_hash()}
                                    for name, mean in zip(columns, importance.importances_mean))
    pd.DataFrame(permutation_rows).to_csv(tables / "permutation_importance.csv", index=False)
    return selected, oof, {"spatial": spatial, "fold_models": fold_models, "primary": primary, "spectral": spectral}


def mapping_stage(settings: dict, supports: dict, cohort: pd.DataFrame, selected: pd.DataFrame,
                  oof: pd.DataFrame, carbon_values: np.ndarray, quantitative: bool) -> None:
    model_root = ROOT / "results/models"
    final_dir = model_root / "final_visualization_models"
    predictions = {}
    for configuration in ("A", "C"):
        saved = joblib.load(final_dir / f"lr_{configuration}.joblib")
        raw = saved["model"].predict_proba(cohort[saved["features"]])[:, 1]
        held_out = oof[oof["model"] == f"lr_{configuration}"].sort_values("sample_index")
        clipped = np.clip(held_out["risk"].to_numpy(), 1e-5, 1 - 1e-5)
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(np.log(clipped / (1 - clipped)).reshape(-1, 1), held_out["target"])
        clipped_full = np.clip(raw, 1e-5, 1 - 1e-5)
        predictions[configuration] = calibrator.predict_proba(
            np.log(clipped_full / (1 - clipped_full)).reshape(-1, 1))[:, 1]
        joblib.dump({"calibrator": calibrator, "analysis_plan_sha256": plan_hash(),
                     "purpose": "full-cohort visualization only"}, final_dir / f"lr_{configuration}_visual_calibrator.joblib")
    variability = []
    for path in sorted((model_root / "model_metadata").glob("lr_C_fold*.joblib")):
        saved = joblib.load(path)
        variability.append(saved["model"].predict_proba(cohort[saved["features"]])[:, 1])
    variability = np.std(np.stack(variability), axis=0)
    at_risk = ~cohort["already_detected_pre_t0"].to_numpy(dtype=bool)
    conventional = np.where(at_risk, predictions["A"], np.nan)
    enhanced = np.where(at_risk, predictions["C"], np.nan)
    carbon_values = np.where(at_risk, carbon_values, np.nan)
    maps = ROOT / "results/maps"
    support_map(conventional, supports, maps / "conventional_risk.tif", "Conventional pre-t0 risk",
                "Visualization-only calibrated risk")
    support_map(enhanced, supports, maps / "hsi_risk.tif", "HSI-enhanced pre-t0 risk",
                "Visualization-only calibrated risk")
    support_map(enhanced - conventional, supports, maps / "hsi_incremental_risk.tif",
                "HSI minus conventional risk", "Risk difference", cmap="coolwarm")
    support_map(np.where(at_risk, variability, np.nan), supports, maps / "model_prediction_variability.tif",
                "Across-fold model prediction variability", "Probability SD")
    support_map(carbon_values, supports, maps / "carbon.tif",
                "Estimated 0-100 cm soil C stock" if quantitative else "Mangrove-area carbon relevance proxy",
                "Mg C per native support" if quantitative else "Mangrove area proxy")
    support_map(enhanced * carbon_values, supports, maps / "carbon_priority.tif",
                "Expected soil C at risk" if quantitative else "Carbon-relevance priority score",
                "Mg C expected at risk" if quantitative else "Proxy-weighted score")
    alerts = oof[(oof["model"] == "lr_C") & (oof["alert_10"])].merge(
        selected[["sample_index", "lon", "lat", "native_support_id", "target", "subtype",
                  "T_detect", "positive_patch_id"]], on="sample_index", suffixes=("", "_cohort"),
        validate="one_to_one")
    alerts["t0"] = settings["t0"]
    alerts["analysis_plan_sha256"] = plan_hash()
    alerts.to_csv(maps / "alerts.csv", index=False)
    alert_geojson(alerts, maps / "alerts.geojson")


def carbon_stage(supports: dict, cohort: pd.DataFrame, selected: pd.DataFrame,
                 oof: pd.DataFrame) -> tuple[np.ndarray, bool]:
    if not (ROOT / "results/tables/primary_comparison.csv").exists():
        raise RuntimeError("Degradation-model evaluation must be frozen before carbon analysis")
    metadata_path = ROOT / "results/tables/carbon_metadata.json"
    if metadata_path.exists() and (ROOT / "results/tables/carbon_targeting.csv").exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        column = "carbon_stock_MgC" if cached["quantitative"] else "carbon_relevance_proxy_ha"
        if cached.get("analysis_plan_sha256") == plan_hash() and column in cohort:
            return cohort[column].to_numpy(dtype=float), bool(cached["quantitative"])
    raw_dir = ROOT / "data/raw/carbon"
    archive_path = raw_dir / "mangroves_SOC30m_0_100cm.zip"
    quantitative = False
    source_paths = []
    access_issue = None
    try:
        ensure_sanderman_soil_carbon_zip(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            import shapefile

            base = "mangroves_SOC30m_0_100cm/mangroves_SOC30m"
            reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                                      shx=io.BytesIO(archive.read(base + ".shx")),
                                      dbf=io.BytesIO(archive.read(base + ".dbf")))
            west, south, east, north = (float(cohort["lon"].min()), float(cohort["lat"].min()),
                                        float(cohort["lon"].max()), float(cohort["lat"].max()))
            members = []
            for shape_record in reader.iterShapeRecords():
                left, bottom, right, top = shape_record.shape.bbox
                if left < east and right > west and bottom < north and top > south:
                    members.append("mangroves_SOC30m_0_100cm/" + shape_record.record.location)
            if not members:
                raise FileNotFoundError("Published soil-carbon tiles do not intersect selected native supports")
            for member in members:
                destination = raw_dir / Path(member).name
                if not destination.exists():
                    with archive.open(member) as source, destination.open("wb") as output:
                        while chunk := source.read(4 * 1024 * 1024):
                            output.write(chunk)
                source_paths.append(destination)
        values = np.full(len(cohort), np.nan)
        for path in source_paths:
            aligned = align_soil_carbon_stock(path, supports)
            values = np.where(np.isfinite(values), values, aligned)
        quantitative = bool(np.isfinite(values).sum() >= 300)
        if not quantitative:
            access_issue = "Official soil-carbon raster has insufficient native-support coverage"
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as error:
        access_issue = str(error)
        values = np.full(len(cohort), np.nan)
    if not quantitative:
        values = support_area_hectares(supports) * cohort["mangrove_fraction"].to_numpy(dtype=float)
    cohort["carbon_stock_MgC" if quantitative else "carbon_relevance_proxy_ha"] = values
    cohort.to_parquet(ROOT / "results/dataset/cohort.parquet", index=False)
    selected_values = cohort.set_index("sample_id").loc[selected["sample_id"],
                                                        "carbon_stock_MgC" if quantitative else "carbon_relevance_proxy_ha"].to_numpy()
    valid = np.isfinite(selected_values) & (selected_values > 0)
    indexed = {name: oof[oof["model"] == name].set_index("sample_index").sort_index()
               for name in ("lr_A", "lr_C")}
    target = (selected["target"] == "Persistent Degradation").to_numpy(dtype=int)
    table = targeting_table(target[valid], selected_values[valid],
                            indexed["lr_A"].loc[np.flatnonzero(valid), "calibrated_risk"].to_numpy(),
                            indexed["lr_C"].loc[np.flatnonzero(valid), "calibrated_risk"].to_numpy(),
                            quantitative)
    table["eligible_supports"] = int(valid.sum())
    table["eligible_positives"] = int(target[valid].sum())
    table["analysis_plan_sha256"] = plan_hash()
    table.to_csv(ROOT / "results/tables/carbon_targeting.csv", index=False)
    plot_targeting(table, ROOT / "results/figures/carbon_capture_curves.png")
    metadata = {"analysis_plan_sha256": plan_hash(), "quantitative": quantitative,
                "unit": "Mg C per native support" if quantitative else "mangrove-area proxy (ha)",
                "dataset": "Sanderman et al. 2018 update, year-2000 mangrove soil organic carbon 0-100 cm" if quantitative else "GMW 2020 mapped mangrove area",
                "raster_reference_year": 2000 if quantitative else 2020,
                "product_publication_year": 2018 if quantitative else 2020,
                "eligible_supports": int(valid.sum()), "eligible_positives": int(target[valid].sum()),
                "source_rasters": [str(path.relative_to(ROOT)) for path in source_paths] if quantitative else [],
                "access_issue": access_issue}
    (ROOT / "results/tables/carbon_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path = ROOT / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"] = [source for source in manifest["sources"] if source.get("role") != "carbon"]
    if quantitative:
        manifest["sources"].append({"source": "Mangrove soil organic carbon 0-100 cm, Sanderman et al. 2018",
                                    "provider": "Zenodo / published research", "scene_id": "record 7727569",
                                    "reference_date": "2000 raster epoch; product published 2018", "url": SANDERMAN_SOC_ZIP_URL,
                                    "local_path": str(archive_path.relative_to(ROOT)),
                                    "md5": "bfb5236878b0b60dee11ca9157e39c38",
                                    "access": "Open, CC BY-SA 4.0", "role": "carbon"})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return values, quantitative


def load_or_train_models(settings: dict, cohort: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model_dir = ROOT / "results/models"
    selected_path = ROOT / "results/dataset/model_cohort.parquet"
    oof_path = model_dir / "oof_predictions.parquet"
    if not force and selected_path.exists() and oof_path.exists() and (ROOT / "results/tables/primary_comparison.csv").exists():
        selected = pd.read_parquet(selected_path)
        oof = pd.read_parquet(oof_path)
        if set(oof["analysis_plan_sha256"]) == {plan_hash()}:
            fold_models = [joblib.load(path) for path in sorted((model_dir / "model_metadata").glob("lr_A_fold*.joblib"))]
            spatial = {"groups": selected["spatial_block"].to_numpy(),
                       "folds": selected["spatial_fold"].to_numpy(),
                       "splits": [(saved["train_indices"], saved["test_indices"]) for saved in fold_models]}
            return selected, oof, {"spatial": spatial}
    return model_stage(settings, cohort)


def summary_stage(settings: dict, summary: dict, selected: pd.DataFrame) -> None:
    tables = ROOT / "results/tables"
    metric = pd.read_csv(tables / "metrics_summary.csv")
    metric = metric[(metric["scope"] == "pooled") & (metric["budget_percent"] == 10)].set_index("model")
    primary = pd.read_csv(tables / "primary_comparison.csv").iloc[0]
    spectral = pd.read_csv(tables / "spectral_control.csv").iloc[0]
    early = pd.read_csv(tables / "early_warning.csv")
    incremental = pd.read_csv(tables / "incremental_warnings.csv")
    added = incremental[(incremental["budget_percent"] == 10) & (incremental["type"] == "HSI_added")]
    baseline_only = incremental[(incremental["budget_percent"] == 10) & (incremental["type"] == "baseline_only")]
    enhanced_alert = early[(early["model"] == "lr_C") & (early["budget_percent"] == 10)].iloc[0]
    baseline_alert = early[(early["model"] == "lr_A") & (early["budget_percent"] == 10)].iloc[0]
    carbon = pd.read_csv(tables / "carbon_targeting.csv")
    carbon_metadata = json.loads((tables / "carbon_metadata.json").read_text(encoding="utf-8"))
    bands = pd.read_csv(ROOT / "results/dataset/hsi_bands.csv")
    spatial = json.loads((ROOT / "results/models/model_metadata/spatial_cv.json").read_text(encoding="utf-8"))
    def score(model: str) -> str:
        return f"{metric.loc[model, 'pr_auc']:.3f}"
    carbon_rows = []
    for budget in (5, 10, 20):
        subset = carbon[carbon["budget_percent"] == budget].set_index("ranking")
        carbon_rows.append(f"| {budget}% | {subset.loc['HSI_risk_x_carbon', 'capture']:.3f} | "
                           f"{subset.loc['conventional_risk_x_carbon', 'capture']:.3f} | "
                           f"{subset.loc['carbon_only', 'capture']:.3f} | {subset.loc['random', 'capture']:.3f} |")
    cnn_file = tables / "cnn_seed_metrics.csv"
    if cnn_file.exists():
        cnn = pd.read_csv(cnn_file)
        cnn_lines = []
        for model, subset in cnn.groupby("model"):
            cnn_lines.append(f"- {model}: mean pooled spatial OOF PR-AUC {subset['pr_auc'].mean():.3f} ± "
                             f"{subset['pr_auc'].std(ddof=1):.3f} over {len(subset)} seeds.")
        cnn_text = "\n".join(cnn_lines)
    else:
        cnn_text = "Tier 2 CNN pending."
    h4_file = tables / "h4_813_ready.csv"
    h4_text = (f"LR F PR-AUC {pd.read_csv(h4_file).iloc[0]['pr_auc']:.3f}; spectral transferability only, "
               "not 20 m spatial equivalence.") if h4_file.exists() else "Tier 2 H4 pending."
    lead = (f"{added['lead_days'].median():.0f} days (IQR {added['lead_days'].quantile(.25):.0f}–"
            f"{added['lead_days'].quantile(.75):.0f})") if len(added) else "not estimable (zero added cases)"
    positive_capture = carbon[carbon["ranking"] == "HSI_risk_x_carbon"].set_index("budget_percent")["capture"]
    conventional_capture = carbon[carbon["ranking"] == "conventional_risk_x_carbon"].set_index("budget_percent")["capture"]
    improves_all = bool((positive_capture > conventional_capture).all())
    primary_supported = primary["ci_low"] > 0
    lines = [
        "# Research summary", "",
        f"**AOI:** {settings['aoi']}. **Scene:** {settings['scene_id']} (NASA EMIT L2A V001, approximately 60 m). "
        f"**t0:** {settings['t0']}. **Outcome:** (t0, 2025-12-27T09:28:43Z]. "
        "**Mask:** Global Mangrove Watch v3 2020, ≥75% mangrove per unique native GLT support.", "",
        f"**Frozen plan SHA-256:** `{plan_hash()}`. Labels and candidate selection preceded model fitting; "
        "correctness amendments are documented in the plan.", "",
        "## Cohort and validation", "",
        f"{summary['native_supports_after_hsi_qa']:,} HSI-QA native supports; {summary['already_detected_pre_t0']:,} "
        f"conventionally detected before t0 excluded; {summary['at_risk']:,} at-risk. "
        f"Future classes: {summary['persistent_positives']} Persistent Degradation "
        f"({summary['positive_patches']} spatial patches; {summary['loss_to_water']} loss-to-water-like, "
        f"{summary['canopy_dieback_like']} canopy-dieback-like), {summary['stable']:,} Stable, "
        f"{summary['transient']:,} Transient Stress excluded, {summary['unresolved']} Unresolved excluded. "
        f"Binary model cohort: {len(selected):,} supports.", "",
        f"EMIT: {len(bands)} original bands; {int(bands['retained'].sum())} retained across "
        f"{bands.loc[bands.retained, 'wavelength_nm'].min():.1f}–"
        f"{bands.loc[bands.retained, 'wavelength_nm'].max():.1f} nm (with QA/atmospheric gaps). "
        "HSI-as-S2 uses official measured ESA Sentinel-2A SRF v4.0, not Gaussian approximations.", "",
        f"Spatial validation: {len(spatial['folds'])} grouped outer folds, "
        f"{spatial['blocks']} event-merged groups, {spatial['block_size_m']} m base blocks and "
        f"{spatial['buffer_m']} m train/test buffer. Positive samples by fold: "
        f"{[fold['positive_samples'] for fold in spatial['folds']]}; positive patches by fold: "
        f"{[fold['positive_patches'] for fold in spatial['folds']]}. Internal spatial CV is not external validation.", "",
        "## Prediction and spectral value", "",
        f"The conventional LR A achieves PR-AUC **{score('lr_A')}**, above binary-cohort prevalence "
        f"{metric.loc['lr_A', 'prevalence']:.3f}; future degradation is therefore predictable above chance "
        "with pre-t0 information at this site. HSI-enhanced LR C achieves "
        f"**{score('lr_C')}**. The prespecified C−A difference is **{primary['pr_auc_difference']:+.3f}** "
        f"(paired spatial-block bootstrap 95% CI **{primary['ci_low']:+.3f} to {primary['ci_high']:+.3f}**, "
        f"{int(primary['valid_bootstraps'])} valid draws). "
        + ("The confirmatory improvement is supported." if primary_supported else
           "The CI includes zero: the confirmatory HSI improvement is **not established**."), "",
        f"HSI-only LR B PR-AUC {score('lr_B')}; official-SRF broadband control LR D "
        f"{score('lr_D')}; physiological HSI LR E {score('lr_E')}. C−D is "
        f"{spectral['pr_auc_difference']:+.3f} (95% CI {spectral['ci_low']:+.3f} to "
        f"{spectral['ci_high']:+.3f}); **full spectral resolution does not outperform the broadband control** "
        "in this experiment. Physiological features are interpretable but do not improve the LR result.", "",
        f"Random Forest, secondary: A {score('rf_A')}, C {score('rf_C')}, D {score('rf_D')}, "
        f"E {score('rf_E')} PR-AUC. Its stronger C versus A result is exploratory and does not "
        "retroactively change the prespecified primary LR conclusion. Held-out permutation importance "
        "is saved separately.", "",
        "## Actual t0 early warnings", "",
        f"At the training-only inner-OOF 10% alert threshold, LR C alerts {int(enhanced_alert['alerts'])} "
        f"supports, including **{int(enhanced_alert['true_future_alerts'])}** future persistent cases "
        f"across {int(enhanced_alert['positive_patches_alerted'])} patches, before real-time T_detect. "
        f"LR A alerts {int(baseline_alert['true_future_alerts'])} true cases. "
        f"**{len(added)}** true cases in {added['positive_patch_id'].nunique()} patches are C-only warnings; "
        f"{len(baseline_only)} true cases are A-only. C-only lead time to conventional confirmation: "
        f"**{lead}**. T_detect is the qualifying confirmation date, not retrospectively inferred onset.", "",
        "## Blue-carbon targeting", "",
        f"Layer: **{carbon_metadata['dataset']}**; "
        + ("quantitative estimated soil stock (Mg C per native support)." if carbon_metadata["quantitative"] else
           "mangrove-area carbon-relevance proxy, **not quantitative carbon stock**.") + " "
        f"Common targeting evaluation covers {carbon_metadata['eligible_supports']} binary-cohort supports "
        f"and {carbon_metadata['eligible_positives']} persistent cases. Rankings use calibrated spatial OOF risk; "
        "carbon never enters degradation-model training.", "",
        "| Budget | HSI risk × carbon | Conventional risk × carbon | Carbon only | Random mean |",
        "|---:|---:|---:|---:|---:|", *carbon_rows, "",
        "These are " + ("carbon-stock-weighted" if carbon_metadata["quantitative"] else "proxy-weighted") +
        " true-case capture fractions, not measured emissions. " +
        ("HSI ranking exceeds the conventional ranking at all three budgets." if improves_all else
         "HSI ranking does **not** exceed the conventional ranking at all three budgets."), "",
        "## Tier 2 and scope", "", cnn_text, "", h4_text, "",
        "No second site met the frozen minimum among accessible evaluated Arab-region scenes; no geographic transfer "
        "validation was executed. Sentinel-1 SAR, thermal, and GEDI are optional extensions and are not included "
        "in Tier 1 conclusions.", "",
        "## Limitations", "",
        "This is a single-scene demonstration PoC. Site selection used data/event feasibility and limits "
        "generalizability. Internal spatial CV is not external geographic validation; one HSI acquisition does "
        "not establish temporal generalization. Observational prediction does not establish causality. "
        "Operational Sentinel-2 labels are not field-confirmed ecological diagnoses; T_detect depends on valid "
        "cloud-free acquisitions. Residual tidal confounding, EMIT ~60 m spatial mixing, and limited independent "
        "positive events remain. CNN results are exploratory, especially with fewer than 100 positives per "
        "some spatial training/validation partitions. Future 813 ~20 m resolution differs from EMIT; H4 tests "
        "only spectral transferability.", "",
        "## Plain-language answer", "",
        ("**Meaningful early-warning value from HSI is supported in the prespecified test.**" if primary_supported else
         "**Meaningful incremental early-warning value from HSI is not established by the prespecified test.**") +
        f" At t0 the HSI model did uniquely alert {len(added)} later-confirmed cases, but this must be "
        "balanced against baseline-only alerts and the null primary CI.",
        ("**HSI improved blue-carbon resource prioritization at every tested budget.**" if improves_all else
         "**HSI did not consistently improve blue-carbon resource prioritization across tested budgets.**") +
        " The result is site-specific and observational.", ""]
    (ROOT / "results/SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def h4_stage(selected: pd.DataFrame, oof: pd.DataFrame, spatial: dict) -> None:
    table_path = ROOT / "results/tables/h4_813_ready.csv"
    if table_path.exists() and set(pd.read_csv(table_path)["analysis_plan_sha256"]) == {plan_hash()}:
        return
    target = (selected["target"] == "Persistent Degradation").to_numpy(dtype=int)
    predictions, _ = train_spatial_oof(selected, target, spatial, "F", "lr",
                                       ROOT / "results/models/model_metadata", plan_hash())
    predictions.to_parquet(ROOT / "results/models/h4_oof_predictions.parquet", index=False)
    result = {"model": "lr_F", "pr_auc": float(average_precision_score(target,
              predictions.sort_values("sample_index")["risk"])),
              "retained_813_bands": len(feature_columns(selected, "F")) - len(feature_columns(selected, "A")),
              "analysis_plan_sha256": plan_hash()}
    pd.DataFrame([result]).to_csv(table_path, index=False)


def cnn_stage(selected: pd.DataFrame, spatial: dict) -> None:
    from src.cnn import run_cnn
    from sklearn.metrics import roc_auc_score, brier_score_loss

    output_path = ROOT / "results/models/cnn_oof_predictions.parquet"
    if output_path.exists() and set(pd.read_parquet(output_path, columns=["analysis_plan_sha256"])["analysis_plan_sha256"]) == {plan_hash()}:
        predictions = pd.read_parquet(output_path)
    else:
        target = (selected["target"] == "Persistent Degradation").to_numpy(dtype=int)
        predictions = run_cnn(selected, target, spatial, plan_hash(),
                              ROOT / "results/models/cnn_training_logs")
        predictions.to_parquet(output_path, index=False)
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"]):
        group = group.sort_values("sample_index")
        rows.append({"model": model, "seed": seed,
                     "pr_auc": float(average_precision_score(group["target"], group["risk"])),
                     "roc_auc": float(roc_auc_score(group["target"], group["risk"])),
                     "brier": float(brier_score_loss(group["target"], group["risk"])),
                     "analysis_plan_sha256": plan_hash()})
    seed_metrics = pd.DataFrame(rows)
    seed_metrics.to_csv(ROOT / "results/tables/cnn_seed_metrics.csv", index=False)
    logs = pd.read_csv(ROOT / "results/models/cnn_training_logs/training_logs.csv")
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, group in logs.groupby("model"):
        curve = group.groupby("epoch")["validation_loss"].mean()
        axis.plot(curve.index, curve.to_numpy(), label=name)
    axis.set(xlabel="Epoch", ylabel="Inner spatial-validation BCE", title="Exploratory CNN training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(ROOT / "results/figures/cnn_training_curves.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    for index, (name, group) in enumerate(seed_metrics.groupby("model")):
        axis.scatter(np.full(len(group), index), group["pr_auc"], label=name)
    axis.set_xticks((0, 1), ["HSI only", "HSI + baseline"])
    axis.set(ylabel="Pooled spatial OOF PR-AUC", title="CNN seed variability")
    figure.tight_layout()
    figure.savefig(ROOT / "results/figures/cnn_seed_variability.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mangrove hyperspectral early-warning PoC")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage", choices=("feasibility", "dataset", "models", "carbon", "maps", "cnn", "813"))
    arguments = parser.parse_args()
    settings = config()
    if arguments.quick:
        quick(settings)
        return
    if arguments.prepare_data:
        prepare_data(settings)
        return
    supports, items, observations, dates, summary, labels = feasibility(settings)
    print(json.dumps(summary, indent=2))
    if arguments.stage == "feasibility":
        return
    if summary["persistent_positives"] < 30 or summary["positive_patches"] < 3 or summary["stable"] + summary["persistent_positives"] < 300:
        raise RuntimeError("Frozen feasibility minimum not met; do not fit models or change thresholds")
    cohort_path = ROOT / "results/dataset/cohort.parquet"
    if cohort_path.exists():
        cohort = pd.read_parquet(cohort_path)
        if set(cohort["analysis_plan_sha256"]) != {plan_hash()}:
            cohort = dataset_stage(settings, supports, items, observations, dates, summary, labels)
    else:
        cohort = dataset_stage(settings, supports, items, observations, dates, summary, labels)
    if arguments.stage == "dataset":
        print(f"Dataset ready: {len(cohort)} QA-valid native supports")
        return
    selected, oof, details = load_or_train_models(settings, cohort, force=arguments.stage == "models")
    if arguments.stage == "models":
        print(f"Model evaluation ready: {len(selected)} binary-cohort supports")
        return
    if arguments.stage in ("cnn", "813"):
        if not (ROOT / "results/SUMMARY.md").exists():
            raise RuntimeError("Tier 1 must finish before Tier 2")
        if arguments.stage == "cnn":
            cnn_stage(selected, details["spatial"])
        else:
            h4_stage(selected, oof, details["spatial"])
        summary_stage(settings, summary, selected)
        return
    carbon_values, quantitative = carbon_stage(supports, cohort, selected, oof)
    if arguments.stage == "carbon":
        print("Carbon targeting complete")
        return
    mapping_stage(settings, supports, cohort, selected, oof, carbon_values, quantitative)
    summary_stage(settings, summary, selected)
    if arguments.stage == "maps":
        print("Tier 1 maps and summary complete")
        return
    cnn_stage(selected, details["spatial"])
    h4_stage(selected, oof, details["spatial"])
    summary_stage(settings, summary, selected)
    if (ROOT / "results/tables/robustness_paired.csv").exists():
        from followup import summary_stage as followup_summary_stage

        followup_summary_stage()
    print("Tier 1 and executable Tier 2 stages complete")


if __name__ == "__main__":
    main()
