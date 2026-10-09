from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling


def targeting_table(target: np.ndarray, carbon: np.ndarray, conventional_risk: np.ndarray,
                    enhanced_risk: np.ndarray, quantitative: bool, random_draws: int = 1000,
                    seed: int = 813) -> pd.DataFrame:
    target = np.asarray(target, dtype=bool)
    carbon = np.asarray(carbon, dtype=float)
    conventional_risk = np.asarray(conventional_risk, dtype=float)
    enhanced_risk = np.asarray(enhanced_risk, dtype=float)
    if not (len(target) == len(carbon) == len(conventional_risk) == len(enhanced_risk)):
        raise ValueError("Targeting arrays differ in length")
    if np.any(~np.isfinite(carbon)) or np.any(carbon < 0):
        raise ValueError("Carbon layer must be complete and nonnegative")
    total = float(carbon[target].sum())
    if total <= 0:
        raise ValueError("No future-degraded carbon relevance to capture")
    rankings = {"HSI_risk_x_carbon": enhanced_risk * carbon,
                "conventional_risk_x_carbon": conventional_risk * carbon,
                "carbon_only": carbon}
    generator = np.random.default_rng(seed)
    rows = []
    for budget in (0.05, 0.10, 0.20):
        selected_count = max(1, int(np.ceil(budget * len(target))))
        for method, score in rankings.items():
            selected = np.argpartition(score, -selected_count)[-selected_count:]
            captured = float(carbon[selected][target[selected]].sum() / total)
            rows.append({"budget_percent": int(budget * 100), "ranking": method,
                         "capture": captured, "carbon_basis": "Mg_C" if quantitative else "proxy"})
        random_capture = np.empty(random_draws)
        for draw in range(random_draws):
            selected = generator.choice(len(target), selected_count, replace=False)
            random_capture[draw] = carbon[selected][target[selected]].sum() / total
        rows.append({"budget_percent": int(budget * 100), "ranking": "random",
                     "capture": float(random_capture.mean()), "random_sd": float(random_capture.std()),
                     "random_q025": float(np.quantile(random_capture, 0.025)),
                     "random_q975": float(np.quantile(random_capture, 0.975)),
                     "carbon_basis": "Mg_C" if quantitative else "proxy"})
    return pd.DataFrame(rows)


def plot_targeting(table: pd.DataFrame, destination) -> None:
    from pathlib import Path

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, group in table.groupby("ranking"):
        group = group.sort_values("budget_percent")
        axis.plot(group["budget_percent"], group["capture"], marker="o", label=method)
    axis.set(xlabel="Intervention budget (% native supports)",
             ylabel="Carbon-weighted capture" if table["carbon_basis"].iloc[0] == "Mg_C" else "Proxy-weighted capture",
             title="OOF intervention targeting")
    axis.set_xticks((5, 10, 20))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def support_area_hectares(supports: dict) -> np.ndarray:
    transform = supports["transform"]
    half_height = abs(transform.e) / 2
    width_radians = np.deg2rad(abs(transform.a))
    latitude = np.asarray(supports["lat"], dtype=float)
    earth_radius_m = 6371008.8
    area_m2 = earth_radius_m ** 2 * width_radians * (
        np.sin(np.deg2rad(latitude + half_height)) - np.sin(np.deg2rad(latitude - half_height)))
    return area_m2 / 10000


def align_soil_carbon_stock(raster_path, supports: dict) -> np.ndarray:
    shape = tuple(supports["ortho_shape"])
    with rasterio.open(raster_path) as source:
        with WarpedVRT(source, crs="EPSG:4326", transform=supports["transform"],
                       width=shape[1], height=shape[0], resampling=Resampling.average,
                       src_nodata=source.nodata if source.nodata is not None else 0,
                       nodata=np.nan) as aligned:
            density = aligned.read(1)
    density = density[np.asarray(supports["row"]), np.asarray(supports["col"])].astype(float)
    density[(density <= 0) | ~np.isfinite(density)] = np.nan
    return density * support_area_hectares(supports) * np.asarray(supports["mangrove_fraction"])


def spatial_capture_bootstrap(target: np.ndarray, carbon: np.ndarray, risk_a: np.ndarray,
                              risk_c: np.ndarray, groups: np.ndarray, repeats: int = 1000,
                              seed: int = 813) -> pd.DataFrame:
    target = np.asarray(target, dtype=bool)
    carbon = np.asarray(carbon, dtype=float)
    risk_a = np.asarray(risk_a, dtype=float)
    risk_c = np.asarray(risk_c, dtype=float)
    groups = np.asarray(groups)
    if not (len(target) == len(carbon) == len(risk_a) == len(risk_c) == len(groups)):
        raise ValueError("Bootstrap arrays differ in length")
    if not np.isfinite(carbon).all() or np.any(carbon <= 0):
        raise ValueError("Carbon stock must be positive and finite")
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    results = {budget: {method: [] for method in ("HSI_risk_x_carbon", "conventional_risk_x_carbon",
                                                   "carbon_only", "random", "C_minus_A")}
               for budget in (5, 10, 20)}
    for _ in range(repeats):
        sampled = generator.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        truth = target[indices]
        stock = carbon[indices]
        denominator = stock[truth].sum()
        if denominator <= 0:
            continue
        scores = {"HSI_risk_x_carbon": risk_c[indices] * stock,
                  "conventional_risk_x_carbon": risk_a[indices] * stock,
                  "carbon_only": stock}
        for budget in (5, 10, 20):
            count = max(1, int(np.ceil(len(indices) * budget / 100)))
            values = {}
            for method, ranking in scores.items():
                chosen = np.argpartition(ranking, -count)[-count:]
                values[method] = float(stock[chosen][truth[chosen]].sum() / denominator)
                results[budget][method].append(values[method])
            chosen = generator.choice(len(indices), count, replace=False)
            results[budget]["random"].append(float(stock[chosen][truth[chosen]].sum() / denominator))
            results[budget]["C_minus_A"].append(values["HSI_risk_x_carbon"] - values["conventional_risk_x_carbon"])
    rows = []
    for budget, methods in results.items():
        for method, values in methods.items():
            rows.append({"budget_percent": budget, "ranking": method,
                         "ci_low": float(np.quantile(values, .025)),
                         "ci_high": float(np.quantile(values, .975)),
                         "bootstrap_mean": float(np.mean(values)), "valid_bootstraps": len(values)})
    return pd.DataFrame(rows)
