> **Public repository note:** This compact submission keeps the files required to reproduce and verify the reported headline results. Some paths referenced below belong to the full internal analysis pipeline and are intentionally not included here.
# FROZEN FINAL RESULTS â€” 2026-10-08T15:18:37Z

Frozen at 2026-10-08T15:18:37Z UTC. **No further analyses after this point.** Frozen primary-plan SHA-256 (plan text before the post-audit amendments): `40d1e7dc8764daf117457b972b5a1061035dbebeaec51acfe36b5b313dca8a32`. Plan file SHA-256 at freeze, including Amendments 1â€“4: `c4cbd47edb8cbe7c08835a56fbcc16d5592243283f9c58e4313e89f0795e2434`. Each result below carries its `analysis_type`. The independent audit is in `diagnostics/INDEPENDENT_AUDIT.md`.

## F1. PRIMARY (unchanged; independently reproduced)
Mahajanga EMIT, 2,220 binary supports (172 Persistent, 2,048 Stable), 97 patches, 135 event-merged CV groups.
- LR A PR-AUC **0.267**, LR C **0.274**, Câˆ’A **+0.007 [âˆ’0.037, +0.061]**: **not established**.
- Câˆ’D âˆ’0.011 [âˆ’0.042, +0.028].
- The audit re-fitted A/C/D from the saved folds and reproduced Câˆ’A exactly (LR C risk jitter â‰¤0.0025, alerts identical).

## F2. Correction: real-time detection date (Amendment 1, outputs `POSTHOC_SENSITIVITY`)
`T_detect_realtime` (`dataset/detection_dates_realtime.csv`) is the first post-t0 date the frozen confirmation rule holds for any onset. The labels are asserted identical for all 10,366 supports, the original `T_detect` is kept, and all frozen files are unchanged. Old vs new: `tables/tdetect_correction_comparison.csv`.

| Quantity | Stored T_detect | T_detect_realtime |
|---|---:|---:|
| C-only true alerts, 10% budget: median lead (IQR), days | 672 (374â€“717) | **82 (62â€“319)** |
| LR C true alerts, 10%: median lead, days | 617 | 82 |
| Positives with <90 d follow-up after detection | 81 | 26 |
| Positives with <3 valid obs after detection | 69 | 25 |
| S2 guard positives; LR Câˆ’A | 91; âˆ’0.003 [âˆ’0.029, +0.106] | 146; +0.033 [âˆ’0.012, +0.094] |
| S2 guard RF Câˆ’A | âˆ’0.004 [âˆ’0.125, +0.092] | +0.071 [âˆ’0.011, +0.156] |
| S3 dieback+guard positives; LR Câˆ’A | 57; âˆ’0.012 [âˆ’0.105, +0.059] | 69; âˆ’0.014 [âˆ’0.121, +0.073] |
| S3 RF Câˆ’A | âˆ’0.000 [âˆ’0.065, +0.058] | +0.038 [âˆ’0.014, +0.131] |

Every guard CI still includes zero. 77/172 positives were already confirmed in real time *before* their stored episode onset, during an earlier, non-persistent episode.

## F3. Cluster-robust incremental warnings (Amendment 3, `POSTHOC_SENSITIVITY`)
Paired 1,000-draw spatial-block bootstrap over 135 groups, plus a patch-level exact sign test (`tables/incremental_warning_inference.csv`, 60 comparisons). The frozen inner-OOF thresholds produce **unequal realized alert volumes** (audit M2), so each comparison is repeated at an equal realized budget.

| LR, primary cohort | Alerts C / A | C-only / A-only TP | Recall Câˆ’A [95% CI] | Precision Câˆ’A [95% CI] | Patches C>A / A>C (sign p) |
|---|---:|---:|---:|---:|---:|
| 10%, frozen threshold | 249 / 227 | 22 / 4 | **+0.105 [+0.031, +0.186]** | +0.047 [âˆ’0.007, +0.106] | 16 / 3 (0.004) |
| 10%, equal budget | 222 / 222 | 15 / 11 | +0.023 [âˆ’0.066, +0.099] | +0.018 [âˆ’0.055, +0.087] | 10 / 9 (1.00) |
| 5%, frozen threshold | 130 / 112 | 15 / 11 | +0.023 [âˆ’0.055, +0.096] | âˆ’0.017 [âˆ’0.120, +0.085] | 11 / 9 (0.82) |
| 5%, equal budget | 111 / 111 | 11 / 11 | 0.000 [âˆ’0.057, +0.064] | 0.000 [âˆ’0.095, +0.112] | 9 / 9 (1.00) |
| 20%, frozen threshold | 457 / 420 | 15 / 3 | +0.070 [+0.012, +0.149] | +0.007 [âˆ’0.021, +0.038] | 13 / 2 (0.007) |
| 20%, equal budget | 444 / 444 | 10 / 5 | +0.029 [âˆ’0.022, +0.122] | +0.011 [âˆ’0.013, +0.042] | 8 / 4 (0.39) |

- LR Câˆ’D at 10%:
  - Frozen threshold: recall +0.052 [+0.010, +0.095], precision âˆ’0.006 [âˆ’0.035, +0.039].
  - Equal budget: recall âˆ’0.017 [âˆ’0.085, +0.039].
- RF Câˆ’A and Câˆ’D: every recall and precision CI includes zero. One exception is borderline: at RF Câˆ’A 5% equal budget the sign test gives p=0.024, but the recall CI is [âˆ’0.094, +0.237].
- Sensitivity cohorts:
  - S1 (Transient as negative), LR 10%, equal budget: recall +0.041 [+0.008, +0.104].
  - S2 with the stored-date guard: all CIs include zero.
  - S2 with the realtime guard, LR frozen threshold: recall +0.075 [+0.022, +0.135] at 10% and +0.062 [+0.018, +0.129] at 20%. At equal budgets the LR CIs include zero.
- **Interpretation:** at the frozen thresholds, HSI-added true positives exceed baseline-only ones, but most of the gain comes from C issuing more alerts. At equal alert volume the gain is small and not established. This is a post-hoc analysis with many comparisons, so isolated CIs that exclude zero are not confirmatory.

## F4. Tides (Amendment 4, `POSTHOC_SENSITIVITY`)
GSFC GOT4.10c (pyTMD) at the support centroid, used in place of EOT20 (2.3 GB; the substitution was logged before computing). Modelled range at the 300 S2 acquisitions: âˆ’2.01 to +0.87 m. **55 of 96 loss-to-water positives have every qualifying structural observation in the top tide quartile.** The median tide on structural observations is +0.40 m, against âˆ’0.69 m across all post-t0 acquisitions (one-sided Mannâ€“Whitney p = 4.5Ã—10â»â¹). Scene-median water fraction does not track tide (Spearman 0.06, p = 0.31). Loss-to-water positives are therefore substantially tide-confounded. The dieback-only analyses (PRESPECIFIED_SECONDARY: LR Câˆ’A âˆ’0.006 [âˆ’0.064, +0.046], RF +0.024 [âˆ’0.004, +0.095]) are the tide-robust endpoint, and they do not establish an HSI gain. Labels were not changed. Files: `tables/tide_diagnostic.json`, `tables/tide_loss_to_water_diagnostic.csv`.

## F5. Arab-region EMIT (Amendments 2/2b)
- **Search:** `EMITL2ARFL` v001, 2022-08-01 to 2025-10-01, over 10 regions. There were 2,763 region-granule hits; overlapping regions share some granules. 417 met both eligibility rules (complete 24-month horizon by 2026-10-01, and GMW-2020 mangroves in the footprint). Every hit is logged in `feasibility/arab_emit_candidates.csv`.
  - Barr Al Hikman (45 hits) and Mahout (48 hits) have **no GMW v3 2020 mangrove tiles**, so all their scenes are ineligible under the frozen mask rule.
- **Processed scenes** (`replication/arab/processed_scenes.csv`):
  - Abu Dhabi 2023-06-25 (rank 1; 7,355 supports) and Saudi Red Sea 2024-07-22 (rank 7): **not processed**. Sentinel-2 processing would have taken about 24 h and about 10 h, which the deadline and memory did not allow (Amendment 2b, logged before any new label was seen).
  - Umm Al Quwain 2023-07-27: 1,976 supports, **3 positives**, so `APPLICATION_DEMO` (not analysed further).
  - Saudi Red Sea 2023-03-30: 1,255 supports, 24 detected pre-t0, **19 positives** (7 dieback-like, 12 loss-to-water) in 15 patches, 824 Stable, 388 Transient. The decision rule gives **`EXTERNAL_TEST`**.
- **EXTERNAL_TEST** of Madagascar-trained models on the Saudi Red Sea 2023-03-30 scene: 843 binary supports, prevalence 0.023, 45 one-km event-merged blocks.

| Model | PR-AUC [95% CI] | ROC-AUC |
|---|---:|---:|
| LR A | 0.136 [0.015, 0.432] | 0.675 |
| LR C | 0.055 [0.017, 0.184] | 0.662 |
| LR D | 0.086 [0.015, 0.344] | 0.631 |
| LR F (813 H4) | 0.054 [0.017, 0.185] | 0.650 |
| RF A / C / D / F | 0.046 / 0.042 / 0.017 / 0.032 | 0.528 / 0.479 / 0.371 / 0.475 |

  - Paired differences:
    - LR: Câˆ’A âˆ’0.081 [âˆ’0.275, +0.025], Câˆ’D âˆ’0.030 [âˆ’0.190, +0.012], Fâˆ’A âˆ’0.082 [âˆ’0.273, +0.026].
    - RF: Câˆ’A âˆ’0.004 [âˆ’0.083, +0.131], Câˆ’D +0.024 [+0.000, +0.166], Fâˆ’A âˆ’0.013 [âˆ’0.083, +0.017].
  - The LR conventional model transfers weakly above prevalence. **HSI does not improve transfer.** RF transfers at or below chance; its Câˆ’D lower bound of 0.000 is relative to a worse-than-chance D (ROC 0.37) and is not evidence of HSI value.
  - Risk maps of the Madagascar models at this site are marked **NOT VALIDATED** (`replication/arab/*_NOT_VALIDATED.png`).
  - This is a small external test: 19 positives, one scene.

## F6. Final conclusion
Pre-t0 conventional information predicts the frozen endpoint above prevalence at Mahajanga, and it transfers weakly to one Saudi Red Sea scene. None of the following establishes an incremental hyperspectral early-warning advantage:
- the prespecified LR test;
- the dieback-only (tide-robust) analysis;
- the follow-up guards (stored or realtime);
- the equal-budget alert comparisons;
- carbon targeting;
- the Arab external test.

The real operational lead of the HSI-only alerts is about **82 days**, not 672. Loss-to-water labels at Mahajanga are largely tidal. The 813-band (400â€“1700 nm) subset performs comparably to full HSI internally (LR F 0.242) and no better externally.

Tests: 45 passing.

---

# Previous audited summary (pre-freeze; kept verbatim; superseded where F1â€“F6 differ, notably Â§5 replication status and the 672-day lead)

# Madagascar PoC: audited research summary

Output classification: `APPLICATION_DEMO` synthesis. Individual analyses retain their own `analysis_type` in the linked result tables. The frozen primary plan and result files were not changed.

## 1. PRIMARY â€” frozen

Mahajanga, Madagascar; NASA EMIT L2A scene `EMIT_L2A_RFL_001_20231227T092843_2336106_021`; t0 2023-12-27T09:28:43Z; full outcome horizon through 2025-12-27T09:28:43Z. GMW v3 2020 mask, 75% native-support mangrove fraction. There are 2,220 binary modeling supports (172 persistent positives, 2,048 Stable), with 97 positive patches.

Frozen LR A PR-AUC **0.267**; LR C **0.274**. Câˆ’A **+0.007** with paired spatial-block bootstrap 95% CI **[-0.037, +0.061]**. The primary HSI gain is not established. Full HSI versus official-SRF broadband control Câˆ’D is -0.011 [-0.042, +0.028].
The primary artifacts are `tables/primary_comparison.csv`, `tables/spectral_control.csv` and `models/oof_predictions.parquet`. The analysis-plan SHA-256 remains `40d1e7dc8764daf117457b972b5a1061035dbebeaec51acfe36b5b313dca8a32`.

## 2. PRESPECIFIED_SECONDARY

The frozen plan explicitly named dieback-only analysis before modeling. With the same spatial-fold assignment and 60 m buffer, dieback versus Stable gives:

- LR Câˆ’A -0.006 [-0.064, +0.046]; Câˆ’D +0.016 [-0.016, +0.056]; Eâˆ’A -0.012 [-0.060, +0.033]. RF Câˆ’A +0.024 [-0.004, +0.095]; Câˆ’D +0.030 [-0.020, +0.142]; Eâˆ’A -0.015 [-0.037, +0.017]. None establishes a dieback-specific HSI advantage.

Full-endpoint RF Câˆ’A +0.058 [-0.042, +0.156]; Câˆ’D +0.022 [-0.026, +0.084]; Eâˆ’A -0.008 [-0.057, +0.035]. The RF Câˆ’A point gain remains positive but its CI includes zero. All LR Aâ€“F, RF A/C/D/E and three-seed CNN PR-AUC, ROC-AUC, Brier and bootstrap intervals appear in `tables/full_metrics_audit.csv`; calibrated OOF reliability curves are in `figures/full_calibration_curves.png`.

The saved inner-OOF thresholds, held-out fold membership, predicted risks, unique native supports and later outcome were independently rechecked. True-positive 10% budget paired counts: C-only **22** (17 patches), A-only **4**, both **61**, neither **85**; exact McNemar p=0.0005335. Pixel correlation makes this p-value exploratory. The primary PR-AUC CI remains the governing inference. Counts at 5% and 20% are in `tables/hsi_added_warning_paired.csv`. For C-only warnings, the median lead from the *first real-time* conventional confirmation is 82 days (IQR 62â€“319); 15 of these 22 stored T_detect dates were later. The previously reported 672-day median based on stored dates overstates the operational lead.

The carbon layer converts a published year-2000 0â€“100 cm soil-carbon density model (Mg C/ha) to an estimated Mg C per native support using the 2020 mangrove fraction. It is a quantitative contextual stock estimate, not an observed 2023 stock or measured emissions. Carbon ranking evaluation covers 2,116 stock-covered binary supports and 129 positive cases; it does not include the 43 positives lacking stock coverage. Paired spatial-block capture Câˆ’A CIs:

| Budget | C capture | A capture | Câˆ’A [95% CI] |
|---:|---:|---:|---:|
| 5% | 0.252 | 0.238 | +0.014 [-0.118, +0.076] |
| 10% | 0.421 | 0.469 | -0.048 [-0.141, +0.085] |
| 20% | 0.734 | 0.649 | +0.085 [-0.014, +0.142] |

All paired Câˆ’A carbon CIs include zero. The ranking advantage is not established at any budget. Prespecified water-anomaly-floor check: floor 0.20: 168 positives, 7640 transient, 75 changed class; floor 0.30: 172 positives, 7726 transient, 15 changed class.

## 3. LABEL DIAGNOSTICS â€” POSTHOC_SENSITIVITY

Cohort reconciliation: **10,366** original supports = **431** pre-t0 detections + **172** Persistent Degradation (76 canopy dieback-like, 96 loss-to-water-like) + **7,710** Transient Stress + **2,048** Stable + **5** Unresolved; no other exclusions. Transient Stress is **77.6% of 9,935 at-risk supports** (74.4% of original supports). Only the 172 Persistent and 2,048 Stable supports entered the frozen binary model.

Among 172 positives, **81** have <90 days remaining after T_detect; **69** have <3 later valid observations; **81** meet either condition and **69** meet both. This is a serious end-of-window persistence limitation: 62 of 96 loss-to-water-like and 19 of 76 dieback-like positives have <90 days after their stored detection. The frozen labels and stored dates have not been altered. The post-detection guard uses those stored dates.

Independent prefix-by-prefix reconstruction finds **109 of 172** stored T_detect dates later than the first time the real-time two-observation/60% rule was satisfied. All 172 stored onset and detection dates correspond to valid degraded observations. The original detector chooses an episode meeting the horizon-end persistence fraction before assigning T_detect, which can miss an earlier real-time confirmation. This is an implementation correctness finding affecting operational lead-time interpretation, while the frozen labels and model PR-AUC results remain unchanged. Using first real-time detection, 26 positives have <90 days until the horizon; the 81-count guard is based on the stored episode date. `tables/label_detection_audit.csv` records every discrepancy.

Fixed-seed NDVI/MNDWI examples, T_onset/T_detect histograms and support-level follow-up counts are in `figures/example_event_timeseries.png`, `figures/Tdetect_distribution.png`, and `tables/post_detection_followup_diagnostic.csv`. No reproducible tide model/harmonics were available locally, so no tide heights were inferred and residual tidal confounding remains.

## 4. POSTHOC_SENSITIVITY

S1 treats resolved Transient Stress as negative. S2 retains positives only with â‰¥90 days and â‰¥3 valid observations after T_detect. S3 applies that guard to dieback-only positives. These definitions change only sensitivity cohorts, never frozen primary labels. S1 uses the same pre-t0 variogram and grouped-buffered CV principles, yielding 2 km blocks; S2/S3 inherit the primary 1 km folds. PR-AUC paired Câˆ’A results:

| Scenario | Model | n | Positives | Patches | Câˆ’A [95% CI] |
|---|---|---:|---:|---:|---:|
| S1_transient_as_negative | LR | 9,930 | 172 | 97 | +0.006 [-0.001, +0.016] |
| S1_transient_as_negative | RF | 9,930 | 172 | 97 | +0.005 [-0.005, +0.041] |
| S2_post_detection_guard | LR | 2,139 | 91 | 45 | -0.003 [-0.029, +0.106] |
| S2_post_detection_guard | RF | 2,139 | 91 | 45 | -0.004 [-0.125, +0.092] |
| S3_dieback_and_guard | LR | 2,105 | 57 | 34 | -0.012 [-0.105, +0.059] |
| S3_dieback_and_guard | RF | 2,105 | 57 | 34 | -0.000 [-0.065, +0.058] |

S1 does not yield a CI excluding zero. S2 and S3 point estimates are non-positive for both LR and RF. The apparent HSI gain is not robust to the follow-up guard.

## 5. REPLICATION STATUS

Abu Dhabi EnMAP L2A scene `ENMAP01-____L2A-DT0000004978_20221101T073017Z_002_V010502_20251109T142510Z` remains provider-disabled for maintenance and is absent from `data/raw/enmap/`. No replication model was fitted. Required files and the unchanged 24-month replication window are specified in `replication/ABU_DHABI_ENMAP_REQUIRED.md`.

## 6. 813 READINESS â€” PRESPECIFIED_SECONDARY

The 400â€“1700 nm H4 subset retains 154 EMIT bands and yields LR F PR-AUC 0.242. It examines spectral transferability only; EMIT ~60 m observations do not reproduce the future 813 ~20 m support. No Arab-site geographic transfer was performed.

## 7. LIMITATIONS

This is one feasibility-selected site and one hyperspectral acquisition. Spatial CV does not establish external geographic or temporal generalization. Native supports and positive patches are spatially correlated. Labels are operational satellite detections, not field-confirmed ecological diagnoses; T_detect is constrained by usable satellite acquisitions. Late-window structural positives, a large Transient Stress class, residual tides, EMIT spatial mixing, limited carbon coverage and unstable CNN results constrain interpretation. Observational prediction does not establish causality.

**Scientifically defensible conclusion:** Pre-t0 information predicts the frozen endpoint above prevalence at this site, and HSI generated some unique t0 alerts. The prespecified LR comparison, dieback-only analysis, RF spatial CIs, and stricter follow-up checks do not establish a reliable incremental HSI early-warning advantage. Quantitative carbon targeting does not show a robust C-over-A advantage at 5%, 10% or 20% budget. Abu Dhabi replication remains necessary before broader claims.


---

# POST-FREEZE ADDENDUM (defined and logged 2026-10-08; reported regardless of direction)

`POSTHOC_SENSITIVITY`. This addendum makes no HSI claims and changes no frozen section above or any frozen label, threshold, configuration, fold or file. Outputs are in `addendum/`.

## A1. Claim 1 robustness (LR, pooled OOF, 1,000-draw block-bootstrap 95% CIs; prevalence 0.077)

| Scenario | Groups | LR A PR-AUC [CI] | LR A ROC-AUC [CI] | LR C PR-AUC [CI] | LR C ROC-AUC [CI] | Câˆ’A PR-AUC [CI] |
|---|---:|---|---|---|---|---|
| Primary (frozen; 8-connected patches, 60 m buffer) | 135 | 0.267 [0.122, 0.403] | 0.797 [0.697, 0.860] | 0.274 [0.107, 0.450] | 0.805 [0.702, 0.870] | +0.007 [âˆ’0.037, +0.061] |
| (a) Events merged across a 1-pixel gap (55 events; folds reassigned) | 131 | **0.187 [0.094, 0.278]** | 0.746 [0.652, 0.804] | **0.156 [0.084, 0.235]** | 0.708 [0.625, 0.786] | âˆ’0.030 [âˆ’0.075, +0.012] |
| (b) Frozen folds, 120 m buffer (73/74/62/83/85 training supports removed per fold) | 135 | 0.256 [0.107, 0.392] | 0.799 [0.704, 0.859] | 0.274 [0.106, 0.451] | 0.803 [0.702, 0.868] | +0.018 [âˆ’0.019, +0.072] |

A wider buffer barely changes the results. Merging near-adjacent events lowers PR-AUC by about a third for A and nearly half for C. Part of the frozen primary performance therefore reflects nearby events that share information across folds. Both models remain above prevalence (both lower CI bounds > 0.077), so Claim 1 holds in a weaker form. The Câˆ’A difference still includes zero, and its point estimate turns negative.

## A2. Claim 3: carbon-weighted capture (2,116 carbon-covered supports, 129 positives; paired block bootstrap)

| Budget | A Ã— carbon | C Ã— carbon | Carbon only | Random (mean) | A âˆ’ carbon-only | C âˆ’ carbon-only | A âˆ’ random | C âˆ’ random | C âˆ’ A |
|---:|---|---|---|---|---|---|---|---|---|
| 5% | 0.238 [0.113, 0.387] | 0.252 [0.071, 0.398] | 0.233 [0.057, 0.431] | 0.050 | +0.005 [âˆ’0.234, +0.251] | +0.019 [âˆ’0.280, +0.296] | **+0.188 [+0.062, +0.335]** | **+0.202 [+0.025, +0.349]** | +0.014 [âˆ’0.109, +0.071] |
| 10% | 0.469 [0.255, 0.595] | 0.421 [0.224, 0.604] | 0.316 [0.094, 0.554] | 0.099 | +0.153 [âˆ’0.124, +0.397] | +0.105 [âˆ’0.214, +0.424] | **+0.370 [+0.156, +0.495]** | **+0.322 [+0.122, +0.503]** | âˆ’0.048 [âˆ’0.134, +0.086] |
| 20% | 0.649 [0.429, 0.784] | 0.734 [0.496, 0.835] | 0.410 [0.176, 0.698] | 0.201 | +0.239 [âˆ’0.049, +0.483] | +0.324 [âˆ’0.035, +0.568] | **+0.448 [+0.230, +0.587]** | **+0.533 [+0.293, +0.633]** | +0.085 [âˆ’0.028, +0.143] |

Point estimates equal the frozen `carbon_targeting.csv` (asserted). Risk Ã— carbon rankings beat random at every budget. They do not reliably beat ranking by carbon alone, and C does not beat A. The Câˆ’A CIs differ slightly from the frozen ones (e.g. 10%: [âˆ’0.134, +0.086] vs [âˆ’0.141, +0.085]) only because this paired procedure draws a different random-number stream.

## A3. System-level lead time (`T_detect_realtime`, frozen 10% threshold)

| Model | Alerts | True alerts (n) | Patches | Median lead (IQR), days |
|---|---:|---:|---:|---|
| LR A (conventional) | 227 | 65 | 41 | 82 (62â€“312) |
| LR C (HSI-enhanced) | 249 | 83 | 48 | 82 (62â€“304) |

As a system, both models warn a median of about 82 days before real-time Sentinel-2 confirmation. The lead time does not depend on HSI.
