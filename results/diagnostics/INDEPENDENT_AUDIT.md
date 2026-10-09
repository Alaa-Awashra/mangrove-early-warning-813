# Independent code audit — Madagascar EMIT PoC

Date: 2026-10-08. Auditor: Claude Code (Opus 5.5), independent of the agent that wrote the code.
Mode: **read-only**. No repository file other than this report was created or changed. Checks ran from scratch
scripts outside the repo against the cached data, saved models and result tables.
Frozen plan SHA-256 on disk: `40d1e7dc8764daf117457b972b5a1061035dbebeaec51acfe36b5b313dca8a32` (matches).

Classification: **BUG (affects results)**, **BUG (cosmetic)**, **RISK** (a design, spec or interpretation weakness;
not an implementation error against the frozen plan), **OK**.

## Headline

1. **Primary result reproduces exactly.** I re-fit LR A/C/D from `model_cohort.parquet` with the saved buffered
   splits. A and D match bit-for-bit. C matches to a maximum of 0.0025 in risk, with identical alerts.
   Recomputed C−A is **+0.0068 [−0.0366, +0.0607]** (1,000/1,000 valid draws). All 44 tests pass.
2. **One result-affecting bug is confirmed: stored `T_detect` is not the first real-time conventional detection.**
   The bug affects lead-time outputs only, not labels or PR-AUC.
3. **Biggest interpretation risk: alert budgets are not equal in realized terms.** At the nominal 10% budget,
   LR C issues 249 alerts and LR A issues 227. Part of the "C-only 22 vs A-only 4" contrast comes from C alerting more.
4. **Biggest label risk: an ambiguity in the persistence rule.** 122 of 172 positives have an episode onset later
   than their first degraded post-t0 observation. All 122 would be Transient Stress under a strict reading of
   "60% … from the first degraded observation". The code follows the plan's own T_onset clause ("first degraded date in the
   eventual qualifying episode"), so this is a RISK. It is not a bug. The frozen labels must not change.
5. **Operational risk for the next tasks: the plan hash covers the whole file.** `run.py`/`followup.py` hash the whole
   plan file and use the hash as a cache key. Appending any amendment changes the hash. Rerunning those scripts would
   then recompute or overwrite the frozen outputs (`run.py`) or fail its asserts (`followup.py:39`). New analyses must
   run from separate scripts that leave the frozen artifacts alone.

## Findings

### Temporal and cohort integrity

| # | Item | Class | Evidence |
|---|---|---|---|
| T1 | S2 near-t0 band window | OK | `src/sentinel.py:212` selects [t0−60 d, t0]. The latest valid date wins (`:290-294`), which is equivalent to the plan's "30 d, else 60 d" rule. Actual dates run 2023-10-30 to 2023-12-24 (t0 = 2023-12-27). `require_pre_t0` is enforced (`src/features.py:77`). |
| T2 | F1 index state / F2 history | OK | `src/features.py:31-71` uses only dates ≤ t0, with history in [t0−24 m, t0]. Age is 3–38 d and the 60-d fallback applies to 0.1% of supports. |
| T3 | Context layers / mask date | OK | GMW v3 **2020** (`src/hsi.py:16`), edge distance from the same mask, no DEM. |
| T4 | Already-detected exclusion | OK | `src/labels.py:350-356` runs a real-time any-onset scan over (t0−6 m, t0] only. |
| T5 | Near-t0 band fallback flag | BUG (cosmetic) | Band predictors get no `fallback_60d` flag. Bands can come from different dates per support if one band is NaN (`src/sentinel.py:293`). |
| T6 | Duplicate S2 items | RISK (low) | 780 items cover 300 unique datetimes, and 180 items repeat a tile+datetime (reprocessed baselines 03.00–05.11). `src/feasibility.py:34-42` picks the best-scoring record per support, deterministically. Offsets are handled per baseline. |

### Spatial handling

| # | Item | Class | Evidence |
|---|---|---|---|
| S1 | GLT dedup | OK | `src/hsi.py:48-62`: GLT values are 1-based, with 0 = nodata (0 negative values in this scene). `np.unique(return_index)` on row-major order gives the lowest row/col representative, deterministically and one per native pixel. Cohort uniqueness is asserted (`run.py:205`). |
| S2 | Mangrove fraction support | RISK | The 75% rule is evaluated on the representative ortho cell (GMW averaged to the 0.000542° grid), not on the native detector footprint (`src/hsi.py:47-48`). |
| S3 | S2→support alignment | RISK | `src/sentinel.py:65-85` uses the lon/lat pixel centre ± half an ortho cell, transformed to the tile UTM CRS (correct). Windows use floor/ceil on the **20 m SCL grid**, so a ~60 m support reads 3–4 × 3–4 pixels (up to ~80 m) with no fractional weights. 10 m bands are averaged to 20 m (`:109`). Median aggregation and the ≥50% clear rule are correct. |
| S4 | Pixel centre vs corner | OK | `src/hsi.py:76-77` adds +0.5 cell, and the transform has a negative e. |
| S5 | QA flags | OK | Cloud, cirrus, spacecraft and dilated cloud (mask bands 0, 1, 3, 4) are applied. The water flag is correctly not applied. |

### Labels

| # | Item | Class | Evidence |
|---|---|---|---|
| L1 | Reference window, ≥6 obs, ≥9 months (274 d) | OK | `src/labels.py:321-326` |
| L2 | Seasonal ±45 d, ≥3 obs, else overall reference | OK | `src/labels.py:342-345` (wrap-around DOY handled) |
| L3 | MAD threshold max(0.10, 2.5·1.4826·MAD) | OK | `src/labels.py:347-348` |
| L4 | Water anomaly max(0.25, 3·1.4826·MAD); excluded from canopy, kept for structural | OK | `src/labels.py:311-314, 336-338`. The baseline uses [t0−24 m, t0], which counts as "pre-t0". |
| L5 | Loss-to-water (wf ≥ 0.50 and Δ ≥ 0.40) | OK | `src/labels.py:336`. The subtype also requires ≥30 d spacing (`:375`), slightly stricter than the plan text. Cosmetic. |
| L6 | SCL classes 4/5/7 canopy, 6 water, others rejected; MNDWI>0 water | OK | `src/sentinel.py:139-147` |
| L7 | Stable / Transient / Unresolved | OK | `src/labels.py:378-381`. Inadequate coverage goes to Unresolved and is never mapped to Stable. |
| L8 | **Persistence onset interpretation** | **RISK (high)** | `_persistent_episode` (`src/labels.py:25-48`) searches successive onsets for one meeting the horizon-end 60% rule. **122/172** positives start after their first degraded future observation, and **all 122** would fail a strict "from the first degraded observation" rule. The plan's T_onset clause supports the implemented reading. The labels are frozen. A strict-onset reading could be added only as a labelled POSTHOC_SENSITIVITY. |
| L9 | **T_detect episode-tied** | **BUG (affects results: lead time only)** | Confirmed. T_detect is chosen *after* selecting the horizon-qualifying episode (`src/labels.py:34-44`). A real-time monitor scanning all onsets confirms earlier in **109/172** cases. Related: **77/172** positives have a first real-time confirmation *before* their stored onset, meaning an earlier, non-persistent episode was already conventionally confirmed. Affected outputs: `tables/early_warning.csv` (median_lead_days and IQR), `tables/incremental_warnings.csv` (lead_days), `figures/lead_time.png`, and the 81-count follow-up guard used by S2/S3. Labels and PR-AUC are unaffected. |
| L10 | Real-time scan scope | RISK (low) | `verify_detection_dates` (`src/audit.py:190-198`) only considers post-t0 onsets. A degraded observation in (t0−6 m, t0] that was not yet confirmed by t0 could make the operational detection slightly earlier still. |

### Hyperspectral processing

| # | Item | Class | Evidence |
|---|---|---|---|
| H1 | Bad bands, water-vapour windows, invalid values | OK | Bands are excluded if officially bad (41), outside 420–2450 nm (12), or in the 1331–1460/1796–1938 nm windows (3 additional). Reflectance <0 or >1 is set to NaN (`src/hsi.py:104-110`). |
| H2 | Retained-band rule outcome-blind | OK | Missingness is computed on all 10,366 QA supports before any label is used (`run.py:143`). 229 bands are retained. |
| H3 | `band_mask` bit order | OK (low impact) | Big-endian unpacking (`src/hsi.py:103`). Only ~0.2% of pixels are flagged, and the data cannot separate big- from little-endian. Either order leaves the retained-band set unchanged. |
| H4 | H2 SRF | OK | Uses the official S2A v4.0 measured SRF (checksum-verified, `src/data.py:21-33`), nm on both grids, trapezoidal weights normalised by in-band valid weight, and ≥95% coverage. The plan names S2A, and the S2 imagery mixes S2A and S2B (noted, matches the plan). h2 missingness is ≤3.6%. |
| H5 | H3 wavelengths | OK | Every target is within 5 nm (e.g. 531→529.5, 570→566.8, 970→969.5). The formulas for PRI, CRI1, ARI, OSAVI, MCARI, REP and water-band depths match the standard definitions. |
| H6 | H4 subset | OK | 400–1700 nm of the retained bands = 154 bands, run through the same PCA branch (`src/models.py:227`). |

### Modelling and evaluation

| # | Item | Class | Evidence |
|---|---|---|---|
| M1 | Fold-local imputer/scaler/PCA/class weights/C selection/calibrator/thresholds | OK | `src/models.py:94-134`. Everything is fit on outer-train only. Inner grouped CV picks C ∈ {0.01, 0.1, 1} by AP. PCA uses ≤10 components. |
| M2 | **Realized alert budget** | **RISK (high, interpretation)** | Following the plan, thresholds are inner-OOF quantiles applied to the full-train model, so the realized rate differs by model and fold. At 10%: lr_A 227, lr_C 249, lr_D 218 alerts. Per-fold rates are 4–24%. The "equal-budget" HSI-added counts (C-only 22 / A-only 4) therefore compare unequal alert volumes. Precision@10 is C 0.333 vs A 0.286. |
| M3 | Block/fold construction | OK | 1 km blocks (variogram floor) with positive-patch union-find merge give 135 groups and 5 folds of 441–448 supports, each containing positives. |
| M4 | Buffer | RISK | A 60 m buffer removes orthogonal neighbours (58 m spacing) but keeps diagonal neighbours (83.5 m). That leaves 16–38 training supports per fold within 100 m of a test support, and the S2 windows (S3) overlap at that distance. |
| M5 | Patch fragmentation | RISK | Patches are 8-connected on representative ortho cells. Allowing a 1-pixel gap merges 97 patches into 55, and 5 of those merged events (34 supports) span more than one fold. That is possible near-event leakage, and it inflates the "patch" counts. |
| M6 | Block-size variogram on the outcome-selected cohort | RISK (low) | It uses a predictor variable only, and the estimate hit the 1 km floor anyway. |
| M7 | Bootstrap | OK | `src/evaluation.py:32-49`: paired, resampling the 135 event-merged groups, percentile 95% CI, 1,000 valid draws. The carbon bootstrap uses the same unit. |
| M8 | PR-AUC on pooled OOF | OK | `average_precision_score` is computed on the concatenated outer OOF. |

### Leakage and determinism

| # | Item | Class | Evidence |
|---|---|---|---|
| D1 | Carbon / outcome columns in features | OK | I scanned all six configurations for carbon, target, T_*, subtype, patch, fold and block names and found none. `feature_columns` also guards against them (`src/models.py:221`). |
| D2 | Determinism | RISK (low) | Seeds are fixed throughout. LR C risk varies by ≤0.0025 between runs (randomized PCA / BLAS threading), while metrics and alerts are identical. |
| D3 | **Plan-hash cache keys** | **RISK (operational)** | `run.py:47-50` and `followup.py:29-30` hash the whole plan, amendments included. Appending an amendment invalidates every cache. `run.py` would then re-derive and overwrite the frozen outputs, and `followup.py:39` would fail its assert. |
| D4 | Hard-coded numbers in summary text | BUG (cosmetic) | `followup.py:415-466` hard-codes "22", "672-day", "2,116/129/43", "62 of 96", "0.242", "154", the cohort counts, and so on. |
| D5 | Project notes | BUG (cosmetic, docs) | CLAUDE.md says no Arab EMIT scene was executed, but `feasibility/candidates.csv` shows Umm Al Quwain EMIT `…20230727T100231_2320807_026` was labelled: 1,976 supports, **3 positives**, rejected under the <30 positive minimum. Under the Task 2 rule, that scene qualifies only as an `APPLICATION_DEMO`. |

## Proposed corrections (awaiting confirmation)

Only **L9** is a confirmed result-affecting implementation bug. Proposed fix (Task 3 item; labels untouched):

- Add `T_detect_realtime` (first real-time any-onset confirmation after t0) beside the original `T_detect`, and
  assert that `target`/`subtype`/`T_onset`/`T_detect` are unchanged.
- Recompute the lead-time outputs as new `*_realtime` files next to the originals: early-warning lead, incremental
  lead, lead-time figure, and the post-detection follow-up counts. Report old and new side by side.
- Log an amendment ("implementation correction, lead-time only, not performance-motivated").

Not proposed as fixes (they would change the frozen design): L8, M2, M4, M5. I recommend reporting them as limitations.
Where they are cheap, they can be quantified under the Task 3 cluster-robust analysis by reporting realized alert counts and
precision differences. D3 is handled procedurally: new analyses run from new scripts and leave `run.py`/`followup.py` alone.

## Post-audit actions (after user confirmation, 2026-10-08)

- **L9 fixed** (Amendment 1). `src/labels.py` gained `_first_realtime_detection` and a new `T_detect_realtime` key; nothing existing was changed. `amendments.py --stage realtime` re-derived all 10,366 trajectories and asserted that `target`, `subtype`, `T_onset`, `T_detect`, `T_last_ok` and `already_detected_pre_t0` are identical to the frozen label cache. `T_detect_realtime` matches the independent reconstruction for 172/172 positives. Corrected outputs carry the `_realtime` suffix, the originals are kept, and `tables/tdetect_correction_comparison.csv` sets old against new.
- **D3 handled procedurally.** New analyses live in `amendments.py` and `arab_emit.py`, which record both the frozen and the current plan SHA. `run.py`/`followup.py` were not rerun. Two tests that compared the whole-file plan hash and mtime now check the hash of the frozen plan prefix (`tests/test_outputs.py`, `tests/test_followup.py`). A new regression test covers realtime detection.
- **M2 quantified** (Amendment 3): every incremental-warning comparison is repeated at an equal realized budget.
- No other code was changed. L8, M4, M5 and S2/S3 remain reported limitations.
