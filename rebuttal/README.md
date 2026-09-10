# Rebuttal experiments — rodent vigilance-state classification (BHI 2026, submission #10)

`run_rebuttal_experiments.py` is a single, self-contained, resumable script that
produces every number, table and figure the reviewers asked for:

| Reviewer request | Where it is answered |
|---|---|
| Leave-one-animal-out / subject-level CV (gKQE 4, tinc 1, y2eZ 3) | stage `loao` (8 folds, group = animal) and `loro` (16 folds, group = animal-day) |
| Quantify the inflation of the random epoch split (gKQE 4, tinc 1) | stage `random_split`: paper's stratified 80/20 split with the engineered set, +animal id, +raw samples, and a replica of the paper's design matrix |
| Comparison with standard / modern approaches (Xn1W 8, gKQE 3, y2eZ 4) | stage `loao`: logistic regression, random forest, RBF-SVM, MLP (128-64), class-balanced XGBoost; stage `cnn`: 1-D CNN on raw EEG (optional, needs `torch`) |
| Ablation of MMD and the other feature groups (gKQE 5, y2eZ 2) | stage `ablation` (8 feature sets, XGBoost, LOAO) |
| Statistical analysis (gKQE 5) | stage `stats`: paired Wilcoxon + paired t-test across folds, McNemar on pooled out-of-fold predictions, Cohen's d_z, rank-biserial r, Holm correction |
| Dispersion measures / error bars (AC, Xn1W 6) | mean ± SD and 95 % t-CI across folds, plus animal-stratified bootstrap 95 % CI on pooled predictions; all bar charts carry fold-SD error bars and no empty gaps |
| Per-class results, especially REM (tinc 3) | `results/table_perclass.tex`, `figures/fig_perclass_f1_loao.*`, confusion matrix |
| Cross-frequency coupling feature that is actually computed (tinc 2, title claim) | `pac_mi_theta_gamma`: Tort et al. (2010) modulation index, theta phase 6-9 Hz, gamma amplitude 30-100 Hz, 18 phase bins |
| Welch PSD, relative power, peak frequency, spectral entropy as described in Methods III.B.1 | stage `features` |
| Calibration with `multi:softmax` (tinc 4) | stage `calibration`: reliability curves, multiclass Brier, ECE; the code comment in `make_xgb` documents that `XGBClassifier.predict_proba` applies a softmax to the raw margins |
| Beeswarm vs bar mismatch (Xn1W 9) | stage `shap`: true beeswarm (one panel per class) **and** mean-abs-SHAP bar, plus gain importance |
| Consistent figure style (AC, Xn1W 5) | all figures: same fonts, colour-blind-safe palette, 300 dpi PNG + vector PDF |

The animal id, the recording day and the raw samples are **never** part of the
engineered feature set (an assertion in the script enforces this); they enter
only the explicitly labelled "leakage" variants of the random-split protocol.

## Requirements

Python >= 3.9 with `numpy pandas scipy scikit-learn xgboost matplotlib shap`
(`pyarrow` optional for a parquet feature cache; `torch` optional for the CNN).
Tested with numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.9, xgboost 3.2,
shap 0.51.

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib shap   # + torch (optional)
```

## Quick start

```bash
cd rebuttal

# 1. verify the feature implementations (MMD is checked against the notebook's
#    calculate_mmd_np, copied verbatim, on random epochs)
python run_rebuttal_experiments.py --selfcheck

# 2. smoke test on synthetic data (no real data needed, ~5-10 min)
python run_rebuttal_experiments.py --synthetic 120 --quick --out-dir /tmp/rebuttal_smoke

# 3. the real thing
python run_rebuttal_experiments.py \
    --data-dir /work/sjajee/stage1_labeled \
    --out-dir ./rebuttal_out \
    --label-map "0:Wake,1:SWS,2:REM" \
    --n-jobs 8
```

`--data-dir` must contain the 16 files `{animal}_{day}.csv` exactly as used in
the notebook (columns `"0"`..`"4999"` = samples of one 10-s epoch at 500 Hz, in
volts; column `"5000"` = integer label).

**Label mapping.** The notebook never states which integer is Wake / SWS / REM.
Without `--label-map` the script prints the class counts, names the classes
`class0/1/2` and warns; the mapping must be confirmed from the BDHSC data
description and passed with `--label-map` before the tables are used (REM should
be the rarest class, about 8 % of epochs). The "REM" class used for the REM-F1
statistics is the class whose name contains "REM", otherwise the rarest class.

## Stages, resuming, run time

Stages run in this order and can be selected with `--stages a,b,c`:

```
data  features  loao  loro  random_split  ablation  cnn  stats  calibration  shap  figures  report
```

Every stage writes its outputs under `--out-dir` and is **skipped if its outputs
already exist**, so an interrupted run is resumed by re-running the same command
(`--force` recomputes). Each (protocol, model) pair is its own checkpoint. All
random seeds are fixed (`--seed`, default 42). A log is written to
`<out-dir>/run.log` and a rough runtime estimate is printed at start-up; every
fold prints its own time and the projected time remaining.

Approximate wall time for the full 138,240-epoch data set on an 8-core
workstation (no GPU):

| stage | time |
|---|---|
| data (16 CSVs, ~9 GB) + features | 10-25 min |
| loao (6 feature models) | 30-60 min |
| loro (2 models × 16 folds) | 25-40 min |
| random_split (4 variants × 3 repeats; the two raw-sample variants have 5,000+ columns) | 1.5-3 h |
| ablation (7 sets × 8 folds) | 1-1.5 h |
| cnn (optional; CPU) | 2-4 h (minutes on a GPU) |
| stats, calibration, shap, figures, report | < 10 min |

i.e. roughly half a day without the CNN, one day with it. Peak memory is about
12 GB (raw epochs cached as float32 = 2.8 GB, plus the raw-sample design
matrices in `random_split`). Useful switches:

* `--quick` — 60 trees instead of 500, fewer bootstraps, 2 random-split repeats;
  for smoke tests only.
* `--models xgb,logreg,rf` — subset of LOAO models; `--loro-models`.
* `--random-repeats 3`, `--bootstrap 1000`, `--shap-n 5000`, `--svm-max-train 20000`, `--cnn-epochs 10`.
* `--raw-dtype float64` — caches the raw epochs in float64 (5.5 GB) so that the
  notebook MMD is reproduced bit-exactly; with the default float32 cache the
  MMD values differ from the notebook's at the 1e-12 relative level only.
* `--n-jobs` — threads for feature extraction / XGBoost / random forest.

If XGBoost is unexpectedly slow on a shared or containerised machine, the
script already sets `OMP_WAIT_POLICY=PASSIVE`; lowering `--n-jobs` below the
number of physical cores also helps.

## What is computed

### Features (stage `features`, cached in `cache/features.npz|csv`)

Computed per epoch in microvolts (the notebook MMD on volts, to reproduce the
published feature exactly):

* Welch PSD (Hann, `nperseg=1000`, 50 % overlap, 0.5 Hz resolution); absolute
  and relative power in delta 0.5-4, theta 4-8, alpha 8-12, beta 12-30, gamma
  30-100 Hz (half-open bins, sum × Δf); per-band peak frequency; total power
  0.5-100 Hz; normalised spectral entropy; theta/delta and delta/gamma ratios.
* Hjorth activity, mobility, complexity; line length; zero-crossing rate (of the
  mean-removed signal).
* `mmd` — exactly the notebook's `calculate_mmd_np` (10 fixed 500-sample
  sub-windows, Σ sqrt(Δindex² + Δamplitude²) on volts), vectorised. On volts the
  amplitude term is ≈1e-13 of the value, i.e. `mmd` ≈ Σ|t_max − t_min| in
  samples; `--selfcheck` prints this.
* `mmd_uV` — the same quantity with amplitudes in µV (dimensionally comparable
  terms), and its two components `mmd_dt` (Σ|Δindex|) and `mmd_damp_uV` (Σ Δamp).
* `pac_mi_theta_gamma` — Tort et al. 2010 modulation index (theta phase 6-9 Hz,
  gamma envelope 30-100 Hz, zero-phase Butterworth + Hilbert, 18 bins, first and
  last 0.5 s discarded). This is the cross-frequency-coupling feature.
* `legacy_fft_*` — the notebook's periodogram band powers (inclusive edges,
  alpha 8-13, beta 13-30). Kept only for the "paper replica" variant; not part
  of the engineered set.

Engineered set = 29 features; feature groups and ablation sets are defined in
`FEATURE_GROUPS` / `ABLATIONS` at the top of the script.

### Protocols

* **A — LOAO**: 8 folds, one animal (both days) held out.
* **B — LORO**: 16 folds, one animal-day held out (the same animal's other day
  stays in training → within-animal, across-day generalisation).
* **C — random stratified 80/20 epoch split** (`random_state=42`, the paper's
  split, plus 2 more seeds for an SD). Variants: engineered set; + animal id;
  + 5,000 raw samples; paper replica (raw + notebook FFT powers + `mmd` + animal id).

Models under identical LOAO conditions: XGBoost with the paper's tuned
configuration (lr 0.1, 500 trees, subsample 0.8, colsample_bytree 0.8,
reg_lambda 1, objective `multi:softmax`, `tree_method=hist`), the same with
class-balanced sample weights, multinomial L2 logistic regression
(log-transformed heavy-tailed columns + standardisation, `max_iter=2000`),
random forest (500 trees), RBF-SVM on a stratified ≤20k-epoch training
subsample, MLP (128-64, early stopping), and — if `torch` is importable — a 1-D
CNN on per-epoch z-scored raw EEG (4 conv blocks, 10 epochs, early stopping on
one held-out training animal).

### Metrics and statistics

Per fold: accuracy, macro and weighted precision / recall / F1, per-class
precision / recall / F1 / support, Cohen's κ, confusion matrix (counts and
row-normalised). Aggregated: mean, SD, 95 % t-CI, min, max across folds
(`results/cv_*_summary.csv`). Pooled out-of-fold predictions
(`results/oof_*.npz`) get an animal-stratified percentile bootstrap (1000
resamples, `results/bootstrap_*.csv`).

`results/stats_paired_tests.csv`: XGBoost vs every baseline and full set vs every
ablation, for macro-F1, REM-F1, accuracy and κ — mean difference with 95 % CI,
Cohen's d_z, paired t-test, Wilcoxon signed-rank (exact for n = 8),
matched-pairs rank-biserial r, Holm-adjusted p-values.
`results/stats_mcnemar.csv`: McNemar test on the pooled out-of-fold predictions
(exact binomial when b + c < 25, otherwise χ² with continuity correction).

### Calibration

Pooled out-of-fold `predict_proba` of the LOAO XGBoost: per-class reliability
curves (10 equal-width bins), top-label reliability, multiclass Brier score,
top-label and per-class ECE, log-loss (`results/calibration_*.csv`,
`figures/fig_calibration_xgb_loao.*`).

## Outputs

```
<out-dir>/
  run.log                      full log
  summary.json                 everything below, machine-readable
  cache/                       raw.npy (float32 epochs), meta.csv, features.npz/.csv
  results/
    cv_<protocol>__<model>_folds.csv      one row per fold, all metrics + confusion matrix
    cv_<protocol>__<model>_summary.csv    mean / SD / 95 % CI across folds
    bootstrap_<protocol>__<model>.csv     bootstrap CIs on pooled predictions
    oof_<protocol>__<model>.npz           pooled out-of-fold predictions and probabilities
    stats_paired_tests.csv, stats_mcnemar.csv
    calibration_bins.csv, calibration_summary.csv
    shap_mean_abs.csv, shap_values.npz, xgb_gain_importance.csv
    table_main.tex, table_perclass.tex, table_ablation.tex   (booktabs, mean ± SD)
    data_summary.json
  figures/  (PNG 300 dpi + PDF)
    fig_models_metrics_loao        grouped bars, models × metrics, fold-SD error bars
    fig_perclass_f1_loao           per-class F1 by model
    fig_protocol_inflation_xgb     XGBoost under LOAO / LORO / random-split variants
    fig_ablation_loao              feature-set ablation
    fig_confusion_xgb_loao         pooled LOAO confusion matrix (counts + row %)
    fig_shap_beeswarm              SHAP beeswarm, one panel per class
    fig_shap_mean_abs_bar          mean |SHAP| per feature, stacked by class
    fig_xgb_gain_importance        XGBoost gain importance
    fig_calibration_xgb_loao       reliability curves + ECE / Brier
  models/xgb_all_animals.json      XGBoost trained on all animals (SHAP model)
```

Tags: `loao__xgb`, `loao__logreg`, ..., `loro__xgb`, `random80_20__xgb__<variant>`,
`loao__xgb_abl_<ablation>`, `loao__cnn`.

## Filling the rebuttal / revised manuscript

* Headline LOAO numbers: `results/cv_loao__xgb_summary.csv` (mean ± SD, 95 % CI)
  and `results/bootstrap_loao__xgb.csv`.
* Inflation of the random split: compare the `random80_20__xgb__*` rows of
  `table_main.tex` with the LOAO row; the `paper_replica` variant reproduces the
  paper's design matrix (raw samples + animal id) on the paper's split.
* MMD ablation: `table_ablation.tex` (ΔF1 and Wilcoxon p vs the full set).
* Per-class / REM results: `table_perclass.tex`.
* Fig. 2 replacement: `fig_models_metrics_loao`; Fig. 3: `fig_xgb_gain_importance`;
  Fig. 4: `fig_shap_beeswarm` (a real beeswarm) or `fig_shap_mean_abs_bar`;
  Fig. 5: `fig_calibration_xgb_loao`.

## Notes and caveats

* `XGBClassifier` with `objective='multi:softmax'`: `predict` returns labels;
  `predict_proba` requests the raw margins and applies a softmax, so the
  calibration analysis is legitimate (the script verifies at run time that the
  rows sum to 1 and are not one-hot). `multi:softmax` and `multi:softprob` fit
  the same objective.
* Constant (drop-out) epochs (value −7.63e−08 in the real data) are kept; their
  spectral features are ≈0, MMD = 0 and PAC = 0. Consider reporting how many
  there are per class (`results/data_summary.json` has per-recording counts).
* The synthetic data (`--synthetic N`) only exercise the pipeline; the numbers
  they produce are meaningless for the paper.
* The CNN stage is skipped with a warning when `torch` is not installed. It has
  been written but could not be executed in the development environment;
  run it once with `--quick --synthetic 120 --stages data,features,cnn` before
  the real run.
