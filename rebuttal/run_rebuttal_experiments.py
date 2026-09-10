#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rebuttal_experiments.py
===========================

Experiments requested by the reviewers of
"XGBoost-Based Automated Vigilance State Classification from Rodent EEG
Using Spectral and Cross-Frequency Features" (IEEE EMBS BHI 2026).

The script is a single, self-contained, resumable pipeline that the authors run
on the real BDHSC 2024 stage-1 data (16 files ``{animal}_{day}.csv``; each row =
one 10-s epoch of 5000 samples at 500 Hz in columns "0".."4999", label in column
"5000").  It produces every number, table and figure needed for the rebuttal and
the revised manuscript:

Stages (run with ``--stages a,b,c`` or the default ``all``)
-----------------------------------------------------------
data          load the CSVs once, cache raw epochs (float32 .npy) + metadata
features      vectorised engineered features (Welch band powers, relative
              powers, peak frequencies, spectral entropy, ratios, Hjorth,
              line length, zero-crossing rate, MMD exactly as in the notebook,
              a dimensionally consistent MMD_uV and its two components, and a
              theta-gamma phase-amplitude-coupling modulation index)
loao          Protocol A: leave-one-animal-out CV (8 folds) for every model
loro          Protocol B: leave-one-recording-day-out CV (16 folds)
random_split  Protocol C: the paper's random stratified 80/20 epoch split, with
              the engineered set, +animal id, +raw samples and a replica of the
              paper's design matrix, to quantify the inflation
ablation      XGBoost LOAO with feature-group ablations
cnn           optional 1-D CNN on raw epochs under LOAO (needs torch)
stats         paired Wilcoxon / t-tests across folds, McNemar on pooled
              out-of-fold predictions, effect sizes, Holm correction
calibration   reliability curves, multiclass Brier score, ECE on pooled
              out-of-fold XGBoost probabilities
shap          SHAP beeswarm (one panel per class) + mean|SHAP| bar + gain bar
figures       all publication figures (300 dpi PNG + PDF, consistent style)
report        summary.json and LaTeX table snippets

Every stage writes its outputs under ``--out-dir`` and is skipped on re-run if
its outputs exist (use ``--force`` to recompute).  Seeds are fixed.

Requirements: Python >= 3.9, numpy, pandas, scipy, scikit-learn, xgboost,
matplotlib, shap.  ``torch`` is optional (CNN baseline only).

Quick smoke test on synthetic data (no real data needed):

    python run_rebuttal_experiments.py --synthetic 120 --quick --out-dir /tmp/rebuttal_smoke

Real data:

    python run_rebuttal_experiments.py --data-dir /path/to/stage1_labeled \\
        --out-dir ./rebuttal_out --label-map "0:Wake,1:SWS,2:REM" --n-jobs 8
"""
from __future__ import annotations

import os

# OpenMP threads that spin-wait can stall XGBoost / OpenBLAS by 50-100x on
# CPU-throttled machines (containers, shared HPC nodes).  A passive wait policy
# costs nothing on a dedicated workstation.  Must be set before numpy is imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import json
import logging
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # optional
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

try:  # optional (CNN baseline only)
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

LOG = logging.getLogger("rebuttal")

# ----------------------------------------------------------------------------
# Constants describing the data and the feature set
# ----------------------------------------------------------------------------
FS = 500                     # Hz
EPOCH_SEC = 10
N_SAMPLES = FS * EPOCH_SEC   # 5000
SAMPLE_COLS = [str(i) for i in range(N_SAMPLES)]
LABEL_COL = str(N_SAMPLES)   # "5000"
FLAT_VALUE = -7.629511e-08   # value of the constant (drop-out) rows seen in the real data
FILE_RE = re.compile(r"^(\d+)_(\d+)\.csv$")

# Band edges used for the engineered features (paper: alpha 8-12, beta 12-30).
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0),
}
# Band edges used by the notebook's periodogram features (kept only for the
# "paper replica" variant of Protocol C; NOT part of the engineered set).
LEGACY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}
TOTAL_BAND = (0.5, 100.0)
PAC_PHASE_BAND = (6.0, 9.0)
PAC_AMP_BAND = (30.0, 100.0)
PAC_N_BINS = 18
EPS = 1e-12

FEATURE_GROUPS: Dict[str, List[str]] = {
    "bandpower_abs": [f"abs_{b}" for b in BANDS],
    "bandpower_rel": [f"rel_{b}" for b in BANDS],
    "peak_freq": [f"peakfreq_{b}" for b in BANDS],
    "spectral_misc": ["total_power", "spectral_entropy", "ratio_theta_delta", "ratio_delta_gamma"],
    "hjorth": ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"],
    "time_domain": ["line_length", "zero_crossing_rate"],
    "mmd": ["mmd"],
    "mmd_uv": ["mmd_uV", "mmd_dt", "mmd_damp_uV"],
    "cfc": ["pac_mi_theta_gamma"],
}
ENGINEERED: List[str] = [f for g in FEATURE_GROUPS.values() for f in g]
LEGACY: List[str] = [f"legacy_fft_{b}" for b in LEGACY_BANDS]
ALL_FEATURES: List[str] = ENGINEERED + LEGACY
META_COLS = ["animal", "day", "epoch_idx", "label"]
# Columns that are strictly positive and heavy tailed: log10-transformed for the
# linear / kernel / MLP baselines (trees are invariant to monotone transforms).
LOG_FEATURES = FEATURE_GROUPS["bandpower_abs"] + ["total_power", "ratio_theta_delta",
                                                   "ratio_delta_gamma", "hjorth_activity",
                                                   "line_length", "mmd_damp_uV"]

# Sanity: the engineered set never contains identifiers or raw samples.
_FORBIDDEN = {"animal", "day", "epoch_idx", "label", "recording_id", "index"}
assert not (_FORBIDDEN & set(ENGINEERED)), "identifier leaked into engineered features"
assert not any(f.isdigit() for f in ENGINEERED), "raw sample column leaked into engineered features"


def _without(groups: Sequence[str]) -> List[str]:
    drop = {f for g in groups for f in FEATURE_GROUPS[g]}
    return [f for f in ENGINEERED if f not in drop]


ABLATIONS: Dict[str, List[str]] = {
    "full": list(ENGINEERED),
    "minus_mmd": _without(["mmd"]),
    "minus_mmd_uv": _without(["mmd_uv"]),
    "minus_hjorth": _without(["hjorth"]),
    "minus_cfc": _without(["cfc"]),
    "bandpowers_only": FEATURE_GROUPS["bandpower_abs"] + FEATURE_GROUPS["bandpower_rel"],
    "mmd_only": FEATURE_GROUPS["mmd"] + FEATURE_GROUPS["mmd_uv"],
    "hjorth_only": list(FEATURE_GROUPS["hjorth"]),
}
ABLATION_LABELS = {
    "full": "Full engineered set",
    "minus_mmd": "- MMD",
    "minus_mmd_uv": "- MMD_uV (+components)",
    "minus_hjorth": "- Hjorth",
    "minus_cfc": "- CFC (PAC)",
    "bandpowers_only": "Band powers only",
    "mmd_only": "MMD only",
    "hjorth_only": "Hjorth only",
}

MODEL_LABELS = {
    "xgb": "XGBoost",
    "xgb_balanced": "XGBoost (class-balanced)",
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "svm": "SVM (RBF)",
    "mlp": "MLP (128-64)",
    "cnn": "1-D CNN (raw EEG)",
}
MODEL_ORDER = list(MODEL_LABELS)
FEATURE_MODELS = ["xgb", "xgb_balanced", "logreg", "rf", "svm", "mlp"]

RANDOM_VARIANTS = {
    "engineered": "Engineered set",
    "engineered_plus_animal": "Engineered + animal id",
    "engineered_plus_raw": "Engineered + raw samples",
    "paper_replica": "Paper design matrix (raw + FFT powers + MMD + animal id; paper's default XGBoost)",
}
# The paper's headline model (notebook cell 23) is XGBClassifier(num_class=3) with library defaults:
# 100 trees, learning rate 0.3, depth 6, no subsampling, objective auto-set to multi:softprob.  The
# replica rows use that configuration so that the leakage quantification reproduces the reported model.
REPLICA_MODEL = "xgb_default"
EXTRA_MODEL_LABELS = {"xgb_default": "XGBoost (paper default config)"}

STAGES = ["data", "features", "loao", "loro", "random_split", "loao_replica", "ablation", "cnn",
          "stats", "calibration", "shap", "figures", "report"]

# Colour-blind-safe categorical palette (validated: adjacent-pair CVD dE >= 8).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLORS = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(MODEL_ORDER)}


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.FileHandler(out_dir / "run.log")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{seconds:.1f}s")


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def parse_label_map(spec: Optional[str]) -> Optional[Dict[int, str]]:
    if not spec:
        return None
    out: Dict[int, str] = {}
    for item in spec.split(","):
        k, v = item.split(":")
        out[int(k.strip())] = v.strip()
    return out


class LogColumns(BaseEstimator, TransformerMixin):
    """log10(x + eps) on selected (non-negative, heavy-tailed) columns."""

    def __init__(self, columns: Sequence[int] = (), eps: float = EPS):
        self.columns = columns          # stored unmodified (sklearn clone contract)
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        if len(self.columns):
            cols = np.asarray(list(self.columns), dtype=int)
            X[:, cols] = np.log10(np.clip(X[:, cols], 0.0, None) + self.eps)
        return X


# ----------------------------------------------------------------------------
# Synthetic data (smoke testing only)
# ----------------------------------------------------------------------------
def _pink_noise(n: int, length: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """Unit-variance 1/f^exponent noise, one row per epoch."""
    white = rng.standard_normal((n, length))
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(length, 1.0 / FS)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.power(freqs[1:], exponent / 2.0)
    scale[0] = 0.0
    x = np.fft.irfft(spec * scale, n=length, axis=-1)
    x /= x.std(axis=-1, keepdims=True) + 1e-12
    return x


def _markov_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """0 = Wake, 1 = SWS, 2 = REM.  REM is entered almost only from SWS."""
    P = np.array([[0.92, 0.08, 0.00],
                  [0.06, 0.90, 0.04],
                  [0.18, 0.02, 0.80]])
    s = np.empty(n, dtype=np.int64)
    s[0] = 0
    for i in range(1, n):
        s[i] = rng.choice(3, p=P[s[i - 1]])
    # make sure each state is present (needed for per-file folds)
    for k in range(3):
        if (s == k).sum() < 3:
            s[rng.choice(n, size=3, replace=False)] = k
    return s


def generate_synthetic_dataset(dest: Path, n_per_file: int, seed: int,
                               n_animals: int = 8, n_days: int = 2) -> None:
    """Write {animal}_{day}.csv files with realistic three-state synthetic EEG.

    SWS : 1-4 Hz high-amplitude oscillation (variable amplitude) + pink noise
    REM : 6-9 Hz theta, low amplitude, with theta-phase-modulated gamma (PAC)
    Wake: broadband (pink + white) low amplitude + unmodulated gamma + theta
          (active wake also carries 5-9 Hz theta, as in real rodents) + artefacts
    Difficulty comes from per-epoch amplitude jitter, overlapping amplitude
    ranges, transitional (mixed) epochs at state changes and strong animal
    effects (gain, frequency offset, noise colour, gamma gain, 50 Hz pickup).
    The animal effects make the animal id a useful leakage feature under a
    random epoch split, as in the real data.  Values are written in volts.
    """
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    sos_gamma = signal.butter(4, [30.0, 100.0], btype="bandpass", fs=FS, output="sos")

    def synth(states: np.ndarray, gain: float, f_off: float, pink_exp: float, gamma_gain: float,
              delta_gain: float, theta_gain: float) -> np.ndarray:
        n = len(states)
        pink = _pink_noise(n, N_SAMPLES, rng, pink_exp)
        gam = signal.sosfilt(sos_gamma, rng.standard_normal((n, N_SAMPLES)), axis=-1)
        gam /= gam.std(axis=-1, keepdims=True) + 1e-12
        white = rng.standard_normal((n, N_SAMPLES))
        phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
        jit = np.exp(rng.normal(0.0, 0.35, size=(n, 1)))     # per-epoch amplitude jitter
        sig = np.zeros((n, N_SAMPLES))
        m = states == 1                                       # SWS
        if m.any():
            k = m.sum()
            f = rng.uniform(1.0, 4.0, size=(k, 1)) + 0.3 * f_off
            env = 1.0 + 0.6 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4, size=(k, 1)) * t + rng.uniform(0, 6.28, size=(k, 1)))
            sig[m] = (delta_gain * rng.uniform(22, 60, size=(k, 1)) * env * np.sin(2 * np.pi * f * t + phase[m])
                      + rng.uniform(4, 14, size=(k, 1)) * np.sin(2 * np.pi * rng.uniform(4, 8, size=(k, 1)) * t)
                      + rng.uniform(14, 26, size=(k, 1)) * pink[m] + 3 * gamma_gain * gam[m])
        m = states == 2                                       # REM: theta + theta-locked gamma
        if m.any():
            k = m.sum()
            f = rng.uniform(6.0, 9.0, size=(k, 1)) + f_off
            th = 2 * np.pi * f * t + phase[m]
            depth = rng.uniform(0.3, 0.9, size=(k, 1))
            sig[m] = (theta_gain * rng.uniform(10, 30, size=(k, 1)) * np.sin(th)
                      + 5 * gamma_gain * (1.0 + depth * np.cos(th)) * gam[m]
                      + rng.uniform(4, 14, size=(k, 1)) * np.sin(2 * np.pi * rng.uniform(1, 4, size=(k, 1)) * t)
                      + rng.uniform(10, 20, size=(k, 1)) * pink[m])
        m = states == 0                                       # Wake
        if m.any():
            k = m.sum()
            f = rng.uniform(5.0, 9.0, size=(k, 1)) + f_off
            sig[m] = (rng.uniform(12, 26, size=(k, 1)) * pink[m] + rng.uniform(4, 11, size=(k, 1)) * gamma_gain * gam[m]
                      + rng.uniform(4, 9, size=(k, 1)) * white[m]
                      + theta_gain * rng.uniform(4, 20, size=(k, 1)) * np.sin(2 * np.pi * f * t + phase[m]))
            art = m & (rng.random(n) < 0.10)                  # movement artefacts in 10 % of wake epochs
            if art.any():
                burst = np.exp(-((t - rng.uniform(1, 9, size=(art.sum(), 1))) ** 2) / (2 * 0.3 ** 2))
                sig[art] += 60 * burst * rng.standard_normal((art.sum(), N_SAMPLES))
        return sig * gain * jit

    for a in range(n_animals):
        gain = rng.uniform(0.6, 1.7)
        f_off = rng.uniform(-1.0, 1.0)
        pink_exp = rng.uniform(0.7, 1.3)
        gamma_gain = rng.uniform(0.5, 1.5)
        delta_gain = rng.uniform(0.6, 1.6)      # animal-specific state signatures
        theta_gain = rng.uniform(0.6, 1.6)
        line50 = rng.uniform(0.0, 5.0)
        for d in range(n_days):
            n = n_per_file
            states = _markov_states(n, rng)
            sig = synth(states, gain, f_off, pink_exp, gamma_gain, delta_gain, theta_gain)
            # transitional epochs: at a state change, the epoch is a mixture of both states
            change = np.flatnonzero(np.r_[False, states[1:] != states[:-1]])
            if len(change):
                prev = synth(states[change - 1], gain, f_off, pink_exp, gamma_gain, delta_gain, theta_gain)
                w = rng.uniform(0.3, 0.6, size=(len(change), 1))
                sig[change] = (1 - w) * sig[change] + w * prev
            sig += line50 * np.sin(2 * np.pi * 50.0 * t)
            # drop-out (flat) epochs as seen in the real data
            flat = rng.random(n) < 0.02
            data = sig * 1e-6
            data[flat] = FLAT_VALUE
            df = pd.DataFrame(data.astype(np.float32), columns=SAMPLE_COLS)
            df[LABEL_COL] = states
            df.to_csv(dest / f"{a}_{d}.csv", index=False, float_format="%.6e")
    LOG.info("synthetic data written to %s (%d files, %d epochs each)", dest, n_animals * n_days, n_per_file)


# ----------------------------------------------------------------------------
# Data loading / caching
# ----------------------------------------------------------------------------
def discover_files(data_dir: Path) -> List[Tuple[int, int, Path]]:
    files = []
    for p in sorted(data_dir.iterdir()):
        m = FILE_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), int(m.group(2)), p))
    files.sort()
    if not files:
        raise FileNotFoundError(f"no {{animal}}_{{day}}.csv files found in {data_dir}")
    return files


def _read_one_csv(path: Path, dtype: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, dtype=np.float32 if dtype == "float32" else np.float64)
    missing = [c for c in SAMPLE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing[:5]}... (expected '0'..'5000')")
    extra = [c for c in df.columns if c not in SAMPLE_COLS and c != LABEL_COL]
    if extra:
        LOG.warning("%s: ignoring unexpected columns %s", path.name, extra[:5])
    x = df[SAMPLE_COLS].to_numpy(dtype=np.float32 if dtype == "float32" else np.float64)
    lab = df[LABEL_COL].to_numpy()
    if not np.all(np.isfinite(lab)) or not np.allclose(lab, np.round(lab)):
        raise ValueError(f"{path.name}: labels in column '5000' are not integers")
    flat = x.max(axis=1) == x.min(axis=1)          # constant (drop-out) epochs
    return x, np.round(lab).astype(np.int64), flat


def load_or_cache_raw(data_dir: Path, cache_dir: Path, dtype: str, n_jobs: int,
                      force: bool = False) -> Tuple[np.ndarray, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = cache_dir / "raw.npy", cache_dir / "meta.csv"
    if raw_path.exists() and meta_path.exists() and not force:
        LOG.info("[data] using cached raw epochs %s", raw_path)
        return np.load(raw_path, mmap_mode="r"), pd.read_csv(meta_path)
    files = discover_files(data_dir)
    LOG.info("[data] reading %d CSV files from %s", len(files), data_dir)
    tm = Timer("data")
    parts = Parallel(n_jobs=max(1, min(n_jobs, 4)))(
        delayed(_read_one_csv)(p, dtype) for _, _, p in files)
    raws, metas = [], []
    for (a, d, p), (x, lab, flat) in zip(files, parts):
        raws.append(x)
        metas.append(pd.DataFrame({"animal": a, "day": d, "epoch_idx": np.arange(len(lab)), "label": lab,
                                   "is_flat": flat}))
        u, c = np.unique(lab, return_counts=True)
        LOG.info("[data]   %s: %d epochs, labels %s, %d constant (flat) epochs", p.name, len(lab),
                 {int(k): int(v) for k, v in zip(u, c)}, int(flat.sum()))
    raw = np.concatenate(raws, axis=0)
    meta = pd.concat(metas, ignore_index=True)
    np.save(raw_path, raw)
    meta.to_csv(meta_path, index=False)
    LOG.info("[data] cached %s epochs x %d samples (%s) in %s", f"{raw.shape[0]:,}", raw.shape[1],
             raw.dtype, hms(tm.elapsed()))
    return np.load(raw_path, mmap_mode="r"), meta


# ----------------------------------------------------------------------------
# Feature extraction (vectorised over epochs)
# ----------------------------------------------------------------------------
def mmd_notebook_vectorised(x: np.ndarray) -> np.ndarray:
    """Maximum-Minimum Distance exactly as ``calculate_mmd_np`` in the notebook,
    vectorised over epochs: 10 fixed non-overlapping 500-sample sub-windows,
    sqrt((argmax-argmin)^2 + (max-min)^2) accumulated sequentially.  Applied to the
    raw signal in volts (as the notebook does) the amplitude term is negligible,
    which is exactly the point the rebuttal has to make.  ``--selfcheck`` asserts
    equality with the verbatim notebook function."""
    n = x.shape[0]
    seg = x.reshape(n, N_SAMPLES // 500, 500)
    min_idx = seg.argmin(axis=2)
    max_idx = seg.argmax(axis=2)
    min_val = seg.min(axis=2)
    max_val = seg.max(axis=2)
    d = np.sqrt((max_idx - min_idx) ** 2 + (max_val - min_val) ** 2)
    total = np.zeros(n, dtype=np.float64)
    for k in range(d.shape[1]):        # sequential accumulation = notebook's loop order
        total = total + d[:, k]
    return total


def mmd_components(x_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dimensionally consistent MMD with amplitudes in microvolts, plus its two
    components: sum |t_max - t_min| (samples) and sum (max - min) (uV)."""
    n = x_uv.shape[0]
    seg = x_uv.reshape(n, N_SAMPLES // 500, 500)
    dt = np.abs(seg.argmax(axis=2) - seg.argmin(axis=2)).astype(np.float64)
    damp = (seg.max(axis=2) - seg.min(axis=2)).astype(np.float64)
    mmd_uv = np.sqrt(dt ** 2 + damp ** 2).sum(axis=1)
    return mmd_uv, dt.sum(axis=1), damp.sum(axis=1)


_SOS_CACHE: Dict[str, np.ndarray] = {}


def _sos(name: str, band: Tuple[float, float], order: int) -> np.ndarray:
    if name not in _SOS_CACHE:
        _SOS_CACHE[name] = signal.butter(order, list(band), btype="bandpass", fs=FS, output="sos")
    return _SOS_CACHE[name]


def pac_modulation_index(x_uv: np.ndarray, phase_band=PAC_PHASE_BAND, amp_band=PAC_AMP_BAND,
                         n_bins: int = PAC_N_BINS, edge_trim: int = 250) -> np.ndarray:
    """Theta-gamma phase-amplitude coupling, modulation index of Tort et al. (2010).

    theta phase: 6-9 Hz band-pass (zero-phase Butterworth) -> Hilbert phase
    gamma amplitude: 30-100 Hz band-pass -> Hilbert envelope
    Mean gamma amplitude in 18 theta-phase bins, normalised to a distribution P;
    MI = (log N - H(P)) / log N in [0, 1].  The first/last 0.5 s are discarded to
    limit filter edge effects.  Constant (drop-out) epochs get MI = 0.
    """
    n = x_uv.shape[0]
    ph = np.angle(signal.hilbert(signal.sosfiltfilt(_sos("pac_phase", phase_band, 2), x_uv, axis=-1), axis=-1))
    am = np.abs(signal.hilbert(signal.sosfiltfilt(_sos("pac_amp", amp_band, 3), x_uv, axis=-1), axis=-1))
    ph = ph[:, edge_trim:-edge_trim]
    am = am[:, edge_trim:-edge_trim]
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    b = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    mean_amp = np.zeros((n, n_bins))
    for k in range(n_bins):
        m = b == k
        cnt = m.sum(axis=1)
        mean_amp[:, k] = (am * m).sum(axis=1) / np.maximum(cnt, 1)
    tot = mean_amp.sum(axis=1, keepdims=True)
    P = mean_amp / np.maximum(tot, 1e-300)
    H = -(P * np.log(P + 1e-300)).sum(axis=1)
    mi = (np.log(n_bins) - H) / np.log(n_bins)
    mi[tot[:, 0] <= 1e-9] = 0.0
    return np.clip(mi, 0.0, 1.0)


def legacy_fft_bandpowers(x_v: np.ndarray) -> Dict[str, np.ndarray]:
    """The notebook's ``calculate_signal_strength``: |FFT|^2 summed over the
    positive-frequency bins with lo <= f <= hi (inclusive edges, alpha 8-13,
    beta 13-30), on the raw signal in volts.  rfft gives identical values."""
    spec = np.abs(np.fft.rfft(x_v, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(N_SAMPLES, 1.0 / FS)
    out = {}
    for b, (lo, hi) in LEGACY_BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        out[f"legacy_fft_{b}"] = spec[:, m].sum(axis=1)
    return out


def extract_features_chunk(x_v: np.ndarray) -> np.ndarray:
    """Engineered features for a chunk of epochs.  ``x_v`` is (n, 5000) in volts.
    All spectral / time-domain features are computed in microvolts; the notebook
    MMD is computed on volts to reproduce the published feature exactly."""
    x_v = np.asarray(x_v, dtype=np.float64)
    n = x_v.shape[0]
    x = x_v * 1e6  # microvolts
    out: Dict[str, np.ndarray] = {}

    # --- Welch PSD (Hann, nperseg = 1000 -> 0.5 Hz resolution, 50 % overlap)
    f, pxx = signal.welch(x, fs=FS, window="hann", nperseg=1000, noverlap=500,
                          detrend="constant", axis=-1)
    df = f[1] - f[0]
    tot_mask = (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])
    total = pxx[:, tot_mask].sum(axis=1) * df
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)          # half-open bins: no double counting at edges
        p = pxx[:, m].sum(axis=1) * df    # uV^2
        out[f"abs_{b}"] = p
        out[f"rel_{b}"] = p / (total + EPS)
        out[f"peakfreq_{b}"] = f[m][np.argmax(pxx[:, m], axis=1)]
    out["total_power"] = total
    pn = pxx[:, tot_mask]
    pn = pn / (pn.sum(axis=1, keepdims=True) + EPS)
    out["spectral_entropy"] = -(pn * np.log(pn + 1e-300)).sum(axis=1) / np.log(pn.shape[1])
    out["ratio_theta_delta"] = out["abs_theta"] / (out["abs_delta"] + EPS)
    out["ratio_delta_gamma"] = out["abs_delta"] / (out["abs_gamma"] + EPS)

    # --- Hjorth parameters
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = x.var(axis=1), d1.var(axis=1), d2.var(axis=1)
    mob = np.sqrt(v1 / (v0 + EPS))
    out["hjorth_activity"] = v0
    out["hjorth_mobility"] = mob
    out["hjorth_complexity"] = np.sqrt(v2 / (v1 + EPS)) / (mob + EPS)

    # --- time domain
    out["line_length"] = np.abs(d1).sum(axis=1)
    xc = x - x.mean(axis=1, keepdims=True)
    out["zero_crossing_rate"] = (np.signbit(xc[:, 1:]) != np.signbit(xc[:, :-1])).mean(axis=1)

    # --- MMD family
    out["mmd"] = mmd_notebook_vectorised(x_v)
    out["mmd_uV"], out["mmd_dt"], out["mmd_damp_uV"] = mmd_components(x)

    # --- cross-frequency coupling
    out["pac_mi_theta_gamma"] = pac_modulation_index(x)

    # --- notebook periodogram powers (paper replica only)
    out.update(legacy_fft_bandpowers(x_v))

    X = np.column_stack([out[name] for name in ALL_FEATURES])
    assert X.shape == (n, len(ALL_FEATURES))
    return X


def compute_features(raw: np.ndarray, chunk_size: int, n_jobs: int) -> np.ndarray:
    n = raw.shape[0]
    bounds = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    LOG.info("[features] extracting %d features for %s epochs in %d chunks (%d threads)",
             len(ALL_FEATURES), f"{n:,}", len(bounds), max(1, n_jobs))
    tm = Timer("features")
    parts = Parallel(n_jobs=max(1, n_jobs), prefer="threads")(
        delayed(extract_features_chunk)(np.asarray(raw[i:j])) for i, j in bounds)
    X = np.vstack(parts)
    bad = ~np.isfinite(X)
    if bad.any():
        LOG.warning("[features] %d non-finite values replaced by 0", int(bad.sum()))
        X[bad] = 0.0
    LOG.info("[features] done in %s (%.1f ms / epoch)", hms(tm.elapsed()), 1000 * tm.elapsed() / n)
    return X


def load_or_compute_features(raw: np.ndarray, meta: pd.DataFrame, cache_dir: Path,
                             chunk_size: int, n_jobs: int, force: bool) -> np.ndarray:
    path = cache_dir / "features.npz"
    if path.exists() and not force:
        z = np.load(path, allow_pickle=False)
        names = list(z["feature_names"])
        if names == ALL_FEATURES and z["X"].shape[0] == len(meta):
            LOG.info("[features] using cached %s", path)
            return z["X"]
        LOG.warning("[features] cache does not match current feature definition; recomputing")
    X = compute_features(raw, chunk_size, n_jobs)
    np.savez(path, X=X, feature_names=np.array(ALL_FEATURES))
    tab = pd.concat([meta.reset_index(drop=True), pd.DataFrame(X, columns=ALL_FEATURES)], axis=1)
    tab.to_csv(cache_dir / "features.csv", index=False, float_format="%.8g")
    try:
        tab.to_parquet(cache_dir / "features.parquet", index=False)
    except Exception:
        pass  # pyarrow / fastparquet not installed: the CSV + NPZ caches are enough
    return X


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def confusion(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(y_true.astype(np.int64) * K + y_pred.astype(np.int64), minlength=K * K).reshape(K, K)


def metrics_from_cm(cm: np.ndarray, class_names: Sequence[str]) -> Dict[str, float]:
    """Accuracy, macro / weighted P-R-F1, per-class P-R-F1 and Cohen's kappa.
    Macro averages run over the classes present in y_true or y_pred (sklearn
    semantics, zero_division = 0)."""
    cm = cm.astype(np.float64)
    N = cm.sum()
    tp = np.diag(cm)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    present = (row > 0) | (col > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(col > 0, tp / col, 0.0)
        rec = np.where(row > 0, tp / row, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        po = tp.sum() / N if N > 0 else np.nan
        pe = (row * col).sum() / (N * N) if N > 0 else np.nan
        kappa = (po - pe) / (1 - pe) if (N > 0 and pe < 1) else np.nan
    w = row / N if N > 0 else np.zeros_like(row)
    out = {
        "accuracy": po,
        "precision_macro": prec[present].mean() if present.any() else np.nan,
        "recall_macro": rec[present].mean() if present.any() else np.nan,
        "f1_macro": f1[present].mean() if present.any() else np.nan,
        "precision_weighted": float((w * prec).sum()),
        "recall_weighted": float((w * rec).sum()),
        "f1_weighted": float((w * f1).sum()),
        "kappa": kappa,
    }
    for k, name in enumerate(class_names):
        out[f"precision_{name}"] = prec[k] if row[k] > 0 else np.nan
        out[f"recall_{name}"] = rec[k] if row[k] > 0 else np.nan
        out[f"f1_{name}"] = f1[k] if row[k] > 0 else np.nan
        out[f"support_{name}"] = row[k]
    return out


def compute_metrics(y_true, y_pred, class_names: Sequence[str]) -> Dict[str, float]:
    K = len(class_names)
    cm = confusion(np.asarray(y_true), np.asarray(y_pred), K)
    m = metrics_from_cm(cm, class_names)
    rown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for i in range(K):
        for j in range(K):
            m[f"cm_{i}_{j}"] = int(cm[i, j])
            m[f"cmn_{i}_{j}"] = float(rown[i, j])
    return m


PRIMARY_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                   "precision_weighted", "recall_weighted", "f1_weighted", "kappa"]


def aggregate_folds(df: pd.DataFrame, class_names: Sequence[str]) -> pd.DataFrame:
    cols = PRIMARY_METRICS + [f"{m}_{c}" for c in class_names for m in ("precision", "recall", "f1")]
    rows = []
    for c in cols:
        v = df[c].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        n = len(v)
        mean = v.mean() if n else np.nan
        sd = v.std(ddof=1) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
        rows.append({"metric": c, "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half,
                     "n_folds": n, "min": v.min() if n else np.nan, "max": v.max() if n else np.nan})
    return pd.DataFrame(rows)


def bootstrap_cluster(y_true: np.ndarray, y_pred: np.ndarray, clusters: np.ndarray, class_names: Sequence[str],
                      n_boot: int, seed: int) -> pd.DataFrame:
    """Cluster (animal-level / recording-level) percentile bootstrap of the pooled
    out-of-fold predictions: whole clusters are resampled with replacement, so the
    interval reflects between-animal variability (the exchangeable unit).  With 8
    animals the interval is coarse and should be read alongside the fold SD."""
    K = len(class_names)
    rng = np.random.default_rng(seed + 1)
    ids = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa"] + [f"f1_{c}" for c in class_names]
    point = metrics_from_cm(confusion(y_true, y_pred, K), class_names)
    samples = np.full((n_boot, len(keys)), np.nan)
    for b in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[c] for c in draw])
        m = metrics_from_cm(confusion(y_true[idx], y_pred[idx], K), class_names)
        samples[b] = [m[k] for k in keys]
    rows = []
    for j, k in enumerate(keys):
        s = samples[:, j]
        s = s[np.isfinite(s)]
        rows.append({"metric": k, "point": point[k],
                     "ci95_low": np.percentile(s, 2.5) if len(s) else np.nan,
                     "ci95_high": np.percentile(s, 97.5) if len(s) else np.nan,
                     "n_boot": int(len(s)), "n_clusters": int(len(ids))})
    return pd.DataFrame(rows)


def bootstrap_pooled(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray, class_names: Sequence[str],
                     n_boot: int, seed: int) -> pd.DataFrame:
    """Within-animal percentile bootstrap (epochs resampled with replacement inside
    each animal) of the pooled out-of-fold predictions.  This captures sampling
    noise at the epoch level only; between-animal variability is reported by the
    fold SD / t-CI and by ``bootstrap_cluster``."""
    K = len(class_names)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa"] + [f"f1_{c}" for c in class_names]
    point = metrics_from_cm(confusion(y_true, y_pred, K), class_names)
    samples = np.full((n_boot, len(keys)), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
        m = metrics_from_cm(confusion(y_true[idx], y_pred[idx], K), class_names)
        samples[b] = [m[k] for k in keys]
    rows = []
    for j, k in enumerate(keys):
        s = samples[:, j]
        s = s[np.isfinite(s)]
        rows.append({"metric": k, "point": point[k],
                     "ci95_low": np.percentile(s, 2.5) if len(s) else np.nan,
                     "ci95_high": np.percentile(s, 97.5) if len(s) else np.nan,
                     "n_boot": int(len(s))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_xgb(n_estimators: int, seed: int, n_jobs: int) -> xgb.XGBClassifier:
    """The paper's tuned configuration (Sec. III.C.3).  ``num_class`` is inferred
    from the labels by the sklearn wrapper.

    NOTE on probabilities: with objective='multi:softmax' the booster's native
    ``predict`` returns class labels, but ``XGBClassifier.predict_proba`` requests
    the raw margins (output_margin=True) and applies scipy's softmax to them, so
    it returns proper class probabilities.  multi:softmax and multi:softprob fit
    the identical softmax cross-entropy objective and differ only in prediction
    post-processing; the calibration analysis below is therefore valid, and the
    manuscript should state this explicitly.  ``check_probabilities`` verifies at
    run time that the returned rows sum to 1 and are not one-hot.
    """
    return xgb.XGBClassifier(
        objective="multi:softmax",
        learning_rate=0.1,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_lambda=1,
        random_state=seed,
        tree_method="hist",
        n_jobs=max(1, n_jobs),
    )


def check_probabilities(proba: np.ndarray, where: str) -> None:
    if proba is None:
        return
    s = proba.sum(axis=1)
    onehot = np.mean(np.isclose(proba.max(axis=1), 1.0)) > 0.999
    if not np.allclose(s, 1.0, atol=1e-4) or onehot:
        LOG.warning("[%s] predict_proba does not look like calibrated probabilities "
                    "(row sums ok=%s, one-hot=%s)", where, bool(np.allclose(s, 1.0, atol=1e-4)), bool(onehot))


def build_feature_model(name: str, feature_names: Sequence[str], cfg: dict, seed: int, n_jobs: int):
    log_idx = [i for i, f in enumerate(feature_names) if f in LOG_FEATURES]
    pre = [("log", LogColumns(log_idx)), ("scale", StandardScaler())]
    if name in ("xgb", "xgb_balanced"):
        return make_xgb(cfg["xgb_trees"], seed, n_jobs)
    if name == "xgb_default":
        # library defaults as in the notebook's headline model (objective -> multi:softprob)
        return xgb.XGBClassifier(n_estimators=cfg["xgb_default_trees"], random_state=seed, tree_method="hist",
                                 n_jobs=max(1, n_jobs))
    if name == "logreg":
        return Pipeline(pre + [("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=2000, random_state=seed))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=cfg["rf_trees"], n_jobs=max(1, n_jobs), random_state=seed,
                                      min_samples_leaf=2)
    if name == "svm":
        return Pipeline(pre + [("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed))])
    if name == "mlp":
        return Pipeline(pre + [("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=cfg["mlp_max_iter"],
                                                     early_stopping=True, n_iter_no_change=10,
                                                     random_state=seed))])
    raise ValueError(name)


def fit_predict_features(name: str, X_tr, y_tr, X_te, feature_names, cfg, seed, n_jobs):
    """Fit one feature-based model and return (y_pred, proba or None)."""
    model = build_feature_model(name, feature_names, cfg, seed, n_jobs)
    if name == "svm" and len(y_tr) > cfg["svm_max_train"]:
        # stratified subsample of the training epochs (kernel SVM is O(n^2))
        sub, _ = train_test_split(np.arange(len(y_tr)), train_size=cfg["svm_max_train"],
                                  stratify=y_tr, random_state=seed)
        X_tr, y_tr = X_tr[sub], y_tr[sub]
    if name == "xgb_balanced":
        model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") and name != "svm" else None
    return np.asarray(y_pred), proba


# ----------------------------------------------------------------------------
# Optional 1-D CNN on raw epochs (torch)
# ----------------------------------------------------------------------------
def fit_predict_cnn(raw: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray,
                    groups_tr: np.ndarray, K: int, cfg: dict, seed: int, n_jobs: int,
                    groups_te: Optional[np.ndarray] = None):
    """Small 1-D CNN (4 conv blocks) on per-epoch z-scored raw EEG.  One training
    animal is held out for early stopping.  Returns (y_pred, proba)."""
    import torch.nn as nn  # type: ignore

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, n_jobs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SmallCNN(nn.Module):
        def __init__(self, n_classes: int, channels=(16, 32, 64, 128), k: int = 7):
            super().__init__()
            layers, cin = [], 1
            for c in channels:
                layers += [nn.Conv1d(cin, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(4)]
                cin = c
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(cin, n_classes)

        def forward(self, x):
            return self.fc(self.drop(self.pool(self.features(x)).flatten(1)))

    def batch_tensor(idx: np.ndarray) -> "torch.Tensor":
        x = np.asarray(raw[idx], dtype=np.float32)
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        return torch.from_numpy(x[:, None, :]).to(device)

    val_group = np.min(groups_tr)          # one TRAINING animal, held out for early stopping
    if groups_te is not None:
        assert val_group not in set(np.unique(groups_te).tolist()), \
            "CNN early-stopping animal must be a training animal, never the test animal"
    val_mask = groups_tr == val_group
    fit_idx, val_idx = tr_idx[~val_mask], tr_idx[val_mask]
    model = SmallCNN(K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    bs = 256
    rng = np.random.default_rng(seed)
    best_state, best_loss, bad, patience = None, np.inf, 0, 3

    def evaluate(idx: np.ndarray) -> Tuple[float, np.ndarray]:
        model.eval()
        losses, probs = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 1024):
                b = idx[i:i + 1024]
                logits = model(batch_tensor(b))
                losses.append(loss_fn(logits, torch.from_numpy(y[b]).to(device)).item() * len(b))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return float(np.sum(losses) / len(idx)), np.vstack(probs)

    for ep in range(cfg["cnn_epochs"]):
        model.train()
        perm = rng.permutation(fit_idx)
        t0 = time.time()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:          # BatchNorm needs more than one sample in training mode
                continue
            opt.zero_grad()
            loss = loss_fn(model(batch_tensor(b)), torch.from_numpy(y[b]).to(device))
            loss.backward()
            opt.step()
        val_loss, _ = evaluate(val_idx)
        LOG.info("[cnn]     epoch %d/%d val_loss=%.4f (%s)", ep + 1, cfg["cnn_epochs"], val_loss, hms(time.time() - t0))
        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                LOG.info("[cnn]     early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, proba = evaluate(te_idx)
    return proba.argmax(axis=1), proba


# ----------------------------------------------------------------------------
# Cross-validation driver
# ----------------------------------------------------------------------------
class Context:
    """Holds configuration, paths and lazily loaded data for the stages."""

    def __init__(self, args: argparse.Namespace, cfg: dict):
        self.args = args
        self.cfg = cfg
        self.out = Path(args.out_dir)
        self.cache = self.out / "cache"
        self.results = self.out / "results"
        self.figures = self.out / "figures"
        self.models = self.out / "models"
        for d in (self.cache, self.results, self.figures, self.models):
            d.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(args.data_dir) if args.data_dir else None
        self._raw = None
        self._meta = None
        self._X = None
        self.class_values: List[int] = []
        self.class_names: List[str] = []
        self.rem_class: Optional[str] = None
        self.timings: Dict[str, float] = {}

    # ---- data
    def raw(self) -> np.ndarray:
        if self._raw is None:
            self._raw, self._meta = load_or_cache_raw(self.data_dir, self.cache, self.args.raw_dtype,
                                                      self.args.n_jobs, self.args.force and "data" in self.args.stage_list)
            self._setup_classes()
        return self._raw

    def meta(self) -> pd.DataFrame:
        if self._meta is None:
            self.raw()
        return self._meta

    def features(self) -> np.ndarray:
        if self._X is None:
            self._X = load_or_compute_features(self.raw(), self.meta(), self.cache, self.args.chunk_size,
                                               self.args.n_jobs, self.args.force and "features" in self.args.stage_list)
        return self._X

    def _setup_classes(self) -> None:
        meta = self._meta
        if "is_flat" not in meta.columns:   # cache written by an older version: derive from the raw epochs
            raw = self._raw
            meta["is_flat"] = np.concatenate([
                np.asarray(raw[i:i + 5000]).max(axis=1) == np.asarray(raw[i:i + 5000]).min(axis=1)
                for i in range(0, raw.shape[0], 5000)])
        flat = meta["is_flat"].to_numpy(dtype=bool)
        vals, counts = np.unique(meta["label"].to_numpy(), return_counts=True)
        self.class_values = [int(v) for v in vals]
        lm = parse_label_map(self.args.label_map)
        if lm is None:
            self.class_names = [f"class{v}" for v in self.class_values]
            LOG.warning("=" * 78)
            LOG.warning("No --label-map given. Class counts: %s", dict(zip(self.class_values, counts.tolist())))
            LOG.warning("Classes are reported as %s. The integer -> {Wake, SWS, REM} mapping MUST be", self.class_names)
            LOG.warning("confirmed by the author (BDHSC data description) and passed as e.g. --label-map \"0:Wake,1:SWS,2:REM\".")
            LOG.warning("=" * 78)
        else:
            missing = [v for v in self.class_values if v not in lm]
            if missing:
                raise ValueError(f"--label-map lacks entries for labels {missing}")
            self.class_names = [lm[v] for v in self.class_values]
        rem = [c for c in self.class_names if "rem" in c.lower()]
        if rem:
            self.rem_class = rem[0]
        else:
            self.rem_class = self.class_names[int(np.argmin(counts))]
            LOG.warning("No class named REM; using the rarest class '%s' as the 'REM' class for REM-F1 statistics",
                        self.rem_class)
        LOG.info("[data] classes %s counts %s (REM class for statistics: %s)", self.class_names,
                 counts.tolist(), self.rem_class)
        lab = meta["label"].to_numpy()
        flat_by_class = {n: int(((lab == v) & flat).sum()) for n, v in zip(self.class_names, self.class_values)}
        if flat.any():
            LOG.warning("[data] %d constant (drop-out) epochs = %.2f%% of the data are kept; by class %s. Their "
                        "spectral features, Hjorth parameters, MMD and PAC are 0 (see results/data_summary.json).",
                        int(flat.sum()), 100.0 * flat.mean(), flat_by_class)
        summary = {"n_epochs": int(len(meta)), "n_animals": int(meta["animal"].nunique()),
                   "n_recordings": int(meta.groupby(["animal", "day"]).ngroups),
                   "class_values": self.class_values, "class_names": self.class_names,
                   "class_counts": {n: int(c) for n, c in zip(self.class_names, counts)},
                   "label_map_confirmed": lm is not None,
                   "n_flat_epochs": int(flat.sum()), "flat_epochs_by_class": flat_by_class,
                   "flat_epochs_per_recording": [{"animal": int(a), "day": int(d), "n_flat": int(c)} for (a, d), c in
                                                 meta.groupby(["animal", "day"])["is_flat"].sum().items()],
                   "per_recording": meta.groupby(["animal", "day"])["label"].value_counts().unstack(fill_value=0)
                   .reset_index().to_dict(orient="records")}
        with open(self.results / "data_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, default=float)

    def y(self) -> np.ndarray:
        lab = self.meta()["label"].to_numpy()
        return np.searchsorted(np.asarray(self.class_values), lab).astype(np.int64)

    def K(self) -> int:
        return len(self.class_values)


def make_splits(protocol: str, meta: pd.DataFrame, y: np.ndarray, repeats: int = 3, seed: int = 42):
    """Return a list of (fold_name, train_idx, test_idx)."""
    n = len(meta)
    idx = np.arange(n)
    if protocol == "loao":
        g = meta["animal"].to_numpy()
        return [(f"animal{a}", idx[g != a], idx[g == a]) for a in np.unique(g)]
    if protocol == "loro":
        g = (meta["animal"].astype(str) + "_" + meta["day"].astype(str)).to_numpy()
        return [(f"rec{r}", idx[g != r], idx[g == r]) for r in np.unique(g)]
    if protocol == "random80_20":
        out = []
        for r in range(repeats):
            tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed + r)
            out.append((f"seed{seed + r}", tr, te))
        return out
    raise ValueError(protocol)


def run_cv(ctx: Context, protocol: str, model_name: str, tag: str, X: Optional[np.ndarray],
           feature_names: Sequence[str], splits, use_raw: bool = False) -> Optional[pd.DataFrame]:
    """Run one model under one protocol; persist per-fold metrics, pooled
    out-of-fold predictions, aggregated summary and bootstrap CIs."""
    folds_csv = ctx.results / f"cv_{tag}_folds.csv"
    oof_npz = ctx.results / f"oof_{tag}.npz"
    if folds_csv.exists() and oof_npz.exists() and not ctx.args.force:
        LOG.info("[%s] outputs exist, skipping (use --force to recompute)", tag)
        return pd.read_csv(folds_csv)
    y = ctx.y()
    meta = ctx.meta()
    K = ctx.K()
    names = ctx.class_names
    rows = []
    oof_idx, oof_true, oof_pred, oof_proba, oof_fold = [], [], [], [], []
    animal = meta["animal"].to_numpy()
    recording = (meta["animal"].astype(str) + "_" + meta["day"].astype(str)).to_numpy()
    tm = Timer(tag)
    for k, (fname, tr, te) in enumerate(splits):
        t0 = time.time()
        # Leakage guards: the test epochs are never in the training fold and, under the group-wise
        # protocols, neither is any epoch of the test animal (LOAO) / test recording (LORO).
        assert np.intersect1d(tr, te).size == 0, f"{tag} {fname}: train/test epoch overlap"
        if protocol == "loao":
            assert not (set(animal[tr].tolist()) & set(animal[te].tolist())), \
                f"{tag} {fname}: test animal present in the training fold"
        elif protocol == "loro":
            assert not (set(recording[tr].tolist()) & set(recording[te].tolist())), \
                f"{tag} {fname}: test recording present in the training fold"
        if use_raw:
            groups_tr = animal[tr]
            y_pred, proba = fit_predict_cnn(ctx.raw(), y, tr, te, groups_tr, K, ctx.cfg, ctx.args.seed,
                                            ctx.args.n_jobs, groups_te=animal[te])
        else:
            y_pred, proba = fit_predict_features(model_name, X[tr], y[tr], X[te], feature_names, ctx.cfg,
                                                 ctx.args.seed, ctx.args.n_jobs)
        if k == 0 and model_name.startswith("xgb"):
            check_probabilities(proba, tag)
        m = compute_metrics(y[te], y_pred, names)
        m.update({"fold": fname, "n_train": int(len(tr)), "n_test": int(len(te)), "fit_seconds": time.time() - t0})
        rows.append(m)
        oof_idx.append(te)
        oof_true.append(y[te])
        oof_pred.append(y_pred)
        oof_proba.append(proba if proba is not None else np.full((len(te), K), np.nan))
        oof_fold.append(np.full(len(te), k))
        el = tm.elapsed()
        LOG.info("[%s] fold %d/%d %-10s acc=%.3f macroF1=%.3f kappa=%.3f (%s; est. remaining %s)", tag, k + 1,
                 len(splits), fname, m["accuracy"], m["f1_macro"], m["kappa"], hms(time.time() - t0),
                 hms(el / (k + 1) * (len(splits) - k - 1)))
    df = pd.DataFrame(rows)
    df.insert(0, "model", model_name)
    df.insert(0, "protocol", protocol)
    df.to_csv(folds_csv, index=False)
    idx = np.concatenate(oof_idx)
    np.savez_compressed(oof_npz, idx=idx, y_true=np.concatenate(oof_true), y_pred=np.concatenate(oof_pred),
                        proba=np.vstack(oof_proba), fold=np.concatenate(oof_fold),
                        animal=meta["animal"].to_numpy()[idx], day=meta["day"].to_numpy()[idx])
    aggregate_folds(df, names).to_csv(ctx.results / f"cv_{tag}_summary.csv", index=False)
    # bootstrap on pooled out-of-fold predictions (first repeat only for random splits)
    first = np.concatenate(oof_fold) == 0 if protocol == "random80_20" else np.ones(len(idx), dtype=bool)
    boot = bootstrap_pooled(np.concatenate(oof_true)[first], np.concatenate(oof_pred)[first],
                            meta["animal"].to_numpy()[idx][first], names, ctx.cfg["bootstrap"], ctx.args.seed)
    boot.to_csv(ctx.results / f"bootstrap_{tag}.csv", index=False)
    cluster = (meta["animal"].astype(str) + "_" + meta["day"].astype(str)).to_numpy()[idx][first] \
        if protocol == "loro" else meta["animal"].to_numpy()[idx][first]
    bootstrap_cluster(np.concatenate(oof_true)[first], np.concatenate(oof_pred)[first], cluster, names,
                      ctx.cfg["bootstrap"], ctx.args.seed).to_csv(ctx.results / f"bootstrap_cluster_{tag}.csv",
                                                                 index=False)
    ctx.timings[tag] = tm.elapsed()
    LOG.info("[%s] done in %s: acc %.3f +- %.3f, macro-F1 %.3f +- %.3f, kappa %.3f +- %.3f", tag, hms(tm.elapsed()),
             df["accuracy"].mean(), df["accuracy"].std(), df["f1_macro"].mean(), df["f1_macro"].std(),
             df["kappa"].mean(), df["kappa"].std())
    return df


def feature_matrix(ctx: Context, cols: Sequence[str]) -> np.ndarray:
    X = ctx.features()
    pos = {f: i for i, f in enumerate(ALL_FEATURES)}
    return np.ascontiguousarray(X[:, [pos[c] for c in cols]])


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def stage_loao(ctx: Context) -> None:
    models = [m for m in ctx.args.models.split(",") if m in FEATURE_MODELS]
    X = feature_matrix(ctx, ENGINEERED)
    splits = make_splits("loao", ctx.meta(), ctx.y())
    LOG.info("[loao] %d folds, %d engineered features, models %s", len(splits), X.shape[1], models)
    for m in models:
        run_cv(ctx, "loao", m, f"loao__{m}", X, ENGINEERED, splits)


def stage_loro(ctx: Context) -> None:
    models = [m for m in ctx.args.loro_models.split(",") if m in FEATURE_MODELS]
    X = feature_matrix(ctx, ENGINEERED)
    splits = make_splits("loro", ctx.meta(), ctx.y())
    LOG.info("[loro] %d folds (leave-one-recording-day-out; the other day of the same animal stays in training)",
             len(splits))
    for m in models:
        run_cv(ctx, "loro", m, f"loro__{m}", X, ENGINEERED, splits)


def stage_random_split(ctx: Context) -> None:
    meta = ctx.meta()
    splits = make_splits("random80_20", meta, ctx.y(), repeats=ctx.cfg["random_repeats"], seed=ctx.args.seed)
    animal = meta["animal"].to_numpy(dtype=np.float32)[:, None]
    for variant in RANDOM_VARIANTS:
        tag = f"random80_20__xgb__{variant}"
        if (ctx.results / f"cv_{tag}_folds.csv").exists() and not ctx.args.force:
            LOG.info("[%s] outputs exist, skipping", tag)
            continue
        if variant == "engineered":
            X, names = feature_matrix(ctx, ENGINEERED), list(ENGINEERED)
        elif variant == "engineered_plus_animal":
            X = np.hstack([feature_matrix(ctx, ENGINEERED), animal])
            names = ENGINEERED + ["animal"]
        elif variant == "engineered_plus_raw":
            X = np.hstack([feature_matrix(ctx, ENGINEERED).astype(np.float32), np.asarray(ctx.raw(), dtype=np.float32)])
            names = ENGINEERED + SAMPLE_COLS
        else:  # paper replica: raw samples + notebook FFT powers + MMD + animal id
            X = np.hstack([np.asarray(ctx.raw(), dtype=np.float32),
                           feature_matrix(ctx, LEGACY + ["mmd"]).astype(np.float32), animal])
            names = SAMPLE_COLS + LEGACY + ["mmd", "animal"]
        LOG.info("[random_split] variant %s: %d columns, %d repeats", variant, X.shape[1], len(splits))
        run_cv(ctx, "random80_20", REPLICA_MODEL if variant == "paper_replica" else "xgb", tag, X, names, splits)
        del X


def replica_matrix(ctx: Context):
    """The paper's design matrix: 5,000 raw samples + notebook FFT band powers + notebook MMD + animal id."""
    animal = ctx.meta()["animal"].to_numpy(dtype=np.float32)[:, None]
    X = np.hstack([np.asarray(ctx.raw(), dtype=np.float32),
                   feature_matrix(ctx, LEGACY + ["mmd"]).astype(np.float32), animal])
    return X, SAMPLE_COLS + LEGACY + ["mmd", "animal"]


def stage_loao_replica(ctx: Context) -> None:
    """Leave-one-animal-out evaluation of the paper's own design matrix and default XGBoost
    configuration: the direct, leakage-free counterpart of the reported 91.5 %.  With the raw
    samples as columns this is the most expensive tabular stage (roughly 8 x 20 min on 4 cores
    for the real data); skip it with --stages if time is short."""
    tag = f"loao__{REPLICA_MODEL}_paper_replica"
    if (ctx.results / f"cv_{tag}_folds.csv").exists() and not ctx.args.force:
        LOG.info("[%s] outputs exist, skipping", tag)
        return
    X, names = replica_matrix(ctx)
    splits = make_splits("loao", ctx.meta(), ctx.y())
    LOG.info("[loao_replica] %d folds, %d columns (paper design matrix), model %s", len(splits), X.shape[1],
             REPLICA_MODEL)
    run_cv(ctx, "loao", REPLICA_MODEL, tag, X, names, splits)
    del X


def stage_ablation(ctx: Context) -> None:
    splits = make_splits("loao", ctx.meta(), ctx.y())
    if not (ctx.results / "cv_loao__xgb_folds.csv").exists():
        run_cv(ctx, "loao", "xgb", "loao__xgb", feature_matrix(ctx, ENGINEERED), ENGINEERED, splits)
    for name, cols in ABLATIONS.items():
        if name == "full":
            continue  # identical to loao__xgb
        run_cv(ctx, "loao", "xgb", f"loao__xgb_abl_{name}", feature_matrix(ctx, cols), cols, splits)


def stage_cnn(ctx: Context) -> None:
    if torch is None:
        LOG.warning("[cnn] torch is not installed: skipping the optional 1-D CNN baseline "
                    "(pip install torch, then re-run with --stages cnn)")
        return
    splits = make_splits("loao", ctx.meta(), ctx.y())
    run_cv(ctx, "loao", "cnn", "loao__cnn", None, [], splits, use_raw=True)


# ---- statistics ---------------------------------------------------------------
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = int(np.isfinite(p).sum())      # NaN p-values (untestable comparisons) are not tests
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i] if np.isfinite(p[i]) else np.nan
        running = max(running, val) if np.isfinite(val) else running
        adj[i] = min(1.0, running) if np.isfinite(val) else np.nan
    return adj


def paired_tests(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """a = reference (XGBoost / full), b = other; d = a - b across folds."""
    d = a - b
    n = len(d)
    out = {"n_folds": n, "mean_ref": a.mean(), "mean_other": b.mean(), "mean_diff": d.mean(),
           "sd_diff": d.std(ddof=1) if n > 1 else np.nan}
    if n > 1:
        half = stats.t.ppf(0.975, n - 1) * out["sd_diff"] / math.sqrt(n)
        out["diff_ci95_low"], out["diff_ci95_high"] = d.mean() - half, d.mean() + half
        out["cohen_dz"] = d.mean() / out["sd_diff"] if out["sd_diff"] > 0 else np.nan
        try:
            t = stats.ttest_rel(a, b)
            out["ttest_stat"], out["ttest_p"] = float(t.statistic), float(t.pvalue)
        except Exception:
            out["ttest_stat"], out["ttest_p"] = np.nan, np.nan
        nz = d[d != 0]
        if len(nz) > 0:
            try:
                try:    # exact null distribution (n <= 16 folds); older scipy: fall back to its default
                    w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact")
                    out["wilcoxon_method"] = "exact"
                except (TypeError, ValueError):
                    w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    out["wilcoxon_method"] = "scipy-auto"
                out["wilcoxon_stat"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
            except Exception:
                out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
            ranks = stats.rankdata(np.abs(nz))
            wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
            out["rank_biserial_r"] = (wp - wm) / (wp + wm)
            # exact two-sided sign test on the non-zero differences (needs only exchangeable signs)
            n_pos, n_nz = int((nz > 0).sum()), int(len(nz))
            out["sign_n_pos"], out["sign_n_nonzero"] = n_pos, n_nz
            out["sign_p"] = float(stats.binomtest(n_pos, n_nz, 0.5).pvalue)
        else:
            out["wilcoxon_stat"], out["wilcoxon_p"], out["rank_biserial_r"] = np.nan, np.nan, 0.0
            out["sign_n_pos"], out["sign_n_nonzero"], out["sign_p"] = 0, 0, np.nan
    return out


def mcnemar(y_true: np.ndarray, pred_ref: np.ndarray, pred_other: np.ndarray) -> Dict[str, float]:
    """McNemar test on the discordant pairs of two classifiers' correctness.
    Exact binomial test when b + c < 25, otherwise chi-square with continuity
    correction (no statsmodels dependency)."""
    cr, co = pred_ref == y_true, pred_other == y_true
    b = int(np.sum(cr & ~co))   # reference right, other wrong
    c = int(np.sum(~cr & co))   # reference wrong, other right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": np.nan, "p_value": 1.0, "method": "none", "odds_ratio": np.nan}
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": p, "method": "exact-binomial",
                "odds_ratio": b / c if c > 0 else np.inf}
    chi2 = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": chi2, "p_value": float(stats.chi2.sf(chi2, 1)),
            "method": "chi2-continuity-corrected", "odds_ratio": b / c if c > 0 else np.inf}


def _load_folds(ctx: Context, tag: str) -> Optional[pd.DataFrame]:
    p = ctx.results / f"cv_{tag}_folds.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_oof(ctx: Context, tag: str):
    p = ctx.results / f"oof_{tag}.npz"
    return np.load(p) if p.exists() else None


def stage_stats(ctx: Context) -> None:
    ref = _load_folds(ctx, "loao__xgb")
    if ref is None:
        LOG.warning("[stats] loao__xgb results missing; run the loao stage first")
        return
    rem_f1 = f"f1_{ctx.rem_class}"
    metrics = {"f1_macro": "macro-F1", rem_f1: "REM-F1", "accuracy": "accuracy", "kappa": "kappa"}
    comparisons = [("baseline", "loao__xgb", f"loao__{m}") for m in MODEL_ORDER if m != "xgb"]
    comparisons += [("ablation", "loao__xgb", f"loao__xgb_abl_{a}") for a in ABLATIONS if a != "full"]
    rows, mc_rows = [], []
    ref_oof = _load_oof(ctx, "loao__xgb")
    for ctype, rtag, otag in comparisons:
        other = _load_folds(ctx, otag)
        if other is None:
            continue
        merged = ref.merge(other, on="fold", suffixes=("_ref", "_oth"))
        for m, label in metrics.items():
            a = merged[f"{m}_ref"].to_numpy(dtype=np.float64)
            b = merged[f"{m}_oth"].to_numpy(dtype=np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            r = {"comparison": ctype, "reference": rtag, "other": otag, "metric": m, "metric_label": label}
            r.update(paired_tests(a[ok], b[ok]))
            rows.append(r)
        oth_oof = _load_oof(ctx, otag)
        if ref_oof is not None and oth_oof is not None and np.array_equal(ref_oof["idx"], oth_oof["idx"]):
            r = {"comparison": ctype, "reference": rtag, "other": otag, "n": int(len(ref_oof["idx"])),
                 "acc_ref": float(np.mean(ref_oof["y_pred"] == ref_oof["y_true"])),
                 "acc_other": float(np.mean(oth_oof["y_pred"] == oth_oof["y_true"]))}
            r.update(mcnemar(ref_oof["y_true"], ref_oof["y_pred"], oth_oof["y_pred"]))
            mc_rows.append(r)
    if not rows:
        LOG.warning("[stats] nothing to compare yet")
        return
    df = pd.DataFrame(rows)
    for m in metrics:
        sel = df["metric"] == m
        df.loc[sel, "wilcoxon_p_holm"] = holm(df.loc[sel, "wilcoxon_p"].to_numpy())
        df.loc[sel, "ttest_p_holm"] = holm(df.loc[sel, "ttest_p"].to_numpy())
    df.to_csv(ctx.results / "stats_paired_tests.csv", index=False)
    if mc_rows:
        mdf = pd.DataFrame(mc_rows)
        mdf["p_value_holm"] = holm(mdf["p_value"].to_numpy())
        mdf.to_csv(ctx.results / "stats_mcnemar.csv", index=False)
    LOG.info("[stats] %d paired comparisons, %d McNemar tests written", len(df), len(mc_rows))
    for _, r in df[df["metric"] == "f1_macro"].iterrows():
        LOG.info("[stats]   macro-F1 %-28s diff=%+.3f [%+.3f, %+.3f] dz=%.2f Wilcoxon p=%.3g t p=%.3g",
                 r["other"], r["mean_diff"], r.get("diff_ci95_low", np.nan), r.get("diff_ci95_high", np.nan),
                 r.get("cohen_dz", np.nan), r.get("wilcoxon_p", np.nan), r.get("ttest_p", np.nan))


# ---- calibration --------------------------------------------------------------
def calibration_tables(y_true: np.ndarray, proba: np.ndarray, class_names: Sequence[str], n_bins: int = 10):
    K = len(class_names)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, summ = [], {}
    onehot = np.eye(K)[y_true]
    summ["brier_multiclass"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    summ["log_loss"] = float(-np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], 1e-12, 1))))
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    b = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            gap = abs((pred[m] == y_true[m]).mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
            rows.append({"class": "top-label", "bin": k, "lower": edges[k], "upper": edges[k + 1], "n": int(m.sum()),
                         "mean_confidence": float(conf[m].mean()), "accuracy": float((pred[m] == y_true[m]).mean())})
    summ["ece_top_label"] = float(ece)
    summ["mce_top_label"] = float(mce)
    for c, name in enumerate(class_names):
        p, yb = proba[:, c], (y_true == c).astype(float)
        bc = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        ece_c, mce_c = 0.0, 0.0
        for k in range(n_bins):
            m = bc == k
            if m.any():
                gap = abs(yb[m].mean() - p[m].mean())
                ece_c += m.mean() * gap
                mce_c = max(mce_c, gap)
                rows.append({"class": name, "bin": k, "lower": edges[k], "upper": edges[k + 1], "n": int(m.sum()),
                             "mean_confidence": float(p[m].mean()), "accuracy": float(yb[m].mean())})
        summ[f"ece_{name}"] = float(ece_c)
        summ[f"mce_{name}"] = float(mce_c)
        summ[f"brier_{name}"] = float(np.mean((p - yb) ** 2))
    summ["n"] = int(len(y_true))
    return pd.DataFrame(rows), summ


def stage_calibration(ctx: Context) -> None:
    oof = _load_oof(ctx, "loao__xgb")
    if oof is None:
        LOG.warning("[calibration] loao__xgb predictions missing")
        return
    proba = oof["proba"]
    if not np.all(np.isfinite(proba)):
        LOG.warning("[calibration] probabilities missing")
        return
    check_probabilities(proba, "calibration")
    bins, summ = calibration_tables(oof["y_true"], proba, ctx.class_names)
    bins.to_csv(ctx.results / "calibration_bins.csv", index=False)
    pd.DataFrame([summ]).to_csv(ctx.results / "calibration_summary.csv", index=False)
    LOG.info("[calibration] Brier=%.4f  ECE(top)=%.4f  %s", summ["brier_multiclass"], summ["ece_top_label"],
             {f"ece_{c}": round(summ[f"ece_{c}"], 4) for c in ctx.class_names})


# ---- SHAP / importance --------------------------------------------------------
def stage_shap(ctx: Context) -> None:
    if shap is None:
        LOG.warning("[shap] shap not installed; skipping")
        return
    y = ctx.y()
    X = pd.DataFrame(feature_matrix(ctx, ENGINEERED), columns=ENGINEERED)
    model_path = ctx.models / "xgb_all_animals.json"
    model = make_xgb(ctx.cfg["xgb_trees"], ctx.args.seed, ctx.args.n_jobs)
    if model_path.exists() and not ctx.args.force:
        model.load_model(model_path)
        LOG.info("[shap] loaded %s", model_path)
    else:
        tm = Timer("shap-fit")
        model.fit(X, y)
        model.save_model(model_path)
        LOG.info("[shap] XGBoost fitted on all animals in %s", hms(tm.elapsed()))
    # gain / weight / total_gain importance
    booster = model.get_booster()
    imp = pd.DataFrame({"feature": ENGINEERED})
    for kind in ("gain", "weight", "total_gain", "cover"):
        sc = booster.get_score(importance_type=kind)
        imp[kind] = [sc.get(f, sc.get(f"f{i}", 0.0)) for i, f in enumerate(ENGINEERED)]
    imp.sort_values("gain", ascending=False).to_csv(ctx.results / "xgb_gain_importance.csv", index=False)
    # stratified sample for SHAP
    n = min(ctx.cfg["shap_n"], len(y))
    if n < len(y):
        sub, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=ctx.args.seed)
    else:
        sub = np.arange(len(y))
    Xs = X.iloc[sub]
    tm = Timer("shap")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv_list = [np.asarray(s) for s in sv]
    else:
        sv = np.asarray(sv)
        sv_list = [sv[:, :, c] for c in range(sv.shape[2])] if sv.ndim == 3 else [sv]
    LOG.info("[shap] SHAP values for %d epochs computed in %s", n, hms(tm.elapsed()))
    np.savez_compressed(ctx.results / "shap_values.npz", shap=np.stack(sv_list, axis=-1), X=Xs.to_numpy(),
                        y=y[sub], idx=sub, feature_names=np.array(ENGINEERED), class_names=np.array(ctx.class_names))
    ma = pd.DataFrame({"feature": ENGINEERED})
    for c, name in enumerate(ctx.class_names):
        ma[f"mean_abs_shap_{name}"] = np.abs(sv_list[c]).mean(axis=0)
    ma["mean_abs_shap_all"] = ma[[f"mean_abs_shap_{n_}" for n_ in ctx.class_names]].mean(axis=1)
    ma.sort_values("mean_abs_shap_all", ascending=False).to_csv(ctx.results / "shap_mean_abs.csv", index=False)
    # figures
    setup_style()
    K = len(ctx.class_names)
    try:
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(1, K, figsize=(4.4 * K, 4.4))
        axes = np.atleast_1d(axes)
        for c, name in enumerate(ctx.class_names):
            plt.sca(axes[c])
            shap.summary_plot(sv_list[c], Xs, feature_names=ENGINEERED, show=False, plot_size=None,
                              max_display=12, color_bar=(c == K - 1))
            axes[c].set_title(f"SHAP: {name}")
            axes[c].set_xlabel("SHAP value (log-odds)")
            axes[c].xaxis.set_major_locator(MaxNLocator(4))
            axes[c].tick_params(axis="y", labelsize=7.5)
        fig.tight_layout(w_pad=2.0)
        save_fig(fig, ctx.figures / "fig_shap_beeswarm")
    except Exception as e:  # pragma: no cover
        LOG.warning("[shap] multi-panel beeswarm failed (%s); writing one file per class", e)
        for c, name in enumerate(ctx.class_names):
            plt.figure(figsize=(4.5, 4.6))
            shap.summary_plot(sv_list[c], Xs, feature_names=ENGINEERED, show=False, max_display=15)
            plt.title(f"SHAP: {name}")
            save_fig(plt.gcf(), ctx.figures / f"fig_shap_beeswarm_{name}")
    # mean |SHAP| bar (stacked by class, fixed class colours)
    top = ma.sort_values("mean_abs_shap_all", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    left = np.zeros(len(top))
    for c, name in enumerate(ctx.class_names):
        vals = top[f"mean_abs_shap_{name}"].to_numpy()
        ax.barh(top["feature"], vals, left=left, color=class_color(c), label=name, height=0.72, linewidth=0)
        left += vals
    ax.set_xlabel("mean |SHAP value| (summed over classes)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    save_fig(fig, ctx.figures / "fig_shap_mean_abs_bar")
    # gain importance bar
    topg = imp.sort_values("gain", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    ax.barh(topg["feature"], topg["gain"], color=PALETTE[0], height=0.72, linewidth=0)
    ax.set_xlabel("XGBoost importance (average gain)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, ctx.figures / "fig_xgb_gain_importance")
    LOG.info("[shap] top features by mean|SHAP|: %s", ma.sort_values("mean_abs_shap_all", ascending=False)["feature"].head(8).tolist())


# ---- figures ------------------------------------------------------------------
def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
        "figure.dpi": 100,
    })


def class_color(c: int) -> str:
    return PALETTE[c % len(PALETTE)]


def save_fig(fig, stem: Path) -> None:
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_bar(ax, group_labels: Sequence[str], series: Sequence[Tuple[str, np.ndarray, np.ndarray, str]],
                ylabel: str, ylim=(0.0, 1.0), legend_cols: int = 4) -> None:
    """Grouped bars with fold-SD error bars; bars fill each group (no empty gap)."""
    G, S = len(group_labels), len(series)
    width = 0.8 / S
    x = np.arange(G)
    for i, (label, means, sds, color) in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        sds = np.where(np.isfinite(sds), sds, 0.0)
        ax.bar(pos, means, width * 0.92, yerr=sds, color=color, label=label, linewidth=0,
               error_kw=dict(elinewidth=0.7, capsize=1.5, capthick=0.7, ecolor="#333333"))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=min(S, legend_cols), loc="upper center", bbox_to_anchor=(0.5, -0.14))


def _summary_value(ctx: Context, tag: str, metric: str) -> Tuple[float, float]:
    p = ctx.results / f"cv_{tag}_summary.csv"
    if not p.exists():
        return np.nan, np.nan
    s = pd.read_csv(p).set_index("metric")
    if metric not in s.index:
        return np.nan, np.nan
    return float(s.loc[metric, "mean"]), float(s.loc[metric, "sd"])


def stage_figures(ctx: Context) -> None:
    setup_style()
    names = ctx.class_names
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa"]
    metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "Cohen's κ"]
    # --- models x metrics (LOAO)
    series = []
    for m in MODEL_ORDER:
        if not (ctx.results / f"cv_loao__{m}_summary.csv").exists():
            continue
        vals = np.array([_summary_value(ctx, f"loao__{m}", c) for c in metric_cols])
        series.append((MODEL_LABELS[m], vals[:, 0], vals[:, 1], MODEL_COLORS[m]))
    if series:
        fig, ax = plt.subplots(figsize=(6.8, 3.0))
        grouped_bar(ax, metric_labels, series, "Score (leave-one-animal-out, mean ± SD over 8 folds)")
        save_fig(fig, ctx.figures / "fig_models_metrics_loao")
        # --- per-class F1 by model
        series_pc = []
        for m in MODEL_ORDER:
            if not (ctx.results / f"cv_loao__{m}_summary.csv").exists():
                continue
            vals = np.array([_summary_value(ctx, f"loao__{m}", f"f1_{c}") for c in names])
            series_pc.append((MODEL_LABELS[m], vals[:, 0], vals[:, 1], MODEL_COLORS[m]))
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        grouped_bar(ax, names, series_pc, "Per-class F1 (LOAO, mean ± SD)")
        save_fig(fig, ctx.figures / "fig_perclass_f1_loao")
    # --- protocols (XGBoost)
    prot = [("loao__xgb", "Leave-one-animal-out"), ("loro__xgb", "Leave-one-recording-out"),
            (f"loao__{REPLICA_MODEL}_paper_replica", "LOAO: paper design matrix")]
    prot += [(f"random80_20__xgb__{v}", f"Random 80/20: {RANDOM_VARIANTS[v].split(' (')[0]}") for v in RANDOM_VARIANTS]
    prot = [(t, l) for t, l in prot if (ctx.results / f"cv_{t}_summary.csv").exists()]
    if prot:
        series_p = []
        for i, (tag, label) in enumerate(prot):
            vals = np.array([_summary_value(ctx, tag, c) for c in ["accuracy", "f1_macro", f"f1_{ctx.rem_class}", "kappa"]])
            series_p.append((label, vals[:, 0], vals[:, 1], PALETTE[i % len(PALETTE)]))
        fig, ax = plt.subplots(figsize=(6.8, 3.0))
        grouped_bar(ax, ["Accuracy", "F1 (macro)", f"F1 ({ctx.rem_class})", "Cohen's κ"], series_p,
                    "XGBoost score (mean ± SD over folds / repeats)", legend_cols=3)
        save_fig(fig, ctx.figures / "fig_protocol_inflation_xgb")
    # --- ablation
    abl = [(f"loao__xgb_abl_{a}" if a != "full" else "loao__xgb", ABLATION_LABELS[a]) for a in ABLATIONS]
    abl = [(t, l) for t, l in abl if (ctx.results / f"cv_{t}_summary.csv").exists()]
    if abl:
        fig, ax = plt.subplots(figsize=(6.8, 3.0))
        x = np.arange(len(abl))
        for j, (metric, label, color) in enumerate([("f1_macro", "F1 (macro)", PALETTE[0]),
                                                     (f"f1_{ctx.rem_class}", f"F1 ({ctx.rem_class})", PALETTE[1]),
                                                     ("kappa", "Cohen's κ", PALETTE[2])]):
            vals = np.array([_summary_value(ctx, t, metric) for t, _ in abl])
            sds = np.where(np.isfinite(vals[:, 1]), vals[:, 1], 0.0)
            ax.bar(x - 0.4 + 0.8 / 3 * (j + 0.5), vals[:, 0], 0.8 / 3 * 0.92, yerr=sds, color=color, label=label,
                   linewidth=0, error_kw=dict(elinewidth=0.7, capsize=1.5, capthick=0.7, ecolor="#333333"))
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in abl], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("XGBoost LOAO score (mean ± SD)")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
        save_fig(fig, ctx.figures / "fig_ablation_loao")
    # --- confusion matrix (pooled OOF, LOAO XGBoost)
    oof = _load_oof(ctx, "loao__xgb")
    if oof is not None:
        K = len(names)
        cm = confusion(oof["y_true"], oof["y_pred"], K)
        rown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * rown[i, j]:.1f}%)", ha="center", va="center",
                        color="white" if rown[i, j] > 0.55 else "#0b0b0b", fontsize=7.5)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Manual label")
        ax.set_title("XGBoost, leave-one-animal-out (pooled)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("row-normalised")
        save_fig(fig, ctx.figures / "fig_confusion_xgb_loao")
    # --- calibration
    p = ctx.results / "calibration_bins.csv"
    if p.exists():
        bins = pd.read_csv(p)
        summ = pd.read_csv(ctx.results / "calibration_summary.csv").iloc[0]
        panels = names + ["top-label"]
        fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 0.4, 2.3), sharey=True)
        for c, name in enumerate(panels):
            ax = axes[c]
            d = bins[bins["class"] == name]
            ax.plot([0, 1], [0, 1], color="#9a9a96", lw=0.8, ls="--")
            ax.plot(d["mean_confidence"], d["accuracy"], marker="o", ms=3.5, lw=1.4,
                    color=class_color(c) if name != "top-label" else "#0b0b0b")
            ece = summ["ece_top_label"] if name == "top-label" else summ[f"ece_{name}"]
            ax.set_title(f"{name}  (ECE {ece:.3f})")
            ax.set_xlabel("Predicted probability")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
        axes[0].set_ylabel("Observed frequency")
        fig.suptitle(f"XGBoost LOAO out-of-fold calibration (Brier {summ['brier_multiclass']:.3f})", y=1.02)
        save_fig(fig, ctx.figures / "fig_calibration_xgb_loao")
    LOG.info("[figures] written to %s", ctx.figures)


# ---- report -------------------------------------------------------------------
def _nz(v) -> float:
    """None (JSON null) -> NaN, otherwise the float value (0.0 stays 0.0)."""
    return np.nan if v is None else float(v)


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    mean, sd = _nz(mean), _nz(sd)
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def tex(s) -> str:
    """Escape the LaTeX special characters that can occur in class / feature / model labels."""
    return re.sub(r"([_%&#])", r"\\\1", str(s))


def _summary_dict(ctx: Context, tag: str) -> Optional[dict]:
    p = ctx.results / f"cv_{tag}_summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    out = {"metrics": {r["metric"]: {k: (None if not np.isfinite(r[k]) else float(r[k]))
                                     for k in ("mean", "sd", "ci95_low", "ci95_high")} for _, r in s.iterrows()},
           "n_folds": int(s["n_folds"].max())}
    b = ctx.results / f"bootstrap_{tag}.csv"
    if b.exists():
        bd = pd.read_csv(b)
        out["bootstrap"] = {r["metric"]: {"point": float(r["point"]), "ci95_low": float(r["ci95_low"]),
                                          "ci95_high": float(r["ci95_high"])} for _, r in bd.iterrows()}
    f = ctx.results / f"cv_{tag}_folds.csv"
    if f.exists():
        fd = pd.read_csv(f)
        out["fit_seconds_total"] = float(fd["fit_seconds"].sum())
        out["folds"] = fd[["fold", "accuracy", "f1_macro", "kappa"]].to_dict(orient="records")
    return out


def stage_report(ctx: Context) -> None:
    names = ctx.class_names
    rem = ctx.rem_class
    res = ctx.results
    summary: Dict[str, object] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(ctx.args).items()},
        "effective_settings": ctx.cfg,
        "feature_set": {"engineered": ENGINEERED, "groups": FEATURE_GROUPS, "ablations": ABLATIONS},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": __import__("sklearn").__version__, "xgboost": xgb.__version__,
                     "shap": getattr(shap, "__version__", None), "torch": getattr(torch, "__version__", None)},
    }
    ds = res / "data_summary.json"
    if ds.exists():
        summary["data"] = json.load(open(ds))
    protocols: Dict[str, dict] = {"loao": {}, "loro": {}, "random80_20": {}, "loao_ablation": {}}
    for m in MODEL_ORDER:
        d = _summary_dict(ctx, f"loao__{m}")
        if d:
            protocols["loao"][m] = d
        d = _summary_dict(ctx, f"loro__{m}")
        if d:
            protocols["loro"][m] = d
    for v in RANDOM_VARIANTS:
        d = _summary_dict(ctx, f"random80_20__xgb__{v}")
        if d:
            protocols["random80_20"][v] = d
    d = _summary_dict(ctx, f"loao__{REPLICA_MODEL}_paper_replica")
    if d:
        protocols["loao"]["paper_replica"] = d
    for a in ABLATIONS:
        d = _summary_dict(ctx, "loao__xgb" if a == "full" else f"loao__xgb_abl_{a}")
        if d:
            d["n_features"] = len(ABLATIONS[a])
            protocols["loao_ablation"][a] = d
    summary["protocols"] = protocols
    for fn, key in (("stats_paired_tests.csv", "stats_paired"), ("stats_mcnemar.csv", "stats_mcnemar"),
                    ("calibration_summary.csv", "calibration"), ("shap_mean_abs.csv", "shap_mean_abs"),
                    ("xgb_gain_importance.csv", "xgb_gain_importance")):
        p = res / fn
        if p.exists():
            df = pd.read_csv(p)
            summary[key] = json.loads(df.head(60).to_json(orient="records"))
    summary["timings_seconds"] = ctx.timings
    with open(ctx.out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o))

    # ---- LaTeX: main table
    lines = ["% Generated by run_rebuttal_experiments.py -- mean +- SD across folds",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "Protocol & Model & Accuracy & Precision (macro) & Recall (macro) & F1 (macro) & Cohen's $\\kappa$ \\\\",
             "\\midrule"]
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa"]

    def row(protocol_label: str, model_label: str, tag: str) -> Optional[str]:
        d = _summary_dict(ctx, tag)
        if not d:
            return None
        cells = [fmt_pm(d["metrics"][c]["mean"], d["metrics"][c]["sd"]) for c in cols]
        return f"{protocol_label} & {tex(model_label)} & " + " & ".join(cells) + " \\\\"

    n_animals, n_rec = ctx.meta()["animal"].nunique(), ctx.meta().groupby(["animal", "day"]).ngroups
    blocks = [(f"LOAO ({n_animals} folds)", [(MODEL_LABELS[m], f"loao__{m}") for m in MODEL_ORDER]
               + [(f"XGBoost, {RANDOM_VARIANTS['paper_replica']}", f"loao__{REPLICA_MODEL}_paper_replica")]),
              (f"LORO ({n_rec} folds)", [(MODEL_LABELS[m], f"loro__{m}") for m in MODEL_ORDER]),
              ("Random 80/20 epochs", [(f"XGBoost, {RANDOM_VARIANTS[v]}", f"random80_20__xgb__{v}") for v in RANDOM_VARIANTS])]
    for label, items in blocks:
        rows = [r for r in (row(label if i == 0 else "", ml, t) for i, (ml, t) in enumerate(items)) if r]
        if rows:
            lines += rows + ["\\midrule"]
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    (res / "table_main.tex").write_text("\n".join(lines) + "\n")

    # ---- LaTeX: per-class table (XGBoost LOAO, plus per-class F1 of every model)
    d = _summary_dict(ctx, "loao__xgb")
    lines = ["% Generated by run_rebuttal_experiments.py", "\\begin{tabular}{lcccc}", "\\toprule",
             "Class & Precision & Recall & F1 & Bootstrap 95\\% CI of F1 \\\\", "\\midrule"]
    if d:
        for c in names:
            b = d.get("bootstrap", {}).get(f"f1_{c}", {})
            ci = f"[{b['ci95_low']:.3f}, {b['ci95_high']:.3f}]" if b else "--"
            lines.append(f"{tex(c)} & " + " & ".join(
                fmt_pm(d["metrics"][f"{m}_{c}"]["mean"], d["metrics"][f"{m}_{c}"]["sd"])
                for m in ("precision", "recall", "f1")) + f" & {ci} \\\\")
    lines += ["\\midrule", "\\multicolumn{5}{l}{\\textit{Per-class F1 of every model (LOAO): "
              + ", ".join(tex(n) for n in names) + "}} \\\\"]
    pad = " & " * max(0, 4 - len(names))
    for m in MODEL_ORDER:
        dm = _summary_dict(ctx, f"loao__{m}")
        if dm:
            lines.append(f"{tex(MODEL_LABELS[m])} & " + " & ".join(
                fmt_pm(dm["metrics"][f"f1_{c}"]["mean"], dm["metrics"][f"f1_{c}"]["sd"])
                for c in names) + pad + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (res / "table_perclass.tex").write_text("\n".join(lines) + "\n")

    # ---- LaTeX: ablation table
    st = pd.read_csv(res / "stats_paired_tests.csv") if (res / "stats_paired_tests.csv").exists() else None
    lines = ["% Generated by run_rebuttal_experiments.py -- XGBoost, leave-one-animal-out",
             "\\begin{tabular}{lccccc}", "\\toprule",
             f"Feature set & \\#feat. & Accuracy & F1 (macro) & F1 ({tex(rem)}) & $\\Delta$F1 vs.\\ full (Wilcoxon $p$) \\\\",
             "\\midrule"]
    for a in ABLATIONS:
        tag = "loao__xgb" if a == "full" else f"loao__xgb_abl_{a}"
        da = _summary_dict(ctx, tag)
        if not da:
            continue
        delta = "--"
        if st is not None and a != "full":
            s = st[(st["other"] == tag) & (st["metric"] == "f1_macro")]
            if len(s):
                r = s.iloc[0]
                delta = f"{-r['mean_diff']:+.3f} ({r['wilcoxon_p']:.3f})" if np.isfinite(r["wilcoxon_p"]) else f"{-r['mean_diff']:+.3f}"
        lines.append(f"{tex(ABLATION_LABELS[a])} & {len(ABLATIONS[a])} & "
                     + " & ".join(fmt_pm(da["metrics"][c]["mean"], da["metrics"][c]["sd"])
                                  for c in ("accuracy", "f1_macro", f"f1_{rem}")) + f" & {delta} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (res / "table_ablation.tex").write_text("\n".join(lines) + "\n")
    LOG.info("[report] summary.json and LaTeX tables written to %s", res)


# ----------------------------------------------------------------------------
# Runtime estimate (rough, printed before running)
# ----------------------------------------------------------------------------
def print_runtime_estimate(n_epochs: int, cfg: dict, stages: Sequence[str], n_jobs: int) -> None:
    scale = n_epochs / 138240.0
    cores = max(1, n_jobs)
    core_f = 8.0 / cores
    tree_f = cfg["xgb_trees"] / 500.0
    est = {
        "data": 8 * scale, "features": 3 * scale * core_f,
        "loao": (8 * 1.5 * tree_f + 8 * 2.5 * cfg["rf_trees"] / 500 + 8 * 1.5 + 8 * 2 + 8 * 0.5) * scale * core_f,
        "loro": 16 * 1.4 * tree_f * scale * core_f * 2,
        "random_split": cfg["random_repeats"] * (2 * 1.5 * tree_f + 2 * 20 * tree_f) * scale * core_f,
        "ablation": 7 * 8 * 1.3 * tree_f * scale * core_f,
        "cnn": 8 * 15 * cfg["cnn_epochs"] / 10 * scale * core_f if torch is not None else 0.0,
        "stats": 0.5, "calibration": 0.2, "shap": 3 * scale * core_f + 1, "figures": 0.5, "report": 0.2,
    }
    total = sum(est[s] for s in stages if s in est)
    LOG.info("Rough runtime estimate for %s epochs on %d cores (minutes): %s  -> total ~%.0f min (%.1f h)",
             f"{n_epochs:,}", cores, {s: round(est[s], 1) for s in stages if s in est}, total, total / 60)
    LOG.info("(The CPU-only CNN is the most variable part; a GPU reduces it to minutes.)")


# ----------------------------------------------------------------------------
# Self-check of the feature implementations
# ----------------------------------------------------------------------------
def calculate_mmd_np(data):
    """
    Calculates the Maximum Minimum Distance (MMD) for a given EEG data segment.

    Args:
      data: A NumPy array of shape (5000,) representing the EEG data segment.

    Returns:
      A float representing the MMD value for the segment.
    """

    mmd = 0
    data = np.array(data)
    segments = data.reshape(-1, 500)  # Reshape data into 10 segments of 500 points each

    for segment in segments:
        min_idx = np.argmin(segment)
        max_idx = np.argmax(segment)
        min_value = segment[min_idx]
        max_value = segment[max_idx]
        mmd += np.sqrt(((max_idx - min_idx) ** 2) + ((max_value - min_value) ** 2))

    return mmd


def selfcheck() -> None:
    """Compare the vectorised MMD with the notebook's ``calculate_mmd_np``
    (copied verbatim above) and sanity-check the other implementations."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, N_SAMPLES)) * 1e-5           # volts, realistic scale
    x[5] = FLAT_VALUE                                            # flat epoch
    x[6, :2500] = FLAT_VALUE                                     # half flat
    x[7] = np.round(x[7], 6)                                     # many ties -> first-occurrence argmax/argmin
    x[8] = rng.standard_normal(N_SAMPLES) * 1e-3                 # large amplitude: amplitude term matters
    x[9] = rng.standard_normal(N_SAMPLES) * 100.0                # microvolt-scale amplitudes
    ref = np.array([calculate_mmd_np(row) for row in x])
    got = mmd_notebook_vectorised(x)
    assert np.allclose(ref, got, rtol=1e-12, atol=0.0), "vectorised MMD differs from the notebook implementation"
    n_exact = int(np.sum(ref == got))
    # MMD_uV must equal the notebook function applied to a microvolt signal
    ref_uv = np.array([calculate_mmd_np(row) for row in x * 1e6])
    got_uv, dt, damp = mmd_components(x * 1e6)
    assert np.allclose(ref_uv, got_uv, rtol=1e-12, atol=0.0)
    assert np.all(dt >= 0) and np.all(damp >= 0) and np.all(got_uv <= dt + damp + 1e-9)
    print(f"[selfcheck] MMD: {len(x)} random epochs, max |diff| = {np.max(np.abs(ref - got)):.3e}, "
          f"{n_exact}/{len(x)} bit-identical  -> PASS")
    share_v = np.median(1.0 - dt[:5] / np.maximum(got[:5], 1e-12))       # realistic-scale epochs only
    share_uv = np.median(damp[:5] / np.maximum(dt[:5] + damp[:5], 1e-12))
    print(f"[selfcheck] MMD on volts is dominated by the index term: median relative amplitude contribution "
          f"= {share_v:.1e} (volts) vs {share_uv:.2f} (microvolts)")
    # PAC: a theta-phase-modulated gamma signal must score higher than an unmodulated one
    t = np.arange(N_SAMPLES) / FS
    sos = signal.butter(4, [30, 100], btype="bandpass", fs=FS, output="sos")
    gam = signal.sosfilt(sos, rng.standard_normal((2, N_SAMPLES)), axis=-1)
    th = 2 * np.pi * 7.5 * t
    coupled = 25 * np.sin(th) + 8 * (1 + 0.9 * np.cos(th)) * gam[0] + 5 * rng.standard_normal(N_SAMPLES)
    uncoupled = 25 * np.sin(th) + 8 * gam[1] + 5 * rng.standard_normal(N_SAMPLES)
    mi = pac_modulation_index(np.vstack([coupled, uncoupled]))
    assert mi[0] > 3 * mi[1] and mi[0] > 0.01, f"PAC modulation index failed: {mi}"
    print(f"[selfcheck] PAC MI coupled={mi[0]:.4f} uncoupled={mi[1]:.4f} -> PASS")
    # metrics from confusion matrix vs sklearn
    yt = rng.integers(0, 3, 500)
    yp = np.where(rng.random(500) < 0.7, yt, rng.integers(0, 3, 500))
    m = compute_metrics(yt, yp, ["a", "b", "c"])
    assert np.isclose(m["accuracy"], accuracy_score(yt, yp))
    assert np.isclose(m["f1_macro"], f1_score(yt, yp, average="macro"))
    assert np.isclose(m["precision_weighted"], precision_score(yt, yp, average="weighted"))
    assert np.isclose(m["recall_macro"], recall_score(yt, yp, average="macro"))
    assert np.isclose(m["kappa"], cohen_kappa_score(yt, yp))
    print("[selfcheck] confusion-matrix metrics match scikit-learn -> PASS")
    # full feature extraction on the test epochs
    F = extract_features_chunk(x)
    assert F.shape == (len(x), len(ALL_FEATURES)) and np.all(np.isfinite(F))
    rel = F[:, [ALL_FEATURES.index(f"rel_{b}") for b in BANDS]].sum(axis=1)
    assert np.all(rel[rel > 0] <= 1.0 + 1e-9)
    flat_row = F[5]
    for name in ["abs_delta", "total_power", "spectral_entropy", "hjorth_activity", "hjorth_mobility",
                 "hjorth_complexity", "line_length", "zero_crossing_rate", "mmd", "mmd_uV", "pac_mi_theta_gamma"]:
        assert abs(flat_row[ALL_FEATURES.index(name)]) < 1e-9, f"flat epoch: {name} = {flat_row[ALL_FEATURES.index(name)]}"
    print("[selfcheck] constant (flat) epoch -> all spectral / Hjorth / MMD / PAC features are exactly 0, no NaN -> PASS")
    print(f"[selfcheck] feature extraction ({len(ALL_FEATURES)} features) finite, relative powers <= 1 -> PASS")
    # XGBoost predict_proba under multi:softmax must return proper probabilities
    clf = make_xgb(20, 0, 1).fit(F[:, :len(ENGINEERED)], yt[:len(x)])
    pr = clf.predict_proba(F[:, :len(ENGINEERED)])
    assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-5) and not np.all(np.isclose(pr.max(axis=1), 1.0))
    print(f"[selfcheck] xgboost {xgb.__version__}: predict_proba with multi:softmax returns softmax "
          f"probabilities (rows sum to 1, not one-hot) -> PASS")
    print("[selfcheck] ALL CHECKS PASSED")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, default=None, help="directory with {animal}_{day}.csv files")
    p.add_argument("--out-dir", type=str, default="./rebuttal_out", help="output directory")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="generate N synthetic epochs per animal-day (16 files) into OUT/synthetic_data and use them")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU threads (-1 = all)")
    p.add_argument("--quick", action="store_true", help="fewer trees / iterations / bootstraps (smoke test)")
    p.add_argument("--stages", type=str, default="all", help="comma list of stages: " + ",".join(STAGES))
    p.add_argument("--label-map", type=str, default=None, help='e.g. "0:Wake,1:SWS,2:REM"')
    p.add_argument("--models", type=str, default=",".join(FEATURE_MODELS), help="LOAO feature models")
    p.add_argument("--loro-models", type=str, default="xgb,xgb_balanced", help="models for the LORO protocol")
    p.add_argument("--random-repeats", type=int, default=3, help="repeats of the random 80/20 split (first = seed 42)")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples on pooled predictions")
    p.add_argument("--shap-n", type=int, default=5000, help="epochs in the stratified SHAP sample")
    p.add_argument("--svm-max-train", type=int, default=20000, help="max training epochs for the RBF SVM")
    p.add_argument("--cnn-epochs", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=2000, help="epochs per feature-extraction chunk")
    p.add_argument("--raw-dtype", choices=["float32", "float64"], default="float32",
                   help="dtype of the cached raw epochs (float64 reproduces the notebook MMD bit-exactly)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="recompute stages whose outputs exist")
    p.add_argument("--selfcheck", action="store_true", help="verify MMD/PAC/metric implementations and exit")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.selfcheck:
        selfcheck()
        return 0
    out = Path(args.out_dir)
    setup_logging(out)
    if args.n_jobs is None or args.n_jobs < 1:
        args.n_jobs = os.cpu_count() or 1
    args.stage_list = STAGES if args.stages.strip() == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in args.stage_list if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; valid: {STAGES}")
    cfg = {
        "xgb_trees": 60 if args.quick else 500,
        "xgb_default_trees": 20 if args.quick else 100,
        "rf_trees": 100 if args.quick else 500,
        "mlp_max_iter": 60 if args.quick else 200,
        "svm_max_train": min(args.svm_max_train, 4000) if args.quick else args.svm_max_train,
        "cnn_epochs": min(args.cnn_epochs, 2) if args.quick else args.cnn_epochs,
        "bootstrap": min(args.bootstrap, 200) if args.quick else args.bootstrap,
        "shap_n": min(args.shap_n, 1000) if args.quick else args.shap_n,
        "random_repeats": min(args.random_repeats, 2) if args.quick else args.random_repeats,
    }
    LOG.info("run_rebuttal_experiments.py  args=%s", {k: v for k, v in vars(args).items() if k != "stage_list"})
    LOG.info("effective settings: %s", cfg)
    np.random.seed(args.seed)

    if args.synthetic > 0:
        sdir = out / "synthetic_data"
        if not sdir.exists() or args.force or len(list(sdir.glob("*.csv"))) < 16:
            generate_synthetic_dataset(sdir, args.synthetic, args.seed)
        args.data_dir = str(sdir)
        if args.label_map is None:
            args.label_map = "0:Wake,1:SWS,2:REM"
        LOG.warning("SYNTHETIC DATA RUN: results are for smoke-testing the pipeline only")
    if args.data_dir is None:
        raise SystemExit("--data-dir is required (or use --synthetic N for a smoke test)")

    ctx = Context(args, cfg)
    t_all = Timer("all")
    ctx.raw()  # data stage (cached) + class setup
    print_runtime_estimate(len(ctx.meta()), cfg, args.stage_list, args.n_jobs)
    runners = {
        "data": lambda: None,
        "features": lambda: ctx.features(),
        "loao": lambda: stage_loao(ctx),
        "loro": lambda: stage_loro(ctx),
        "random_split": lambda: stage_random_split(ctx),
        "ablation": lambda: stage_ablation(ctx),
        "loao_replica": lambda: stage_loao_replica(ctx),
        "cnn": lambda: stage_cnn(ctx),
        "stats": lambda: stage_stats(ctx),
        "calibration": lambda: stage_calibration(ctx),
        "shap": lambda: stage_shap(ctx),
        "figures": lambda: stage_figures(ctx),
        "report": lambda: stage_report(ctx),
    }
    for s in STAGES:
        if s not in args.stage_list:
            continue
        LOG.info("=" * 30 + f" stage: {s} " + "=" * 30)
        tm = Timer(s)
        runners[s]()
        ctx.timings[f"stage_{s}"] = tm.elapsed()
        LOG.info("stage %s finished in %s (total elapsed %s)", s, hms(tm.elapsed()), hms(t_all.elapsed()))
    LOG.info("ALL DONE in %s. Outputs: %s", hms(t_all.elapsed()), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
