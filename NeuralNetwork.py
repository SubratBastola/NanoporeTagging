# FAST 2-PHASE PIPELINE (RESUMABLE)
# Phase A: Excel+ABF -> NPZ (exact window + stride decimation, optional LPF)
# Phase B: DL inference (optical) + ±5 ms refinement + electrical + ±5 ms refinement -> one Excel per source Excel
#
# RESUMABILITY:
#   - Phase A: each row's NPZ is named deterministically from its event_id and written
#     atomically (tmp file + os.replace). On restart, if the NPZ already exists on disk
#     it is NOT rebuilt.
#   - Phase B: after every row, results are saved into a small JSON checkpoint file next
#     to the output Excel (also written atomically). On restart, rows already present in
#     the checkpoint are skipped and processing continues from the next unfinished row.
#     The final .xlsx is (re)written from the checkpoint at the end of each run.
#   - The temp NPZ folder is NO LONGER deleted on startup, so a crash (e.g. an
#     out-of-memory kill) never throws away work that's already on disk.
#
# OPTICAL LANDMARK DETECTION (Dr. Oguz's z-score edge detector + level-crossing
# refinement) is merged directly into this file below, in the section headed
# "OPTICAL LANDMARK DETECTION -- merged inline". No separate module needed.
#
# Needs: pandas, numpy, scipy, torch, pyabf, openpyxl/xlrd, tkinter

import os, glob, math, random, json, tempfile
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
import pyabf
from scipy import signal
from scipy.signal import find_peaks
import torch, torch.nn as nn


# ============================================================
# OPTICAL LANDMARK DETECTION -- merged inline (previously a
# separate optical_landmarks.py module; now self-contained, no
# external import needed beyond the standard imports above).
#
# What this does: locates the base->plateau->drop landmarks in a
# noisy optical trace.
#   1. zscore_edge_candidates()    -- Dr. Oguz's transition detector
#      (ported from dc_zscore_qc_v7.m): anti-alias resample -> zero-
#      phase Bessel LPF -> lag-difference -> trailing historical MAD
#      z-score -> persistence gate. Finds every RAPID, PERSISTENT
#      change in the window.
#   2. find_primary_event_edges()  -- pairs a rise with a fall (biggest
#      level jump, with a check that the signal genuinely stays
#      elevated in between) to directly identify the real event, even
#      inside an oversized window that also contains smaller/slower
#      decoy features.
#   3. refine_landmarks_by_level() -- pinpoints the precise corner via
#      a level-crossing threshold (X% of the way from baseline to
#      plateau), since after filtering a real step has no sharp
#      corner and the z-score edge alone still lands into the slope.
#   4. coarse_locate_optical_event() -- fallback used only when no
#      clean edge pair is found at all, to still give the DL model a
#      sane crop instead of the full oversized window.
#
# Validated by test_optical_landmarks.py (a separate regression-test
# script, not required at runtime) against synthetic data with known
# ground truth. Re-run that after changing EDGE_CFG or this section.
# ============================================================

# ============================== CONFIG ==============================
@dataclass
class EdgeConfig:
    # Dr. Oguz's z-score transition/edge detector
    EDGE_ANALYSIS_FS: float = 2000.0     # Hz, matches Oguz's analysisFsTarget_Hz
    EDGE_BESSEL_CUTOFF_HZ: float = 25.0
    EDGE_BESSEL_ORDER: int = 8
    EDGE_LAG_MS: float = 30.0
    EDGE_STATS_WINDOW_S: float = 8.0     # trailing history length; auto-shrinks for short windows
    EDGE_THRESHOLD_Z: float = 6.0
    EDGE_PERSISTENCE_MS: float = 20.0
    EDGE_MIN_MAD: float = 1e-4
    EDGE_SNAP_TOL_MS: float = 15.0       # how close a single-sided snap may search

    # Level-crossing corner refinement (see refine_landmarks_by_level)
    LEVEL_CROSS_FRAC: float = 0.08
    LEVEL_CROSS_SEARCH_S: float = 0.05
    LEVEL_CROSS_SMOOTH_MS: float = 4.0   # light denoise before threshold-crossing search

    # Fallback coarse localization (deviation-sum), used only when no clean
    # rise/fall pair is found at all
    PRECROP_PAD_S: float = 0.05
    PRECROP_DEV_MAD: float = 3.0
    PRECROP_MIN_SPAN_S: float = 0.02
    SMOOTH_FS: float = 500.0
    MA_OPT_SLOW_MS: float = 80.0

    # Minimum plausible interior-consistency fraction for a rise/fall pair
    # to be accepted as one real, continuous event (see find_primary_event_edges)
    PAIR_INTERIOR_CONSISTENCY: float = 0.8


DEFAULT_CFG = EdgeConfig()


# ============================== GENERIC DSP UTILS ==============================
def moving_average_same_len(x, win):
    win = int(max(1, round(win)))
    if win % 2 == 0: win -= 1
    pad = win // 2
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), k, mode="valid").astype(np.float32, copy=False)

def stride_decimate(x, fs_in, fs_target):
    if fs_target is None:
        return np.asarray(x, dtype=np.float32), float(fs_in)
    dec = max(1, int(round(float(fs_in) / float(fs_target))))
    return np.asarray(x[::dec], dtype=np.float32), float(fs_in / dec)

def apply_filtfilt(x, sos):
    x = np.asarray(x, dtype=np.float32)
    if sos is None or len(x) < 8:
        return x
    try:
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        y = signal.sosfilt(sos, x)
        y = signal.sosfilt(sos, y[::-1])[::-1]
        return y.astype(np.float32)

def bessel_lowpass_sos(order, cutoff_hz, fs):
    return signal.bessel(order, cutoff_hz, btype="low", output="sos", fs=fs)

def resample_poly_to_fs(x, fs_in, fs_target, max_denom=1000):
    """Anti-alias polyphase resample to (approximately) fs_target, matching
    MATLAB's resample(x, p, q) via rat(). Returns (y, actual_fs_out)."""
    frac = Fraction(fs_target / fs_in).limit_denominator(max_denom)
    p, q = frac.numerator, frac.denominator
    y = signal.resample_poly(np.asarray(x, dtype=np.float64), p, q)
    return y.astype(np.float32), float(fs_in * p / q)


# ============================== EDGE DETECTOR ==============================
def zscore_edge_candidates(t, opt, fs, t_lo, t_hi, cfg=DEFAULT_CFG):
    """Port of Dr. Oguz's transition-focused QC pipeline (dc_zscore_qc_v7.m):
    anti-alias resample -> zero-phase Bessel low-pass -> lag-difference ->
    trailing historical MAD z-score -> persistence gate. Returns candidate
    edges as a list of dicts:
      {'t': steepest-point time, 'z': signed z-score, 'direction': 'rise'|'fall',
       't_start': run start (corner), 't_end': run end (corner)}

    Unlike a single global baseline over the window, the historical stats are
    TRAILING (causal) and adaptive, so this stays robust even when the window
    contains other, smaller/slower features besides the real event.
    """
    m = (t >= t_lo) & (t <= t_hi)
    if not np.any(m):
        return []
    opt_win = np.asarray(opt[m], dtype=np.float64)
    if opt_win.size < 16:
        return []

    y, fs_a = resample_poly_to_fs(opt_win, fs, cfg.EDGE_ANALYSIS_FS)
    if y.size < 16:
        return []
    td = t_lo + np.arange(y.size) / fs_a

    sos = bessel_lowpass_sos(cfg.EDGE_BESSEL_ORDER, cfg.EDGE_BESSEL_CUTOFF_HZ, fs_a)
    y_f = apply_filtfilt(y, sos)

    lag_n = max(1, int(round((cfg.EDGE_LAG_MS / 1000.0) * fs_a)))
    lag_diff = np.full(y_f.size, np.nan, dtype=np.float64)
    if y_f.size > lag_n:
        lag_diff[lag_n:] = y_f[lag_n:] - y_f[:-lag_n]

    # Trailing (causal) historical median/MAD, excluding the current sample.
    # The stats window auto-shrinks to fit short per-event segments instead of
    # assuming a whole, minutes-long sweep like the original MATLAB script.
    window_s = min(cfg.EDGE_STATS_WINDOW_S, max(0.5, (t_hi - t_lo) * 0.4))
    win_n = max(5, int(round(window_s * fs_a)))
    win_n = min(win_n, max(5, y_f.size - 1))

    s = pd.Series(lag_diff)
    med = s.rolling(win_n, min_periods=5).median()
    mad = (s - med).abs().rolling(win_n, min_periods=5).median()
    hist_med = med.shift(1).to_numpy()
    hist_mad = mad.shift(1).to_numpy()
    hist_mad = np.where(np.isnan(hist_mad), np.nan, np.maximum(hist_mad, cfg.EDGE_MIN_MAD))

    with np.errstate(invalid="ignore", divide="ignore"):
        z = 0.6745 * (lag_diff - hist_med) / hist_mad
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    persist_n = max(1, int(round((cfg.EDGE_PERSISTENCE_MS / 1000.0) * fs_a)))
    above = np.abs(z) >= cfg.EDGE_THRESHOLD_Z
    if not np.any(above):
        return []
    padded = np.concatenate(([False], above, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.nonzero(changes == 1)[0]
    ends = np.nonzero(changes == -1)[0] - 1

    out = []
    for s0, s1 in zip(starts, ends):
        if (s1 - s0 + 1) < persist_n:
            continue
        j = s0 + int(np.argmax(np.abs(z[s0:s1 + 1])))
        direction = "rise" if lag_diff[j] > 0 else "fall"
        out.append({
            "t": float(td[j]), "z": float(z[j]), "direction": direction,
            "t_start": float(td[s0]), "t_end": float(td[s1]),
        })
    return out


# ============================== PAIRING (find the real event) ==============================
def find_primary_event_edges(t, opt, fs, t_lo, t_hi, cfg=DEFAULT_CFG):
    """Find the best rise/fall PAIR of detected transitions (by largest level
    change) -- this is the direct, physically-grounded detection of the real
    event's onset and offset corners. Returns (rise_cand, fall_cand) dicts
    (each with 't_start'/'t_end'), or (None, None) if no clean pair exists.

    Critically, a candidate pair is only accepted if the signal actually
    STAYS at the new level for essentially the entire span between them --
    otherwise a rise from one unrelated feature could get paired with a fall
    from a completely different one elsewhere in a large window just because
    the two, in isolation, have a big level difference.
    """
    cands = zscore_edge_candidates(t, opt, fs, t_lo, t_hi, cfg)
    rises = [c for c in cands if c["direction"] == "rise"]
    falls = [c for c in cands if c["direction"] == "fall"]
    if not rises or not falls:
        return None, None

    best = None; best_score = -1.0
    for r in rises:
        for f in falls:
            if f["t_start"] <= r["t_end"]:
                continue
            # True pre-event baseline, measured right before the transition
            # actually STARTS (t_start) -- not before its steepest point (t),
            # which is already partway into the ramp and gives a contaminated
            # "baseline" estimate.
            base_level = level_before(t, opt, r["t_start"])
            interior_m = (t >= r["t_end"]) & (t <= f["t_start"])
            if not np.any(interior_m):
                continue
            interior = opt[interior_m]
            plateau_level = float(np.median(interior))
            amp = abs(plateau_level - base_level)
            if amp <= 0:
                continue
            # Require the INTERIOR between the two edges to actually sit on
            # the plateau side essentially throughout -- i.e. this must be one
            # continuous elevated segment, not two disconnected edges from
            # separate features that happen to bracket a big difference.
            mid_level = (base_level + plateau_level) / 2.0
            on_plateau_side = (interior > mid_level) if plateau_level > base_level else (interior < mid_level)
            if np.mean(on_plateau_side) < cfg.PAIR_INTERIOR_CONSISTENCY:
                continue  # signal dips back toward baseline in between -> not a real single event
            if amp > best_score:
                best_score = amp; best = (r, f)
    return best if best is not None else (None, None)


def locate_event_via_edges(t, opt, fs, t_lo, t_hi, cfg=DEFAULT_CFG):
    """(lo, hi) crop span around the best detected rise/fall pair, padded --
    for use as a tight input to a downstream model. Returns (span or None, candidates)."""
    r, f = find_primary_event_edges(t, opt, fs, t_lo, t_hi, cfg)
    if r is None:
        return None, []
    lo = max(t_lo, r["t_start"] - cfg.PRECROP_PAD_S)
    hi = min(t_hi, f["t_end"] + cfg.PRECROP_PAD_S)
    if hi <= lo:
        return None, [r, f]
    return (lo, hi), [r, f]


# ============================== LEVEL-CROSSING REFINEMENT ==============================
def level_before(t, opt, t_point, span_s=0.03):
    m = (t >= t_point - span_s) & (t < t_point)
    vals = opt[m]
    return float(np.median(vals)) if vals.size else float(np.interp(t_point, t, opt))

def level_after(t, opt, t_point, span_s=0.03):
    m = (t > t_point) & (t <= t_point + span_s)
    vals = opt[m]
    return float(np.median(vals)) if vals.size else float(np.interp(t_point, t, opt))

def find_level_crossing(t, opt, lo, hi, threshold, rising, want_first=True):
    """Search t in [lo, hi] for where opt crosses `threshold`. rising=True
    looks for the signal going from below to above; False for above to below.
    want_first=True returns the earliest such crossing in the window, else
    the latest. Sub-sample precision via linear interpolation between the two
    bracketing samples. Returns None if no crossing is found."""
    m = (t >= lo) & (t <= hi)
    idx = np.nonzero(m)[0]
    if idx.size < 2:
        return None
    seg_t = t[idx]; seg_y = opt[idx]
    cond = (seg_y >= threshold) if rising else (seg_y <= threshold)
    hits = np.nonzero(cond)[0]
    if hits.size == 0:
        return None
    j = hits[0] if want_first else hits[-1]
    if j == 0:
        return float(seg_t[0])
    y0, y1 = seg_y[j - 1], seg_y[j]
    t0, t1 = seg_t[j - 1], seg_t[j]
    frac = 0.0 if y1 == y0 else min(max((threshold - y0) / (y1 - y0), 0.0), 1.0)
    return float(t0 + frac * (t1 - t0))

def refine_landmarks_by_level(t, opt, r, f, cfg=DEFAULT_CFG):
    """Given a confirmed rise/fall pair, pinpoint the onset/offset using a
    level-based threshold crossing (the 'X% of amplitude' rise/fall-time
    convention) rather than the z-score run boundary -- because after
    low-pass filtering a real step has no sharp corner, the derivative-based
    run boundary still lands measurably into the curve. Returns (onset_t, offset_t).

    cfg.LEVEL_CROSS_FRAC is the tuning knob:
      - LOWER pulls the landmark closer to flat baseline (further from plateau)
      - 0.5 is the least-biased estimate of a true INSTANTANEOUS transition's
        real timing, but is NOT generally what a human eye marks as "the corner"
      - HIGHER moves toward the plateau
    """
    frac = cfg.LEVEL_CROSS_FRAC
    search_pad_s = cfg.LEVEL_CROSS_SEARCH_S

    # Light denoise before threshold-crossing search. This runs on RAW samples,
    # not the filtered/resampled series the z-score edge detector uses -- so
    # for low-SNR data, a random noise spike can cross a threshold that sits
    # close to the noise floor, well before the real transition. A small
    # moving average (much lighter than the Bessel filter -- just enough to
    # suppress single-sample noise) fixes this without smearing the timing much.
    fs_est = 1.0 / np.median(np.diff(t)) if t.size > 1 else 1000.0
    smooth_win = max(1, int(round((cfg.LEVEL_CROSS_SMOOTH_MS / 1000.0) * fs_est)))
    opt_s = moving_average_same_len(np.asarray(opt, dtype=np.float32), smooth_win) if smooth_win > 1 else opt

    pre_base = level_before(t, opt_s, r["t_start"])
    post_base = level_after(t, opt_s, f["t_end"])
    interior_m = (t >= r["t_end"]) & (t <= f["t_start"])
    plateau_level = float(np.median(opt_s[interior_m])) if np.any(interior_m) else pre_base

    # Clamp each search window so it can never reach into the OTHER
    # transition's territory. Without this, a very short event (brief
    # plateau) lets the offset search window extend backward past the rise
    # and into pre-event baseline -- which is trivially already below the
    # offset threshold, producing a false "already crossed" result before
    # the event has even happened (and vice versa for onset on the other side).
    onset_hi = min(r["t_end"] + search_pad_s, f["t_start"])
    offset_lo = max(f["t_start"] - search_pad_s, r["t_end"])

    rising_to_plateau = plateau_level > pre_base
    onset_thresh = pre_base + frac * (plateau_level - pre_base)
    onset_t = find_level_crossing(
        t, opt_s, r["t_start"] - search_pad_s, onset_hi,
        onset_thresh, rising=rising_to_plateau, want_first=True
    )
    if onset_t is None:
        onset_t = r["t_start"]

    falling_to_base = plateau_level > post_base
    offset_thresh = post_base + frac * (plateau_level - post_base)
    offset_t = find_level_crossing(
        t, opt_s, offset_lo, f["t_end"] + search_pad_s,
        offset_thresh, rising=(not falling_to_base), want_first=True
    )
    if offset_t is None:
        offset_t = f["t_end"]

    return onset_t, offset_t


# ============================== SINGLE-SIDED FALLBACK / ORDERING ==============================
def snap_single_edge(t, opt, fs, t_lo, t_hi, c, want_dir, boundary_key, cfg=DEFAULT_CFG, tol=None):
    """Fallback for a single landmark (used only when find_primary_event_edges
    couldn't form a full rise+fall pair): look for one matching-direction edge
    near estimate c and return its corner boundary time, or None."""
    tol = (cfg.EDGE_SNAP_TOL_MS / 1000.0) if tol is None else tol
    ctx_lo = max(t_lo, c - max(1.0, tol * 10))
    ctx_hi = min(t_hi, c + max(1.0, tol * 10))
    cands = zscore_edge_candidates(t, opt, fs, ctx_lo, ctx_hi, cfg)
    cands = [x for x in cands if x["direction"] == want_dir and abs(x[boundary_key] - c) <= tol]
    if not cands:
        return None
    return min(cands, key=lambda x: abs(x[boundary_key] - c))[boundary_key]

def enforce_opt_order(pred):
    """Keep base <= plateau <= end after any override (edge-snap, primary
    edges, etc.) may have moved base/end independently of the plateau guess."""
    out = dict(pred)
    if out["opt_plateau"] < out["opt_base"]: out["opt_plateau"] = out["opt_base"]
    if out["opt_end"] < out["opt_plateau"]: out["opt_end"] = out["opt_plateau"]
    return out


# ============================== FALLBACK COARSE LOCALIZATION ==============================
def coarse_locate_optical_event(t, opt, fs, t_lo, t_hi, cfg=DEFAULT_CFG):
    """Roughly find the base->plateau->drop region inside a (possibly much
    larger) window, using robust-statistics changepoint detection on the
    slow-smoothed optical trace. Returns (lo, hi) or None if nothing stands
    out from baseline. Only used when find_primary_event_edges finds no clean
    rise/fall pair at all -- it does NOT replace the edge detector or the
    level-crossing refinement.
    """
    m = (t >= t_lo) & (t <= t_hi)
    if not np.any(m):
        return None
    opt_d, fs_d = stride_decimate(opt[m], fs, cfg.SMOOTH_FS)
    if opt_d.size < 8:
        return None
    td = t_lo + np.arange(opt_d.size) / fs_d
    w80 = max(1, int(round((cfg.MA_OPT_SLOW_MS / 1000.0) * fs_d)))
    sm = moving_average_same_len(opt_d, w80)
    med = float(np.median(sm))
    mad = float(max(1e-9, 1.4826 * np.median(np.abs(sm - med))))
    dev = np.abs(sm - med)
    over = dev > (cfg.PRECROP_DEV_MAD * mad)
    if not np.any(over):
        return None
    idx = np.nonzero(over)[0]
    # Merge into contiguous runs. Score each run by TOTAL deviation "evidence"
    # (sum of |deviation| across the run), not by how long it lasts. A brief,
    # shallow bump can otherwise tie or beat the real event on pure duration.
    splits = np.nonzero(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    scores = [float(np.sum(dev[g])) for g in groups]
    best = groups[int(np.argmax(scores))]
    lo, hi = float(td[best[0]]), float(td[best[-1]])
    if (hi - lo) < cfg.PRECROP_MIN_SPAN_S:
        return None
    lo = max(t_lo, lo - cfg.PRECROP_PAD_S)
    hi = min(t_hi, hi + cfg.PRECROP_PAD_S)
    if hi <= lo:
        return None
    return lo, hi


# ============================== TOP-LEVEL ENTRY POINT ==============================
def locate_optical_landmarks(t, opt, fs, t_lo, t_hi, cfg=DEFAULT_CFG,
                              fallback_pred=None):
    """The full decision path used by the pipeline for one event window:
      1. Try to find a clean rise/fall pair and refine it via level-crossing.
      2. If only one side is found, snap that side; keep the fallback for the other.
      3. If neither side is found, return the fallback unchanged (or a
         degenerate all-None result if no fallback was given), plus a crop
         span a caller can use to give a secondary model a tight window.

    fallback_pred: optional dict with 'opt_base'/'opt_plateau'/'opt_end' to
    fall back on (e.g. a DL model's own estimate) when edges aren't found.

    Returns (pred_dict, method_notes, crop_span_or_None).
    """
    primary_r, primary_f = find_primary_event_edges(t, opt, fs, t_lo, t_hi, cfg)
    notes = []
    crop_span = None

    if fallback_pred is not None:
        pred = dict(fallback_pred)
    else:
        pred = {"opt_base": t_lo, "opt_plateau": (t_lo + t_hi) / 2.0, "opt_end": t_hi}

    if primary_r is not None and primary_f is not None:
        onset_t, offset_t = refine_landmarks_by_level(t, opt, primary_r, primary_f, cfg)
        pred["opt_base"] = onset_t
        pred["opt_end"] = offset_t
        crop_span = (max(t_lo, primary_r["t_start"] - cfg.PRECROP_PAD_S),
                     min(t_hi, primary_f["t_end"] + cfg.PRECROP_PAD_S))
        notes.append("edges: full pair + level-crossing")
    else:
        if primary_r is not None:
            pred["opt_base"] = primary_r["t_start"]
            notes.append("edges: rise only")
        else:
            snapped = snap_single_edge(t, opt, fs, t_lo, t_hi, pred["opt_base"], "rise", "t_start", cfg)
            if snapped is not None:
                pred["opt_base"] = snapped
                notes.append("edges: rise snap")
            else:
                notes.append("edges: no rise found, fallback used")
        if primary_f is not None:
            pred["opt_end"] = primary_f["t_end"]
            notes.append("edges: fall only")
        else:
            snapped = snap_single_edge(t, opt, fs, t_lo, t_hi, pred["opt_end"], "fall", "t_end", cfg)
            if snapped is not None:
                pred["opt_end"] = snapped
                notes.append("edges: fall snap")
            else:
                notes.append("edges: no fall found, fallback used")
        found = coarse_locate_optical_event(t, opt, fs, t_lo, t_hi, cfg)
        if found is not None:
            crop_span = found

    pred = enforce_opt_order(pred)
    return pred, "; ".join(notes), crop_span


# ============================== CONFIG ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NPZ build (fast, like your script)
PAD_S          = 0.01  # 10 ms on each side of (window_start, window_end)
LPF_HZ         = 100.0 # set None to disable LPF
BESSEL_ORDER   = 6
TARGET_FS      = 5000.0
MIN_SAMPLES    = 32
MAX_WINDOW_S   = 60.0

#
USE_FIXED_CHANNELS = True
CH_ELEC, CH_OPT, CH_OPTREF = 0, 2, 3

# DL preprocess
SMOOTH_FS    = 500.0
N_SAMPLES    = 256
MA_OPT_FAST  = 20.0  # ms
MA_OPT_SLOW  = 80.0  # ms
MA_ELEC      = 8.0   # ms

# Refinement windows (WIDENED: optical ±10 ms, electrical ±20 ms)
REFINE_MS_OPT  = 10.0
REFINE_MS_ELEC = 20.0

# ---- Coarse pre-crop trigger for the optical DL model --------------------
# The model always maps whatever window it's given onto a fixed N_SAMPLES grid,
# so a window that's much bigger than the actual event loses resolution. If no
# clean edge pair is found (see EDGE_CFG below) AND the NPZ window's duration
# exceeds PRECROP_TRIGGER_S, coarse_locate_optical_event() gives the DL model a
# tight crop instead of the full oversized window as a last-resort fallback.
PRECROP_ENABLE     = True
PRECROP_TRIGGER_S  = 2.0     # only pre-crop windows longer than this

# Minimum enforced separation between electrical landmark timestamps, so that
# e.g. exit-spike and exit-baseline can never be reported as the same instant.
MIN_ELEC_GAP_S = 0.002  # 2 ms

# ---- Optical landmark detection config -----------------------------------
# All the actual tuning knobs (Dr. Oguz's z-score edge detector, the
# level-crossing corner refinement, and the deviation-sum fallback) live in
# the EdgeConfig class defined above (merged inline in this same file).
# Override defaults here only if your own hand-labeled data calls for it.
EDGE_ENABLE = True
EDGE_CFG = EdgeConfig(
    # LEVEL_CROSS_FRAC=0.08 is the main one worth tuning against your own
    # hand-labeled examples: lower pulls landmarks closer to flat baseline,
    # 0.5 is the least-biased estimate of an INSTANTANEOUS transition's true
    # timing but not what a human eye calls "the corner", higher moves toward
    # the plateau.
    #
    # EDGE_THRESHOLD_Z lowered from 6.0 -> 4.5: on synthetic tests matching a
    # noisy, continuously-wandering baseline with small/modest-amplitude
    # events, 6.0 missed ~60% of real (but borderline) transitions outright
    # (the true event's z-score landed at 5.8, just under threshold) with
    # ZERO measured false positives on pure noise down to 4.0. 4.5 splits the
    # difference. If you start seeing spurious detections on flat/noisy
    # stretches with no real event, raise this back up.
    EDGE_THRESHOLD_Z=4.5,
)

# How often (in rows) to (re)write a partial <name>__predicted.xlsx while a
# long Phase B run is still in progress, so you have something to look at /
# resume-check without waiting for every row in the Excel to finish. The
# checkpoint .json is still the authoritative resume state either way.
XLSX_FLUSH_EVERY_ROWS = 10

# If you've changed the inference/landmark logic (as opposed to the NPZ-building
# logic) since a previous run, set this to True to ignore any existing
# <name>__predicted.checkpoint.json and recompute every row's PREDICTIONS from
# scratch. NPZs are untouched either way -- they're still reused from disk, since
# NPZ *content* didn't change, only how landmarks are inferred from it.
FORCE_REDO_PHASE_B = False

# Excel columns
REQUIRED_IN_COLS = ["event_id","file_name","sensor","analytes","solution","window_start","window_end"]
OUT_COLS = [
    "event_id","file_name","sensor","analytes","solution",
    "event_start (s)","event_end (s)","notes",
    "Base (V)","Step (V)","RefBase (V)","RefStep (V)",
    "entry Base (pA)","entry Peak (pA)","exit Peak (pA)","exit Base (pA)",
    "duration","OSC","RefOSC","entry spike","exit spike",
    "event_plateau","event_plateau_t","entry_base_t","entry_peak_t","exit_peak_t","exit_base_t"
]

# ============================== UTILS ==============================
def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def design_sos(order, cutoff_hz, fs):
    if cutoff_hz is None: return None
    try:
        return signal.butter(order, cutoff_hz, btype="low", output="sos", fs=fs)
    except Exception:
        wn = float(cutoff_hz) / (fs * 0.5)
        return signal.butter(order, wn, btype="low", output="sos")

filt = apply_filtfilt  # alias: NPZ building uses this name for the Butterworth LPF too

def robust_norm(x, axis=1):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1e-9 + mad)

def safe_id(x):
    """Filesystem-safe stringification of an arbitrary event_id."""
    s = str(x).strip()
    for ch in ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s or "unknown"

def atomic_write_bytes_via(write_fn, final_path):
    """Write to a tmp file in the same directory then atomically replace."""
    d = os.path.dirname(final_path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
        raise

def atomic_write_json(path, data):
    def _write(tmp_path):
        with open(tmp_path, "w") as f:
            json.dump(data, f)
    atomic_write_bytes_via(_write, path)

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            # Corrupt/partial checkpoint (e.g. process died mid non-atomic write from
            # an older run) -> start this Excel's Phase B over rather than crash.
            print(f"  [warn] checkpoint {os.path.basename(path)} unreadable, ignoring it.")
            return {}
    return {}

# ============================== DL MODEL + INPUTS ==============================
class StrongOpt1D(nn.Module):
    def __init__(self, in_ch=4, hidden=96, n_out=3, p_drop=0.1):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 9, padding=4), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, 128), nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    def forward(self,x): return self.head(self.fe(x))

OPT_KEYS  = ["opt_base","opt_plateau","opt_end"]
OPT_IDX   = {k:i for i,k in enumerate(OPT_KEYS)}
ELEC_KEYS = ["elec_entry_base","elec_entry_peak","elec_exit_peak","elec_exit_base"]

def to_fixed_length(series_t, series_x, t_lo, t_hi, n_out):
    if t_hi <= t_lo: t_hi = t_lo + 1e-6
    grid = np.linspace(t_lo, t_hi, n_out, dtype=np.float32)
    x = np.interp(grid, series_t, series_x).astype(np.float32)
    return grid, x

def build_opt_inputs(t, opt, fs, t_lo, t_hi):
    opt_d, fs_d = stride_decimate(opt, fs, SMOOTH_FS)
    td = t[0] + np.arange(opt_d.size)/fs_d
    w20 = max(1, int(round((MA_OPT_FAST/1000.0)*fs_d)))
    w80 = max(1, int(round((MA_OPT_SLOW/1000.0)*fs_d)))
    opt_ma20 = moving_average_same_len(opt_d, w20)
    opt_ma80 = moving_average_same_len(opt_d, w80)
    grid,  opt_raw = to_fixed_length(td, opt_d,    t_lo, t_hi, N_SAMPLES)
    _,     opt_f   = to_fixed_length(td, opt_ma20, t_lo, t_hi, N_SAMPLES)
    _,     opt_s   = to_fixed_length(td, opt_ma80, t_lo, t_hi, N_SAMPLES)
    pos = np.linspace(0,1,N_SAMPLES, dtype=np.float32)
    x = np.stack([opt_raw, opt_f, opt_s, pos], axis=0)
    x = robust_norm(x, axis=1)
    return grid, x

@torch.no_grad()
def infer_opt_only(model, t, opt, fs, t_lo, t_hi):
    _, x = build_opt_inputs(t,opt,fs,t_lo,t_hi)
    x = torch.from_numpy(x[None]).to(DEVICE, dtype=torch.float32)
    frac = model(x).cpu().numpy()[0]
    dur = (t_hi - t_lo)
    ob, os_, oe = [t_lo + float(frac[OPT_IDX[k]])*dur for k in OPT_KEYS]
    def clamp(v): return float(min(max(v, np.nextafter(t_lo,t_hi)), np.nextafter(t_hi,t_lo)))
    ob=clamp(ob); os_=clamp(os_); oe=clamp(oe)
    os_ = max(ob, os_); oe = max(os_, oe)
    return {"opt_base":ob, "opt_plateau":os_, "opt_end":oe}

# ============================== REFINEMENT (± window) ==============================
def refine_optical_on_raw(opt_t, opt_y, pred):
    def widx(c, ms):
        r = ms/1000.0; m = (opt_t>=c-r)&(opt_t<=c+r); return np.nonzero(m)[0]
    out = dict(pred)
    for k in ["opt_base","opt_end"]:
        idx = widx(pred[k], REFINE_MS_OPT)
        if idx.size >= 5:
            v=[]; win=5
            for i in range(idx[0], idx[-1]-win+2): v.append(np.var(opt_y[i:i+win]))
            if v: out[k] = float(opt_t[idx[0] + int(np.argmin(v))])
    idx = widx(pred["opt_plateau"], REFINE_MS_OPT)
    if idx.size >= 3:
        seg = opt_y[idx]; pk,_ = find_peaks(seg)
        j = int(pk[np.argmax(seg[pk])]) if pk.size else int(np.argmax(seg))
        out["opt_plateau"] = float(opt_t[idx[j]])
    return out

def compute_electrical_from_opt(t, elec, fs, t_lo, t_hi, opt_pred):
    m=(t>=t_lo)&(t<=t_hi)
    if not np.any(m):
        return {k:float(t_lo) for k in ELEC_KEYS}
    ed, fs_ed = stride_decimate(elec[m], fs, SMOOTH_FS)
    elec_td = t_lo + np.arange(ed.size)/fs_ed
    w = max(1, int(round((MA_ELEC/1000.0)*fs_ed)))
    eS = moving_average_same_len(ed, w)
    b = max(6, int(0.20*len(eS)))
    base = float(np.median(eS[:b])); sig = float(max(1e-9, 1.4826*np.median(np.abs(eS[:b]-np.median(eS[:b])))))
    run = max(6, int(round((8e-3)*fs_ed)))
    mask_left = elec_td <= opt_pred["opt_base"]
    if np.any(mask_left):
        idx=None; c=0; left=np.nonzero(mask_left)[0]
        for i in left[::-1]:
            if abs(eS[i]-base) < 2.5*sig: c+=1
            else: c=0
            if c>=run: idx=i; break
        if idx is None: idx = left[-1]
        entry_base = elec_td[idx]
    else:
        entry_base = float(t_lo)
    step_t = opt_pred["opt_plateau"]
    seg = (elec_td>=step_t-0.020)&(elec_td<=step_t)
    if not np.any(seg): entry_peak = step_t
    else:
        s=eS[seg]; i0=np.nonzero(seg)[0][0]; pk,_=find_peaks(s)
        entry_peak = elec_td[i0 + (int(pk[np.argmax(s[pk])]) if pk.size else int(np.argmax(s)))]
    end_t = opt_pred["opt_end"]
    seg2 = (elec_td>=end_t-0.020)&(elec_td<=end_t)
    if not np.any(seg2): exit_peak = end_t
    else:
        s=-eS[seg2]; i0=np.nonzero(seg2)[0][0]; pk,_=find_peaks(s)
        exit_peak = elec_td[i0 + (int(pk[np.argmax(s[pk])]) if pk.size else int(np.argmax(-s)))]
    start = int(np.searchsorted(elec_td, exit_peak, side="left"))
    near = np.abs(eS - base) < 2.5*sig
    eb_idx=None; c=0
    for i in range(start, len(near)):
        if near[i]: c+=1
        else: c=0
        if c>=run: eb_idx=i; break
    exit_base = elec_td[eb_idx] if eb_idx is not None else t_hi
    def clamp(v): return float(min(max(v, np.nextafter(t_lo,t_hi)), np.nextafter(t_hi,t_lo)))
    entry_peak = min(clamp(entry_peak), clamp(step_t))
    exit_peak  = min(clamp(exit_peak),  clamp(end_t))
    entry_base = clamp(entry_base)
    exit_base  = clamp(exit_base)
    # Enforce strict ordering with a minimum separation -- previously this used
    # max(exit_base, exit_peak), which could leave them exactly equal whenever
    # the sustained-baseline search failed or fell back to t_hi. Now exit_base
    # is always pushed a real gap past exit_peak (clamped back into the window).
    if entry_peak < entry_base + MIN_ELEC_GAP_S:
        entry_peak = min(entry_base + MIN_ELEC_GAP_S, clamp(step_t))
    if exit_base < exit_peak + MIN_ELEC_GAP_S:
        exit_base = min(exit_peak + MIN_ELEC_GAP_S, t_hi)
    return {"elec_entry_base":entry_base,"elec_entry_peak":entry_peak,
            "elec_exit_peak":exit_peak,"elec_exit_base":exit_base}

def refine_electrical_on_raw(elec_t, elec_y, pred):
    def widx(c, ms, lo=None, hi=None):
        r=ms/1000.0; m=(elec_t>=c-r)&(elec_t<=c+r)
        if lo is not None: m &= (elec_t >= lo)
        if hi is not None: m &= (elec_t <= hi)
        return np.nonzero(m)[0]
    out=dict(pred)

    # Refine the two peaks first -- these searches are symmetric around the
    # coarse estimate, same as before.
    idx = widx(pred["elec_entry_peak"], REFINE_MS_ELEC)
    if idx.size>=3:
        seg=elec_y[idx]; pk,_=find_peaks(seg)
        j=int(pk[np.argmax(seg[pk])]) if pk.size else int(np.argmax(seg))
        out["elec_entry_peak"]=float(elec_t[idx[j]])
    idx = widx(pred["elec_exit_peak"], REFINE_MS_ELEC)
    if idx.size>=3:
        seg=-elec_y[idx]; pk,_=find_peaks(seg)
        j=int(pk[np.argmax(seg[pk])]) if pk.size else int(np.argmax(-seg))
        out["elec_exit_peak"]=float(elec_t[idx[j]])

    # Refine the baselines, but restrict the search to the side of the (now
    # refined) peak where they actually belong -- entry_base must precede
    # entry_peak, exit_base must follow exit_peak. This is what stops the
    # variance-minimization search from ever landing on/before the peak it's
    # supposed to be on the other side of.
    idx = widx(pred["elec_entry_base"], REFINE_MS_ELEC, hi=out["elec_entry_peak"] - MIN_ELEC_GAP_S)
    if idx.size>=5:
        v=[]; win=5
        for i in range(idx[0], idx[-1]-win+2): v.append(np.var(elec_y[i:i+win]))
        if v: out["elec_entry_base"]=float(elec_t[idx[0]+int(np.argmin(v))])
    idx = widx(pred["elec_exit_base"], REFINE_MS_ELEC, lo=out["elec_exit_peak"] + MIN_ELEC_GAP_S)
    if idx.size>=5:
        v=[]; win=5
        for i in range(idx[0], idx[-1]-win+2): v.append(np.var(elec_y[i:i+win]))
        if v: out["elec_exit_base"]=float(elec_t[idx[0]+int(np.argmin(v))])

    eb,ep,xp,xb=out["elec_entry_base"],out["elec_entry_peak"],out["elec_exit_peak"],out["elec_exit_base"]
    # Final safety net: guarantee real separation, not just non-decreasing order.
    if ep < eb + MIN_ELEC_GAP_S: ep = eb + MIN_ELEC_GAP_S
    if xp < ep + MIN_ELEC_GAP_S: xp = ep + MIN_ELEC_GAP_S
    if xb < xp + MIN_ELEC_GAP_S: xb = xp + MIN_ELEC_GAP_S
    out.update({"elec_entry_base":eb,"elec_entry_peak":ep,"elec_exit_peak":xp,"elec_exit_base":xb})
    return out

# ============================== ABF FAST CUT ==============================
def resolve_sweep_range(abf: pyabf.ABF):
    """Return list of (sweep_idx, t0, t1) in absolute seconds across whole file."""
    ranges=[]
    t = 0.0
    for s in range(abf.sweepCount):
        abf.setSweep(sweepNumber=s, channel=0)
        dur = float(abf.sweepLengthSec)
        ranges.append((s, t, t+dur))
        t += dur
    return ranges

def cut_exact_from_sweep(abf: pyabf.ABF, fs: float, sweep_idx: int, t0_abs: float, t1_abs: float):
    """Cut [t0_abs, t1_abs] but **within a single sweep** (fast)."""
    s, s0, s1 = sweep_idx
    abf.setSweep(sweepNumber=s, channel=0); dur = float(abf.sweepLengthSec)
    fs_ok = float(abf.dataRate) if hasattr(abf,"dataRate") else fs
    t0_loc = max(0.0, t0_abs - s0)
    t1_loc = min(dur, t1_abs - s0)
    i0 = int(round(t0_loc * fs_ok))
    i1 = int(round(t1_loc * fs_ok))
    if i1 <= i0: i1 = i0 + MIN_SAMPLES
    ys=[]
    for ch in range(abf.channelCount):
        abf.setSweep(sweepNumber=s, channel=ch)
        y = abf.sweepY
        ys.append(np.asarray(y[i0:i1], dtype=np.float32))
    return ys, fs_ok, (s0 + t0_loc)

def pick_channels(names, units):
    if USE_FIXED_CHANNELS:
        return CH_ELEC, CH_OPT, CH_OPTREF
    cand_e = [i for i,u in enumerate(units) if "pa" in str(u).lower() or str(u).lower().endswith("a")]
    elec = cand_e[0] if cand_e else 0
    volt = [i for i,u in enumerate(units) if str(u).lower()=="v"]
    if volt:
        ref=None
        for i in volt:
            if "ref" in str(names[i]).lower(): ref=i; break
        if ref is None and len(volt)>=2: ref=volt[1]
        if ref is None: ref=volt[0]
        opt = volt[0] if volt[0]!=ref else (volt[1] if len(volt)>=2 else volt[0])
    else:
        others=[i for i in range(len(names)) if i!=elec]
        opt = others[0] if others else elec
        ref = others[1] if len(others)>1 else opt
    return int(elec), int(opt), int(ref)

# ============================== SCAN + GUI PREVIEW ==============================
def _excel_paths(folder):
    pats=["*.xlsx","*.xls","*.xlsm","*.csv"]
    out=[]
    for pat in pats: out+=glob.glob(os.path.join(folder,pat))
    return sorted(set(out))

def scan_folder(folder):
    excels=_excel_paths(folder)
    abfs=sorted(glob.glob(os.path.join(folder,"**","*.abf"), recursive=True))
    abf_map={os.path.splitext(os.path.basename(p))[0].lower():p for p in abfs}
    matches=[]; unmatched=[]; mentioned=set()
    for xp in excels:
        try: df = pd.read_csv(xp) if xp.lower().endswith(".csv") else pd.read_excel(xp)
        except Exception: df=pd.DataFrame()
        bases=[]
        if not df.empty:
            col=None
            for c in df.columns:
                if str(c).strip().lower()=="file_name": col=c; break
            if col is not None:
                for v in df[col].dropna().astype(str).values:
                    b=os.path.splitext(os.path.basename(v.strip()))[0].lower()
                    if b: bases.append(b)
        bases=sorted(set(bases))
        this=[]
        for b in bases:
            if b in abf_map:
                this.append((b,abf_map[b])); mentioned.add(b)
            else:
                unmatched.append((xp,b))
        matches.append((xp,this))
    orphans=[p for b,p in abf_map.items() if b not in mentioned]
    print(f"[scan] Found {len(excels)} Excel(s), {len(abfs)} ABF(s), "
          f"{sum(len(m) for _,m in matches)} matched pair(s).")
    return excels, abfs, matches, unmatched, orphans

def preview_gui(folder, excels, abfs, matches, unmatched, orphans):
    root=tk.Tk(); root.title("Preview: Excel/ABF scan"); root.geometry("1100x600")
    try: root.call('wm','attributes','.','-topmost',True)
    except: pass
    frm=ttk.Frame(root,padding=10); frm.pack(fill="both",expand=True)
    info=f"Folder: {folder}\nExcels: {len(excels)}   ABFs: {len(abfs)}   Matched pairs: {sum(len(m) for _,m in matches)}   Unmatched refs: {len(unmatched)}   Orphans: {len(orphans)}"
    ttk.Label(frm,text=info).pack(anchor="w",pady=(0,8))
    paned=ttk.PanedWindow(frm,orient="horizontal"); paned.pack(fill="both",expand=True)
    left=ttk.Frame(paned,padding=6)
    lf1=ttk.Labelframe(left,text="Excels")
    lb1=tk.Listbox(lf1); sb1=ttk.Scrollbar(lf1,orient="vertical",command=lb1.yview); lb1.config(yscrollcommand=sb1.set)
    lb1.pack(side="left",fill="both",expand=True); sb1.pack(side="right",fill="y")
    for p in excels: lb1.insert("end", os.path.basename(p))
    lf1.pack(fill="both",expand=True,pady=(0,6))
    lf2=ttk.Labelframe(left,text="ABFs"); lb2=tk.Listbox(lf2); sb2=ttk.Scrollbar(lf2,orient="vertical",command=lb2.yview); lb2.config(yscrollcommand=sb2.set)
    lb2.pack(side="left",fill="both",expand=True); sb2.pack(side="right",fill="y")
    for p in abfs: lb2.insert("end", p)
    lf2.pack(fill="both",expand=True)
    paned.add(left,weight=1)
    mid=ttk.Frame(paned,padding=6); lf3=ttk.Labelframe(mid,text="Matches (excel → base → abf)")
    lb3=tk.Listbox(lf3); sb3=ttk.Scrollbar(lf3,orient="vertical",command=lb3.yview); lb3.config(yscrollcommand=sb3.set)
    lb3.pack(side="left",fill="both",expand=True); sb3.pack(side="right",fill="y")
    for xp,ml in matches:
        x=os.path.basename(xp)
        if not ml: lb3.insert("end", f"[{x}] — no matches")
        for b,ap in ml: lb3.insert("end", f"[{x}] {b} → {ap}")
    lf3.pack(fill="both",expand=True); paned.add(mid,weight=2)
    right=ttk.Frame(paned,padding=6)
    lf4=ttk.Labelframe(right,text="Unmatched Excel refs"); lb4=tk.Listbox(lf4); sb4=ttk.Scrollbar(lf4,orient="vertical",command=lb4.yview); lb4.config(yscrollcommand=sb4.set)
    lb4.pack(side="left",fill="both",expand=True); sb4.pack(side="right",fill="y")
    for xp,b in unmatched: lb4.insert("end", f"[{os.path.basename(xp)}]  {b}")
    lf4.pack(fill="both",expand=True,pady=(0,6))
    lf5=ttk.Labelframe(right,text="Orphan ABFs"); lb5=tk.Listbox(lf5); sb5=ttk.Scrollbar(lf5,orient="vertical",command=lb5.yview); lb5.config(yscrollcommand=sb5.set)
    lb5.pack(side="left",fill="both",expand=True); sb5.pack(side="right",fill="y")
    for ap in orphans: lb5.insert("end", ap)
    lf5.pack(fill="both",expand=True); paned.add(right,weight=1)
    proceed={"ok":False}
    b=ttk.Frame(frm); b.pack(fill="x",pady=8)
    ttk.Button(b,text="Cancel",command=lambda:(root.destroy())).pack(side="right",padx=(0,6))
    ttk.Button(b,text="Proceed",command=lambda:(proceed.update(ok=True),root.destroy())).pack(side="right")
    root.mainloop()
    return proceed["ok"]

def pick_folder_dialog():
    root=tk.Tk(); root.withdraw()
    try: root.call('wm','attributes','.','-topmost',True)
    except: pass
    d=filedialog.askdirectory(title="Select folder with Excel + ABF")
    root.destroy(); return d

def pick_ckpt_dialog():
    root=tk.Tk(); root.withdraw()
    try: root.call('wm','attributes','.','-topmost',True)
    except: pass
    p=filedialog.askopenfilename(title="Select trained checkpoint (.pt)", filetypes=[("PyTorch","*.pt"),("All","*.*")])
    root.destroy(); return p

# ============================== PHASE A: FAST NPZ CREATION ==============================
def npz_path_for(npz_dir, base, event_id):
    """Deterministic NPZ filename so re-runs can detect already-built segments."""
    return os.path.join(npz_dir, f"{base}__evt{safe_id(event_id)}.npz")

def open_abf(abf_path):
    """Open an ABF file and resolve its sweep boundaries ONCE.

    This is the expensive part (parses the header and loads the recording into
    memory, then walks every sweep). It should be called once per ABF file and
    reused for every event/row that references that file — not once per row.
    """
    A = pyabf.ABF(abf_path)
    fs = float(A.dataRate) if hasattr(A,"dataRate") else (1.0/float(A.dataSecPerPoint))
    ranges = resolve_sweep_range(A)
    return A, fs, ranges

def build_npz_for_row(A, fs, ranges, base, row, out_path):
    """Cut+process ONE event's window from an already-open ABF and write the
    NPZ atomically to out_path. Cheap — no file I/O beyond the final write."""
    ws = float(row["window_start"]); we = float(row["window_end"])
    t0 = max(0.0, ws - PAD_S); t1 = we + PAD_S
    if (t1 - t0) > MAX_WINDOW_S:
        raise ValueError("window too long")

    sweep = None
    for s_idx, s_lo, s_hi in ranges:
        if t0 >= s_lo and t1 <= s_hi:
            sweep = (s_idx, s_lo, s_hi); break
    if sweep is None:
        overlaps = [(s_idx, max(0.0, min(s_hi,t1)-max(s_lo,t0)), s_lo, s_hi) for s_idx,s_lo,s_hi in ranges]
        s_idx, ov, s_lo, s_hi = max(overlaps, key=lambda z:z[1])
        if ov <= 0:
            raise ValueError("window not in any sweep")
        sweep = (s_idx, s_lo, s_hi)
        t0 = max(t0, s_lo); t1 = min(t1, s_hi)

    ys, fs_ok, seg_t0_abs = cut_exact_from_sweep(A, fs, sweep, t0, t1)
    names = [A.adcNames[i] if hasattr(A,"adcNames") else f"ch{i}" for i in range(A.channelCount)]
    units = [A.adcUnits[i] if hasattr(A,"adcUnits") else "" for i in range(A.channelCount)]
    elec_i, opt_i, ref_i = pick_channels(names, units)

    elec = ys[elec_i]
    opt  = ys[opt_i]
    ref  = ys[ref_i]
    L = min(len(elec), len(opt), len(ref))
    elec, opt, ref = elec[:L], opt[:L], ref[:L]

    sos = design_sos(BESSEL_ORDER, LPF_HZ, fs_ok)
    elec = filt(elec, sos); opt = filt(opt, sos); ref = filt(ref, sos)
    elec, fs_out = stride_decimate(elec, fs_ok, TARGET_FS)
    opt,  _      = stride_decimate(opt,  fs_ok, TARGET_FS)
    ref,  _      = stride_decimate(ref,  fs_ok, TARGET_FS)
    L = min(len(elec), len(opt), len(ref))
    elec, opt, ref = elec[:L], opt[:L], ref[:L]

    if L < MIN_SAMPLES:
        raise ValueError("too few samples after cut/decimate")

    x_raw = np.stack([elec, opt, ref], axis=0).astype(np.float32)

    def _write(tmp_path):
        # np.savez_compressed insists on a .npz suffix to avoid double-appending it
        tmp_npz = tmp_path if tmp_path.endswith(".npz") else tmp_path + ".npz"
        np.savez_compressed(
            tmp_npz,
            x_raw=x_raw, fs=float(fs_out), t0=float(seg_t0_abs), length=int(L), base_name=base
        )
        if tmp_npz != tmp_path:
            os.replace(tmp_npz, tmp_path)

    atomic_write_bytes_via(_write, out_path)
    return out_path

# ============================== PHASE B: INFERENCE ==============================
def run_inference_on_npz(model, npz_path):
    z = np.load(npz_path, allow_pickle=True)
    with z:
        x_raw=z["x_raw"].astype(np.float32); fs=float(z["fs"]); t0=float(z["t0"]); L=int(z["length"])
        base=str(z.get("base_name", os.path.basename(npz_path)))
    t = t0 + np.arange(L, dtype=np.float32)/fs
    elec, opt, optr = x_raw[0], x_raw[1], x_raw[2]
    t_lo, t_hi = float(t[0]), float(t[-1])

    # Try to find the real event directly as a clean, persistent rise+fall
    # PAIR (Dr. Oguz's z-score edge detector). When this succeeds, it's a
    # direct physical detection of the transition corners and is trusted over
    # the DL model's answer for opt_base/opt_end -- the DL model does a global
    # fractional regression over the window and, even after variance-based
    # refinement, can still land mid-slope rather than at the true corner
    # (this is what "point sits partway up/down the ramp instead of at the
    # flat corner" looks like). The DL model + refinement are still used for
    # opt_plateau, and as the fallback for opt_base/opt_end when no clean edge
    # pair is found at all (e.g. a faint/noisy transition).
    primary_r, primary_f = (None, None)
    if EDGE_ENABLE:
        primary_r, primary_f = find_primary_event_edges(t, opt, fs, t_lo, t_hi, EDGE_CFG)

    # If the window is much larger than a real event, feeding it straight to the
    # DL model (which always maps the window onto a fixed-size grid) can point it
    # at the wrong feature entirely. Roughly localize the event first and hand the
    # model a tight crop instead. The electrical stage below still uses the FULL
    # window (t_lo/t_hi), since it needs real baseline on both sides of the event.
    dl_lo, dl_hi = t_lo, t_hi
    if primary_r is not None and primary_f is not None:
        dl_lo = max(t_lo, primary_r["t_start"] - EDGE_CFG.PRECROP_PAD_S)
        dl_hi = min(t_hi, primary_f["t_end"] + EDGE_CFG.PRECROP_PAD_S)
    elif PRECROP_ENABLE and (t_hi - t_lo) > PRECROP_TRIGGER_S:
        found = coarse_locate_optical_event(t, opt, fs, t_lo, t_hi, EDGE_CFG)
        if found is not None:
            dl_lo, dl_hi = found

    opt_pred  = infer_opt_only(model, t, opt, fs, dl_lo, dl_hi)
    opt_pred  = refine_optical_on_raw(t, opt, opt_pred)

    # Prefer the directly-detected edge corner for base/end. Fall back to a
    # single-sided edge snap (tight tolerance) if only that one landmark
    # lacks a clean paired detection, and only fall back to the DL+refine
    # answer as-is when no matching edge can be found nearby at all.
    #
    # opt_method records WHICH path was taken for base/end -- this is what
    # actually answers "why is this particular row still off": if you see
    # struggling rows, check this field first. "edges: full pair" is the
    # trusted, corner-precise path; anything mentioning "DL fallback" means
    # no clean transition was detected at all and the weaker DL+variance-
    # search answer was used instead -- that's very likely why those rows
    # look wrong, not a bug. If MANY of your real events show DL fallback,
    # EDGE_THRESHOLD_Z / EDGE_PERSISTENCE_MS are probably tuned too strict
    # for your signal-to-noise ratio; lower them and re-check this field.
    opt_method = []
    if primary_r is not None and primary_f is not None:
        # Both corners found -- refine with the level-crossing method, which
        # is less biased than the raw z-score run boundary (see EDGE_CFG's
        # LEVEL_CROSS_FRAC for the tuning tradeoff; validated in
        # test_optical_landmarks.py against synthetic ground truth).
        onset_t, offset_t = refine_landmarks_by_level(t, opt, primary_r, primary_f, EDGE_CFG)
        opt_pred["opt_base"] = onset_t
        opt_pred["opt_end"]  = offset_t
        opt_method.append("edges: full pair")
    else:
        if primary_r is not None:
            opt_pred["opt_base"] = primary_r["t_start"]
            opt_method.append("base: edge (rise only)")
        else:
            snapped = snap_single_edge(t, opt, fs, t_lo, t_hi, opt_pred["opt_base"], "rise", "t_start", EDGE_CFG)
            if snapped is not None:
                opt_pred["opt_base"] = snapped
                opt_method.append("base: edge snap")
            else:
                opt_method.append("base: DL fallback")
        if primary_f is not None:
            opt_pred["opt_end"] = primary_f["t_end"]
            opt_method.append("end: edge (fall only)")
        else:
            snapped = snap_single_edge(t, opt, fs, t_lo, t_hi, opt_pred["opt_end"], "fall", "t_end", EDGE_CFG)
            if snapped is not None:
                opt_pred["opt_end"] = snapped
                opt_method.append("end: edge snap")
            else:
                opt_method.append("end: DL fallback")
    opt_pred = enforce_opt_order(opt_pred)
    elec_pred = compute_electrical_from_opt(t, elec, fs, t_lo, t_hi, opt_pred)
    elec_pred = refine_electrical_on_raw(t, elec, elec_pred)

    def interp(sig, tt): return float(np.interp(tt, t, sig))
    v_base=interp(opt,  opt_pred["opt_base"])
    v_step=interp(opt,  opt_pred["opt_plateau"])
    vr_b =interp(optr, opt_pred["opt_base"])
    vr_s =interp(optr, opt_pred["opt_plateau"])
    eb_pa=interp(elec, elec_pred["elec_entry_base"])
    ep_pa=interp(elec, elec_pred["elec_entry_peak"])
    xp_pa=interp(elec, elec_pred["elec_exit_peak"])
    xb_pa=interp(elec, elec_pred["elec_exit_base"])

    duration = opt_pred["opt_end"] - opt_pred["opt_base"]
    osc      = abs((v_step - v_base) / ((v_step + v_base) / 2) * 100) if (v_step + v_base) != 0 else 0.0
    refosc   =  abs((vr_s - vr_b) / ((vr_s + vr_b) / 2) * 100) if (vr_s + vr_b) != 0 else 0.0
    ent_spk  = ep_pa - eb_pa
    ex_spk   = xp_pa - xb_pa

    out = {
        "event_start (s)": opt_pred["opt_base"],
        "event_end (s)":   opt_pred["opt_end"],
        "notes": "; ".join(opt_method),
        "Base (V)": v_base, "Step (V)": v_step,
        "RefBase (V)": vr_b, "RefStep (V)": vr_s,
        "entry Base (pA)": eb_pa, "entry Peak (pA)": ep_pa,
        "exit Peak (pA)":  xp_pa, "exit Base (pA)":  xb_pa,
        "duration": duration, "OSC": osc, "RefOSC": refosc,
        "event_plateau": opt_pred["opt_plateau"],
        "event_plateau_t": opt_pred["opt_plateau"],
        "entry_base_t": elec_pred["elec_entry_base"],
        "entry_peak_t": elec_pred["elec_entry_peak"],
        "exit_peak_t":  elec_pred["elec_exit_peak"],
        "exit_base_t":  elec_pred["elec_exit_base"],
        "_base_name": base
    }
    return out

def atomic_write_xlsx(df, path):
    def _write(tmp_path):
        tmp_xlsx = tmp_path if tmp_path.endswith(".xlsx") else tmp_path + ".xlsx"
        df.to_excel(tmp_xlsx, index=False)
        if tmp_xlsx != tmp_path:
            os.replace(tmp_xlsx, tmp_path)
    atomic_write_bytes_via(_write, path)

def blank_out_row(row, note):
    return {
        "event_id": row["event_id"], "file_name": row["file_name"],
        "sensor": row["sensor"], "analytes": row["analytes"], "solution": row["solution"],
        "event_start (s)": None, "event_end (s)": None, "notes": note,
        "Base (V)": None, "Step (V)": None, "RefBase (V)": None, "RefStep (V)": None,
        "entry Base (pA)": None,"entry Peak (pA)": None,"exit Peak (pA)": None,"exit Base (pA)": None,
        "duration": None,"OSC": None,"RefOSC": None,"entry spike": None,"exit spike": None,
        "event_plateau": None,"event_plateau_t": None,"entry_base_t": None,"entry_peak_t": None,"exit_peak_t": None,"exit_base_t": None
    }

# ============================== MAIN ==============================
def main():
    set_seed()
    folder = pick_folder_dialog()
    ckpt   = pick_ckpt_dialog()
    if not folder or not os.path.isdir(folder): raise FileNotFoundError("Folder not found")
    if not ckpt   or not os.path.isfile(ckpt):  raise FileNotFoundError("Checkpoint not found")

    excels, abfs, matches, unmatched, orphans = scan_folder(folder)
    ok = preview_gui(folder, excels, abfs, matches, unmatched, orphans)
    if not ok:
        print("Cancelled."); return
    print(f"\nOutput files will be written next to each source Excel, "
          f"as '<name>__predicted.xlsx', inside: {folder}")

    # Phase A: build ALL NPZs (fast) — NOTE: temp folder is preserved across runs so
    # that a crash mid-way (e.g. an OOM kill) doesn't throw away finished work.
    npz_dir = os.path.join(folder, "_tmp_npz")
    os.makedirs(npz_dir, exist_ok=True)

    # ---- Pass 1: figure out what needs building, WITHOUT opening any ABF yet. ----
    # Rows whose NPZ already exists on disk are resolved immediately (no ABF open
    # at all). Everything else is grouped by which ABF file it needs, so that file
    # only gets opened once no matter how many events/rows reference it.
    jobs_by_excel = {}      # xp -> list of dict(row=..., npz=..., note=...), same order as df
    pending_by_abf = {}     # abf_path -> list of (xp, row_index, rec, target_npz, base)

    for xp, pair_list in matches:
        if not pair_list: continue
        df = pd.read_csv(xp) if xp.lower().endswith(".csv") else pd.read_excel(xp)
        miss = [c for c in REQUIRED_IN_COLS if c not in df.columns]
        if miss:
            print(f"[{os.path.basename(xp)}] missing required cols {miss} -> skip file.")
            continue
        base_to_abf = {os.path.splitext(os.path.basename(ap))[0].lower(): ap for _,ap in pair_list}
        jobs = [None] * len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            rec = {k: row.get(k, None) for k in REQUIRED_IN_COLS}
            base = os.path.splitext(os.path.basename(str(rec["file_name"]).strip()))[0].lower()
            if base not in base_to_abf:
                jobs[i] = {"row":rec, "npz":None, "note":"abf not found"}
                continue
            abf_path = base_to_abf[base]
            target_npz = npz_path_for(npz_dir, base, rec["event_id"])
            if os.path.exists(target_npz) and os.path.getsize(target_npz) > 0:
                # Already built in a previous (possibly crashed) run -> reuse it, no ABF open.
                jobs[i] = {"row":rec, "npz":target_npz, "note":""}
            else:
                pending_by_abf.setdefault(abf_path, []).append((xp, i, rec, target_npz, base))
        jobs_by_excel[xp] = jobs

    n_reused = sum(1 for jobs in jobs_by_excel.values() for j in jobs if j and j.get("npz"))
    n_pending = sum(len(v) for v in pending_by_abf.values())
    print(f"\n[Phase A] {n_reused} NPZ(s) already on disk (reused, no ABF re-read), "
          f"{n_pending} event(s) to build across {len(pending_by_abf)} ABF file(s).")

    # ---- Pass 2: for each ABF file that still has pending events, open it ONCE
    # and cut every event that needs it before moving to the next file. ----
    for abf_path, items in pending_by_abf.items():
        print(f"  [Phase A] {os.path.basename(abf_path)}: opening once for {len(items)} event(s)...")
        try:
            A, fs, ranges = open_abf(abf_path)
        except Exception as e:
            for xp, i, rec, target_npz, base in items:
                jobs_by_excel[xp][i] = {"row":rec, "npz":None, "note":f"abf open failed: {e}"}
            continue
        for xp, i, rec, target_npz, base in items:
            try:
                build_npz_for_row(A, fs, ranges, base, rec, target_npz)
                jobs_by_excel[xp][i] = {"row":rec, "npz":target_npz, "note":""}
            except Exception as e:
                jobs_by_excel[xp][i] = {"row":rec, "npz":None, "note":f"npz build failed: {e}"}
        del A  # release this file's data before opening the next one

    # Phase B: load model once, run inference+refine
    print("\n[Phase B] Inference with trained model ...")
    ck = torch.load(ckpt, map_location="cpu")
    model = StrongOpt1D().to(DEVICE).eval()
    model.load_state_dict(ck["model"])

    for xp, jobs in jobs_by_excel.items():
        out_xlsx = os.path.splitext(xp)[0] + "__predicted.xlsx"
        ckpt_path = os.path.splitext(xp)[0] + "__predicted.checkpoint.json"
        checkpoint = {} if FORCE_REDO_PHASE_B else load_checkpoint(ckpt_path)  # str(row index) -> out_row dict
        if FORCE_REDO_PHASE_B:
            print(f"\n[Phase B] {os.path.basename(xp)}: FORCE_REDO_PHASE_B=True, "
                  f"ignoring any existing checkpoint and recomputing all rows.")

        n_already = sum(1 for i in range(len(jobs)) if str(i) in checkpoint)
        if n_already:
            print(f"\n[Phase B] {os.path.basename(xp)}: resuming, "
                  f"{n_already}/{len(jobs)} row(s) already done.")
        else:
            print(f"\n[Phase B] {os.path.basename(xp)}: {len(jobs)} row(s) to process.")

        for i, j in enumerate(jobs):
            key = str(i)
            if key in checkpoint:
                continue  # done in a previous run

            row = j["row"]; note = j["note"]
            if not j["npz"]:
                out_row = blank_out_row(row, note)
            else:
                try:
                    pred = run_inference_on_npz(model, j["npz"])
                    out_row = {
                        "event_id": row["event_id"], "file_name": row["file_name"],
                        "sensor": row["sensor"], "analytes": row["analytes"], "solution": row["solution"],
                        **{k: pred[k] for k in OUT_COLS if k in pred}
                    }
                except Exception as e:
                    out_row = blank_out_row(row, f"infer fail: {e}")

            checkpoint[key] = out_row
            # Persist after every row so a crash loses at most the in-flight row.
            atomic_write_json(ckpt_path, checkpoint)

            # Periodically (re)write a partial .xlsx so there's something to
            # look at during a long run instead of only at the very end.
            # Not-yet-processed rows are shown with notes="pending".
            if (i + 1) % XLSX_FLUSH_EVERY_ROWS == 0:
                partial_rows = [
                    checkpoint[str(k)] if str(k) in checkpoint else blank_out_row(jobs[k]["row"], "pending")
                    for k in range(len(jobs))
                ]
                atomic_write_xlsx(pd.DataFrame(partial_rows, columns=OUT_COLS), out_xlsx)
                print(f"  [flush] {os.path.basename(out_xlsx)}  ({i+1}/{len(jobs)} rows done so far)")

        out_rows = [checkpoint[str(i)] for i in range(len(jobs))]
        df_out = pd.DataFrame(out_rows, columns=OUT_COLS)
        atomic_write_xlsx(df_out, out_xlsx)
        print(f"[write] {os.path.basename(out_xlsx)}  ({len(df_out)} rows)  ->  {out_xlsx}")

    print("\nDone. Temp NPZs:", npz_dir)

if __name__ == "__main__":
    main()