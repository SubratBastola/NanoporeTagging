import os
import sys
import io
import json
import base64
import html
import webbrowser
import queue
import threading
import traceback
import warnings
from datetime import datetime
from itertools import combinations

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import matplotlib
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ================================ ENGINE ======================================
COLUMN_ALIASES = {
    "duration": ["duration", "duration_val", "dur", "dwell", "dwell_time"],
    "OSC": ["osc", "osc_pct", "step_osc", "optical_scatter"],
    "RefOSC": ["refosc", "ref_osc", "refosc_pct", "reference_osc"],
    # Raw peak values and peak-minus-base spike amplitudes are deliberately
    # separate clustering features.
    "entry_peak": ["entry Peak (pA)", "entry_peak", "entry peak", "entry peak (pA)", "entry_peak_pa"],
    "exit_peak": ["exit Peak (pA)", "exit_peak", "exit peak", "exit peak (pA)", "exit_peak_pa"],
    "entry_spike": ["entry_spike", "entry spike", "entryspike", "entry_current", "ispike_entry"],
    "exit_spike": ["exit_spike", "exit spike", "exitspike", "exit_current", "ispike_exit"],
}
CANONICAL_FEATURES = list(COLUMN_ALIASES.keys())

# Raw-current columns used to calculate absolute spike amplitudes.
# The requested definitions are:
#   entry_spike = |entry_peak - entry_base|
#   exit_spike  = |exit_peak  - exit_base|
RAW_SPIKE_ALIASES = {
    "entry_base": [
        "entry Base (pA)", "entry_base", "entry base", "entry baseline",
        "entry_baseline", "entry baseline (pA)", "entry_base_pa",
    ],
    "entry_peak": [
        "entry Peak (pA)", "entry_peak", "entry peak",
        "entry peak (pA)", "entry_peak_pa",
    ],
    "exit_peak": [
        "exit Peak (pA)", "exit_peak", "exit peak",
        "exit peak (pA)", "exit_peak_pa",
    ],
    "exit_base": [
        "exit Base (pA)", "exit_base", "exit base", "exit baseline",
        "exit_baseline", "exit baseline (pA)", "exit_base_pa",
    ],
}


def _norm(s):
    return str(s).strip().lower().replace(" ", "_")


def _find_alias_column(df, aliases):
    """Return the first matching actual column name for a list of aliases."""
    lower = {_norm(c): c for c in df.columns}
    return next((lower[_norm(a)] for a in aliases if _norm(a) in lower), None)


def detect_columns(df):
    """Map canonical feature names and raw spike ingredients to actual file columns."""
    out = {}
    for canon, aliases in COLUMN_ALIASES.items():
        out[canon] = _find_alias_column(df, aliases)
    for canon, aliases in RAW_SPIKE_ALIASES.items():
        out[canon] = _find_alias_column(df, aliases)
    return out


def derive_spike_series(df, col_map, which):
    """
    Build an absolute spike-amplitude series from raw peak/base columns.

    entry_spike = |entry_peak - entry_base|
    exit_spike  = |exit_peak  - exit_base|
    """
    if which == "entry_spike":
        peak_col = col_map.get("entry_peak")
        base_col = col_map.get("entry_base")
    elif which == "exit_spike":
        peak_col = col_map.get("exit_peak")
        base_col = col_map.get("exit_base")
    else:
        raise ValueError(f"Unknown spike feature: {which}")

    if peak_col is None or base_col is None:
        return None, None

    peak = pd.to_numeric(df[peak_col], errors="coerce")
    base = pd.to_numeric(df[base_col], errors="coerce")
    return (peak - base).abs(), f"|{peak_col} - {base_col}|"


def read_table(path, sheet_name=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
    raise ValueError(f"Unsupported file type: {ext}")


def extract_file_features(
    df, path, duration_min=0.0, duration_max=0.0,
    selected_features=None, min_nonnull=2
):
    """
    Extract only the features selected in the GUI.

    Entry/exit peaks are used as absolute current magnitudes.

    For entry_spike and exit_spike, raw current measurements are preferred:
        entry_spike = |entry_peak - entry_base|
        exit_spike  = |exit_peak  - exit_base|

    If the raw peak/base columns are unavailable, a populated precomputed
    spike column is used as a fallback.
    """
    selected_features = list(selected_features or CANONICAL_FEATURES)
    col_map = detect_columns(df)
    usable = {}
    skipped = {}
    source_description = {}

    # Duration is also read independently for min/max filtering, even if the
    # user does not choose duration as a clustering feature.
    duration_actual = col_map.get("duration")
    duration_for_filter = (
        pd.to_numeric(df[duration_actual], errors="coerce")
        if duration_actual is not None else None
    )

    for canon in selected_features:
        numeric = None
        source = None

        # Prefer deriving the spike from its actual raw peak/base measurements.
        if canon in ("entry_spike", "exit_spike"):
            derived, formula_source = derive_spike_series(df, col_map, canon)
            if derived is not None and int(derived.notna().sum()) >= min_nonnull:
                numeric = derived
                source = f"DERIVED: {formula_source}"
            else:
                # Fall back to a populated precomputed spike column.
                actual = col_map.get(canon)
                if actual is not None:
                    candidate = pd.to_numeric(df[actual], errors="coerce").abs()
                    if int(candidate.notna().sum()) >= min_nonnull:
                        numeric = candidate
                        source = actual
        else:
            actual = col_map.get(canon)
            if actual is not None:
                numeric = pd.to_numeric(df[actual], errors="coerce")
                if canon in ("entry_peak", "exit_peak"):
                    numeric = numeric.abs()
                    source = f"|{actual}|"
                else:
                    source = actual

        if numeric is None:
            if canon == "entry_spike":
                skipped[canon] = "need entry Peak and entry Base (or populated entry spike)"
            elif canon == "exit_spike":
                skipped[canon] = "need exit Peak and exit Base (or populated exit spike)"
            else:
                skipped[canon] = "column not found"
            continue

        n_valid = int(numeric.notna().sum())
        if n_valid < min_nonnull:
            skipped[canon] = f"only {n_valid} numeric value(s)"
            continue
        if numeric.dropna().nunique() < 2:
            skipped[canon] = "no variation"
            continue

        usable[canon] = numeric
        source_description[canon] = source

    # Apply duration limits before clustering whether or not duration itself
    # is one of the selected clustering features.
    keep = pd.Series(True, index=df.index)
    if duration_for_filter is not None:
        if duration_min > 0:
            keep &= duration_for_filter > duration_min
        if duration_max > 0:
            keep &= duration_for_filter <= duration_max

    out = pd.DataFrame(index=df.index)
    for canon, ser in usable.items():
        out[canon] = ser

    out = out.loc[keep].copy()
    out.insert(0, "source_row", out.index.astype(int) + 2)
    out.insert(0, "source_file", os.path.basename(path))
    out = out.reset_index(drop=True)

    return out, usable.keys(), skipped, col_map, source_description


def build_combined_dataset(
    paths, duration_min=0.0, duration_max=0.0,
    selected_features=None, log=None
):
    """Load all selected files and build one pooled event table."""
    log = log or (lambda *_: None)
    selected_features = list(selected_features or CANONICAL_FEATURES)
    frames = []
    file_info = []

    log(f"Requested clustering features: {', '.join(selected_features)}")

    for path in paths:
        log(f"\nLoading: {os.path.basename(path)}")
        df = read_table(path)
        log(f"  Raw shape: {len(df)} rows x {len(df.columns)} columns")

        feat_df, usable, skipped, col_map, source_desc = extract_file_features(
            df, path, duration_min, duration_max, selected_features
        )
        usable = list(usable)

        log("  Selected feature resolution:")
        for canon in selected_features:
            if canon in usable:
                log(f"    {canon:<13} -> {source_desc.get(canon, ''):<45} USE")
            else:
                log(f"    {canon:<13} -> SKIP ({skipped.get(canon, 'not usable')})")

        if not usable:
            log("  WARNING: none of the selected features are usable in this file; file skipped.")
            file_info.append({
                "path": path, "rows": len(df), "usable": [],
                "skipped_file": True
            })
            continue

        feat_df["_file_order"] = len(file_info)
        frames.append(feat_df)
        file_info.append({
            "path": path, "rows": len(df), "usable": usable,
            "skipped_file": False
        })
        log(f"  Kept {len(feat_df)} candidate events; usable selected features: {', '.join(usable)}")

    if not frames:
        raise ValueError(
            "None of the selected files contained usable values for the selected clustering features."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)

    features = []
    for f in selected_features:
        if (
            f in combined.columns
            and combined[f].notna().sum() >= 2
            and combined[f].dropna().nunique() >= 2
        ):
            features.append(f)

    if not features:
        raise ValueError("No usable selected clustering features remained after pooling the files.")

    log(f"\nPooled events before NA handling: {len(combined)}")
    log(f"Pooled usable selected features: {', '.join(features)}")
    return combined, features, file_info


def preprocess_combined(combined, features, impute_missing=True, log=None):
    log = log or (lambda *_: None)
    data = combined[["source_file", "source_row", "_file_order"] + features].copy()

    # Remove rows that are empty across ALL selected clustering features.
    before = len(data)
    data = data.dropna(subset=features, how="all").reset_index(drop=True)
    log(f"Removed {before - len(data)} rows empty across all clustering features.")

    if len(data) < 3:
        raise ValueError("Fewer than 3 usable events remain after NA filtering.")

    if impute_missing:
        log("NA handling: median-imputing occasional missing values within each usable feature.")
        for f in features:
            nmiss = int(data[f].isna().sum())
            if nmiss:
                med = float(data[f].median())
                data[f] = data[f].fillna(med)
                log(f"  {f}: filled {nmiss} NA value(s) with median {med:.6g}")
    else:
        before = len(data)
        data = data.dropna(subset=features).reset_index(drop=True)
        log(f"NA handling: complete-case only; removed {before - len(data)} row(s).")

    # Remove features that became constant after cleaning.
    features = [f for f in features if data[f].nunique(dropna=True) >= 2]
    if not features:
        raise ValueError("All usable feature columns are constant after cleaning.")

    Xraw = data[features].astype(float).copy()
    Xmodel = Xraw.copy()

    # Keep the original duration transform, but only if duration exists.
    if "duration" in features:
        Xmodel["duration"] = np.log1p(np.clip(Xmodel["duration"].values, 0, None))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xmodel.values)
    return data, Xraw, Xs, features


def fit_gmm(X, k, rs=42, n_init=10):
    if k >= len(X):
        raise ValueError(f"K={k} is too large for only {len(X)} events.")
    g = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=n_init,
        random_state=rs,
        reg_covar=1e-6,
    )
    g.fit(X)
    return g


def _lr(gs, gb, X):
    return 2.0 * (gb.score(X) - gs.score(X)) * len(X)


def bootstrap_lrt(X, ks, kb, n_boot, rs, log):
    """
    Parametric bootstrap for the K=ks vs K=kb likelihood-ratio test.

    NOTE: scikit-learn's GaussianMixture.sample() reseeds from the model's
    stored `random_state` on every call (via check_random_state), so calling
    g0.sample() repeatedly with a fixed random_state returns the IDENTICAL
    synthetic dataset every time -- not a fresh draw. Left alone, that makes
    every bootstrap replicate in this loop resample the exact same data,
    collapsing the null distribution's spread and making the resulting
    p-value hit its floor (1/(n_boot+1)) far more often than it should. We
    give g0.random_state a new seed immediately before each sample() call so
    every replicate is a genuinely different bootstrap draw.
    """
    g0 = fit_gmm(X, ks, rs, 10)
    g1 = fit_gmm(X, kb, rs + 1, 10)
    obs = _lr(g0, g1, X)
    nulls = []

    for b in range(n_boot):
        g0.random_state = rs + 500 + b
        Xb, _ = g0.sample(len(X))
        try:
            a = fit_gmm(Xb, ks, rs + 100 + b, 5)
            c = fit_gmm(Xb, kb, rs + 200 + b, 5)
            nulls.append(_lr(a, c, Xb))
        except Exception:
            nulls.append(np.nan)
        if log:
            log(f"    bootstrap {b + 1:>2}/{n_boot}   (K={ks} vs {kb})")

    nulls = np.asarray(nulls, dtype=float)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0:
        return obs, np.nan
    p = (1.0 + np.sum(nulls >= obs)) / (len(nulls) + 1.0)
    return obs, p


def build_k_breakdown(labels_by_k, k_bic, final_k):
    """
    Long-format table of how the pooled events split into clusters at every
    K tried during the sweep: one row per (K, cluster) pair, with N and Pct.

    `Recommended` flags the row's K as the algorithm's final chosen K (from
    the sequential bootstrap LRT), and `BIC_best` flags the row's K as the
    lowest-BIC K from the sweep (these can differ).
    """
    rows = []
    for k in sorted(labels_by_k):
        lbl = labels_by_k[k]
        n_total = len(lbl)
        for c in range(k):
            m = lbl == c
            rows.append({
                "K": k,
                "Cluster": f"C{c}",
                "N": int(m.sum()),
                "Pct": round(100.0 * float(m.mean()), 2) if n_total else 0.0,
                "Recommended": (k == final_k),
                "BIC_best": (k == k_bic),
            })
    return pd.DataFrame(rows)


def _run_clustering_once(paths, alpha=0.05, max_k=8, duration_min=0.0, duration_max=0.0,
                          n_boot=25, impute_missing=True, selected_features=None,
                          excluded_points=None, log=None):
    log = log or print

    if duration_min > 0:
        log(f"Minimum duration filter: > {duration_min:g} s")
    if duration_max > 0:
        log(f"Maximum duration filter: <= {duration_max:g} s")
    else:
        log("Maximum duration filter: OFF")

    combined, features, file_info = build_combined_dataset(
        paths, duration_min, duration_max, selected_features, log
    )

    # Remove points manually excluded from the Point Inspector.
    excluded_points = set(excluded_points or [])
    if excluded_points:
        keys = list(zip(combined["source_file"], combined["source_row"]))
        keep = np.array([key not in excluded_points for key in keys], dtype=bool)
        removed = int((~keep).sum())
        combined = combined.loc[keep].reset_index(drop=True)
        log(f"Manual point exclusions applied: removed {removed} event(s).")

    meta, X_raw, Xs, features = preprocess_combined(combined, features, impute_missing, log)

    n = len(X_raw)
    max_k = max(1, min(int(max_k), n - 1))
    log(f"\nFinal modeling matrix: {n} events x {len(features)} feature(s)")
    log(f"Features used for clustering: {', '.join(features)}")

    log(f"\n[3] GMM sweep K = 1 .. {max_k}")
    log(f"  {'K':>3}  {'BIC':>12}  {'AIC':>12}")
    bic, aic, gmms = {}, {}, {}
    for k in range(1, max_k + 1):
        g = fit_gmm(Xs, k, 42 + k, 15)
        gmms[k] = g
        bic[k] = g.bic(Xs)
        aic[k] = g.aic(Xs)
        log(f"  {k:>3}  {bic[k]:>12.1f}  {aic[k]:>12.1f}")

    k_bic = min(bic, key=bic.get)
    log(f"  BIC-best K = {k_bic}")

    # Stable-numbered cluster labels at EVERY K in the sweep (largest = C0,
    # next largest = C1, ...), used to draw the "how clusters split as K
    # grows" Sankey/alluvial chart alongside the BIC curve.
    labels_by_k = {}
    for k in range(1, max_k + 1):
        raw_labels = gmms[k].predict(Xs)
        order_k = sorted(np.unique(raw_labels), key=lambda c: -(raw_labels == c).sum())
        remap_k = {old: new for new, old in enumerate(order_k)}
        labels_by_k[k] = np.array([remap_k[x] for x in raw_labels], dtype=int)

    log(f"\n[4] Sequential bootstrap LRT (alpha={alpha}, boot={n_boot})")
    lrt_rows = []
    final_k = 1
    for k in range(2, max_k + 1):
        obs, p = bootstrap_lrt(Xs, k - 1, k, n_boot, 1000 + k, log)
        sig = bool(np.isfinite(p) and p < alpha)
        lrt_rows.append({
            "k_small": k - 1,
            "k_big": k,
            "obs_lr": obs,
            "p_boot": p,
            "significant": sig,
        })
        log(f"  K={k-1} vs {k}: obs_LR={obs:9.2f} p={p:.4f} {'KEEP' if sig else 'STOP'}")
        if sig:
            final_k = k
        else:
            break

    labels = labels_by_k[final_k]

    labeled = meta[["source_file", "source_row"]].copy()
    for f in features:
        labeled[f] = X_raw[f].values
    labeled["cluster"] = labels

    rows = []
    for k in range(final_k):
        m = labels == k
        row = {
            "Cluster": k,
            "N": int(m.sum()),
            "Pct": round(100.0 * float(m.mean()), 2),
        }
        for f in features:
            v = X_raw.loc[m, f].to_numpy(dtype=float)
            row[f + "_mean"] = float(np.mean(v))
            row[f + "_std"] = float(np.std(v, ddof=0))
            row[f + "_median"] = float(np.median(v))
        rows.append(row)

    summary = pd.DataFrame(rows)
    per_file = (
        labeled.groupby(["source_file", "cluster"], dropna=False)
        .size().rename("N").reset_index()
    )
    totals = labeled.groupby("source_file").size().rename("file_total")
    per_file = per_file.join(totals, on="source_file")
    per_file["Pct_in_file"] = 100.0 * per_file["N"] / per_file["file_total"]

    k_breakdown = build_k_breakdown(labels_by_k, k_bic, final_k)

    log(f"\n[5] Final K = {final_k}")
    for _, r in summary.iterrows():
        parts = [f"C{int(r['Cluster'])} (n={int(r['N'])}, {r['Pct']:.1f}%)"]
        for f in features[:3]:
            parts.append(f"{f}={r[f + '_mean']:.3g}")
        log("  " + "  ".join(parts))

    return {
        "X_raw": X_raw,
        "features": features,
        "labels": labels,
        "labels_by_k": labels_by_k,
        "labeled": labeled,
        "bic": bic,
        "aic": aic,
        "k_bic": k_bic,
        "K": final_k,
        "max_k": max_k,
        "summary": summary,
        "per_file": per_file,
        "lrt": pd.DataFrame(lrt_rows),
        "file_info": file_info,
        "k_breakdown": k_breakdown,
    }


def run_clustering(paths, alpha=0.05, max_k=8, duration_min=0.0, duration_max=0.0,
                    n_boot=25, impute_missing=True, selected_features=None,
                    excluded_points=None, min_cluster_pct=0.0, log=None,
                    max_size_filter_iters=5):
    """
    Wraps _run_clustering_once() with an optional minimum-cluster-size filter.

    After a normal fit, any final cluster holding less than `min_cluster_pct`
    percent of that run's events is treated as noise: its events are added to
    the exclusion set and the ENTIRE clustering (GMM sweep, bootstrap LRT,
    final K, everything) is refit from scratch on the reduced data -- the
    same mechanism the Point Inspector's "Remove cluster(s) + recompute"
    button uses, just applied automatically before you ever see a result.
    This keeps the K-selection diagnostics (BIC curve, Sankey, K Breakdown)
    fully consistent with the final clusters actually reported: nothing you
    see anywhere in the app or the HTML report still contains a sub-threshold
    cluster.

    Repeats (capped at `max_size_filter_iters`) since removing one small
    cluster can occasionally leave another cluster newly below the
    threshold. `min_cluster_pct <= 0` disables the filter entirely (default),
    preserving the exact prior behavior.
    """
    log = log or print
    excluded_points = set(excluded_points or [])

    for iteration in range(max_size_filter_iters):
        result = _run_clustering_once(
            paths, alpha, max_k, duration_min, duration_max,
            n_boot, impute_missing, selected_features, excluded_points, log,
        )

        if min_cluster_pct <= 0:
            return result

        labels, K = result["labels"], result["K"]
        total = len(labels)
        if total == 0 or K == 0:
            return result

        counts = np.array([(labels == k).sum() for k in range(K)])
        pcts = 100.0 * counts / total
        small_ks = [k for k in range(K) if pcts[k] < min_cluster_pct]

        if not small_ks:
            return result
        if len(small_ks) == K:
            log(f"\n[Cluster size filter] All {K} cluster(s) are below {min_cluster_pct:g}% "
                f"of events; filter skipped for this run to avoid discarding everything.")
            return result

        labeled = result["labeled"]
        removed_mask = labeled["cluster"].isin(small_ks)
        newly_excluded = set(
            zip(labeled.loc[removed_mask, "source_file"], labeled.loc[removed_mask, "source_row"])
        )
        before = len(excluded_points)
        excluded_points |= newly_excluded
        if len(excluded_points) == before:
            # No new points to remove (shouldn't normally happen) -- stop to avoid looping forever.
            return result

        small_desc = ", ".join(f"C{k} ({pcts[k]:.2f}%)" for k in small_ks)
        log(f"\n[Cluster size filter] Removing {len(newly_excluded)} event(s) in cluster(s) "
            f"{small_desc} — below the {min_cluster_pct:g}% threshold. "
            f"Re-fitting ({iteration + 1}/{max_size_filter_iters})...")

    log(f"\n[Cluster size filter] Reached {max_size_filter_iters} refit(s) without every cluster "
        f"clearing {min_cluster_pct:g}%; using the latest result.")
    return result


# User-facing feature labels only. Internal feature names remain unchanged.
FEATURE_DISPLAY_LABELS = {
    "duration": "Duration (s)",
    "OSC": "Optical step change (%)",
    "RefOSC": "Ref Optical step change (%)",
    "entry_peak": "Entry peak |pA|",
    "exit_peak": "Exit peak |pA|",
    "entry_spike": "Entry spike = |peak − base| (pA)",
    "exit_spike": "Exit spike = |peak − base| (pA)",
}


def _feature_label(name):
    return FEATURE_DISPLAY_LABELS.get(name, name)


def _cluster_label(names, k):
    """Custom cluster name if one was set for index k, else the default 'Ck'."""
    if names and names.get(k):
        return names[k]
    return f"C{k}"


# ================================ THEME =======================================
BG, PANEL, BORDER = "#f5f7fa", "#eef1f8", "#dde3f0"
ACCENT, SUCCESS, ERR, WARN = "#2563eb", "#16a34a", "#dc2626", "#d97706"
TEXT, TEXT_MED, TEXT_DIM, WHITE = "#111827", "#4b5563", "#9ca3af", "#ffffff"

# Fixed, high-contrast, rank-stable cluster colors.
# Clusters are always numbered by size (C0 = largest, C1 = next largest, ...;
# see the `remap` step in run_clustering), so PALETTE[0] is always the color
# of the biggest cluster in any given run, PALETTE[1] the second biggest, etc.
# 10 maximally distinguishable colors, in the requested order:
# Red, Blue, Green, Purple, Black, Orange, Magenta, Brown, Gold, Lime.
PALETTE = [
    "#E31A1C",  # 1  Red
    "#1F4FE0",  # 2  Blue
    "#158000",  # 3  Green
    "#8E24AA",  # 4  Purple
    "#000000",  # 5  Black
    "#FF7F00",  # 6  Orange
    "#E619D6",  # 7  Magenta
    "#8B4513",  # 8  Brown
    "#D4A600",  # 9  Gold
    "#39B300",  # 10 Lime
]


def _style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.8)
    ax.tick_params(labelsize=9, colors=TEXT_MED)


def fig_kselection(R, names=None):
    """
    BIC curve and cluster-proportions panels, captioned with the plain-English
    reason Final K was chosen (the sequential bootstrap LRT itself still runs
    in run_clustering() and its numbers are still saved to bootstrap_lrt.csv;
    this figure just doesn't plot the LRT bars anymore).

    The cluster-proportions panel is a pie chart (previously a bar chart);
    each wedge is one final cluster, sized by its share of pooled events.

    `names` is an optional {cluster_index: custom_name} dict used to label
    the cluster-proportions wedges; indices without a custom name fall back
    to "C0", "C1", etc.
    """
    fig = Figure(figsize=(10, 5.3), facecolor=WHITE, dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14, wspace=0.28)
    bic, K = R["bic"], R["K"]

    ax = fig.add_subplot(1, 2, 1)
    _style(ax)
    ks = sorted(bic)
    ax.plot(ks, [bic[k] for k in ks], "o-", color=ACCENT, lw=2.5, ms=8, mfc=WHITE, mew=2.2)
    ax.axvline(R["k_bic"], color=TEXT_DIM, lw=2, ls="--", label=f"BIC best = {R['k_bic']}")
    ax.axvline(K, color=SUCCESS, lw=2, label=f"Final K = {K}")
    ax.set_xlabel("K", color=TEXT_MED)
    ax.set_ylabel("BIC", color=TEXT_MED)
    ax.set_title("BIC curve", fontweight="bold", color=TEXT)
    ax.legend(fontsize=9, edgecolor=BORDER)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_facecolor(WHITE)
    labels = R["labels"]
    counts = [(labels == k).sum() for k in range(K)]
    n_total = len(labels) if len(labels) else 1
    colors = [PALETTE[k % len(PALETTE)] for k in range(K)]
    wedge_labels = [_cluster_label(names, k) for k in range(K)]
    ax.pie(
        counts,
        labels=wedge_labels,
        autopct=lambda p: (f"{p:.1f}%\nn={int(round(p * n_total / 100.0))}" if p > 0 else ""),
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor=WHITE, linewidth=1.4),
        textprops=dict(fontsize=8.5, color=TEXT),
        pctdistance=0.72,
    )
    ax.set_title("Cluster proportions", fontweight="bold", color=TEXT)
    ax.axis("equal")

    fig.suptitle(
        f"Final K = {K} is where clusters were statistically significant",
        fontsize=12.5, fontweight="bold", color=SUCCESS, y=0.97
    )
    return fig


# --------------------------- Cluster-split Sankey -----------------------------
# Ported from the MATLAB "plotSankeyFlowChart" alluvial-flow routine
# (sankey_alluvialflow / get_curves) so the same smooth-curve flow patches
# can be drawn with matplotlib, using the app's own color palette.
#
# The category ORDER and COLOR at each K stage are computed separately from
# the raw GMM fit via `_compute_sankey_layout` below. Simply stacking
# categories in size-rank order (as the very first version of this chart
# did) causes heavy visual crisscrossing whenever the size ranking reshuffles
# between adjacent K's, even though the underlying clusters are stable --
# a stable cluster just happened to have a different size rank at K vs K+1.
# `_compute_sankey_layout` instead orders each stage's categories so that a
# parent's children stay contiguous and sit where the parent was (a
# parent-then-descending-size ordering), and assigns colors by LINEAGE: a
# cluster's largest/dominant child keeps its parent's color, so following a
# single color across the chart traces one continuous line of descent
# rather than "whichever cluster happens to be biggest at this K".

def _sankey_curve_points(x1, y1, x2, y2, n_points=15):
    """
    Smooth interpolation between (x1, y1) and (x2, y2) for a single flow edge,
    matching the cosine-eased curve used by the original MATLAB get_curves().
    y1/y2 are scalars here (one edge at a time).
    """
    t = np.linspace(0, np.pi, n_points)
    c = (1 - np.cos(t)) / 2.0
    y = y1 + (y2 - y1) * c
    x = np.linspace(x1, x2, n_points)
    return x, y


def _compute_sankey_layout(labels_by_k, final_k):
    """
    Compute a low-crossing display ORDER and a lineage-based COLOR for every
    K stage in the sweep.

    order[k]    -> list of original cluster ids (as they appear in
                   labels_by_k[k], i.e. already size-rank remapped for that
                   K alone) giving the top-to-bottom stacking order to use
                   when drawing stage k.
    color_of[k] -> {cluster_id: palette_index} for stage k.

    Ordering rule: at K=1 there is one category. Going from K to K+1, every
    new-stage cluster is assigned a "parent" = whichever K-stage cluster
    contributes the most events to it. The new stage's order is built by
    walking the previous stage's order and, for each parent in turn,
    inserting its children (sorted by descending inherited event count)
    contiguously. This keeps a split visually "opening up" in place instead
    of jumping across the chart. This part is independent of `final_k`.

    Returns (order, color_of, parent_of_stage, children_of_stage). The last
    two are also handed back (not just used internally) so other views --
    e.g. a "Cluster Hierarchy (3D)" tab -- can describe, in plain terms,
    which cluster split into which at each step without recomputing the
    same overlap matrices.
    """
    ks = sorted(labels_by_k)
    order = {}
    parent_of_stage = {}    # k -> {child cluster id: parent cluster id at k-1}
    children_of_stage = {}  # k -> {parent cluster id at k-1: [child ids at k, desc by shared count]}

    k0 = ks[0]
    n0 = len(np.unique(labels_by_k[k0]))
    order[k0] = list(range(n0))

    for i in range(1, len(ks)):
        kprev, kcur = ks[i - 1], ks[i]
        lprev, lcur = labels_by_k[kprev], labels_by_k[kcur]
        n_prev = len(np.unique(lprev))
        n_cur = len(np.unique(lcur))

        counts = np.zeros((n_prev, n_cur))
        for p in range(n_prev):
            mp = lprev == p
            if not mp.any():
                continue
            for c in range(n_cur):
                counts[p, c] = np.sum(mp & (lcur == c))

        parent_of = {}
        for c in range(n_cur):
            col = counts[:, c]
            parent_of[c] = int(np.argmax(col)) if col.sum() > 0 else None
        parent_of_stage[kcur] = parent_of

        children_of = {}
        for p in range(n_prev):
            children = [c for c in range(n_cur) if parent_of.get(c) == p]
            children.sort(key=lambda c: -counts[p, c])
            children_of[p] = children
        children_of_stage[kcur] = children_of

        new_order = []
        used = set()
        for p in order[kprev]:
            for c in children_of.get(p, []):
                new_order.append(c)
                used.add(c)
        leftover = [c for c in range(n_cur) if c not in used]
        leftover.sort(key=lambda c: -(lcur == c).sum())
        new_order.extend(leftover)
        order[kcur] = new_order

    # ---- color assignment, anchored at final_k ----
    color_of = {}
    n_final = len(np.unique(labels_by_k[final_k]))
    color_of[final_k] = {c: c % len(PALETTE) for c in range(n_final)}
    next_color = n_final

    idx_final = ks.index(final_k)

    # Backward: final_k -> ks[0]. Each cluster inherits the color of its
    # dominant (largest-shared-count) child one stage up, since that child
    # already has its color from the pass above/this same backward walk.
    for i in range(idx_final - 1, -1, -1):
        k = ks[i]
        kchild = ks[i + 1]
        n_k = len(np.unique(labels_by_k[k]))
        children_of = children_of_stage.get(kchild, {})
        color_of[k] = {}
        for p in range(n_k):
            kids = children_of.get(p, [])
            if kids:
                dominant = kids[0]
                color_of[k][p] = color_of[kchild].get(dominant, p % len(PALETTE))
            else:
                color_of[k][p] = p % len(PALETTE)

    # Forward: final_k -> ks[-1]. The dominant child of a parent inherits the
    # parent's color; any other (genuinely new) child gets a fresh color.
    for i in range(idx_final + 1, len(ks)):
        k = ks[i]
        kprev = ks[i - 1]
        n_k = len(np.unique(labels_by_k[k]))
        parent_of = parent_of_stage.get(k, {})
        children_of = children_of_stage.get(k, {})
        color_of[k] = {}
        for c in range(n_k):
            p = parent_of.get(c)
            is_dominant = p is not None and children_of.get(p, [None])[0] == c
            if is_dominant:
                color_of[k][c] = color_of[kprev].get(p, next_color % len(PALETTE))
            else:
                color_of[k][c] = next_color % len(PALETTE)
                next_color += 1

    return order, color_of, parent_of_stage, children_of_stage


def _sankey_draw_layer(ax, order1, order2, counts, x1, x2, y1_points,
                        color_of1, flow_alpha, bar_width):
    """
    Draw one x1->x2 alluvial segment using explicit, precomputed per-stage
    category orders (lists of original cluster ids, front-to-back =
    top-to-bottom) instead of assuming size-rank index order.

    `counts` is the full [n_prev x n_cur] overlap matrix, indexed by
    ORIGINAL cluster id (not display position). `color_of1` maps a stage-1
    cluster id to a palette index; every flow leaving that cluster, and its
    own bar segment, are drawn in that one color -- so a lineage keeps a
    single color as it moves rightward across stages.

    Returns the top-y position of each category in order2 (for chaining
    into the next segment) and the gap used between blocks.
    """
    bars1 = np.array([counts[c, :].sum() for c in order1], dtype=float)
    bars2 = np.array([counts[:, c].sum() for c in order2], dtype=float)
    total2 = bars2.sum()
    gap2 = (total2 * 0.12) / max(1, len(order2) - 1) if len(order2) > 1 else 0.0

    y2_points = np.concatenate([[0], np.cumsum(bars2)])[:-1] + np.arange(len(order2)) * gap2
    right_running = y2_points.copy()

    for i1, c1 in enumerate(order1):
        if bars1[i1] <= 0:
            continue
        row = np.array([counts[c1, c2] for c2 in order2], dtype=float)
        left_edges = np.concatenate([[0], np.cumsum(row)]) + y1_points[i1]
        top_lefts = left_edges[:-1]
        bottom_lefts = left_edges[1:]
        color = PALETTE[color_of1.get(c1, i1) % len(PALETTE)]

        for i2, c2 in enumerate(order2):
            amt = row[i2]
            if amt <= 0:
                continue
            top_r = right_running[i2]
            bot_r = top_r + amt
            right_running[i2] = bot_r

            bx, by = _sankey_curve_points(x1 + 0.12, bottom_lefts[i2], x2 - 0.12, bot_r)
            tx, ty = _sankey_curve_points(x2 - 0.12, top_r, x1 + 0.12, top_lefts[i2])
            xs = np.concatenate([[x1, x1], bx, [x2, x2], tx])
            ys = np.concatenate([[top_lefts[i2], bottom_lefts[i2]], by, [bot_r, top_r], ty])
            ax.fill(xs, ys, color=color, alpha=flow_alpha, linewidth=0)

    for i1, c1 in enumerate(order1):
        if bars1[i1] <= 0:
            continue
        color = PALETTE[color_of1.get(c1, i1) % len(PALETTE)]
        ax.plot([x1, x1], [y1_points[i1], y1_points[i1] + bars1[i1]],
                color=color, linewidth=bar_width, solid_capstyle="butt")

    return y2_points, gap2


def fig_cluster_sankey(R, flow_alpha=0.35, bar_width=22, show_perc=True):
    """
    Alluvial/Sankey chart showing how the pooled events regroup as K sweeps
    from 1 up to the max K tried.

    Category order at each K is chosen (via `_compute_sankey_layout`) to
    keep a parent cluster's children contiguous and positioned where the
    parent was, instead of always re-sorting by size -- this is what
    minimizes crisscrossing between adjacent K's. Color is anchored at the
    recommended Final K to match the pie chart / K Breakdown / 3D Explorer
    exactly (cluster c there is always PALETTE[c]), then propagated outward
    along each cluster's dominant lineage, so a cluster that persists keeps
    its color across K's and a genuine new split gets a genuinely new color.
    When `show_perc` is True, each block is labeled with the percentage of
    all events it holds at that K.
    """
    labels_by_k = R["labels_by_k"]
    ks = sorted(labels_by_k)
    n_layers = len(ks)
    n_obs = len(labels_by_k[ks[0]])

    order, color_of, _parent_of_stage, _children_of_stage = _compute_sankey_layout(labels_by_k, R["K"])

    fig = Figure(figsize=(13, 5.5), facecolor=WHITE, dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    x_positions = np.arange(n_layers, dtype=float)

    k0 = ks[0]
    lbl0 = labels_by_k[k0]
    order0 = order[k0]
    bars0 = np.array([(lbl0 == c).sum() for c in order0], dtype=float)
    gap0 = (n_obs * 0.12) / max(1, len(order0) - 1) if len(order0) > 1 else 0.0
    y_points = np.concatenate([[0], np.cumsum(bars0)])[:-1] + np.arange(len(order0)) * gap0

    y_points_by_stage = [y_points.copy()]
    order_by_stage = [order0]

    for t in range(n_layers - 1):
        kprev, kcur = ks[t], ks[t + 1]
        lprev, lcur = labels_by_k[kprev], labels_by_k[kcur]
        n_prev = len(np.unique(lprev))
        n_cur = len(np.unique(lcur))
        counts = np.zeros((n_prev, n_cur))
        for p in range(n_prev):
            mp = lprev == p
            if not mp.any():
                continue
            for c in range(n_cur):
                counts[p, c] = np.sum(mp & (lcur == c))

        y_points, _ = _sankey_draw_layer(
            ax, order[kprev], order[kcur], counts,
            x_positions[t], x_positions[t + 1], y_points,
            color_of[kprev], flow_alpha, bar_width,
        )
        order_by_stage.append(order[kcur])
        y_points_by_stage.append(y_points.copy())

    # Each stage's bar is normally drawn as the "from" side of the segment
    # leading OUT of it -- but the very last stage never plays that role
    # (there's no segment after it), so its bar has to be drawn explicitly
    # here or it's left blank.
    k_last = ks[-1]
    lbl_last = labels_by_k[k_last]
    order_last = order_by_stage[-1]
    y_last = y_points_by_stage[-1]
    color_last = color_of[k_last]
    for i, c in enumerate(order_last):
        cnt = int((lbl_last == c).sum())
        if cnt <= 0:
            continue
        color = PALETTE[color_last.get(c, i) % len(PALETTE)]
        ax.plot([x_positions[-1], x_positions[-1]], [y_last[i], y_last[i] + cnt],
                color=color, linewidth=bar_width, solid_capstyle="butt")

    if show_perc:
        for t, k in enumerate(ks):
            lbl_t = labels_by_k[k]
            order_t = order_by_stage[t]
            ypts_t = y_points_by_stage[t]
            for i, c in enumerate(order_t):
                cnt = int((lbl_t == c).sum())
                if cnt <= 0:
                    continue
                pct = 100.0 * cnt / n_obs
                y_mid = ypts_t[i] + cnt / 2.0
                ax.text(
                    x_positions[t] + 0.16, y_mid, f"{pct:.0f}%",
                    ha="left", va="center", fontsize=8, color=TEXT,
                    fontweight="bold",
                )

    ymax = n_obs * 1.12
    for t, k in enumerate(ks):
        weight = "bold" if k == R["K"] else "normal"
        color = SUCCESS if k == R["K"] else TEXT_MED
        label = f"K={k}" + ("  (final)" if k == R["K"] else "")
        ax.text(x_positions[t], ymax, label, ha="center", va="bottom",
                fontsize=9.5, fontweight=weight, color=color)
        if k == R["K"]:
            ax.axvline(x_positions[t], color=SUCCESS, lw=1.2, ls="--", alpha=0.5, zorder=0)

    ax.set_xlim(x_positions[0] - 0.3, x_positions[-1] + 0.3)
    ax.set_ylim(ymax * 1.08, -ymax * 0.03)  # inverted y, like the MATLAB "axis ij"
    ax.set_title("How clusters split as K increases  ·  color = lineage, not size rank",
                 fontweight="bold", color=TEXT, pad=18)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.04)
    return fig


def fig_scatter_grid(R, names=None):
    features = R["features"]
    X = R["X_raw"]
    labels, K = R["labels"], R["K"]
    pairs = list(combinations(range(len(features)), 2))

    if not pairs:
        fig = Figure(figsize=(10, 5), facecolor=WHITE, dpi=100)
        ax = fig.add_subplot(111)
        _style(ax)
        f = features[0]
        for k in range(K):
            vals = X.loc[labels == k, f].values
            ax.scatter(vals, np.zeros_like(vals) + k, s=25, alpha=0.6, color=PALETTE[k % len(PALETTE)], label=_cluster_label(names, k))
        ax.set_xlabel(_feature_label(f))
        ax.set_yticks(range(K))
        ax.set_yticklabels([_cluster_label(names, k) for k in range(K)])
        ax.set_title("One usable feature: cluster positions")
        return fig

    ncols = 2
    nrows = int(np.ceil(len(pairs) / ncols))
    fig = Figure(figsize=(13, max(4.3, nrows * 4.3)), facecolor=WHITE, dpi=100)
    fig.subplots_adjust(hspace=0.34, wspace=0.26, left=0.07, right=0.97, top=0.98, bottom=0.05)
    for idx, (fi, fj) in enumerate(pairs):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        _style(ax)
        for k in range(K):
            m = labels == k
            ax.scatter(X.iloc[m, fi], X.iloc[m, fj], s=18, alpha=0.6, linewidths=0,
                       color=PALETTE[k % len(PALETTE)], label=f"{_cluster_label(names, k)} (n={m.sum()})")
        ax.set_xlabel(_feature_label(features[fi]), color=TEXT_MED)
        ax.set_ylabel(_feature_label(features[fj]), color=TEXT_MED)
        if features[fi] == "duration":
            ax.set_xlim(left=0, right=float(X["duration"].max()) + 5.0)
        if features[fj] == "duration":
            ax.set_ylim(bottom=0, top=float(X["duration"].max()) + 5.0)
        ax.legend(fontsize=7.5, markerscale=1.5, edgecolor=BORDER, loc="best")
    return fig


def fig_report_pairs(R, names=None):
    """
    Fixed 2x3 grid of specific feature-pair scatter plots for the HTML
    report, colored by final cluster assignment:

        Duration vs OSC, OSC vs RefOSC, OSC vs Entry spike,
        Duration vs Entry spike, OSC vs Exit spike, Duration vs Exit spike.

    Any pair whose two features are not both present among R["features"]
    is skipped (not drawn). If none of the pairs are available, a small
    placeholder figure explaining that is returned instead.
    """
    features = R["features"]
    X = R["X_raw"]
    labels, K = R["labels"], R["K"]

    pairs = [
        ("duration", "OSC"),
        ("OSC", "RefOSC"),
        ("OSC", "entry_spike"),
        ("duration", "entry_spike"),
        ("OSC", "exit_spike"),
        ("duration", "exit_spike"),
    ]
    usable_pairs = [(a, b) for a, b in pairs if a in features and b in features]

    if not usable_pairs:
        fig = Figure(figsize=(9, 3), facecolor=WHITE, dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "None of the requested feature pairs (Duration/OSC/RefOSC/Entry spike/Exit spike)\n"
            "are available among this run's usable clustering features.",
            ha="center", va="center", color=TEXT_DIM, fontsize=10.5
        )
        return fig

    ncols = 2
    nrows = int(np.ceil(len(usable_pairs) / ncols))
    fig = Figure(figsize=(13, max(4.3, nrows * 4.3)), facecolor=WHITE, dpi=100)
    fig.subplots_adjust(hspace=0.34, wspace=0.26, left=0.07, right=0.97, top=0.95, bottom=0.06)
    for idx, (fx, fy) in enumerate(usable_pairs):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        _style(ax)
        for k in range(K):
            m = labels == k
            ax.scatter(X.loc[m, fx], X.loc[m, fy], s=18, alpha=0.6, linewidths=0,
                       color=PALETTE[k % len(PALETTE)], label=f"{_cluster_label(names, k)} (n={m.sum()})")
        ax.set_xlabel(_feature_label(fx), color=TEXT_MED)
        ax.set_ylabel(_feature_label(fy), color=TEXT_MED)
        if fx == "duration":
            ax.set_xlim(left=0, right=float(X["duration"].max()) + 5.0)
        if fy == "duration":
            ax.set_ylim(bottom=0, top=float(X["duration"].max()) + 5.0)
        ax.legend(fontsize=7.5, markerscale=1.5, edgecolor=BORDER, loc="best")
    fig.suptitle("Feature relationships", fontsize=13, fontweight="bold", color=TEXT, y=0.99)
    return fig


def fig_3d(R, xname, yname, zname, limits=None, names=None):
    """
    3D scatter of the pooled events colored by cluster.

    `limits` is an optional dict like {"x": (lo, hi), "y": (lo, hi),
    "z": (lo, hi)} where either side of a tuple may be None (no bound on
    that side). When given, only events whose xname/yname/zname values fall
    within ALL supplied bounds are plotted, and axis ranges are pinned to
    match. Feature keys not present in `limits`, or with both bounds None,
    are left unfiltered/auto-ranged.

    `names` is an optional {cluster_index: custom_name} dict used for the
    legend labels; indices without a custom name fall back to "C0", "C1", etc.
    """
    X, labels, K = R["X_raw"], R["labels"], R["K"]

    mask = pd.Series(True, index=X.index)
    limits = limits or {}
    axis_names = {"x": xname, "y": yname, "z": zname}
    for axis_key, fname in axis_names.items():
        lo, hi = limits.get(axis_key, (None, None))
        if lo is not None:
            mask &= X[fname] >= lo
        if hi is not None:
            mask &= X[fname] <= hi

    Xf = X.loc[mask]
    labelsf = labels[mask.values]

    fig = Figure(figsize=(12, 7.5), facecolor=WHITE, dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.fill = False
        p.pane.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.5)

    if len(Xf) == 0:
        ax.text2D(0.5, 0.5, "No points within the selected limits",
                  ha="center", va="center", transform=ax.transAxes, color=TEXT_DIM)
    else:
        for k in range(K):
            m = labelsf == k
            if not np.any(m):
                continue
            ax.scatter(Xf.loc[m, xname], Xf.loc[m, yname], Xf.loc[m, zname], s=55, alpha=0.78,
                       linewidths=0, color=PALETTE[k % len(PALETTE)], depthshade=True,
                       label=f"{_cluster_label(names, k)} (n={int(m.sum())}, {100*m.mean():.0f}%)")
        ax.legend(fontsize=10, edgecolor=BORDER)

    ax.set_xlabel(_feature_label(xname), color=TEXT_MED, labelpad=10)
    ax.set_ylabel(_feature_label(yname), color=TEXT_MED, labelpad=10)
    ax.set_zlabel(_feature_label(zname), color=TEXT_MED, labelpad=10)

    duration_top = float(X["duration"].max()) + 5.0 if "duration" in X.columns else None

    def _axis_range(axis_key, fname, default_top):
        lo, hi = limits.get(axis_key, (None, None))
        if lo is not None or hi is not None:
            auto_lo = float(X[fname].min())
            auto_hi = float(X[fname].max())
            return (lo if lo is not None else auto_lo, hi if hi is not None else auto_hi)
        if fname == "duration":
            return (0, default_top)
        return None

    xr = _axis_range("x", xname, duration_top)
    yr = _axis_range("y", yname, duration_top)
    zr = _axis_range("z", zname, duration_top)
    if xr is not None:
        ax.set_xlim(*xr)
    if yr is not None:
        ax.set_ylim(*yr)
    if zr is not None:
        ax.set_zlim(*zr)

    ax.tick_params(colors=TEXT_MED)
    ax.view_init(elev=18, azim=-60)
    title = "3D Explorer"
    if any(limits.get(k, (None, None)) != (None, None) for k in ("x", "y", "z")):
        title += f"  ·  showing {len(Xf)} of {len(X)} events within limits"
    ax.set_title(title, color=TEXT, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05)
    return fig


# --------------------------- Cluster hierarchy (locked selection) --------

def fig_3d_hierarchy(R, xname, yname, zname, k, anchor=None, limits=None):
    """
    3D scatter at an arbitrary K from the sweep.

    - No cluster selected (`anchor` is None): show EVERY cluster at K, one
      color each -- exactly like the normal 3D Explorer, just at whichever
      K the slider is on.

    - A cluster is selected (`anchor` = (k0, c0), the K/cluster where the
      user clicked): the exact set of events belonging to C(c0) at K=k0 is
      LOCKED once, at selection time, and that same fixed set of points is
      what's drawn at every K from then on -- nothing is ever added to it
      or dropped from it as the slider moves. What changes is only the
      COLOR: at the current slider K, each locked point is colored by
      whatever cluster label the independent K-cluster fit actually gives
      it (`labels_by_k[k]`, restricted to the locked points). So moving K
      up shows the locked set splitting into more colors (as more of the
      independent fit's clusters intersect it), and moving K down shows it
      collapsing into fewer colors -- reflecting the real per-K clustering
      rather than an assumed clean split/merge tree.
    """
    X = R["X_raw"]
    labels_k = pd.Series(R["labels_by_k"][k], index=X.index)

    limits = limits or {}
    axis_mask = pd.Series(True, index=X.index)
    axis_names = {"x": xname, "y": yname, "z": zname}
    for axis_key, fname in axis_names.items():
        lo, hi = limits.get(axis_key, (None, None))
        if lo is not None:
            axis_mask &= X[fname] >= lo
        if hi is not None:
            axis_mask &= X[fname] <= hi

    fig = Figure(figsize=(12, 7.5), facecolor=WHITE, dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.fill = False
        p.pane.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.5)

    groups = []  # list of (label, boolean_member_mask, color)
    if anchor is None:
        n_k = len(np.unique(labels_k))
        for c in range(n_k):
            groups.append((f"C{c}", labels_k == c, PALETTE[c % len(PALETTE)]))
        subtitle = "no cluster selected -- showing all clusters at this K"
    else:
        k0, c0 = anchor
        locked_mask = pd.Series(R["labels_by_k"][k0] == c0, index=X.index)
        n_locked = int(locked_mask.sum())
        sub_labels_here = labels_k[locked_mask]
        distinct = sorted(sub_labels_here.unique().tolist())
        for i, c in enumerate(distinct):
            member_mask = locked_mask & (labels_k == c)
            groups.append((f"C{c}", member_mask, PALETTE[i % len(PALETTE)]))
        if k == k0:
            subtitle = f"C{c0} @ K={k0} (n={n_locked}) -- the selected cluster itself"
        elif len(distinct) > 1:
            subtitle = f"the {n_locked} events of C{c0} @ K={k0} split into {len(distinct)} group(s) at K={k}"
        elif distinct:
            subtitle = f"the {n_locked} events of C{c0} @ K={k0} are all one group (C{distinct[0]}) at K={k}"
        else:
            subtitle = f"the {n_locked} events of C{c0} @ K={k0} could not be located at K={k}"

    any_drawn = False
    for label, member_mask, color in groups:
        mask = member_mask & axis_mask
        sub = X.loc[mask]
        if sub.empty:
            continue
        any_drawn = True
        ax.scatter(
            sub[xname], sub[yname], sub[zname], s=50, alpha=0.80, linewidths=0,
            color=color, depthshade=True, label=f"{label} (n={len(sub)})",
        )
    if any_drawn:
        ax.legend(fontsize=10, edgecolor=BORDER)
    else:
        ax.text2D(0.5, 0.5, "No points to show at this K",
                  ha="center", va="center", transform=ax.transAxes, color=TEXT_DIM)

    ax.set_xlabel(_feature_label(xname), color=TEXT_MED, labelpad=10)
    ax.set_ylabel(_feature_label(yname), color=TEXT_MED, labelpad=10)
    ax.set_zlabel(_feature_label(zname), color=TEXT_MED, labelpad=10)

    duration_top = float(X["duration"].max()) + 5.0 if "duration" in X.columns else None
    if xname == "duration" and duration_top is not None:
        ax.set_xlim(0, duration_top)
    if yname == "duration" and duration_top is not None:
        ax.set_ylim(0, duration_top)
    if zname == "duration" and duration_top is not None:
        ax.set_zlim(0, duration_top)

    ax.tick_params(colors=TEXT_MED)
    ax.view_init(elev=18, azim=-60)
    ax.set_title(f"Cluster Hierarchy (3D)  ·  K = {k}\n{subtitle}", color=TEXT, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.05)
    return fig


def fig_distributions(R, names=None):
    features = R["features"]
    X, labels, K = R["X_raw"], R["labels"], R["K"]
    n = len(features)
    fig = Figure(figsize=(max(7, 2.8 * n), 5.5), facecolor=WHITE, dpi=100)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.12, wspace=0.35)
    for fi, fn in enumerate(features):
        ax = fig.add_subplot(1, n, fi + 1)
        _style(ax)
        data = [X.loc[labels == k, fn].values for k in range(K)]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color=WHITE, lw=2),
                        whiskerprops=dict(color=TEXT_MED, lw=1), capprops=dict(color=TEXT_MED, lw=1),
                        flierprops=dict(marker=".", ms=3, alpha=0.4))
        for patch, k in zip(bp["boxes"], range(K)):
            patch.set_facecolor(PALETTE[k % len(PALETTE)])
            patch.set_alpha(0.75)
            patch.set_edgecolor(BORDER)
        ax.set_title(_feature_label(fn), fontsize=10, fontweight="bold", color=TEXT)
        ax.set_xticks(range(1, K + 1))
        ax.set_xticklabels([_cluster_label(names, k) for k in range(K)], fontsize=8, color=TEXT_MED)
        if fn == "duration":
            ax.set_ylim(0, float(X["duration"].max()) + 5.0)
    fig.suptitle("Feature distributions per cluster", fontsize=13, color=TEXT, y=0.98)
    return fig



def fig_compare_3d(
    RA, RB, xname, yname, zname,
    label_a="Folder A", label_b="Folder B",
    alpha_a=0.72, alpha_b=0.72
):
    """
    Overlay two independently clustered datasets in one 3D explorer.

    Shape encodes dataset:
      Folder A = circle
      Folder B = triangle

    Color encodes rank (largest = PALETTE[0], etc.) within that dataset, using
    the same fixed 10-color palette as everywhere else, so C0 in A and C0 in B
    both use the same color. Cluster numbers from A and B are independent fits
    and should not be interpreted as matched classes.
    """
    fig = Figure(figsize=(12.5, 7.8), facecolor=WHITE, dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.fill = False
        p.pane.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.5)

    alpha_a = float(np.clip(alpha_a, 0.03, 1.0))
    alpha_b = float(np.clip(alpha_b, 0.03, 1.0))

    datasets = [
        (RA, label_a, "o", alpha_a),
        (RB, label_b, "^", alpha_b),
    ]

    for R, group_label, marker, point_alpha in datasets:
        X = R["X_raw"]
        labels = R["labels"]
        for k in range(R["K"]):
            m = labels == k
            ax.scatter(
                X.loc[m, xname],
                X.loc[m, yname],
                X.loc[m, zname],
                s=58,
                alpha=point_alpha,
                linewidths=0.45,
                edgecolors=WHITE,
                marker=marker,
                color=PALETTE[k % len(PALETTE)],
                depthshade=True,
                label=f"{group_label} · C{k} (n={int(m.sum())})",
            )

    ax.set_xlabel(_feature_label(xname), color=TEXT_MED, labelpad=10)
    ax.set_ylabel(_feature_label(yname), color=TEXT_MED, labelpad=10)
    ax.set_zlabel(_feature_label(zname), color=TEXT_MED, labelpad=10)

    # Keep duration axes compact: maximum displayed duration + 5 seconds.
    if "duration" in RA["X_raw"].columns and "duration" in RB["X_raw"].columns:
        duration_top = max(
            float(RA["X_raw"]["duration"].max()),
            float(RB["X_raw"]["duration"].max())
        ) + 5.0
        if xname == "duration":
            ax.set_xlim(0, duration_top)
        if yname == "duration":
            ax.set_ylim(0, duration_top)
        if zname == "duration":
            ax.set_zlim(0, duration_top)

    ax.tick_params(colors=TEXT_MED)
    ax.view_init(elev=18, azim=-60)
    ax.legend(fontsize=8.5, edgecolor=BORDER, loc="best")
    ax.set_title(
        f"{label_a}: circles   ·   {label_b}: triangles",
        fontweight="bold", color=TEXT, pad=16
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.05)
    return fig


# --------------------------- Explore-other-K precompute ------------------------

def _compute_all_k_stats(R):
    """
    Per-K (N, Pct, and per-feature mean) for every K tried during the sweep,
    used to populate the HTML report's "Explore other K" tab. Only the final
    recommended K is used to decide the reported result -- this is
    precomputed data for side-by-side visual comparison only.
    """
    X = R["X_raw"]
    features = R["features"]
    out = {}
    for k, labels in R["labels_by_k"].items():
        n_total = len(labels)
        rows = []
        for c in range(k):
            m = labels == c
            row = {
                "cluster": c,
                "n": int(m.sum()),
                "pct": round(100.0 * float(m.mean()), 2) if n_total else 0.0,
            }
            for f in features:
                v = X.loc[m, f].to_numpy(dtype=float)
                row[f] = float(np.mean(v)) if len(v) else None
            rows.append(row)
        out[k] = rows
    return out


def _fig_k_pie(labels, k):
    """Small standalone proportions pie chart for one K, used in the
    Explore-other-K tab (swapped in via JS as the user changes K)."""
    fig = Figure(figsize=(4.2, 4.2), facecolor=WHITE, dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor(WHITE)
    counts = [(labels == c).sum() for c in range(k)]
    n_total = len(labels) if len(labels) else 1
    colors = [PALETTE[c % len(PALETTE)] for c in range(k)]
    ax.pie(
        counts,
        labels=[f"C{c}" for c in range(k)],
        autopct=lambda p: (f"{p:.1f}%" if p > 0 else ""),
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor=WHITE, linewidth=1.0),
        textprops=dict(fontsize=9, color=TEXT),
    )
    ax.set_title(f"K = {k}", fontsize=11, fontweight="bold", color=TEXT)
    ax.axis("equal")
    return fig


def _build_explore_k_plotly(R, xname, yname, zname, limits):
    """
    One Plotly figure holding a trace per (K, cluster) pair across the whole
    sweep, with only the recommended final K's traces visible by default.
    The HTML report's "Explore other K" tab flips visibility between K's via
    Plotly.restyle rather than rebuilding the figure, so switching K is
    instant and needs no server/recompute.

    Returns (figure, trace_k) where trace_k[i] is the K that trace i belongs
    to, so the page's JS can build the boolean visibility mask.
    """
    import plotly.graph_objects as go

    X = R["X_raw"]
    ks = sorted(R["labels_by_k"])

    mask = pd.Series(True, index=X.index)
    limits = limits or {}
    axis_names = {"x": xname, "y": yname, "z": zname}
    for axis_key, fname in axis_names.items():
        lo, hi = limits.get(axis_key, (None, None))
        if lo is not None:
            mask &= X[fname] >= lo
        if hi is not None:
            mask &= X[fname] <= hi

    Xf = X.loc[mask]
    traces = []
    trace_k = []
    for k in ks:
        labf = R["labels_by_k"][k][mask.values]
        for c in range(k):
            m = labf == c
            sub = Xf.loc[m]
            traces.append(go.Scatter3d(
                x=sub[xname], y=sub[yname], z=sub[zname],
                mode="markers",
                marker=dict(size=3.5, color=PALETTE[c % len(PALETTE)], opacity=0.8, line=dict(width=0)),
                name=f"C{c} (n={int(m.sum())})",
                visible=(k == R["K"]),
            ))
            trace_k.append(k)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=_feature_label(xname)),
            yaxis=dict(title=_feature_label(yname)),
            zaxis=dict(title=_feature_label(zname)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=600,
        paper_bgcolor=WHITE,
        showlegend=True,
    )
    return fig, trace_k


# --------------------------- HTML report export -------------------------------

def _k_breakdown_to_html_table(df, final_k, k_bic, names=None):
    """
    Horizontal K-breakdown table: one COLUMN per K tried during the sweep,
    each column stacking that K's clusters top-to-bottom (C0, C1, ...), each
    cell showing the cluster's percentage of pooled events -- matching the
    "K=1 | K=2 | K=3 | K=4" layout used in the app's own K Breakdown view.

    The recommended final K's column is highlighted green; the lowest-BIC
    K's column (when it differs from the final K) is highlighted amber.

    Custom cluster names (if any) are only applied to the final-K column,
    since cluster indices at other K stages in the sweep are separate GMM
    fits and don't correspond 1:1 to the final clustering.
    """
    ks_sorted = sorted(int(k) for k in df["K"].unique().tolist())

    by_k = {}
    for k in ks_sorted:
        rows_k = df[df["K"] == k].copy()
        rows_k["_cidx"] = rows_k["Cluster"].apply(lambda c: int(str(c)[1:]))
        rows_k = rows_k.sort_values("_cidx")
        cells = []
        for _, row in rows_k.iterrows():
            cidx = int(row["_cidx"])
            label = _cluster_label(names, cidx) if k == final_k else str(row["Cluster"])
            cells.append((label, float(row["Pct"])))
        by_k[k] = cells

    max_rows = max((len(v) for v in by_k.values()), default=0)

    def _col_class(k):
        if k == final_k:
            return "final"
        if k == k_bic:
            return "bic"
        return ""

    header_cells = []
    for k in ks_sorted:
        cls = _col_class(k)
        suffix = " (recommended)" if k == final_k else (" (lowest BIC)" if k == k_bic else "")
        header_cells.append(f"<th class='{cls}'>K={k}{suffix}</th>")

    body_rows = []
    for r in range(max_rows):
        cells = []
        for k in ks_sorted:
            cls = _col_class(k)
            col = by_k[k]
            if r < len(col):
                label, pct = col[r]
                cells.append(f"<td class='{cls}'>{html.escape(label)} &middot; {pct:.1f}%</td>")
            else:
                cells.append(f"<td class='{cls}'></td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<table class='data-table kbreak-table'><thead><tr>"
        f"{''.join(header_cells)}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def build_html_report(R, xname, yname, zname, limits, output_path, run_meta=None, names=None):
    """
    Write a single self-contained HTML report for the CURRENT clustering
    result `R`, using whichever 3D-Explorer axis choices and axis limits are
    passed in (i.e. exactly what the GUI's 3D Explorer tab is showing right
    now). Nothing is recomputed here -- if points were deleted and the model
    re-run before calling this, the report reflects that latest run.

    The report has two tabs:

      "Report" (default) embeds, top to bottom:
        - Files: the source file name(s) that went into this run
        - K Selection: the BIC / cluster-proportions (pie) figure, captioned
          with the plain-English reason Final K was chosen
        - Sankey Plot: the "how clusters split as K increases" chart
        - K Breakdown: a horizontal, one-column-per-K table (K=1, K=2, ...)
          with the recommended final K's column highlighted green and the
          lowest-BIC K's column highlighted amber
        - Feature Relationships: a fixed 2x3 grid of Duration vs OSC, OSC vs
          RefOSC, OSC vs Entry spike, Duration vs Entry spike, OSC vs Exit
          spike, and Duration vs Exit spike (pairs not available for this
          run's features are simply omitted)
        - an INTERACTIVE Plotly 3D scatter of the currently filtered events,
          respecting the given axis choices and min/max limits
        - a "Save as PDF" button (top right, next to the tabs) that opens
          the browser's print dialog with a print stylesheet tuned to
          produce a clean PDF of this Report tab

      "Explore other K" lets the reader pick any K tried during the sweep
      (not just the recommended Final K) and preview -- via precomputed,
      client-side-only data, no recompute -- that K's cluster proportions
      (pie chart), per-cluster feature averages (table), and an interactive
      3D scatter colored by that K's cluster assignment. This is for
      visualization only; it does not change the reported/recommended
      result.

    `run_meta` is an optional dict (key: "sample_name") used only for the
    short run-description panel at the top of the report.

    `names` is an optional {cluster_index: custom_name} dict; when given, it
    replaces "C0", "C1", etc. with the custom names in the K-breakdown table
    (final-K column only) and in the interactive 3D legend.
    """
    run_meta = run_meta or {}
    limits = limits or {}
    names = names or {}
    features = R["features"]
    ks_sorted = sorted(R["labels_by_k"])

    # A short, editable description shown at the top of the report. Pre-filled
    # with a plain-English sentence built from the source files so there's
    # something sensible to start from, but the reader can click into it and
    # replace it with their own note (sample name, context, whatever) before
    # printing/saving as PDF -- no per-file table is shown by default.
    _file_names = [
        os.path.basename(fi["path"]) for fi in R.get("file_info", [])
        if not fi.get("skipped_file")
    ]
    if len(_file_names) == 1:
        default_note = f"Data from {_file_names[0]} ({len(R['X_raw'])} events)."
    elif _file_names:
        default_note = (
            f"Data from {len(_file_names)} files: {', '.join(_file_names)} "
            f"({len(R['X_raw'])} events total)."
        )
    else:
        default_note = ""
    default_note_html = html.escape(default_note)

    # ---- static matplotlib figures, embedded as base64 PNG ----
    def _fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    k_sel_b64 = _fig_to_b64(fig_kselection(R, names=names))
    sankey_b64 = _fig_to_b64(fig_cluster_sankey(R))
    pairs_b64 = _fig_to_b64(fig_report_pairs(R, names=names))

    # ---- interactive Plotly 3D scatter (only if 3+ usable features exist) ----
    has_3d = len(features) >= 3
    if has_3d and not (xname and yname and zname and all(f in features for f in (xname, yname, zname))):
        xname, yname, zname = features[0], features[1], features[2]

    filt_note = ""
    trace_feature_data_json = "[]"
    features_3d_json = "[]"
    feature_labels_3d_json = "{}"
    duration_range_3d_json = "null"
    if has_3d:
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError(
                "The HTML report needs the 'plotly' package for the interactive 3D plot.\n"
                "Install it with:  pip install plotly"
            ) from e

        X, labels, K = R["X_raw"], R["labels"], R["K"]
        labeled = R["labeled"]

        mask = pd.Series(True, index=X.index)
        axis_names = {"x": xname, "y": yname, "z": zname}
        for axis_key, fname in axis_names.items():
            lo, hi = limits.get(axis_key, (None, None))
            if lo is not None:
                mask &= X[fname] >= lo
            if hi is not None:
                mask &= X[fname] <= hi

        Xf = X.loc[mask]
        labf = labels[mask.values]
        labeledf = labeled.loc[mask].reset_index(drop=True)

        traces = []
        trace_feature_data = []  # one {feature: [values...]} dict per trace, for the axis dropdowns
        for k in range(K):
            m = labf == k
            if not np.any(m):
                continue
            sub = Xf.loc[m]
            subl = labeledf.loc[np.asarray(m)]
            hover = []
            for row in subl.itertuples(index=False):
                lines = [f"{row.source_file}, row {int(row.source_row)}"]
                for f in features:
                    lines.append(f"{f}: {getattr(row, f):.4g}")
                hover.append("<br>".join(lines))
            traces.append(go.Scatter3d(
                x=sub[xname], y=sub[yname], z=sub[zname],
                mode="markers",
                marker=dict(size=4, color=PALETTE[k % len(PALETTE)], opacity=0.8, line=dict(width=0)),
                name=f"{_cluster_label(names, k)} (n={int(m.sum())}, {100*np.mean(m):.0f}%)",
                text=hover,
                hoverinfo="text",
            ))
            trace_feature_data.append({f: sub[f].round(6).tolist() for f in features})

        fig3d = go.Figure(data=traces)

        def _axis_range(axis_key, fname):
            lo, hi = limits.get(axis_key, (None, None))
            if lo is not None or hi is not None:
                auto_lo = float(X[fname].min())
                auto_hi = float(X[fname].max())
                return [lo if lo is not None else auto_lo, hi if hi is not None else auto_hi]
            if fname == "duration":
                return [0, float(X["duration"].max()) + 5.0]
            return None

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title=_feature_label(xname), range=_axis_range("x", xname)),
                yaxis=dict(title=_feature_label(yname), range=_axis_range("y", yname)),
                zaxis=dict(title=_feature_label(zname), range=_axis_range("z", zname)),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(itemsizing="constant"),
            height=650,
            paper_bgcolor=WHITE,
        )
        if any(limits.get(k, (None, None)) != (None, None) for k in ("x", "y", "z")):
            filt_note = f"Showing {len(Xf)} of {len(X)} events within the selected axis limits."
        plot_section = fig3d.to_html(full_html=False, include_plotlyjs="inline", div_id="explorer3d")

        trace_feature_data_json = json.dumps(trace_feature_data)
        features_3d_json = json.dumps(features)
        feature_labels_3d_json = json.dumps({f: _feature_label(f) for f in features})
        duration_range_3d_json = (
            json.dumps([0, float(X["duration"].max()) + 5.0]) if "duration" in features else "null"
        )
        axis_controls_html = f"""<div style="margin: 0 0 12px;">
          <label style="font-weight:600; margin-right:6px;">X:</label>
          <select id="axis-x-select" class="axis-select" onchange="updateAxis3D()"></select>
          <label style="font-weight:600; margin:0 6px 0 16px;">Y:</label>
          <select id="axis-y-select" class="axis-select" onchange="updateAxis3D()"></select>
          <label style="font-weight:600; margin:0 6px 0 16px;">Z:</label>
          <select id="axis-z-select" class="axis-select" onchange="updateAxis3D()"></select>
        </div>"""
    else:
        plot_section = (
            "<div class='legend-note'>3D Explorer not available for this run "
            "(fewer than 3 usable clustering features).</div>"
        )
        axis_controls_html = ""

    # ---- Explore-other-K precomputed data ----
    all_k_stats = _compute_all_k_stats(R)
    k_pies_b64 = {k: _fig_to_b64(_fig_k_pie(R["labels_by_k"][k], k)) for k in ks_sorted}

    explore_plot_section = (
        "<div class='legend-note'>Interactive 3D preview not available for this run "
        "(fewer than 3 usable clustering features).</div>"
    )
    trace_k_json = "[]"
    if has_3d:
        try:
            explore_fig, trace_k = _build_explore_k_plotly(R, xname, yname, zname, limits)
            explore_plot_section = explore_fig.to_html(
                full_html=False, include_plotlyjs=False, div_id="exploreK3d"
            )
            trace_k_json = json.dumps(trace_k)
        except Exception:
            explore_plot_section = (
                "<div class='legend-note'>Interactive 3D preview unavailable for the Explore-K tab.</div>"
            )

    kbreak_html = _k_breakdown_to_html_table(R["k_breakdown"], R["K"], R["k_bic"], names=names)

    sample_name = (run_meta.get("sample_name") or "").strip()
    features_used = ", ".join(features)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    axes_note = f"X = {_feature_label(xname)}, Y = {_feature_label(yname)}, Z = {_feature_label(zname)}." if has_3d else ""
    sample_line = (
        f'<div class="meta"><b>Sample:</b> {html.escape(sample_name)}</div>' if sample_name else ""
    )

    k_options_html = "".join(
        f'<option value="{k}"{" selected" if k == R["K"] else ""}>'
        f'{k}{" (recommended)" if k == R["K"] else ""}</option>'
        for k in ks_sorted
    )
    feature_headers_html = "".join(f"<th>{html.escape(_feature_label(f))}</th>" for f in features)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cluster Viewer Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; background:{BG}; color:{TEXT}; margin:0; padding:0; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 32px 60px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 17px; margin-top: 40px; border-bottom: 2px solid {BORDER}; padding-bottom: 6px; color:{TEXT}; }}
  .meta {{ color:{TEXT_MED}; font-size: 13px; margin-bottom: 6px; }}
  .card {{ background:{WHITE}; border:1px solid {BORDER}; border-radius: 10px; padding: 18px 20px; margin-top: 14px; }}
  .legend-note {{ font-size: 12.5px; color:{TEXT_MED}; margin: 6px 0 12px; }}
  .legend-swatch {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:5px; vertical-align:middle; }}
  img.report-fig {{ max-width: 100%; height: auto; display:block; margin: 10px auto; }}
  table.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.data-table th {{ background:{PANEL}; color:{TEXT_MED}; text-align:center; padding: 7px 10px; border-bottom: 2px solid {BORDER}; }}
  table.data-table td {{ text-align:center; padding: 6px 10px; border-bottom: 1px solid {BORDER}; }}
  tr.even {{ background:{WHITE}; }}
  tr.odd {{ background:#f8faff; }}
  tr.final {{ background:#dcfce7; font-weight:600; }}
  tr.bic {{ background:#fef3c7; }}
  table.kbreak-table th, table.kbreak-table td {{ text-align:center; }}
  table.kbreak-table th.final, table.kbreak-table td.final {{ background:#dcfce7; font-weight:600; }}
  table.kbreak-table th.bic, table.kbreak-table td.bic {{ background:#fef3c7; }}
  .footer-note {{ margin-top: 40px; font-size: 12px; color:{TEXT_DIM}; }}
  .tabs-nav {{ display:flex; align-items:center; gap:8px; margin-bottom: 6px; }}
  .tab-btn {{ background:{PANEL}; color:{TEXT_MED}; border:1px solid {BORDER}; border-radius: 8px 8px 0 0;
              padding: 9px 18px; font-size: 14px; font-weight:600; cursor:pointer; }}
  .tab-btn.active {{ background:{WHITE}; color:{ACCENT}; border-bottom-color:{WHITE}; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}
  select#k-select, select.axis-select {{ font-size: 14px; padding: 5px 10px; border-radius:6px; border:1px solid {BORDER}; }}
  .explore-row {{ display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start; }}
  #save-pdf-btn {{ margin-left:auto; background:{ACCENT}; color:{WHITE}; border:none; border-radius:8px;
                    padding: 9px 18px; font-size: 14px; font-weight:600; cursor:pointer; }}
  #save-pdf-btn:hover {{ opacity: 0.9; }}
  .note-box {{ border: 1px solid {BORDER}; border-radius: 8px; padding: 12px 14px; font-size: 14px;
               line-height: 1.5; min-height: 24px; background: {WHITE}; color: {TEXT}; }}
  .note-box:focus {{ outline: 2px solid {ACCENT}; outline-offset: 1px; }}
  .note-box:empty:before {{ content: "Click to add a description…"; color: {TEXT_DIM}; }}
  .print-only {{ display: none; }}
  @media print {{
    .tabs-nav {{ display:none !important; }}
    #tab-report {{ display:block !important; }}
    #tab-explore {{ display:none !important; }}
    .card {{ break-inside: avoid; border: 1px solid #ccc; }}
    body {{ background: {WHITE}; }}
    .note-box {{ border: none; padding: 0; }}
    .screen-only {{ display: none !important; }}
    .print-only {{ display: block !important; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Real Data Cluster Viewer — Report</h1>
  {sample_line}
  <div class="meta">Generated {generated_at} &middot; Final K = {R['K']} (BIC-best K = {R['k_bic']}) &middot; {len(R['X_raw'])} events &middot; features: {features_used}</div>

  <div class="tabs-nav">
    <button type="button" class="tab-btn active" id="btn-report" onclick="showReportTab('report')">Report</button>
    <button type="button" class="tab-btn" id="btn-explore" onclick="showReportTab('explore')">Explore other K</button>
    <button type="button" id="save-pdf-btn" onclick="saveAsPdf()">🖨️ Save as PDF</button>
  </div>

  <div id="tab-report" class="tab-content active">

    <h2>Notes</h2>
    <div class="card">
      <div class="legend-note">Click below to add or edit a short description of this run (sample name, files used, anything you'd like noted). This text prints/saves with the report.</div>
      <div id="report-note" class="note-box" contenteditable="true">{default_note_html}</div>
    </div>

    <h2>K Selection</h2>
    <div class="card">
      <img class="report-fig" src="data:image/png;base64,{k_sel_b64}">
    </div>

    <h2>Sankey Plot</h2>
    <div class="card">
      <div class="legend-note">How clusters split as K increases.</div>
      <img class="report-fig" src="data:image/png;base64,{sankey_b64}">
    </div>

    <h2>K Breakdown</h2>
    <div class="card">
      <div class="legend-note">
        <span class="legend-swatch" style="background:#86efac;"></span>Recommended final K = {R['K']} — statistically significant (sequential bootstrap LRT)
        &nbsp;&nbsp;
        <span class="legend-swatch" style="background:#fde68a;"></span>Lowest-BIC K = {R['k_bic']}
      </div>
      {kbreak_html}
    </div>

    <h2>Feature Relationships</h2>
    <div class="card">
      <div class="legend-note">Duration vs OSC, OSC vs RefOSC, OSC vs Entry spike, Duration vs Entry spike, OSC vs Exit spike, Duration vs Exit spike.</div>
      <img class="report-fig" src="data:image/png;base64,{pairs_b64}">
    </div>

    <h2>3D Explorer (interactive)</h2>
    <div class="card">
      <div class="legend-note">{axes_note}{(' ' + filt_note) if filt_note else ''} Drag to rotate, scroll to zoom, hover a point for its source file/row and feature values. Use the X/Y/Z dropdowns below to plot any other feature on any axis -- this only changes what's drawn, it doesn't re-filter which events are shown.{' A snapshot of the current view is used when you print or save as PDF, since the interactive 3D plot itself does not print.' if has_3d else ''}</div>
      {axis_controls_html}
      <div id="explorer3d-wrap" class="screen-only">
        {plot_section}
      </div>
      {'<img id="explorer3d-print-img" class="report-fig print-only" alt="3D Explorer snapshot">' if has_3d else ''}
    </div>

  </div>

  <div id="tab-explore" class="tab-content">
    <h2>Explore other K</h2>
    <div class="card">
      <div class="legend-note">
        The Report tab above always reflects the recommended Final K = {R['K']} (chosen by the sequential
        bootstrap LRT) — that does not change here. This tab lets you preview, from precomputed data only,
        what the clustering would have looked like at any other K tried during the sweep.
      </div>
      <div style="margin: 6px 0 16px;">
        <label for="k-select" style="font-weight:600; margin-right:8px;">Preview K =</label>
        <select id="k-select" onchange="updateExploreK()">
          {k_options_html}
        </select>
      </div>
      <div class="explore-row">
        <div>
          <img id="k-pie-img" class="report-fig" style="max-width:380px;" src="data:image/png;base64,{k_pies_b64[R['K']]}">
        </div>
        <div style="flex:1; min-width:340px;">
          <table class="data-table" id="k-summary-table">
            <thead><tr><th>Cluster</th><th>N</th><th>Pct</th>{feature_headers_html}</tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <div class="legend-note" style="margin-top:16px;">{axes_note} Interactive 3D preview at the selected K (drag to rotate, hover a point for details):</div>
      {explore_plot_section}
    </div>
  </div>

  <div class="footer-note">Generated by Real Data Cluster Viewer.</div>
</div>

<script>
  const ALL_K_STATS = {json.dumps(all_k_stats)};
  const K_PIES = {json.dumps(k_pies_b64)};
  const TRACE_K = {trace_k_json};
  const FEATURES = {json.dumps(features)};

  const TRACE_FEATURE_DATA_3D = {trace_feature_data_json};
  const FEATURES_3D = {features_3d_json};
  const FEATURE_LABELS_3D = {feature_labels_3d_json};
  const DURATION_RANGE_3D = {duration_range_3d_json};
  const INITIAL_AXES_3D = {json.dumps({"x": xname, "y": yname, "z": zname} if has_3d else {})};

  function showReportTab(name) {{
    document.getElementById('tab-report').classList.toggle('active', name === 'report');
    document.getElementById('tab-explore').classList.toggle('active', name === 'explore');
    document.getElementById('btn-report').classList.toggle('active', name === 'report');
    document.getElementById('btn-explore').classList.toggle('active', name === 'explore');
  }}

  function populateAxisSelect3D(id, selected) {{
    const sel = document.getElementById(id);
    if (!sel) return;
    FEATURES_3D.forEach(function(f) {{
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = FEATURE_LABELS_3D[f] || f;
      if (f === selected) opt.selected = true;
      sel.appendChild(opt);
    }});
  }}

  function axisRangeFor3D(f) {{
    if (f === 'duration' && DURATION_RANGE_3D) return DURATION_RANGE_3D;
    return null;
  }}

  function updateAxis3D() {{
    const gd = document.getElementById('explorer3d');
    const xsel = document.getElementById('axis-x-select');
    const ysel = document.getElementById('axis-y-select');
    const zsel = document.getElementById('axis-z-select');
    if (!gd || !gd.data || !window.Plotly || !xsel || !ysel || !zsel) return;

    const xf = xsel.value, yf = ysel.value, zf = zsel.value;
    const xs = TRACE_FEATURE_DATA_3D.map(function(d) {{ return d[xf]; }});
    const ys = TRACE_FEATURE_DATA_3D.map(function(d) {{ return d[yf]; }});
    const zs = TRACE_FEATURE_DATA_3D.map(function(d) {{ return d[zf]; }});
    window.Plotly.restyle(gd, {{x: xs, y: ys, z: zs}});

    const xr = axisRangeFor3D(xf), yr = axisRangeFor3D(yf), zr = axisRangeFor3D(zf);
    const sceneUpdate = {{
      'scene.xaxis.title.text': FEATURE_LABELS_3D[xf] || xf,
      'scene.yaxis.title.text': FEATURE_LABELS_3D[yf] || yf,
      'scene.zaxis.title.text': FEATURE_LABELS_3D[zf] || zf,
    }};
    sceneUpdate['scene.xaxis.autorange'] = xr ? false : true;
    if (xr) sceneUpdate['scene.xaxis.range'] = xr;
    sceneUpdate['scene.yaxis.autorange'] = yr ? false : true;
    if (yr) sceneUpdate['scene.yaxis.range'] = yr;
    sceneUpdate['scene.zaxis.autorange'] = zr ? false : true;
    if (zr) sceneUpdate['scene.zaxis.range'] = zr;
    window.Plotly.relayout(gd, sceneUpdate);
    // Keep the print snapshot in sync with whatever the reader last set the
    // axes to, so printing right after changing an axis still shows the
    // current view rather than a stale one.
    refreshExplorer3DPrintImage();
  }}

  function initAxis3D() {{
    if (!FEATURES_3D.length) return;
    populateAxisSelect3D('axis-x-select', INITIAL_AXES_3D.x);
    populateAxisSelect3D('axis-y-select', INITIAL_AXES_3D.y);
    populateAxisSelect3D('axis-z-select', INITIAL_AXES_3D.z);
  }}

  // The interactive 3D Explorer is WebGL-based and prints as a blank box in
  // every major browser, so for printing/PDF we swap in a plain PNG
  // screenshot of the plot's current view (Plotly.toImage), shown only in
  // the print stylesheet while the live plot stays hidden there. Refreshed
  // on load, whenever the axes change, and right before printing.
  function refreshExplorer3DPrintImage() {{
    const gd = document.getElementById('explorer3d');
    const img = document.getElementById('explorer3d-print-img');
    if (!gd || !gd.data || !window.Plotly || !img) return Promise.resolve();
    const w = Math.max(600, gd.offsetWidth || 900);
    const h = Math.max(400, gd.offsetHeight || 650);
    return window.Plotly.toImage(gd, {{format: 'png', width: w, height: h, scale: 2}})
      .then(function(url) {{ img.src = url; }})
      .catch(function() {{ /* leave any previous snapshot in place */ }});
  }}

  function saveAsPdf() {{
    // Refresh the snapshot right before printing so a reader who never
    // touched the axis controls (or just rotated the plot) still gets a
    // current view, then open the print dialog once it's ready.
    Promise.resolve(refreshExplorer3DPrintImage()).finally(function() {{
      window.print();
    }});
  }}

  function updateExploreK() {{
    const sel = document.getElementById('k-select');
    if (!sel) return;
    const k = sel.value;

    const pieImg = document.getElementById('k-pie-img');
    if (pieImg && K_PIES[k]) {{
      pieImg.src = 'data:image/png;base64,' + K_PIES[k];
    }}

    const tbody = document.querySelector('#k-summary-table tbody');
    if (tbody) {{
      tbody.innerHTML = '';
      const rows = ALL_K_STATS[k] || [];
      rows.forEach(function(row) {{
        const tr = document.createElement('tr');
        let rowHtml = '<td>C' + row.cluster + '</td><td>' + row.n + '</td><td>' + row.pct.toFixed(2) + '%</td>';
        FEATURES.forEach(function(f) {{
          const v = row[f];
          rowHtml += '<td>' + ((v === null || v === undefined) ? '-' : Number(v).toPrecision(4)) + '</td>';
        }});
        tr.innerHTML = rowHtml;
        tbody.appendChild(tr);
      }});
    }}

    const gd = document.getElementById('exploreK3d');
    if (gd && gd.data && window.Plotly) {{
      const kNum = parseInt(k, 10);
      const vis = TRACE_K.map(function(tk) {{ return tk === kNum; }});
      window.Plotly.restyle(gd, {{visible: vis}});
    }}
  }}

  window.addEventListener('DOMContentLoaded', function() {{
    updateExploreK();
    initAxis3D();
    // Give the WebGL 3D plot a moment to finish its first paint before
    // grabbing the initial print snapshot.
    setTimeout(refreshExplorer3DPrintImage, 400);
  }});
</script>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    return output_path


# ================================= GUI =========================================
def launch():
    matplotlib.use("TkAgg")
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    # --- Silence noisy "main thread is not in main loop" shutdown errors ---
    # When this app is launched from a Jupyter/IPython notebook (ipykernel)
    # rather than a plain `python script.py` process, the notebook kernel
    # keeps running after the Tk window is closed and `mainloop()` returns.
    # Any lingering reference to the App (e.g. held by the notebook's cell
    # output or a variable) means Python's garbage collector eventually
    # frees the app's tk.Variable objects (IntVar/DoubleVar/StringVar/
    # BooleanVar) *after* the Tcl interpreter behind them has already been
    # torn down. tkinter.Variable.__del__ then tries to call into that dead
    # interpreter and raises RuntimeError: "main thread is not in main
    # loop". Since this happens inside __del__, Python doesn't crash -- it
    # just prints "Exception ignored in: ..." for every leftover Variable.
    # This is harmless (nothing in the app depends on that cleanup
    # succeeding), so swallow only that specific error here.
    _orig_var_del = tk.Variable.__del__

    def _quiet_var_del(self):
        try:
            _orig_var_del(self)
        except RuntimeError:
            pass

    tk.Variable.__del__ = _quiet_var_del

    # The same dead-interpreter-at-shutdown problem hits tkinter.Image
    # (PhotoImage) objects too, not just Variables. NavigationToolbar2Tk
    # creates a handful of PhotoImage icons per toolbar, and this app creates
    # one toolbar per embedded chart (and now stacks two charts -- BIC/pie
    # and the Sankey -- on the K Selection tab alone), so there are more of
    # these images alive than before. If any of them are still referenced
    # when the notebook/interpreter tears down the Tcl runtime, Image.__del__
    # tries to call into that dead interpreter and raises the same "main
    # thread is not in main loop" RuntimeError, just for images instead of
    # variables. Harmless, so swallow it the same way.
    _orig_image_del = tk.Image.__del__

    def _quiet_image_del(self):
        try:
            _orig_image_del(self)
        except RuntimeError:
            pass

    tk.Image.__del__ = _quiet_image_del
    # -----------------------------------------------------------------------

    # --- Defensive patch for a Windows/Tk/matplotlib mousewheel crash ----
    # This app destroys and recreates FigureCanvasTkAgg widgets every time
    # clustering is re-run. On some Windows Tk builds this leaves a stale
    # <MouseWheel> binding that later delivers an event whose
    # `event.widget` is a raw string path (the widget it names is already
    # destroyed) rather than a live widget. matplotlib's own built-in
    # handler, FigureCanvasTk.scroll_event_windows, assumes event.widget
    # is always a real widget and calls `.winfo_containing(...)` on it,
    # raising `AttributeError: 'str' object has no attribute
    # 'winfo_containing'`. This app never uses matplotlib's built-in
    # scroll-to-zoom, so it's safe to make that handler a no-op whenever
    # it gets a widget reference it can't use.
    try:
        from matplotlib.backends import _backend_tk as _mpl_backend_tk

        _orig_scroll_event_windows = _mpl_backend_tk.FigureCanvasTk.scroll_event_windows

        def _safe_scroll_event_windows(self, event, *a, **kw):
            if not hasattr(getattr(event, "widget", None), "winfo_containing"):
                return
            try:
                return _orig_scroll_event_windows(self, event, *a, **kw)
            except AttributeError:
                return

        _mpl_backend_tk.FigureCanvasTk.scroll_event_windows = _safe_scroll_event_windows
    except Exception:
        # If matplotlib's internals ever differ from what's patched here,
        # skip the patch rather than fail app startup over it.
        pass
    # -----------------------------------------------------------------------

    base = "Segoe UI" if sys.platform == "win32" else "Helvetica Neue" if sys.platform == "darwin" else "DejaVu Sans"
    FONT, FONT_B, FONT_SM, FONT_H = (base, 10), (base, 10, "bold"), (base, 9), (base, 13, "bold")
    MONO = ("Consolas", 10) if sys.platform == "win32" else ("Menlo", 10)

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Real Data Cluster Viewer · Multi-file GMM + Bootstrap LRT")
            self.configure(bg=BG)
            self.geometry("1480x920")
            self.minsize(1080, 700)

            self.paths = []
            self.sample_name = tk.StringVar(value="")

            # Custom cluster names: {cluster_index: name}. Cluster indices are
            # rank-by-size stable (largest = 0, ...), so names generally carry
            # over across re-runs of the same file set. self.cluster_name_vars
            # holds the live Entry StringVars for the current K, rebuilt each
            # time the K Breakdown tab is (re)built.
            self.cluster_names = {}
            self.cluster_name_vars = {}

            self.alpha = tk.DoubleVar(value=0.05)
            self.maxk = tk.IntVar(value=8)
            self.durmin = tk.DoubleVar(value=0.0)
            self.durmax = tk.StringVar(value="0")
            self.boot = tk.IntVar(value=25)
            self.min_cluster_pct = tk.DoubleVar(value=0.0)
            self.impute = tk.BooleanVar(value=True)
            self.feature_vars = {
                "duration": tk.BooleanVar(value=True),
                "OSC": tk.BooleanVar(value=True),
                "RefOSC": tk.BooleanVar(value=True),
                "entry_peak": tk.BooleanVar(value=False),
                "exit_peak": tk.BooleanVar(value=False),
                "entry_spike": tk.BooleanVar(value=True),
                "exit_spike": tk.BooleanVar(value=True),
            }
            self.x3d = tk.StringVar()
            self.y3d = tk.StringVar()
            self.z3d = tk.StringVar()

            # 3D Explorer axis-limit filter. Each bound is a StringVar; blank
            # means "no bound on that side". Applied only when the user
            # presses "Apply limits" (stored in self.lim3d as parsed floats).
            self.x3d_min = tk.StringVar(value="")
            self.x3d_max = tk.StringVar(value="")
            self.y3d_min = tk.StringVar(value="")
            self.y3d_max = tk.StringVar(value="")
            self.z3d_min = tk.StringVar(value="")
            self.z3d_max = tk.StringVar(value="")
            self.lim3d = {"x": (None, None), "y": (None, None), "z": (None, None)}

            self.R = None
            self.q = queue.Queue()

            # Interactive point inspection / deletion state.
            # Keys are (source_file, source_row), so deleted points remain excluded
            # when the clustering is recomputed.
            self.excluded_points = set()
            self.removed_cluster_history = []
            self.selected_point_index = None
            self.inspect_x = tk.StringVar()
            self.inspect_y = tk.StringVar()

            # ---------------- Folder comparison mode ----------------
            self.compare_paths_a = []
            self.compare_paths_b = []
            self.compare_folder_a = tk.StringVar(value="")
            self.compare_folder_b = tk.StringVar(value="")

            # Independent clustering settings for each folder/group.
            self.comp_a_alpha = tk.DoubleVar(value=0.05)
            self.comp_a_maxk = tk.IntVar(value=8)
            self.comp_a_durmin = tk.DoubleVar(value=0.0)
            self.comp_a_durmax = tk.StringVar(value="0")
            self.comp_a_boot = tk.IntVar(value=25)
            self.comp_a_impute = tk.BooleanVar(value=True)

            self.comp_b_alpha = tk.DoubleVar(value=0.05)
            self.comp_b_maxk = tk.IntVar(value=8)
            self.comp_b_durmin = tk.DoubleVar(value=0.0)
            self.comp_b_durmax = tk.StringVar(value="0")
            self.comp_b_boot = tk.IntVar(value=25)
            self.comp_b_impute = tk.BooleanVar(value=True)

            # Shared feature choices ensure the overlaid 3D axes are comparable.
            self.compare_feature_vars = {
                "duration": tk.BooleanVar(value=True),
                "OSC": tk.BooleanVar(value=True),
                "RefOSC": tk.BooleanVar(value=True),
                "entry_peak": tk.BooleanVar(value=False),
                "exit_peak": tk.BooleanVar(value=False),
                "entry_spike": tk.BooleanVar(value=True),
                "exit_spike": tk.BooleanVar(value=True),
            }
            self.compare_x3d = tk.StringVar()
            self.compare_y3d = tk.StringVar()
            self.compare_z3d = tk.StringVar()

            # Per-folder 3D point opacity. 1.00 = fully opaque;
            # lower values make that folder more transparent.
            self.compare_alpha_a = tk.DoubleVar(value=0.72)
            self.compare_alpha_b = tk.DoubleVar(value=0.72)

            self.compare_RA = None
            self.compare_RB = None
            self.compare_common_features = []

            self._styles()
            self._build()
            self._poll()

        def _styles(self):
            s = ttk.Style(self)
            s.theme_use("clam")
            s.configure("App.TNotebook", background=BG, borderwidth=0)
            s.configure("App.TNotebook.Tab", background=PANEL, foreground=TEXT_MED, padding=[16, 7], font=FONT)
            s.map("App.TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", ACCENT)])
            s.configure("Accent.Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT, borderwidth=0)
            s.configure("Data.Treeview", background=WHITE, fieldbackground=WHITE, foreground=TEXT, rowheight=26, font=MONO)
            s.configure("Data.Treeview.Heading", background=PANEL, foreground=TEXT_MED, font=FONT_B, relief="flat")

        def _build(self):
            sb = tk.Frame(self, bg=WHITE, width=390)
            sb.pack(side=tk.LEFT, fill=tk.Y)
            sb.pack_propagate(False)
            tk.Frame(sb, bg=ACCENT, height=3).pack(fill=tk.X)
            tk.Label(sb, text="Real Data Cluster Viewer", font=FONT_H, bg=WHITE, fg=TEXT).pack(anchor="w", padx=20, pady=(14, 0))
            tk.Label(sb, text="Multiple files · NA-safe · GMM/LRT", font=FONT_SM, bg=WHITE, fg=TEXT_DIM).pack(anchor="w", padx=20)
            self._hr(sb)

            box = tk.Frame(sb, bg=WHITE, padx=16)
            box.pack(fill=tk.X)
            self.file_list = tk.Listbox(box, height=7, font=FONT_SM, selectmode=tk.EXTENDED,
                                        bg=PANEL, fg=TEXT, relief=tk.FLAT, highlightthickness=1,
                                        highlightbackground=BORDER)
            self.file_list.pack(fill=tk.X)
            br = tk.Frame(box, bg=WHITE)
            br.pack(fill=tk.X, pady=6)
            self._btn(br, "Add files…", self._browse, ACCENT, WHITE).pack(side=tk.LEFT)
            self._btn(br, "Remove", self._remove_selected, PANEL, TEXT).pack(side=tk.LEFT, padx=5)
            self._btn(br, "Clear", self._clear_files, PANEL, TEXT).pack(side=tk.LEFT)

            tk.Label(
                box, text="Sample name (shown on HTML report)", bg=WHITE, fg=TEXT_MED,
                font=FONT_SM, anchor="w"
            ).pack(fill=tk.X, pady=(6, 2))
            tk.Entry(
                box, textvariable=self.sample_name, font=FONT, relief=tk.SOLID, bd=1,
                highlightthickness=1, highlightbackground=BORDER
            ).pack(fill=tk.X)
            self._hr(sb)

            pf = tk.Frame(sb, bg=WHITE, padx=16)
            pf.pack(fill=tk.X)
            self._param(pf, "Significance α", self.alpha, 0.001, 0.20, "{:.3f}")
            self._param(pf, "Max K", self.maxk, 2, 10, "{:.0f}")
            self._param(pf, "Min duration (s)", self.durmin, 0.0, 5.0, "{:.2f}")

            # Maximum duration is a typed box so any cutoff can be entered.
            maxrow = tk.Frame(pf, bg=WHITE)
            maxrow.pack(fill=tk.X, pady=5)
            tk.Label(maxrow, text="Max duration (s)", bg=WHITE, fg=TEXT, font=FONT).pack(side=tk.LEFT)
            self.durmax_entry = tk.Entry(
                maxrow, textvariable=self.durmax, width=9, justify="center",
                font=(MONO[0], MONO[1], "bold"), relief=tk.SOLID, bd=1,
                highlightthickness=1, highlightbackground=BORDER
            )
            self.durmax_entry.pack(side=tk.RIGHT)
            tk.Label(
                pf, text="0 = no manual maximum; e.g. 30 keeps duration ≤ 30 s",
                bg=WHITE, fg=TEXT_DIM, font=FONT_SM, anchor="w"
            ).pack(fill=tk.X, pady=(0, 3))

            self._param(pf, "Bootstraps", self.boot, 10, 60, "{:.0f}")

            self._param(pf, "Min cluster size (%)", self.min_cluster_pct, 0.0, 20.0, "{:.1f}%")
            tk.Label(
                pf,
                text=("0 = keep every cluster. Otherwise, any final cluster holding less than "
                      "this % of events is treated as noise, its events excluded, and the "
                      "clustering automatically refit without them."),
                bg=WHITE, fg=TEXT_DIM, font=FONT_SM, wraplength=345, justify=tk.LEFT, anchor="w"
            ).pack(fill=tk.X, pady=(0, 3))

            # User-selectable clustering features.
            tk.Label(
                pf, text="Clustering features", bg=WHITE, fg=TEXT,
                font=FONT_B, anchor="w"
            ).pack(fill=tk.X, pady=(10, 3))

            featbox = tk.Frame(
                pf, bg=PANEL, highlightbackground=BORDER, highlightthickness=1
            )
            featbox.pack(fill=tk.X, pady=(0, 4))

            feature_labels = [
                ("duration", "Duration (s)"),
                ("OSC", "Optical step change (%)"),
                ("RefOSC", "Ref Optical step change (%)"),
                ("entry_peak", "Entry peak |pA|"),
                ("exit_peak", "Exit peak |pA|"),
                ("entry_spike", "Entry spike = |peak − base| (pA)"),
                ("exit_spike", "Exit spike = |peak − base| (pA)"),
            ]
            for feat, label in feature_labels:
                tk.Checkbutton(
                    featbox, text=label, variable=self.feature_vars[feat],
                    bg=PANEL, fg=TEXT, activebackground=PANEL,
                    font=FONT_SM, anchor="w", justify=tk.LEFT
                ).pack(fill=tk.X, padx=7, pady=1)

            tk.Label(
                pf,
                text=("Peak features use the raw peak values. Spike Δ features are calculated "
                      "as signed peak − base from the raw pA columns."),
                bg=WHITE, fg=TEXT_DIM, font=FONT_SM,
                wraplength=345, justify=tk.LEFT, anchor="w"
            ).pack(fill=tk.X, pady=(0, 3))

            ck = tk.Checkbutton(pf, text="Median-impute partial NA values", variable=self.impute,
                                bg=WHITE, fg=TEXT, activebackground=WHITE, font=FONT, anchor="w")
            ck.pack(fill=tk.X, pady=(8, 2))
            tk.Label(pf, text="Entirely empty feature columns are always ignored.", bg=WHITE,
                     fg=TEXT_DIM, font=FONT_SM, wraplength=340, justify=tk.LEFT).pack(anchor="w")
            self._hr(sb)

            af = tk.Frame(sb, bg=WHITE, padx=16)
            af.pack(fill=tk.X)
            self.run_btn = self._btn(af, "▶  Run Clustering", self._run, SUCCESS, WHITE, big=True)
            self.save_btn = self._btn(af, "⬇  Save Results", self._save, ACCENT, WHITE, big=True)
            self.report_btn = self._btn(af, "📄  Generate HTML Report", self._generate_html_report, ACCENT, WHITE, big=True)
            self.run_btn.pack(fill=tk.X, pady=(0, 6))
            self.save_btn.pack(fill=tk.X, pady=(0, 6))
            self.save_btn.config(state=tk.DISABLED)
            self.report_btn.pack(fill=tk.X)
            self.report_btn.config(state=tk.DISABLED)
            self.prog = ttk.Progressbar(sb, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
            self.prog.pack(fill=tk.X, padx=16, pady=(10, 4))
            self.status = tk.Label(sb, text="Add one or more files to begin.", bg=WHITE, fg=TEXT_DIM,
                                   font=FONT_SM, wraplength=350, justify=tk.LEFT, anchor="w")
            self.status.pack(fill=tk.X, padx=16)

            tk.Frame(self, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
            main = tk.Frame(self, bg=BG)
            main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.nb = ttk.Notebook(main, style="App.TNotebook")
            self.nb.pack(fill=tk.BOTH, expand=True)

            self.t_console = tk.Frame(self.nb, bg=WHITE)
            self.nb.add(self.t_console, text="  Console  ")
            self.log = scrolledtext.ScrolledText(self.t_console, bg=WHITE, fg=TEXT, font=MONO,
                                                  relief=tk.FLAT, state=tk.DISABLED, wrap=tk.NONE,
                                                  padx=16, pady=12)
            self.log.pack(fill=tk.BOTH, expand=True)
            self.log.tag_config("head", foreground=ACCENT, font=(MONO[0], MONO[1], "bold"))
            self.log.tag_config("ok", foreground=SUCCESS)
            self.log.tag_config("err", foreground=ERR)
            self.log.tag_config("warn", foreground=WARN)

            self.tabs = {}
            for key, name in [
                ("k", "  K Selection  "),
                ("kbreak", "  K Breakdown  "),
                ("scatter", "  Scatter Grid  "),
                ("inspect", "  Point Inspector  "),
                ("3d", "  3D Explorer  "),
                ("hierarchy", "  Cluster Hierarchy (3D)  "),
                ("dist", "  Distributions  "),
                ("sum", "  Summary  "),
                ("file", "  Per-file  "),
                ("compare", "  Folder Comparison  "),
                ("compare_full", "  Folder Comparison · Full 3D  "),
            ]:
                f = tk.Frame(self.nb, bg=BG)
                self.nb.add(f, text=name)
                self.tabs[key] = f
                if key != "compare":
                    self._ph(f, "Run clustering to view")

            # Comparison is self-contained and can be used even before a main analysis.
            self._build_compare_tab()
            self._build_compare_full_tab()

        def _hr(self, p):
            tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=10)

        def _btn(self, p, txt, cmd, bg, fg, big=False):
            return tk.Button(p, text=txt, command=cmd, bg=bg, fg=fg, relief=tk.FLAT,
                             font=FONT_B if big else FONT, cursor="hand2", padx=12,
                             pady=9 if big else 5, bd=0)

        def _param(self, p, label, var, lo, hi, fmt):
            row = tk.Frame(p, bg=WHITE)
            row.pack(fill=tk.X, pady=5)
            top = tk.Frame(row, bg=WHITE)
            top.pack(fill=tk.X)
            tk.Label(top, text=label, bg=WHITE, fg=TEXT, font=FONT).pack(side=tk.LEFT)
            vl = tk.Label(top, text=fmt.format(var.get()), bg="#eff4ff", fg=ACCENT,
                          font=(MONO[0], MONO[1], "bold"), padx=6)
            vl.pack(side=tk.RIGHT)
            ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                      command=lambda v: vl.config(text=fmt.format(float(v)))).pack(fill=tk.X, pady=(3, 0))

        def _clabel(self, k):
            """Custom name for cluster k if one was set, else the default 'Ck'."""
            return self.cluster_names.get(int(k)) or f"C{int(k)}"

        def _ph(self, tab, txt):
            for w in tab.winfo_children():
                w.destroy()
            tk.Label(tab, text=txt, bg=BG, fg=TEXT_DIM, font=FONT_H).pack(expand=True)

        def _embed(self, fig, parent):
            c = FigureCanvasTkAgg(fig, master=parent)
            c.draw()
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tbf = tk.Frame(parent, bg=PANEL)
            tbf.pack(fill=tk.X)
            NavigationToolbar2Tk(c, tbf).update()
            return c

        def _embed_scroll(self, fig, parent):
            """
            Embed a figure inside a scrollable canvas.

            Fix: the previous version bound the mousewheel with `bind_all`,
            which registers the handler on the *entire application*, not just
            this canvas. Every time this method ran again (e.g. re-running
            clustering rebuilds the Scatter Grid tab), the old canvas was
            destroyed but its `bind_all` callback stayed registered and kept
            firing on every scroll anywhere in the app, referencing a canvas
            that no longer existed -> "invalid command name ...!canvas"
            TclErrors flooding the console.

            Fix: only bind while the mouse is actually over this canvas
            (bind on <Enter>, unbind on <Leave>/<Destroy>), and guard the
            handler itself against a destroyed widget.
            """
            wrap = tk.Frame(parent, bg=BG)
            wrap.pack(fill=tk.BOTH, expand=True)
            vs = ttk.Scrollbar(wrap, orient=tk.VERTICAL)
            vs.pack(side=tk.RIGHT, fill=tk.Y)
            cv = tk.Canvas(wrap, bg=BG, yscrollcommand=vs.set, highlightthickness=0)
            cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vs.config(command=cv.yview)
            inner = tk.Frame(cv, bg=WHITE)
            cv.create_window((0, 0), window=inner, anchor="nw")
            FigureCanvasTkAgg(fig, master=inner).get_tk_widget().pack()
            inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

            def _on_mousewheel(event):
                if not cv.winfo_exists():
                    return
                # Guard against the same kind of stale-widget event that
                # can otherwise crash matplotlib's own scroll handler.
                if not hasattr(getattr(event, "widget", None), "winfo_containing"):
                    return
                try:
                    if getattr(event, "num", None) == 4:
                        cv.yview_scroll(-1, "units")
                    elif getattr(event, "num", None) == 5:
                        cv.yview_scroll(1, "units")
                    else:
                        cv.yview_scroll(int(-event.delta / 120), "units")
                except tk.TclError:
                    pass

            def _bind_wheel(_e=None):
                cv.bind_all("<MouseWheel>", _on_mousewheel)
                cv.bind_all("<Button-4>", _on_mousewheel)
                cv.bind_all("<Button-5>", _on_mousewheel)

            def _unbind_wheel(_e=None):
                try:
                    cv.unbind_all("<MouseWheel>")
                    cv.unbind_all("<Button-4>")
                    cv.unbind_all("<Button-5>")
                except tk.TclError:
                    pass

            cv.bind("<Enter>", _bind_wheel)
            cv.bind("<Leave>", _unbind_wheel)
            cv.bind("<Destroy>", _unbind_wheel)

        def _embed_scroll_multi(self, figs, parent):
            """
            Like _embed_scroll, but stacks several figures vertically inside
            one scrollable canvas. Used for the K Selection tab so the BIC/
            proportions/LRT figure and the cluster-split Sankey chart can
            both be visible, one above the other, without either being
            squeezed to fit.
            """
            wrap = tk.Frame(parent, bg=BG)
            wrap.pack(fill=tk.BOTH, expand=True)
            vs = ttk.Scrollbar(wrap, orient=tk.VERTICAL)
            vs.pack(side=tk.RIGHT, fill=tk.Y)
            cv = tk.Canvas(wrap, bg=BG, yscrollcommand=vs.set, highlightthickness=0)
            cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vs.config(command=cv.yview)
            inner = tk.Frame(cv, bg=BG)
            cv.create_window((0, 0), window=inner, anchor="nw")

            for i, fig in enumerate(figs):
                holder = tk.Frame(inner, bg=WHITE)
                holder.pack(fill=tk.X, pady=(0 if i == 0 else 10, 0))
                canvas = FigureCanvasTkAgg(fig, master=holder)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                tbf = tk.Frame(holder, bg=PANEL)
                tbf.pack(fill=tk.X)
                NavigationToolbar2Tk(canvas, tbf).update()

            inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

            def _on_mousewheel(event):
                if not cv.winfo_exists():
                    return
                if not hasattr(getattr(event, "widget", None), "winfo_containing"):
                    return
                try:
                    if getattr(event, "num", None) == 4:
                        cv.yview_scroll(-1, "units")
                    elif getattr(event, "num", None) == 5:
                        cv.yview_scroll(1, "units")
                    else:
                        cv.yview_scroll(int(-event.delta / 120), "units")
                except tk.TclError:
                    pass

            def _bind_wheel(_e=None):
                cv.bind_all("<MouseWheel>", _on_mousewheel)
                cv.bind_all("<Button-4>", _on_mousewheel)
                cv.bind_all("<Button-5>", _on_mousewheel)

            def _unbind_wheel(_e=None):
                try:
                    cv.unbind_all("<MouseWheel>")
                    cv.unbind_all("<Button-4>")
                    cv.unbind_all("<Button-5>")
                except tk.TclError:
                    pass

            cv.bind("<Enter>", _bind_wheel)
            cv.bind("<Leave>", _unbind_wheel)
            cv.bind("<Destroy>", _unbind_wheel)

        def _browse(self):
            p = filedialog.askopenfilenames(
                title="Select one or more data files",
                filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("CSV", "*.csv"),
                           ("Excel", "*.xlsx *.xls"), ("All files", "*.*")],
            )
            if not p:
                return
            existing = set(self.paths)
            for path in p:
                if path not in existing:
                    self.paths.append(path)
                    existing.add(path)
            self.excluded_points.clear()
            self.removed_cluster_history.clear()
            self.selected_point_index = None
            self.cluster_names.clear()
            self.cluster_name_vars.clear()
            self._refresh_file_list()
            self.status.config(text=f"{len(self.paths)} file(s) selected — press Run.", fg=SUCCESS)

        def _refresh_file_list(self):
            self.file_list.delete(0, tk.END)
            for path in self.paths:
                self.file_list.insert(tk.END, os.path.basename(path))

        def _remove_selected(self):
            inds = list(self.file_list.curselection())
            for i in reversed(inds):
                del self.paths[i]
            self.excluded_points.clear()
            self.removed_cluster_history.clear()
            self.selected_point_index = None
            self.cluster_names.clear()
            self.cluster_name_vars.clear()
            self._refresh_file_list()
            self.status.config(text=f"{len(self.paths)} file(s) selected.", fg=TEXT_DIM)

        def _clear_files(self):
            self.paths.clear()
            self.excluded_points.clear()
            self.removed_cluster_history.clear()
            self.selected_point_index = None
            self.cluster_names.clear()
            self.cluster_name_vars.clear()
            self._refresh_file_list()
            self.status.config(text="Add one or more files to begin.", fg=TEXT_DIM)

        def _run(self):
            if not self.paths:
                messagebox.showwarning("No files", "Select one or more CSV/XLSX files first.")
                return

            self.run_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.prog.start(10)
            self._logclear()
            for key, f in self.tabs.items():
                if key != "compare":
                    self._ph(f, "Running…")

            paths = list(self.paths)
            a = round(self.alpha.get(), 4)
            mk = int(self.maxk.get())
            dm = round(self.durmin.get(), 3)
            try:
                dx = float(self.durmax.get().strip() or "0")
            except ValueError:
                messagebox.showerror("Invalid maximum duration",
                                     "Max duration must be a number. Use 0 for no maximum.")
                self.run_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                self.prog.stop()
                return
            if dx < 0:
                messagebox.showerror("Invalid maximum duration",
                                     "Max duration cannot be negative. Use 0 for no maximum.")
                self.run_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                self.prog.stop()
                return
            if dx > 0 and dx <= dm:
                messagebox.showerror("Invalid duration range",
                                     "Max duration must be greater than Min duration.")
                self.run_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                self.prog.stop()
                return

            nb = int(self.boot.get())
            imp = bool(self.impute.get())
            mcp = round(self.min_cluster_pct.get(), 2)
            selected_features = [
                f for f in CANONICAL_FEATURES if self.feature_vars[f].get()
            ]
            if not selected_features:
                messagebox.showwarning(
                    "No clustering features",
                    "Select at least one clustering feature."
                )
                self.run_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                self.prog.stop()
                return

            def work():
                try:
                    self.q.put(("=" * 70 + "\n", "head"))
                    self.q.put((f"[1] Selected {len(paths)} file(s)\n", None))
                    for p in paths:
                        self.q.put((f"    {os.path.basename(p)}\n", None))
                    self.q.put((f"    Features selected: {', '.join(selected_features)}\n", None))
                    self.q.put(("\n[2] Reading real data, deriving spikes, and resolving NA values…\n", None))

                    self.R = run_clustering(
                        paths=paths,
                        alpha=a,
                        max_k=mk,
                        duration_min=dm,
                        duration_max=dx,
                        n_boot=nb,
                        impute_missing=imp,
                        selected_features=selected_features,
                        excluded_points=set(self.excluded_points),
                        min_cluster_pct=mcp,
                        log=lambda m: self.q.put((m + "\n", None)),
                    )
                    self.q.put((f"\nDone — final K = {self.R['K']}\n", "ok"))
                    self.after(0, self._ok)
                except Exception:
                    self.q.put(("\nERROR\n" + traceback.format_exc(), "err"))
                    self.after(0, self._err)

            threading.Thread(target=work, daemon=True).start()

        def _ok(self):
            self.prog.stop()
            self.run_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.report_btn.config(state=tk.NORMAL)
            feats = ", ".join(self.R["features"])
            excl_txt = f" · {len(self.excluded_points)} point(s) excluded" if self.excluded_points else ""
            self.status.config(
                text=f"Done — {self.R['K']} clusters · features: {feats}{excl_txt}",
                fg=SUCCESS
            )

            for key, f in self.tabs.items():
                if key == "compare":
                    continue
                for w in f.winfo_children():
                    w.destroy()

            self._embed_scroll_multi(
                [fig_kselection(self.R, names=self.cluster_names), fig_cluster_sankey(self.R)],
                self.tabs["k"],
            )
            self._build_k_breakdown_table(self.tabs["kbreak"], self.R["k_breakdown"], self.R["K"], self.R["k_bic"])
            self._embed_scroll(fig_scatter_grid(self.R, names=self.cluster_names), self.tabs["scatter"])
            self._build_point_inspector()
            # Axis limits from a previous run don't necessarily apply to a
            # newly recomputed model (different events, different ranges).
            self.lim3d = {"x": (None, None), "y": (None, None), "z": (None, None)}
            for v in (self.x3d_min, self.x3d_max, self.y3d_min, self.y3d_max, self.z3d_min, self.z3d_max):
                v.set("")
            self._build_3d()
            self._build_hierarchy_tab()
            self._embed(fig_distributions(self.R, names=self.cluster_names), self.tabs["dist"])
            self._build_table(self.tabs["sum"], self.R["summary"], cluster_col="Cluster")
            self._build_table(self.tabs["file"], self.R["per_file"], cluster_col="cluster")
            self.nb.select(1)

        def _err(self):
            self.prog.stop()
            self.run_btn.config(state=tk.NORMAL)
            self.status.config(text="Error — see Console", fg=ERR)
            self.nb.select(0)

        # ======================= K BREAKDOWN TABLE =======================

        def _build_k_breakdown_table(self, tab, df, final_k, k_bic):
            """
            Table of cluster sizes/percentages at every K tried in the sweep.
            Rows for the algorithm's recommended final K are highlighted green;
            rows for the BIC-best K (when different from the final K) are
            highlighted amber so both can be told apart at a glance.

            Also hosts the cluster-naming panel: custom names entered here are
            applied to the final-K rows in this table (and everywhere else in
            the app / HTML report that labels clusters) once "Apply Names" is
            pressed.
            """
            tab.config(bg=WHITE)
            for w in tab.winfo_children():
                w.destroy()

            name_panel = tk.Frame(tab, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
            name_panel.pack(fill=tk.X, padx=12, pady=(12, 4))
            tk.Label(
                name_panel,
                text="Name your clusters (used in tables & the 3D explorer legend, incl. the HTML report):",
                bg=PANEL, fg=TEXT, font=FONT_B, anchor="w"
            ).pack(fill=tk.X, padx=10, pady=(8, 4))

            entries_row = tk.Frame(name_panel, bg=PANEL)
            entries_row.pack(fill=tk.X, padx=10, pady=(0, 4))

            self.cluster_name_vars = {}
            for k in range(final_k):
                cell = tk.Frame(entries_row, bg=PANEL)
                cell.pack(side=tk.LEFT, padx=(0, 10), pady=2)
                swatch = tk.Frame(cell, bg=PALETTE[k % len(PALETTE)], width=12, height=12)
                swatch.pack(side=tk.LEFT, padx=(0, 4))
                swatch.pack_propagate(False)
                tk.Label(cell, text=f"C{k}:", bg=PANEL, fg=TEXT_MED, font=FONT_SM).pack(side=tk.LEFT)
                var = tk.StringVar(value=self.cluster_names.get(k, ""))
                self.cluster_name_vars[k] = var
                tk.Entry(cell, textvariable=var, width=12, font=FONT_SM, relief=tk.SOLID, bd=1).pack(
                    side=tk.LEFT, padx=(4, 0)
                )

            self._btn(name_panel, "Apply Names", self._apply_cluster_names, ACCENT, WHITE).pack(
                anchor="w", padx=10, pady=(0, 8)
            )

            note = tk.Label(
                tab,
                text=(f"Green rows = recommended final K = {final_k} (sequential bootstrap LRT).  "
                      f"Amber rows = lowest-BIC K = {k_bic}."
                      + ("" if k_bic == final_k else "  These differ for this run.")),
                bg=WHITE, fg=TEXT_MED, font=FONT_SM, anchor="w",
                wraplength=1100, justify=tk.LEFT
            )
            note.pack(fill=tk.X, padx=14, pady=(8, 6))

            wrap = tk.Frame(tab, bg=WHITE)
            wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

            cols = ["K", "Cluster", "N", "Pct"]
            tree = ttk.Treeview(wrap, columns=cols, show="headings", style="Data.Treeview")
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=110, anchor="center")

            tree.tag_configure("normal_even", background=WHITE)
            tree.tag_configure("normal_odd", background="#f8faff")
            tree.tag_configure("final", background="#dcfce7")  # light green
            tree.tag_configure("bic", background="#fef3c7")    # light amber

            for i, row in df.iterrows():
                k = int(row["K"])
                if k == final_k:
                    tag = "final"
                    idx = int(str(row["Cluster"])[1:])
                    label = self._clabel(idx)
                elif k == k_bic:
                    tag = "bic"
                    label = row["Cluster"]
                else:
                    tag = "normal_even" if i % 2 == 0 else "normal_odd"
                    label = row["Cluster"]
                tree.insert("", tk.END, values=[row["K"], label, row["N"], f"{row['Pct']:.2f}"], tags=(tag,))

            vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(fill=tk.BOTH, expand=True)

        def _apply_cluster_names(self):
            """
            Read the current cluster-name entries, store non-blank ones in
            self.cluster_names, and redraw every view that labels clusters
            (Scatter Grid, 3D Explorer, Distributions, K Selection, K
            Breakdown, Summary, Per-file, Point Inspector) so the new names
            take effect immediately without re-running the clustering.
            """
            if not self.R:
                return

            K = self.R["K"]
            self.cluster_names = {}
            for k in range(K):
                var = self.cluster_name_vars.get(k)
                if var:
                    val = var.get().strip()
                    if val:
                        self.cluster_names[k] = val

            for w in self.tabs["k"].winfo_children():
                w.destroy()
            self._embed_scroll_multi(
                [fig_kselection(self.R, names=self.cluster_names), fig_cluster_sankey(self.R)],
                self.tabs["k"],
            )

            self._build_k_breakdown_table(self.tabs["kbreak"], self.R["k_breakdown"], self.R["K"], self.R["k_bic"])

            for w in self.tabs["scatter"].winfo_children():
                w.destroy()
            self._embed_scroll(fig_scatter_grid(self.R, names=self.cluster_names), self.tabs["scatter"])

            self._refresh_3d()

            for w in self.tabs["dist"].winfo_children():
                w.destroy()
            self._embed(fig_distributions(self.R, names=self.cluster_names), self.tabs["dist"])

            self._build_table(self.tabs["sum"], self.R["summary"], cluster_col="Cluster")
            self._build_table(self.tabs["file"], self.R["per_file"], cluster_col="cluster")

            if hasattr(self, "cluster_remove_list"):
                self.cluster_remove_list.delete(0, tk.END)
                for k in range(self.R["K"]):
                    n_k = int((self.R["labels"] == k).sum())
                    self.cluster_remove_list.insert(tk.END, f"{self._clabel(k)}   n={n_k}")
            self._refresh_point_inspector()

            self.status.config(text="Cluster names updated.", fg=SUCCESS)

        # ======================= FOLDER COMPARISON =======================

        def _build_compare_tab(self):
            """
            Final-tab workflow for comparing two folders of real event files.

            Each folder is clustered independently with its own alpha, Max K,
            duration limits, bootstrap count, and NA policy. The selected
            clustering features are shared so the final 3D overlay is directly
            comparable.
            """
            tab = self.tabs["compare"]
            for w in tab.winfo_children():
                w.destroy()
            tab.config(bg=BG)

            # Scrollable controls on top because this tab has two parameter panels.
            outer = tk.Frame(tab, bg=BG)
            outer.pack(fill=tk.BOTH, expand=True)

            controls = tk.Frame(outer, bg=WHITE)
            controls.pack(fill=tk.X, padx=10, pady=(8, 5))

            tk.Label(
                controls,
                text="Compare two folders",
                bg=WHITE, fg=TEXT, font=FONT_H, anchor="w"
            ).pack(fill=tk.X, padx=12, pady=(10, 1))
            tk.Label(
                controls,
                text=("Each folder may contain multiple CSV/XLSX files. "
                      "A and B are clustered independently; the final 3D plot overlays them. "
                      "Use the A/B opacity sliders to fade either folder."),
                bg=WHITE, fg=TEXT_DIM, font=FONT_SM, anchor="w",
                justify=tk.LEFT, wraplength=1050
            ).pack(fill=tk.X, padx=12, pady=(0, 8))

            groups = tk.Frame(controls, bg=WHITE)
            groups.pack(fill=tk.X, padx=8)

            self.comp_panel_a = self._comparison_group_panel(
                groups, "A", "Folder A · circles",
                self.compare_folder_a,
                self.comp_a_alpha, self.comp_a_maxk,
                self.comp_a_durmin, self.comp_a_durmax,
                self.comp_a_boot, self.comp_a_impute
            )
            self.comp_panel_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

            self.comp_panel_b = self._comparison_group_panel(
                groups, "B", "Folder B · triangles",
                self.compare_folder_b,
                self.comp_b_alpha, self.comp_b_maxk,
                self.comp_b_durmin, self.comp_b_durmax,
                self.comp_b_boot, self.comp_b_impute
            )
            self.comp_panel_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

            # Shared comparison features.
            common = tk.Frame(
                controls, bg=PANEL,
                highlightbackground=BORDER, highlightthickness=1
            )
            common.pack(fill=tk.X, padx=8, pady=(8, 5))

            tk.Label(
                common, text="Shared comparison features",
                bg=PANEL, fg=TEXT, font=FONT_B
            ).pack(side=tk.LEFT, padx=(10, 8), pady=7)

            labels = [
                ("duration", "Duration (s)"),
                ("OSC", "Optical step change (%)"),
                ("RefOSC", "Ref Optical step change (%)"),
                ("entry_peak", "Entry peak |pA|"),
                ("exit_peak", "Exit peak |pA|"),
                ("entry_spike", "Entry spike = |peak − base| (pA)"),
                ("exit_spike", "Exit spike = |peak − base| (pA)"),
            ]
            for feat, label in labels:
                tk.Checkbutton(
                    common, text=label,
                    variable=self.compare_feature_vars[feat],
                    bg=PANEL, fg=TEXT, activebackground=PANEL,
                    font=FONT_SM
                ).pack(side=tk.LEFT, padx=4, pady=5)

            actions = tk.Frame(controls, bg=WHITE)
            actions.pack(fill=tk.X, padx=8, pady=(3, 10))

            self.compare_run_btn = self._btn(
                actions, "▶  Cluster A + B and Compare",
                self._run_comparison, SUCCESS, WHITE, big=True
            )
            self.compare_run_btn.pack(side=tk.LEFT)

            self.compare_save_btn = self._btn(
                actions, "⬇  Save Comparison",
                self._save_comparison, ACCENT, WHITE, big=True
            )
            self.compare_save_btn.pack(side=tk.LEFT, padx=7)
            self.compare_save_btn.config(state=tk.DISABLED)

            self.compare_prog = ttk.Progressbar(
                actions, mode="indeterminate",
                style="Accent.Horizontal.TProgressbar", length=190
            )
            self.compare_prog.pack(side=tk.LEFT, padx=10)

            self.compare_status = tk.Label(
                actions,
                text="Choose Folder A and Folder B.",
                bg=WHITE, fg=TEXT_DIM, font=FONT_SM,
                justify=tk.LEFT, anchor="w"
            )
            self.compare_status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            # 3D axes controls. Disabled conceptually until comparison is run.
            axisbar = tk.Frame(outer, bg=PANEL, pady=7)
            axisbar.pack(fill=tk.X, padx=10, pady=(0, 0))

            tk.Label(
                axisbar, text="Combined 3D:",
                bg=PANEL, fg=TEXT, font=FONT_B
            ).pack(side=tk.LEFT, padx=(10, 7))

            self.compare_axis_boxes = []
            for lab, var in [
                ("X", self.compare_x3d),
                ("Y", self.compare_y3d),
                ("Z", self.compare_z3d),
            ]:
                tk.Label(
                    axisbar, text=f"{lab}:",
                    bg=PANEL, fg=TEXT_MED, font=FONT_B
                ).pack(side=tk.LEFT, padx=(6, 3))
                cb = ttk.Combobox(
                    axisbar, textvariable=var,
                    values=[], state="readonly", font=FONT, width=13
                )
                cb.pack(side=tk.LEFT)
                self.compare_axis_boxes.append(cb)

            self.compare_update_btn = self._btn(
                axisbar, "Update 3D",
                self._refresh_comparison_3d, ACCENT, WHITE
            )
            self.compare_update_btn.pack(side=tk.LEFT, padx=(10, 14))
            self.compare_update_btn.config(state=tk.DISABLED)

            # Folder A opacity slider.
            tk.Label(
                axisbar, text="A opacity",
                bg=PANEL, fg=TEXT_MED, font=FONT_SM
            ).pack(side=tk.LEFT, padx=(2, 3))
            self.compare_alpha_a_lbl = tk.Label(
                axisbar, text=f"{self.compare_alpha_a.get():.2f}",
                bg="#eff4ff", fg=ACCENT, font=FONT_SM, padx=4
            )
            self.compare_alpha_a_lbl.pack(side=tk.LEFT)
            self.compare_alpha_a_scale = ttk.Scale(
                axisbar, from_=0.03, to=1.0,
                variable=self.compare_alpha_a,
                orient=tk.HORIZONTAL, length=105,
                command=lambda v: self._on_compare_alpha_change(
                    "A", float(v)
                )
            )
            self.compare_alpha_a_scale.pack(side=tk.LEFT, padx=(3, 9))

            # Folder B opacity slider.
            tk.Label(
                axisbar, text="B opacity",
                bg=PANEL, fg=TEXT_MED, font=FONT_SM
            ).pack(side=tk.LEFT, padx=(2, 3))
            self.compare_alpha_b_lbl = tk.Label(
                axisbar, text=f"{self.compare_alpha_b.get():.2f}",
                bg="#eff4ff", fg=ACCENT, font=FONT_SM, padx=4
            )
            self.compare_alpha_b_lbl.pack(side=tk.LEFT)
            self.compare_alpha_b_scale = ttk.Scale(
                axisbar, from_=0.03, to=1.0,
                variable=self.compare_alpha_b,
                orient=tk.HORIZONTAL, length=105,
                command=lambda v: self._on_compare_alpha_change(
                    "B", float(v)
                )
            )
            self.compare_alpha_b_scale.pack(side=tk.LEFT, padx=(3, 8))

            self.compare_plot_frame = tk.Frame(outer, bg=WHITE)
            self.compare_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

            self._comparison_placeholder(
                "Select two folders, choose settings, then click “Cluster A + B and Compare”."
            )


        def _build_compare_full_tab(self):
            """Dedicated last tab with only the comparison 3D plot in large format."""
            tab = self.tabs["compare_full"]
            for w in tab.winfo_children():
                w.destroy()
            tab.config(bg=WHITE)

            self.compare_full_plot_frame = tk.Frame(tab, bg=WHITE)
            self.compare_full_plot_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            self._comparison_full_placeholder(
                "Run Folder Comparison to view the large 3D overlay here."
            )

        def _comparison_full_placeholder(self, text):
            if not hasattr(self, "compare_full_plot_frame"):
                return
            for w in self.compare_full_plot_frame.winfo_children():
                w.destroy()
            tk.Label(
                self.compare_full_plot_frame,
                text=text,
                bg=WHITE,
                fg=TEXT_DIM,
                font=FONT_H,
                justify=tk.CENTER,
            ).pack(expand=True)

        def _draw_comparison_3d_into(self, parent):
            """Render the current comparison 3D figure into the given parent frame."""
            if self.compare_RA is None or self.compare_RB is None:
                return

            x = self.compare_x3d.get()
            y = self.compare_y3d.get()
            z = self.compare_z3d.get()
            common = self.compare_common_features
            if not all(v in common for v in (x, y, z)):
                return

            for w in parent.winfo_children():
                w.destroy()

            name_a = os.path.basename(
                os.path.normpath(self.compare_folder_a.get())
            ) or "Folder A"
            name_b = os.path.basename(
                os.path.normpath(self.compare_folder_b.get())
            ) or "Folder B"

            fig = fig_compare_3d(
                self.compare_RA, self.compare_RB,
                x, y, z, name_a, name_b,
                alpha_a=self.compare_alpha_a.get(),
                alpha_b=self.compare_alpha_b.get()
            )
            self._embed(fig, parent)

        def _comparison_group_panel(
            self, parent, group, title, folder_var,
            alpha_var, maxk_var, durmin_var, durmax_var, boot_var, impute_var
        ):
            panel = tk.Frame(
                parent, bg=PANEL,
                highlightbackground=BORDER, highlightthickness=1
            )

            head = tk.Frame(panel, bg=PANEL)
            head.pack(fill=tk.X, padx=9, pady=(8, 4))
            tk.Label(
                head, text=title, bg=PANEL, fg=TEXT,
                font=FONT_B
            ).pack(side=tk.LEFT)
            self._btn(
                head, "Choose folder…",
                lambda g=group: self._choose_compare_folder(g),
                ACCENT, WHITE
            ).pack(side=tk.RIGHT)

            path_lbl = tk.Label(
                panel, textvariable=folder_var,
                bg=PANEL, fg=TEXT_DIM, font=FONT_SM,
                anchor="w", justify=tk.LEFT, wraplength=470
            )
            path_lbl.pack(fill=tk.X, padx=9)

            listbox = tk.Listbox(
                panel, height=4, bg=WHITE, fg=TEXT,
                font=FONT_SM, relief=tk.FLAT,
                highlightthickness=1, highlightbackground=BORDER
            )
            listbox.pack(fill=tk.X, padx=9, pady=(4, 6))
            if group == "A":
                self.compare_file_list_a = listbox
            else:
                self.compare_file_list_b = listbox

            grid = tk.Frame(panel, bg=PANEL)
            grid.pack(fill=tk.X, padx=9, pady=(0, 6))

            fields = [
                ("Significance α", alpha_var),
                ("Max K", maxk_var),
                ("Min duration (s)", durmin_var),
                ("Max duration (s)", durmax_var),
                ("Bootstraps", boot_var),
            ]
            for r, (label, var) in enumerate(fields):
                tk.Label(
                    grid, text=label, bg=PANEL, fg=TEXT_MED,
                    font=FONT_SM, anchor="w"
                ).grid(row=r // 3, column=(r % 3) * 2, sticky="w", padx=(0, 4), pady=3)
                ent = tk.Entry(
                    grid, textvariable=var, width=8, justify="center",
                    font=FONT_SM, relief=tk.SOLID, bd=1
                )
                ent.grid(row=r // 3, column=(r % 3) * 2 + 1, sticky="w", padx=(0, 12), pady=3)

            tk.Checkbutton(
                panel, text="Median-impute partial NA",
                variable=impute_var, bg=PANEL, fg=TEXT,
                activebackground=PANEL, font=FONT_SM
            ).pack(anchor="w", padx=9, pady=(0, 7))

            return panel

        def _choose_compare_folder(self, group):
            folder = filedialog.askdirectory(
                title=f"Choose Folder {group} containing CSV/XLSX files"
            )
            if not folder:
                return

            # Direct files in the chosen folder. This avoids accidentally reading
            # old output subfolders or unrelated nested directories.
            supported = {".csv", ".xlsx", ".xls"}
            paths = sorted(
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, name))
                and os.path.splitext(name)[1].lower() in supported
                and not name.startswith("~$")
            )

            if not paths:
                messagebox.showwarning(
                    "No data files",
                    f"No CSV/XLSX files were found directly inside:\n{folder}"
                )
                return

            if group == "A":
                self.compare_paths_a = paths
                self.compare_folder_a.set(folder)
                lb = self.compare_file_list_a
            else:
                self.compare_paths_b = paths
                self.compare_folder_b.set(folder)
                lb = self.compare_file_list_b

            lb.delete(0, tk.END)
            for p in paths:
                lb.insert(tk.END, os.path.basename(p))

            self.compare_RA = None
            self.compare_RB = None
            self.compare_save_btn.config(state=tk.DISABLED)
            self.compare_update_btn.config(state=tk.DISABLED)
            self.compare_status.config(
                text=(f"Folder A: {len(self.compare_paths_a)} file(s) · "
                      f"Folder B: {len(self.compare_paths_b)} file(s)"),
                fg=TEXT_MED
            )
            self._comparison_placeholder(
                "Folder selection changed. Re-run Folder Comparison to update both 3D views."
            )

        def _parse_compare_settings(
            self, group, alpha_var, maxk_var, durmin_var, durmax_var, boot_var
        ):
            try:
                alpha = float(alpha_var.get())
                maxk = int(maxk_var.get())
                durmin = float(durmin_var.get())
                durmax = float(str(durmax_var.get()).strip() or "0")
                boot = int(boot_var.get())
            except Exception:
                raise ValueError(f"Folder {group}: one or more clustering settings are not valid numbers.")

            if not (0 < alpha < 1):
                raise ValueError(f"Folder {group}: significance α must be between 0 and 1.")
            if maxk < 2:
                raise ValueError(f"Folder {group}: Max K must be at least 2.")
            if durmin < 0 or durmax < 0:
                raise ValueError(f"Folder {group}: duration limits cannot be negative.")
            if durmax > 0 and durmax <= durmin:
                raise ValueError(f"Folder {group}: Max duration must be greater than Min duration.")
            if boot < 1:
                raise ValueError(f"Folder {group}: Bootstraps must be at least 1.")

            return {
                "alpha": alpha,
                "max_k": maxk,
                "duration_min": durmin,
                "duration_max": durmax,
                "n_boot": boot,
            }

        def _run_comparison(self):
            if not self.compare_paths_a or not self.compare_paths_b:
                messagebox.showwarning(
                    "Two folders required",
                    "Choose both Folder A and Folder B first."
                )
                return

            selected_features = [
                f for f in CANONICAL_FEATURES
                if self.compare_feature_vars[f].get()
            ]
            if len(selected_features) < 3:
                messagebox.showwarning(
                    "Need 3 comparison features",
                    "Select at least three shared features for the combined 3D comparison."
                )
                return

            try:
                sa = self._parse_compare_settings(
                    "A", self.comp_a_alpha, self.comp_a_maxk,
                    self.comp_a_durmin, self.comp_a_durmax, self.comp_a_boot
                )
                sb = self._parse_compare_settings(
                    "B", self.comp_b_alpha, self.comp_b_maxk,
                    self.comp_b_durmin, self.comp_b_durmax, self.comp_b_boot
                )
            except ValueError as e:
                messagebox.showerror("Invalid comparison settings", str(e))
                return

            paths_a = list(self.compare_paths_a)
            paths_b = list(self.compare_paths_b)
            impute_a = bool(self.comp_a_impute.get())
            impute_b = bool(self.comp_b_impute.get())

            self.compare_run_btn.config(state=tk.DISABLED)
            self.compare_save_btn.config(state=tk.DISABLED)
            self.compare_update_btn.config(state=tk.DISABLED)
            self.compare_prog.start(10)
            self.compare_status.config(
                text="Clustering Folder A and Folder B independently…",
                fg=ACCENT
            )
            self._comparison_placeholder("Running comparison clustering…")

            def clog(prefix):
                return lambda m: self.q.put((f"[Compare {prefix}] {m}\n", None))

            def work():
                try:
                    RA = run_clustering(
                        paths=paths_a,
                        alpha=sa["alpha"],
                        max_k=sa["max_k"],
                        duration_min=sa["duration_min"],
                        duration_max=sa["duration_max"],
                        n_boot=sa["n_boot"],
                        impute_missing=impute_a,
                        selected_features=selected_features,
                        excluded_points=None,
                        log=clog("A"),
                    )

                    RB = run_clustering(
                        paths=paths_b,
                        alpha=sb["alpha"],
                        max_k=sb["max_k"],
                        duration_min=sb["duration_min"],
                        duration_max=sb["duration_max"],
                        n_boot=sb["n_boot"],
                        impute_missing=impute_b,
                        selected_features=selected_features,
                        excluded_points=None,
                        log=clog("B"),
                    )

                    common = [
                        f for f in selected_features
                        if f in RA["features"] and f in RB["features"]
                    ]
                    if len(common) < 3:
                        raise ValueError(
                            "Fewer than three selected features contain usable data in BOTH folders.\n"
                            f"Folder A usable: {', '.join(RA['features'])}\n"
                            f"Folder B usable: {', '.join(RB['features'])}"
                        )

                    self.compare_RA = RA
                    self.compare_RB = RB
                    self.compare_common_features = common
                    self.after(0, self._comparison_ok)

                except Exception:
                    self.q.put((
                        "\nCOMPARISON ERROR\n" + traceback.format_exc(),
                        "err"
                    ))
                    self.after(0, self._comparison_err)

            threading.Thread(target=work, daemon=True).start()

        def _comparison_ok(self):
            self.compare_prog.stop()
            self.compare_run_btn.config(state=tk.NORMAL)
            self.compare_save_btn.config(state=tk.NORMAL)
            self.compare_update_btn.config(state=tk.NORMAL)

            common = self.compare_common_features
            defaults = common[:3]
            self.compare_x3d.set(defaults[0])
            self.compare_y3d.set(defaults[1])
            self.compare_z3d.set(defaults[2])

            for cb in self.compare_axis_boxes:
                cb["values"] = common

            na = len(self.compare_RA["X_raw"])
            nb = len(self.compare_RB["X_raw"])
            ka = self.compare_RA["K"]
            kb = self.compare_RB["K"]

            self.compare_status.config(
                text=(f"Done · A: n={na}, K={ka}, circles   |   "
                      f"B: n={nb}, K={kb}, triangles"),
                fg=SUCCESS
            )
            self._refresh_comparison_3d()

        def _comparison_err(self):
            self.compare_prog.stop()
            self.compare_run_btn.config(state=tk.NORMAL)
            self.compare_status.config(
                text="Comparison error — see the main Console tab for details.",
                fg=ERR
            )
            self._comparison_placeholder(
                "Comparison failed. See Console for the full error."
            )

        def _comparison_placeholder(self, text):
            if hasattr(self, "compare_plot_frame"):
                for w in self.compare_plot_frame.winfo_children():
                    w.destroy()
                tk.Label(
                    self.compare_plot_frame, text=text,
                    bg=WHITE, fg=TEXT_DIM, font=FONT_H,
                    justify=tk.CENTER
                ).pack(expand=True)

            self._comparison_full_placeholder(text)


        def _on_compare_alpha_change(self, group, value):
            """Update the numeric opacity label and redraw the comparison plot."""
            value = float(np.clip(value, 0.03, 1.0))
            if group == "A":
                self.compare_alpha_a_lbl.config(text=f"{value:.2f}")
            else:
                self.compare_alpha_b_lbl.config(text=f"{value:.2f}")

            # Once comparison results exist, redraw immediately so the slider
            # behaves interactively without requiring the Update 3D button.
            if self.compare_RA is not None and self.compare_RB is not None:
                self._refresh_comparison_3d()

        def _refresh_comparison_3d(self):
            if self.compare_RA is None or self.compare_RB is None:
                return

            x = self.compare_x3d.get()
            y = self.compare_y3d.get()
            z = self.compare_z3d.get()
            common = self.compare_common_features
            if not all(v in common for v in (x, y, z)):
                return

            self._draw_comparison_3d_into(self.compare_plot_frame)
            self._draw_comparison_3d_into(self.compare_full_plot_frame)

        def _save_comparison(self):
            if self.compare_RA is None or self.compare_RB is None:
                return

            d = filedialog.askdirectory(title="Save folder comparison results")
            if not d:
                return

            RA, RB = self.compare_RA, self.compare_RB

            RA["labeled"].to_csv(
                os.path.join(d, "folder_A_cluster_labels.csv"), index=False
            )
            RB["labeled"].to_csv(
                os.path.join(d, "folder_B_cluster_labels.csv"), index=False
            )
            RA["summary"].to_csv(
                os.path.join(d, "folder_A_cluster_summary.csv"), index=False
            )
            RB["summary"].to_csv(
                os.path.join(d, "folder_B_cluster_summary.csv"), index=False
            )
            RA["lrt"].to_csv(
                os.path.join(d, "folder_A_bootstrap_lrt.csv"), index=False
            )
            RB["lrt"].to_csv(
                os.path.join(d, "folder_B_bootstrap_lrt.csv"), index=False
            )

            # Save one workbook for easy side-by-side review.
            try:
                with pd.ExcelWriter(
                    os.path.join(d, "folder_comparison_results.xlsx"),
                    engine="openpyxl"
                ) as xw:
                    RA["labeled"].to_excel(xw, sheet_name="A_labels", index=False)
                    RB["labeled"].to_excel(xw, sheet_name="B_labels", index=False)
                    RA["summary"].to_excel(xw, sheet_name="A_summary", index=False)
                    RB["summary"].to_excel(xw, sheet_name="B_summary", index=False)
                    RA["lrt"].to_excel(xw, sheet_name="A_LRT", index=False)
                    RB["lrt"].to_excel(xw, sheet_name="B_LRT", index=False)
            except Exception:
                pass

            messagebox.showinfo(
                "Comparison saved",
                f"Saved Folder A/B comparison results to:\n{d}"
            )

        def _build_point_inspector(self):
            """Interactive 2D plot for identifying and manually excluding events."""
            tab = self.tabs["inspect"]
            for w in tab.winfo_children():
                w.destroy()
            tab.config(bg=BG)

            features = self.R["features"]
            if not features:
                self._ph(tab, "No usable clustering features.")
                return

            # Top controls
            ctrl = tk.Frame(tab, bg=PANEL, pady=8)
            ctrl.pack(fill=tk.X, padx=12, pady=(8, 0))

            default_x = features[0]
            default_y = features[1] if len(features) > 1 else features[0]
            self.inspect_x.set(default_x)
            self.inspect_y.set(default_y)

            tk.Label(ctrl, text="X:", bg=PANEL, fg=TEXT_MED, font=FONT_B).pack(side=tk.LEFT, padx=(10, 4))
            xcb = ttk.Combobox(ctrl, textvariable=self.inspect_x, values=features,
                               state="readonly", font=FONT, width=14)
            xcb.pack(side=tk.LEFT)

            tk.Label(ctrl, text="Y:", bg=PANEL, fg=TEXT_MED, font=FONT_B).pack(side=tk.LEFT, padx=(12, 4))
            ycb = ttk.Combobox(ctrl, textvariable=self.inspect_y, values=features,
                               state="readonly", font=FONT, width=14)
            ycb.pack(side=tk.LEFT)

            self._btn(ctrl, "Update plot", self._refresh_point_inspector, ACCENT, WHITE).pack(
                side=tk.LEFT, padx=10
            )

            self.deleted_lbl = tk.Label(
                ctrl, text=f"Excluded: {len(self.excluded_points)}",
                bg=PANEL, fg=WARN if self.excluded_points else TEXT_DIM, font=FONT_B
            )
            self.deleted_lbl.pack(side=tk.RIGHT, padx=10)

            body = tk.Frame(tab, bg=BG)
            body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))

            # Plot occupies the main area.
            self.inspect_plot_frame = tk.Frame(body, bg=WHITE)
            self.inspect_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Point-details panel.
            info = tk.Frame(
                body, bg=WHITE, width=330,
                highlightbackground=BORDER, highlightthickness=1
            )
            info.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
            info.pack_propagate(False)

            tk.Label(
                info, text="Selected event", bg=WHITE, fg=TEXT,
                font=FONT_H, anchor="w"
            ).pack(fill=tk.X, padx=14, pady=(14, 3))

            tk.Label(
                info,
                text="Click a point in the plot to identify its source and values.",
                bg=WHITE, fg=TEXT_DIM, font=FONT_SM,
                wraplength=295, justify=tk.LEFT, anchor="w"
            ).pack(fill=tk.X, padx=14, pady=(0, 10))

            self.point_info = scrolledtext.ScrolledText(
                info, height=18, bg=PANEL, fg=TEXT, font=MONO,
                relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD,
                padx=9, pady=9
            )
            self.point_info.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

            # Whole-cluster exclusion: select one or more current clusters and
            # recompute the model after removing every event in those clusters.
            tk.Label(
                info, text="Remove cluster(s) and recompute",
                bg=WHITE, fg=TEXT, font=FONT_B, anchor="w"
            ).pack(fill=tk.X, padx=12, pady=(0, 3))

            self.cluster_remove_list = tk.Listbox(
                info, height=min(5, max(2, int(self.R["K"]))),
                selectmode=tk.EXTENDED, exportselection=False,
                bg=PANEL, fg=TEXT, font=MONO, relief=tk.FLAT,
                highlightthickness=1, highlightbackground=BORDER
            )
            self.cluster_remove_list.pack(fill=tk.X, padx=12, pady=(0, 5))
            for k in range(self.R["K"]):
                n_k = int((self.R["labels"] == k).sum())
                self.cluster_remove_list.insert(tk.END, f"{self._clabel(k)}   n={n_k}")

            self.remove_cluster_btn = self._btn(
                info, "Remove selected cluster(s) + recompute",
                self._remove_selected_clusters, WARN, WHITE
            )
            self.remove_cluster_btn.pack(fill=tk.X, padx=12, pady=(0, 9))

            self.delete_point_btn = self._btn(
                info, "Delete selected + recompute",
                self._delete_selected_point, ERR, WHITE, big=True
            )
            self.delete_point_btn.pack(fill=tk.X, padx=12, pady=(0, 6))
            self.delete_point_btn.config(state=tk.DISABLED)

            self.restore_points_btn = self._btn(
                info, "Restore all excluded points / clusters",
                self._restore_deleted_points, PANEL, TEXT
            )
            self.restore_points_btn.pack(fill=tk.X, padx=12, pady=(0, 12))
            if not self.excluded_points:
                self.restore_points_btn.config(state=tk.DISABLED)

            self.selected_point_index = None
            self._refresh_point_inspector()

        def _refresh_point_inspector(self):
            if not hasattr(self, "inspect_plot_frame"):
                return

            for w in self.inspect_plot_frame.winfo_children():
                w.destroy()

            xname = self.inspect_x.get()
            yname = self.inspect_y.get()
            if not xname or not yname:
                return

            X = self.R["X_raw"]
            labels = self.R["labels"]
            K = self.R["K"]

            fig = Figure(figsize=(9.5, 7.0), facecolor=WHITE, dpi=100)
            ax = fig.add_subplot(111)
            _style(ax)

            # Each cluster is a separate pickable artist. Store original row indices
            # on the artist so Matplotlib's pick event can be mapped back exactly.
            for k in range(K):
                row_indices = np.flatnonzero(labels == k)
                artist = ax.scatter(
                    X.iloc[row_indices][xname],
                    X.iloc[row_indices][yname],
                    s=45, alpha=0.75, linewidths=0.5,
                    edgecolors=WHITE,
                    color=PALETTE[k % len(PALETTE)],
                    label=f"{self._clabel(k)} (n={len(row_indices)})",
                    picker=7,
                )
                artist._cluster_row_indices = row_indices

            # Highlight currently selected point.
            if self.selected_point_index is not None and 0 <= self.selected_point_index < len(X):
                idx = self.selected_point_index
                ax.scatter(
                    [X.iloc[idx][xname]], [X.iloc[idx][yname]],
                    s=160, facecolors="none", edgecolors=TEXT,
                    linewidths=2.2, zorder=10
                )

            ax.set_xlabel(_feature_label(xname), color=TEXT_MED)
            ax.set_ylabel(_feature_label(yname), color=TEXT_MED)
            ax.set_title("Click any event to inspect it", fontweight="bold", color=TEXT)

            if "duration" in X.columns:
                duration_top = float(X["duration"].max()) + 5.0
                if xname == "duration":
                    ax.set_xlim(0, duration_top)
                if yname == "duration":
                    ax.set_ylim(0, duration_top)

            ax.legend(fontsize=9, edgecolor=BORDER, loc="best")
            fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.11)

            self.inspect_canvas = FigureCanvasTkAgg(fig, master=self.inspect_plot_frame)
            self.inspect_canvas.draw()
            self.inspect_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            tbf = tk.Frame(self.inspect_plot_frame, bg=PANEL)
            tbf.pack(fill=tk.X)
            NavigationToolbar2Tk(self.inspect_canvas, tbf).update()

            self.inspect_canvas.mpl_connect("pick_event", self._on_point_pick)

        def _on_point_pick(self, event):
            artist = event.artist
            row_indices = getattr(artist, "_cluster_row_indices", None)
            if row_indices is None or len(event.ind) == 0:
                return

            # If several points overlap within the pick radius, use the first match.
            local_idx = int(event.ind[0])
            if local_idx >= len(row_indices):
                return

            idx = int(row_indices[local_idx])
            self.selected_point_index = idx
            self._show_selected_point_info(idx)
            self._refresh_point_inspector()

        def _show_selected_point_info(self, idx):
            labeled = self.R["labeled"]
            if idx < 0 or idx >= len(labeled):
                return

            row = labeled.iloc[idx]
            source_file = str(row["source_file"])
            source_row = int(row["source_row"])
            cluster = int(row["cluster"])

            # Resolve full path when possible.
            full_path = next(
                (p for p in self.paths if os.path.basename(p) == source_file),
                source_file
            )

            lines = [
                f"FILE",
                f"{source_file}",
                "",
                f"Full path:",
                f"{full_path}",
                "",
                f"Original row: {source_row}",
                f"Current cluster: C{cluster}",
                "",
                "FEATURE VALUES",
            ]
            for f in self.R["features"]:
                try:
                    lines.append(f"{f}: {float(row[f]):.6g}")
                except Exception:
                    lines.append(f"{f}: {row[f]}")

            key = (source_file, source_row)
            if key in self.excluded_points:
                lines.extend(["", "STATUS: excluded"])

            self.point_info.config(state=tk.NORMAL)
            self.point_info.delete("1.0", tk.END)
            self.point_info.insert(tk.END, "\n".join(lines))
            self.point_info.config(state=tk.DISABLED)

            self.delete_point_btn.config(state=tk.NORMAL)

        def _remove_selected_clusters(self):
            """Exclude all events in one or more selected current clusters, then refit."""
            if self.R is None or not hasattr(self, "cluster_remove_list"):
                return

            selections = list(self.cluster_remove_list.curselection())
            if not selections:
                messagebox.showwarning(
                    "No clusters selected",
                    "Select one or more clusters in the list first."
                )
                return

            cluster_ids = sorted(int(i) for i in selections)
            labeled = self.R["labeled"]
            mask = labeled["cluster"].isin(cluster_ids)
            sub = labeled.loc[mask, ["source_file", "source_row", "cluster"]].copy()
            if sub.empty:
                return

            new_keys = {
                (str(r.source_file), int(r.source_row))
                for r in sub.itertuples(index=False)
            }
            before = len(self.excluded_points)
            self.excluded_points.update(new_keys)
            newly_removed = len(self.excluded_points) - before

            self.removed_cluster_history.append({
                "clusters": ",".join(f"C{k}" for k in cluster_ids),
                "events_in_selected_clusters": int(len(sub)),
                "newly_excluded_events": int(newly_removed),
            })

            self.selected_point_index = None
            cluster_text = ", ".join(self._clabel(k) for k in cluster_ids)
            self.status.config(
                text=(f"Removed {cluster_text} ({newly_removed} new event(s)) — "
                      "recomputing clusters…"),
                fg=WARN
            )
            self._run()

        def _delete_selected_point(self):
            if self.R is None or self.selected_point_index is None:
                return

            row = self.R["labeled"].iloc[self.selected_point_index]
            source_file = str(row["source_file"])
            source_row = int(row["source_row"])
            key = (source_file, source_row)

            # Explicit button press is the deletion confirmation.
            self.excluded_points.add(key)
            self.selected_point_index = None

            self.status.config(
                text=f"Excluded {source_file}, row {source_row} — recomputing clusters…",
                fg=WARN
            )
            self._run()

        def _restore_deleted_points(self):
            if not self.excluded_points:
                return
            n = len(self.excluded_points)
            self.excluded_points.clear()
            self.removed_cluster_history.clear()
            self.selected_point_index = None
            self.status.config(
                text=f"Restored {n} excluded point(s)/cluster event(s) — recomputing clusters…",
                fg=SUCCESS
            )
            self._run()

        def _build_3d(self):
            tab = self.tabs["3d"]
            features = self.R["features"]

            if len(features) < 3:
                self._ph(tab, f"3D Explorer needs at least 3 usable features.\nCurrent features: {', '.join(features)}")
                return

            self.x3d.set(features[0])
            self.y3d.set(features[1])
            self.z3d.set(features[2])

            for w in tab.winfo_children():
                w.destroy()

            ctrl = tk.Frame(tab, bg=PANEL, pady=8)
            ctrl.pack(fill=tk.X, padx=12, pady=(8, 0))
            for lab, var in [("X:", self.x3d), ("Y:", self.y3d), ("Z:", self.z3d)]:
                tk.Label(ctrl, text=lab, bg=PANEL, fg=TEXT_MED, font=FONT_B).pack(side=tk.LEFT, padx=(12, 4))
                cb = ttk.Combobox(ctrl, textvariable=var, values=features, state="readonly", font=FONT, width=13)
                cb.pack(side=tk.LEFT)
            self._btn(ctrl, "Update", self._refresh_3d, ACCENT, WHITE).pack(side=tk.LEFT, padx=12)

            # Axis-limit filter row: min/max entries per axis plus apply/clear.
            limrow = tk.Frame(tab, bg=PANEL, pady=6)
            limrow.pack(fill=tk.X, padx=12, pady=(4, 0))
            tk.Label(
                limrow, text="Show only points within:",
                bg=PANEL, fg=TEXT, font=FONT_B
            ).pack(side=tk.LEFT, padx=(12, 10))

            for lab, minvar, maxvar in [
                ("X", self.x3d_min, self.x3d_max),
                ("Y", self.y3d_min, self.y3d_max),
                ("Z", self.z3d_min, self.z3d_max),
            ]:
                tk.Label(limrow, text=f"{lab} min:", bg=PANEL, fg=TEXT_MED, font=FONT_SM).pack(side=tk.LEFT, padx=(6, 3))
                tk.Entry(limrow, textvariable=minvar, width=8, justify="center",
                         font=FONT_SM, relief=tk.SOLID, bd=1).pack(side=tk.LEFT)
                tk.Label(limrow, text=f"{lab} max:", bg=PANEL, fg=TEXT_MED, font=FONT_SM).pack(side=tk.LEFT, padx=(8, 3))
                tk.Entry(limrow, textvariable=maxvar, width=8, justify="center",
                         font=FONT_SM, relief=tk.SOLID, bd=1).pack(side=tk.LEFT)

            self._btn(limrow, "Apply limits", self._apply_3d_limits, SUCCESS, WHITE).pack(side=tk.LEFT, padx=(14, 6))
            self._btn(limrow, "Clear limits", self._clear_3d_limits, PANEL, TEXT).pack(side=tk.LEFT)

            tk.Label(
                tab,
                text="Leave a box blank for no bound on that side. Limits filter which points are drawn; they don't change the clustering.",
                bg=BG, fg=TEXT_DIM, font=FONT_SM, anchor="w"
            ).pack(fill=tk.X, padx=16, pady=(4, 0))

            self.f3d = tk.Frame(tab, bg=WHITE)
            self.f3d.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))
            self._refresh_3d()

        def _parse_3d_limit_box(self, var, axis_label, side_label):
            txt = var.get().strip()
            if txt == "":
                return None
            try:
                return float(txt)
            except ValueError:
                raise ValueError(f"{axis_label} {side_label} must be a number (or blank).")

        def _apply_3d_limits(self):
            try:
                xlo = self._parse_3d_limit_box(self.x3d_min, "X", "min")
                xhi = self._parse_3d_limit_box(self.x3d_max, "X", "max")
                ylo = self._parse_3d_limit_box(self.y3d_min, "Y", "min")
                yhi = self._parse_3d_limit_box(self.y3d_max, "Y", "max")
                zlo = self._parse_3d_limit_box(self.z3d_min, "Z", "min")
                zhi = self._parse_3d_limit_box(self.z3d_max, "Z", "max")
            except ValueError as e:
                messagebox.showerror("Invalid limit", str(e))
                return

            for label, lo, hi in [("X", xlo, xhi), ("Y", ylo, yhi), ("Z", zlo, zhi)]:
                if lo is not None and hi is not None and lo >= hi:
                    messagebox.showerror("Invalid limit", f"{label} min must be less than {label} max.")
                    return

            self.lim3d = {"x": (xlo, xhi), "y": (ylo, yhi), "z": (zlo, zhi)}
            self._refresh_3d()
            self._refresh_hierarchy()

        def _clear_3d_limits(self):
            for v in (self.x3d_min, self.x3d_max, self.y3d_min, self.y3d_max, self.z3d_min, self.z3d_max):
                v.set("")
            self.lim3d = {"x": (None, None), "y": (None, None), "z": (None, None)}
            self._refresh_3d()
            self._refresh_hierarchy()

        def _refresh_3d(self):
            if not hasattr(self, "f3d"):
                return
            for w in self.f3d.winfo_children():
                w.destroy()
            self._embed(
                fig_3d(self.R, self.x3d.get(), self.y3d.get(), self.z3d.get(),
                       limits=self.lim3d, names=self.cluster_names),
                self.f3d
            )

        # ======================= CLUSTER HIERARCHY (3D) =======================

        def _build_hierarchy_tab(self):
            """
            Step through K with a slider. Click any cluster to LOCK its exact
            set of events; that fixed set of points is then always what's
            drawn, at every K, no matter how far you move the slider --
            moving K up recolors it by however many groups it now splits
            into, moving K down recolors it by however many (fewer) groups
            it collapses into. With nothing locked, the slider just shows
            every cluster at that K, like a plain 3D view.
            """
            tab = self.tabs["hierarchy"]
            for w in tab.winfo_children():
                w.destroy()
            tab.config(bg=BG)

            features = self.R["features"]
            if len(features) < 3:
                self._ph(tab, f"Cluster Hierarchy (3D) needs at least 3 usable features.\nCurrent features: {', '.join(features)}")
                return

            ks = sorted(self.R["labels_by_k"])
            self.hier_ks = ks

            self.hier_k = tk.IntVar(value=self.R["K"])
            self.hier_anchor = None  # (k0, c0): K and cluster whose points are locked, or None

            ctrl = tk.Frame(tab, bg=PANEL, pady=8)
            ctrl.pack(fill=tk.X, padx=12, pady=(8, 0))

            tk.Label(ctrl, text="K:", bg=PANEL, fg=TEXT_MED, font=FONT_B).pack(side=tk.LEFT, padx=(12, 6))
            self.hier_k_label = tk.Label(
                ctrl, text=str(self.hier_k.get()), bg="#eff4ff", fg=ACCENT,
                font=(MONO[0], MONO[1], "bold"), padx=6
            )
            self.hier_k_scale = ttk.Scale(
                ctrl, from_=min(ks), to=max(ks), variable=self.hier_k, orient=tk.HORIZONTAL,
                length=200, command=lambda v: self._on_hier_k_change(v)
            )
            self.hier_k_scale.pack(side=tk.LEFT, padx=(0, 6))
            self.hier_k_label.pack(side=tk.LEFT, padx=(0, 18))

            for lab, var in [("X:", self.x3d), ("Y:", self.y3d), ("Z:", self.z3d)]:
                tk.Label(ctrl, text=lab, bg=PANEL, fg=TEXT_MED, font=FONT_B).pack(side=tk.LEFT, padx=(6, 4))
                cb = ttk.Combobox(ctrl, textvariable=var, values=features, state="readonly", font=FONT, width=13)
                cb.pack(side=tk.LEFT)
                cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_hierarchy())

            self._btn(ctrl, "Update", self._refresh_hierarchy, ACCENT, WHITE).pack(side=tk.LEFT, padx=12)

            # Axis-limit filter row -- shared with the 3D Explorer tab (same
            # StringVars and self.lim3d), so setting limits in either tab
            # applies to both.
            limrow = tk.Frame(tab, bg=PANEL, pady=6)
            limrow.pack(fill=tk.X, padx=12, pady=(4, 0))
            tk.Label(
                limrow, text="Show only points within:",
                bg=PANEL, fg=TEXT, font=FONT_B
            ).pack(side=tk.LEFT, padx=(12, 10))

            for lab, minvar, maxvar in [
                ("X", self.x3d_min, self.x3d_max),
                ("Y", self.y3d_min, self.y3d_max),
                ("Z", self.z3d_min, self.z3d_max),
            ]:
                tk.Label(limrow, text=f"{lab} min:", bg=PANEL, fg=TEXT_MED, font=FONT_SM).pack(side=tk.LEFT, padx=(6, 3))
                tk.Entry(limrow, textvariable=minvar, width=8, justify="center",
                         font=FONT_SM, relief=tk.SOLID, bd=1).pack(side=tk.LEFT)
                tk.Label(limrow, text=f"{lab} max:", bg=PANEL, fg=TEXT_MED, font=FONT_SM).pack(side=tk.LEFT, padx=(8, 3))
                tk.Entry(limrow, textvariable=maxvar, width=8, justify="center",
                         font=FONT_SM, relief=tk.SOLID, bd=1).pack(side=tk.LEFT)

            self._btn(limrow, "Apply limits", self._apply_3d_limits, SUCCESS, WHITE).pack(side=tk.LEFT, padx=(14, 6))
            self._btn(limrow, "Clear limits", self._clear_3d_limits, PANEL, TEXT).pack(side=tk.LEFT)

            tk.Label(
                tab,
                text="Click a cluster on the right to LOCK its exact set of events. That same fixed "
                     "set of points stays on screen as you move the K slider -- only its coloring "
                     "changes, showing how many groups it splits into (K up) or collapses into (K "
                     "down). Click a different cluster to lock onto that one instead, or clear the "
                     "selection to see every cluster at the current K again.",
                bg=BG, fg=TEXT_DIM, font=FONT_SM, anchor="w", wraplength=1100, justify=tk.LEFT
            ).pack(fill=tk.X, padx=16, pady=(4, 0))

            body = tk.Frame(tab, bg=BG)
            body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))

            self.hier_plot_frame = tk.Frame(body, bg=WHITE)
            self.hier_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            info = tk.Frame(
                body, bg=WHITE, width=340,
                highlightbackground=BORDER, highlightthickness=1
            )
            info.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
            info.pack_propagate(False)

            tk.Label(
                info, text="Clusters shown", bg=WHITE, fg=TEXT, font=FONT_H, anchor="w"
            ).pack(fill=tk.X, padx=14, pady=(14, 3))
            self.hier_status_label = tk.Label(
                info, text="", bg=WHITE, fg=TEXT_DIM, font=FONT_SM, anchor="w",
                wraplength=305, justify=tk.LEFT
            )
            self.hier_status_label.pack(fill=tk.X, padx=14, pady=(0, 8))

            self.hier_listbox_frame = tk.Frame(info, bg=WHITE)
            self.hier_listbox_frame.pack(fill=tk.X, padx=12)

            self._btn(
                info, "Clear selection (show all)", lambda: self._set_hier_anchor(None), PANEL, TEXT
            ).pack(fill=tk.X, padx=12, pady=(10, 10))

            self._refresh_hierarchy()

        def _on_hier_k_change(self, value):
            if not hasattr(self, "hier_ks") or not self.hier_ks:
                return
            target = float(value)
            k = min(self.hier_ks, key=lambda x: abs(x - target))
            self.hier_k.set(k)
            if hasattr(self, "hier_k_label"):
                self.hier_k_label.config(text=str(k))
            self._refresh_hierarchy()

        def _set_hier_anchor(self, anchor):
            """anchor is (k0, c0): lock onto the exact events of cluster c0 at
            K=k0, or None to show every cluster at the current K again."""
            self.hier_anchor = anchor
            self._refresh_hierarchy()

        def _refresh_hierarchy(self):
            if not hasattr(self, "hier_plot_frame") or self.R is None:
                return

            k = int(self.hier_k.get())
            labels_by_k = self.R["labels_by_k"]
            labels_k = labels_by_k[k]
            n_k = len(np.unique(labels_k))
            n_total = len(labels_k)
            anchor = self.hier_anchor

            # ---- figure out what's actually being displayed at this K ----
            if anchor is None:
                display = [(c, labels_k == c, PALETTE[c % len(PALETTE)]) for c in range(n_k)]
                self.hier_status_label.config(text=f"Showing all {n_k} cluster(s) at K={k}.")
            else:
                k0, c0 = anchor
                locked_mask = labels_by_k[k0] == c0
                n_locked = int(locked_mask.sum())
                distinct = sorted(set(labels_k[locked_mask].tolist()))
                display = [
                    (c, locked_mask & (labels_k == c), PALETTE[i % len(PALETTE)])
                    for i, c in enumerate(distinct)
                ]
                if k == k0:
                    self.hier_status_label.config(text=f"Locked on C{c0} @ K={k0} (n={n_locked}).")
                elif len(distinct) > 1:
                    self.hier_status_label.config(
                        text=(f"The {n_locked} locked events (C{c0} @ K={k0}) split into "
                              f"{len(distinct)} group(s) at K={k}.")
                    )
                elif distinct:
                    self.hier_status_label.config(
                        text=(f"The {n_locked} locked events (C{c0} @ K={k0}) are all one "
                              f"group (C{distinct[0]}) at K={k}.")
                    )
                else:
                    self.hier_status_label.config(text="Could not locate this selection at this K.")

            # ---- listbox: groups currently visible, clickable to lock a NEW
            # selection (the full cluster at this K, not just the shown
            # subset -- clicking always starts a fresh, genuine lock) ----
            for w in self.hier_listbox_frame.winfo_children():
                w.destroy()
            for c, member_mask, color in display:
                cnt = int(member_mask.sum())
                pct = 100.0 * cnt / n_total if n_total else 0.0
                row = tk.Frame(self.hier_listbox_frame, bg=WHITE)
                row.pack(fill=tk.X, pady=1)
                swatch = tk.Frame(row, bg=color, width=14, height=14)
                swatch.pack(side=tk.LEFT, padx=(4, 6), pady=3)
                swatch.pack_propagate(False)
                tk.Button(
                    row, text=f"C{c}   n={cnt} ({pct:.1f}%)", bg=WHITE, fg=TEXT, font=FONT_SM,
                    relief=tk.FLAT, cursor="hand2", anchor="w",
                    command=lambda kk=k, cc=c: self._set_hier_anchor((kk, cc))
                ).pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3)

            # ---- plot ----
            for w in self.hier_plot_frame.winfo_children():
                w.destroy()
            fig = fig_3d_hierarchy(
                self.R, self.x3d.get(), self.y3d.get(), self.z3d.get(),
                k, anchor=anchor, limits=self.lim3d,
            )
            self._embed(fig, self.hier_plot_frame)

        def _build_table(self, tab, df, cluster_col=None):
            """Generic Treeview table. If `cluster_col` names a column holding
            integer cluster indices, its cells display the current custom
            cluster name (or the default "Ck") instead of the bare index."""
            tab.config(bg=WHITE)
            for w in tab.winfo_children():
                w.destroy()
            wrap = tk.Frame(tab, bg=WHITE)
            wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
            cols = list(df.columns)
            tree = ttk.Treeview(wrap, columns=cols, show="headings", style="Data.Treeview")
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=max(90, min(220, len(str(c)) * 10)), anchor="center")
            for i, (_, row) in enumerate(df.iterrows()):
                vals = []
                for c in cols:
                    v = row[c]
                    if cluster_col is not None and c == cluster_col:
                        try:
                            vals.append(self._clabel(int(v)))
                        except (TypeError, ValueError):
                            vals.append(v)
                    elif isinstance(v, float):
                        vals.append(f"{v:.4g}")
                    else:
                        vals.append(v)
                tree.insert("", tk.END, values=vals, tags=("odd" if i % 2 else "even",))
            tree.tag_configure("even", background=WHITE)
            tree.tag_configure("odd", background="#f8faff")
            vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(fill=tk.BOTH, expand=True)

        def _save(self):
            if not self.R:
                return
            d = filedialog.askdirectory(title="Output folder")
            if not d:
                return

            self.R["labeled"].to_csv(os.path.join(d, "pooled_cluster_labels.csv"), index=False)
            self.R["summary"].to_csv(os.path.join(d, "cluster_summary.csv"), index=False)
            self.R["per_file"].to_csv(os.path.join(d, "cluster_counts_by_file.csv"), index=False)
            self.R["lrt"].to_csv(os.path.join(d, "bootstrap_lrt.csv"), index=False)
            self.R["k_breakdown"].to_csv(os.path.join(d, "k_breakdown.csv"), index=False)

            # Save an audit trail of points manually excluded in the GUI.
            excluded_df = pd.DataFrame(
                sorted(self.excluded_points),
                columns=["source_file", "source_row"]
            )
            excluded_df.to_csv(os.path.join(d, "manually_excluded_points.csv"), index=False)

            cluster_history_df = pd.DataFrame(self.removed_cluster_history)
            cluster_history_df.to_csv(
                os.path.join(d, "removed_cluster_history.csv"), index=False
            )

            # Also write one labeled CSV per source file.
            for source, sub in self.R["labeled"].groupby("source_file"):
                stem = os.path.splitext(os.path.basename(source))[0]
                sub.to_csv(os.path.join(d, f"{stem}_clustered.csv"), index=False)

            # Convenience Excel workbook.
            try:
                with pd.ExcelWriter(os.path.join(d, "clustering_results.xlsx"), engine="openpyxl") as xw:
                    self.R["labeled"].to_excel(xw, sheet_name="labels", index=False)
                    self.R["summary"].to_excel(xw, sheet_name="cluster_summary", index=False)
                    self.R["per_file"].to_excel(xw, sheet_name="per_file", index=False)
                    self.R["lrt"].to_excel(xw, sheet_name="LRT", index=False)
                    self.R["k_breakdown"].to_excel(xw, sheet_name="k_breakdown", index=False)
                    excluded_df.to_excel(xw, sheet_name="excluded_points", index=False)
                    cluster_history_df.to_excel(xw, sheet_name="removed_clusters", index=False)
            except Exception:
                pass

            self.status.config(text=f"Saved results to {d}", fg=SUCCESS)
            messagebox.showinfo("Saved", f"Saved clustering results to:\n{d}")

        def _generate_html_report(self):
            """
            Export a self-contained HTML report for the CURRENT clustering
            result (self.R) -- i.e. whatever the app most recently finished
            computing, including any manual point/cluster exclusions already
            applied. The interactive 3D section uses exactly the axis
            selections and axis limits currently set in the 3D Explorer tab.
            The report includes a "Save as PDF" button that opens the
            browser's print dialog (print-to-PDF), styled via the report's
            own @media print rules.
            """
            if not self.R:
                return

            path = filedialog.asksaveasfilename(
                title="Save HTML report",
                defaultextension=".html",
                filetypes=[("HTML file", "*.html"), ("All files", "*.*")],
                initialfile="cluster_report.html",
            )
            if not path:
                return

            run_meta = {
                "sample_name": self.sample_name.get(),
            }

            try:
                build_html_report(
                    self.R,
                    self.x3d.get(), self.y3d.get(), self.z3d.get(),
                    dict(self.lim3d),
                    path,
                    run_meta=run_meta,
                    names=dict(self.cluster_names),
                )
            except ImportError as e:
                messagebox.showerror("Missing dependency", str(e))
                return
            except Exception:
                messagebox.showerror("Report error", traceback.format_exc())
                return

            self.status.config(text=f"Saved HTML report to {path}", fg=SUCCESS)
            if messagebox.askyesno("Report saved", f"Saved HTML report to:\n{path}\n\nOpen it now?"):
                webbrowser.open("file://" + os.path.abspath(path))

        def _poll(self):
            try:
                while True:
                    m, t = self.q.get_nowait()
                    self.log.config(state=tk.NORMAL)
                    self.log.insert(tk.END, m, t or "")
                    self.log.see(tk.END)
                    self.log.config(state=tk.DISABLED)
            except queue.Empty:
                pass
            self.after(40, self._poll)

        def _logclear(self):
            self.log.config(state=tk.NORMAL)
            self.log.delete("1.0", tk.END)
            self.log.config(state=tk.DISABLED)

    App().mainloop()


if __name__ == "__main__":
    launch()