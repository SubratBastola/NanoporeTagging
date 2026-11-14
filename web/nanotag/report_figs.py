"""Friendly cluster names in the pictures of a clustering HTML report.

The report's figures are PNG pictures made when the clustering ran, so a name given later
("C0 (Full)") cannot be written into them. This module draws the affected figures again with
clustering.py's own figure functions, from the files the run saved (pooled_cluster_labels.csv,
k_breakdown.csv), and hands them to the report page as it is served. Nothing on disk changes
and the clustering is not run again. The Sankey plot shows no cluster names and is never redrawn.

The K Selection figure also needs the BIC curve, which the run did not save as a file. It is read
from the run's job log (the sweep table clustering.py prints); if that log is gone, the GMM sweep
alone is fitted again with the same seeds. Either way the result is checked against the run
(BIC-best K, cluster sizes) and the figure is left as it was if the check fails.
"""
import base64
import io
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger("nanotag.report_figs")

_lock = threading.Lock()
_cache = OrderedDict()          # (run id, report file stamp, figure, names) -> value
_CACHE_MAX = 96
REFIT_MAX_EVENTS = 60000        # above this the sweep is not refitted inside a web request


def _remember(key, make):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    val = make()
    _cache[key] = val
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return val


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)     # same as clustering.py's report
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _display(names_k):
    """{"0": "Full"} -> {0: "C0 (Full)"}"""
    out = {}
    for c, n in (names_k or {}).items():
        if str(n).strip():
            out[int(c)] = f"C{int(c)} ({n})"
    return out


def _bic_from_log(job_log, ks):
    """BIC per K from the last 'GMM sweep' table in the job log, or None."""
    if not job_log:
        return None
    parts = re.split(r"\[3\] GMM sweep K = 1 \.\. \d+", job_log)
    if len(parts) < 2:
        return None
    bic = {}
    for line in parts[-1].splitlines()[1:]:
        m = re.match(r"^(?:\[\d\d:\d\d:\d\d\])?\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", line)
        if m:
            bic[int(m.group(1))] = float(m.group(2))
        elif bic:
            break
    return bic if sorted(bic) == list(ks) else None


def _bic_refit(CL, pooled, features, ks, final_k, final_counts):
    """Fit the GMM sweep again exactly as clustering.py did (same preprocessing and seeds)."""
    import numpy as np
    if len(pooled) > REFIT_MAX_EVENTS:
        return None
    combined = pooled[["source_file", "source_row"] + features].copy()
    combined["_file_order"] = 0
    _, _, Xs, feats = CL.preprocess_combined(combined, list(features), True, None)
    if list(feats) != list(features) or len(Xs) != len(pooled):
        return None
    bic = {}
    for k in ks:
        g = CL.fit_gmm(Xs, k, 42 + k, 15)
        bic[k] = float(g.bic(Xs))
        if k == final_k:                      # the refit must reproduce the run's clusters
            sizes = sorted(np.bincount(g.predict(Xs), minlength=k).tolist(), reverse=True)
            if sizes != sorted(final_counts, reverse=True):
                return None
    return bic


def figures(run, names, job_log=None):
    """Figures of one clustering run redrawn with friendly names.

    names: {"<K>": {"<cluster>": "name"}}.  Returns {} when there are no names, otherwise
    {"ksel": png_b64, "pairs": png_b64, "pies": {"<K>": png_b64}, "skipped": [..]} holding only
    the figures that changed.
    """
    names = {str(k): v for k, v in (names or {}).items() if _display(v)}
    if not names:
        return {}
    out_dir = Path(run.get("out_dir") or "")
    rep, pooled_p, kb_p = out_dir / "cluster_report.html", out_dir / "pooled_cluster_labels.csv", out_dir / "k_breakdown.csv"
    if not (rep.exists() and pooled_p.exists() and kb_p.exists()):
        return {"skipped": ["ksel", "pairs", "pies"]}
    st = rep.stat()
    stamp = (int(run["id"]), st.st_mtime_ns, st.st_size)
    out = {"pies": {}, "skipped": []}
    with _lock:
        import numpy as np
        import pandas as pd
        from . import vendor
        CL = vendor.clustering()

        def data():
            pooled = pd.read_csv(pooled_p)
            feats = [c for c in pooled.columns if c not in ("source_file", "source_row", "cluster")]
            kb = pd.read_csv(kb_p)
            counts = {int(k): [int(n) for n in g.sort_values("_c")["N"]]
                      for k, g in kb.assign(_c=kb["Cluster"].astype(str).str[1:].astype(int)).groupby("K")}
            return pooled, feats, counts
        pooled, feats, counts = _remember(stamp + ("data",), data)
        ks = sorted(counts)
        labels = pooled["cluster"].to_numpy(dtype=int)
        final_k = int(run.get("final_k") or (labels.max() + 1))

        # ---- per-K pies of the "Explore other K" tab (clustering.py's pie, relabelled)
        for k_s, nm in names.items():
            k = int(k_s)
            if k not in counts:
                continue
            disp = _display(nm)

            def pie(k=k, disp=disp):
                fig = CL._fig_k_pie(np.repeat(np.arange(k), counts[k]), k)
                for t in fig.axes[0].texts:
                    m = re.fullmatch(r"C(\d+)", t.get_text())
                    if m and int(m.group(1)) in disp:
                        t.set_text(disp[int(m.group(1))])
                return _b64(fig)
            out["pies"][k_s] = _remember(stamp + ("pie", k, tuple(sorted(disp.items()))), pie)

        # ---- figures of the recommended K: Feature Relationships and K Selection
        disp = _display(names.get(str(final_k)))
        if disp:
            key = tuple(sorted(disp.items()))
            R = {"features": feats, "X_raw": pooled[feats].astype(float), "labels": labels, "K": final_k}
            out["pairs"] = _remember(stamp + ("pairs", key), lambda: _b64(CL.fig_report_pairs(R, names=disp)))

            def bic():
                k_bic = int(run.get("k_bic") or 0)
                for how, b in (("job log", lambda: _bic_from_log(job_log, ks)),
                               ("refit", lambda: _bic_refit(CL, pooled, feats, ks, final_k, counts.get(final_k, [])))):
                    try:
                        v = b()
                    except Exception as e:      # noqa: BLE001 - never break the report over a picture
                        log.warning("run %s: BIC curve from %s failed: %s", run["id"], how, e)
                        v = None
                    if v and min(v, key=v.get) == k_bic:
                        return v
                return None
            b = _remember(stamp + ("bic",), bic)
            if b:
                Rk = dict(R, bic=b, k_bic=int(run["k_bic"]))
                out["ksel"] = _remember(stamp + ("ksel", key), lambda: _b64(CL.fig_kselection(Rk, names=disp)))
            else:
                out["skipped"].append("ksel")
    return out
