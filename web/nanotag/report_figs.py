"""Friendly cluster names in the pictures of a clustering HTML report.

The report's figures are PNG pictures made when the clustering ran, so a name given later
("C0 (Full)") cannot be written into them. This module draws the affected figures again with
clustering.py's own figure functions, from the files the run saved (pooled_cluster_labels.csv,
k_breakdown.csv), and hands them to the report page as it is served. Nothing on disk changes
and the clustering is not run again. The Sankey plot shows no cluster names and is never redrawn.

The K Selection figure has two halves: the BIC curve, which the run did not save as a file, and the
cluster-proportions pie, which carries the names. So that the pie always gets its names, the figure
is put together from the report's own picture (the BIC half and the heading, pixel for pixel) and
a freshly drawn pie (clustering.py's fig_kselection, right half only). The geometry is checked
against the original picture first. Only if that check fails (e.g. a report made with an older
plotting library) is the whole figure redrawn, with the BIC curve read from the run's job log or,
if the log is gone, from fitting the GMM sweep again with the same seeds.
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


def _fit_pie_labels(fig, disp):
    """Wrap long names on the K Selection pie onto several lines so that a label on the pie's left
    side stays clear of the BIC panel (clustering.py writes each label on one line)."""
    import textwrap
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if fig.canvas is None or not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    rend = fig.canvas.get_renderer()
    limit = fig.axes[0].get_window_extent(rend).x1 + 0.08 * fig.dpi
    texts = [(t, t.get_text()) for t in fig.axes[1].texts if t.get_text() in set(disp.values())]
    for t, full in texts:                       # only a label that reaches the BIC panel is wrapped
        for width in (None, 28, 22, 18, 15, 12, 10, 8):
            t.set_text(full if width is None else textwrap.fill(full, width))
            if t.get_window_extent(rend).x0 >= limit:
                break


def _ksel_composite(CL, R, disp, ks, orig_b64):
    """K Selection figure = the original picture's BIC half and heading + the pie drawn with names.

    Returns PNG base64, or None when the original picture does not line up with what
    clustering.py draws today (then the caller redraws the whole figure instead).
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox
    from PIL import Image
    DPI = 110
    O = np.asarray(Image.open(io.BytesIO(base64.b64decode(orig_b64))).convert("RGB"))
    h0, w0 = O.shape[:2]
    Rd = dict(R, bic={k: 0.0 for k in ks}, k_bic=ks[0])       # the BIC half drawn here is thrown away

    def draw(names):
        fig = CL.fig_kselection(Rd, names=names)
        FigureCanvasAgg(fig)
        fig.dpi = DPI
        if names:
            _fit_pie_labels(fig, names)
        return fig, fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)

    def render(fig, box):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches=box)
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))

    figA, tA = draw({})
    fw, fh = figA.get_size_inches()
    split = figA.axes[0].get_position().x1 * fw + 0.06        # just right of the BIC panel (inches)
    band = 0.90 * fh                                          # above this: only the figure heading
    # The original picture shares its right, top and bottom edges with figure A (only the width of
    # the BIC tick labels differs), so its left edge follows from its width.
    xO0 = tA.x1 - w0 / DPI
    if xO0 > split - 0.5:
        return None
    A = render(figA, Bbox.from_extents(xO0, tA.y0, tA.x1, tA.y1))
    if A.shape != O.shape:
        return None
    xs = int(round((split - xO0) * DPI))
    diff = np.abs(A[:, xs:].astype(int) - O[:, xs:].astype(int)).max(axis=2)
    if (diff > 96).mean() > 0.004:                            # pie half must be the same picture
        return None

    figB, tB = draw(disp)
    U = Bbox.from_extents(xO0, min(tA.y0, tB.y0), max(tA.x1, tB.x1), max(tA.y1, tB.y1))
    out = render(figB, U).copy()
    oy = int(round((U.y1 - tA.y1) * DPI))                     # where the original sits in the new canvas
    yb = oy + int(round((tA.y1 - band) * DPI))
    out[:, :xs] = 255
    out[:yb, :] = 255
    hh = min(h0, out.shape[0] - oy)
    out[oy:oy + hh, :xs] = O[:hh, :xs]                        # BIC half, exactly as in the report
    ww = min(w0, out.shape[1])
    out[oy:yb, :ww] = O[:yb - oy, :ww]                        # heading, exactly as in the report
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
            m = re.search(r'<h2>K Selection</h2>\s*<div class="card">\s*<img class="report-fig" '
                          r'src="data:image/png;base64,([^"]+)"', rep.read_text(encoding="utf-8", errors="replace"))
            return pooled, feats, counts, (m.group(1) if m else None)
        pooled, feats, counts, ksel_orig = _remember(stamp + ("data",), data)
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

            def composite():
                if not ksel_orig:
                    return None
                try:
                    return _ksel_composite(CL, R, disp, ks, ksel_orig)
                except Exception as e:          # noqa: BLE001
                    log.warning("run %s: K Selection composite failed: %s", run["id"], e)
                    return None
            ksel = _remember(stamp + ("ksel-composite", key), composite)
            if ksel:
                out["ksel"] = ksel
                return out

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
                def redraw():
                    fig = CL.fig_kselection(Rk, names=disp)
                    _fit_pie_labels(fig, disp)
                    return _b64(fig)
                out["ksel"] = _remember(stamp + ("ksel", key), redraw)
            else:
                out["skipped"].append("ksel")
    return out
