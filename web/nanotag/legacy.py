"""Legacy CSV/XLSX compatibility (column names used by Tagger_GUI.py,
NeuralNetwork.py and clustering.py).

The column mapping (`load_legacy_frame`) and the derived-value formulas
(`tagger_derived`, `nn_derived`) are copied from the original scripts so that
files round-trip exactly.
"""
import io
import math
import os

import numpy as np
import pandas as pd

from .db import jload

# Tagger_GUI.py LEGACY_SAVE_ORDER
LEGACY_SAVE_ORDER = [
    "event_id", "file_name", "sensor", "analytes", "solution",
    "event_start (s)", "event_end (s)", "notes",
    "Base (V)", "Step (V)", "RefBase (V)", "RefStep (V)",
    "entry Base (pA)", "entry Peak (pA)", "exit Peak (pA)", "exit Base (pA)",
    "duration", "OSC", "RefOSC", "entry spike", "exit spike",
    "event_plateau",
    "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
    "window_start", "window_end",
]

# NeuralNetwork.py OUT_COLS (the <name>__predicted.xlsx layout)
NN_OUT_COLS = [
    "event_id", "file_name", "sensor", "analytes", "solution",
    "event_start (s)", "event_end (s)", "notes",
    "Base (V)", "Step (V)", "RefBase (V)", "RefStep (V)",
    "entry Base (pA)", "entry Peak (pA)", "exit Peak (pA)", "exit Base (pA)",
    "duration", "OSC", "RefOSC", "entry spike", "exit spike",
    "event_plateau", "event_plateau_t", "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
]

DERIVED_COLS = ["duration", "OSC", "RefOSC", "entry spike", "exit spike"]
AUDIT_COLS = ["db_event_id", "recording", "created_by", "created_at", "updated_by", "updated_at"]


def _norm(s):
    s = (str(s) if s is not None else "").strip().lower()
    for ch in [" ", "\t", "(", ")", ",", "[", "]"]:
        s = s.replace(ch, "")
    return s


def read_table_bytes(data: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(data))
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError(f"Unsupported file type: {ext} (use .csv or .xlsx)")


def load_legacy_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Tagger_GUI.load_legacy_or_internal_events, operating on a DataFrame.
    Returns INTERNAL column names plus `_derived` (dict of any derived columns present)."""
    file_cols = {_norm(c): c for c in df_raw.columns}

    def get_series(*candidates, as_float=False):
        for cand in candidates:
            key = _norm(cand)
            if key in file_cols:
                s = df_raw[file_cols[key]]
                return pd.to_numeric(s, errors="coerce") if as_float else s
        return pd.Series([np.nan] * len(df_raw))

    out = pd.DataFrame(index=range(len(df_raw)))
    out["event_id"] = get_series("event_id")
    out["file_name"] = get_series("file_name", "file name")
    out["sensor"] = get_series("sensor")
    out["analytes"] = get_series("analytes")
    out["solution"] = get_series("solution")
    out["notes"] = get_series("notes")
    out["event_start"] = get_series("event_start (s)", "event_start_s", "event_start", as_float=True)
    out["event_end"] = get_series("event_end (s)", "event_end_s", "event_end", as_float=True)
    out["event_plateau"] = get_series("event_plateau", as_float=True)
    out["optical_base"] = get_series("Base (V)", "base (v)", "optical_base", as_float=True)
    out["optical_rise"] = get_series("Step (V)", "step (v)", "optical_rise", as_float=True)
    out["optical_end"] = get_series("optical_end", as_float=True)
    out["opticalre_base"] = get_series("RefBase (V,)", "refbase (v,)", "opticalre_base", as_float=True)
    out["opticalre_rise"] = get_series("RefStep (V)", "refstep (v)", "opticalre_rise", as_float=True)
    out["opticalre_end"] = get_series("opticalre_end", as_float=True)
    out["entry_base"] = get_series("entry Base (pA)", "entry base (pa)", "entry_base", as_float=True)
    out["entry_peak"] = get_series("entry Peak (pA)", "entry peak (pa)", "entry_peak", as_float=True)
    out["exit_peak"] = get_series("exit Peak (pA)", "exit peak (pa)", "exit_peak", as_float=True)
    out["exit_base"] = get_series("exit Base (pA)", "exit base (pa)", "exit_base", as_float=True)
    out["entry_base_t"] = get_series("entry_base_t", as_float=True)
    out["entry_peak_t"] = get_series("entry_peak_t", as_float=True)
    out["exit_peak_t"] = get_series("exit_peak_t", as_float=True)
    out["exit_base_t"] = get_series("exit_base_t", as_float=True)
    out["window_start"] = get_series("window_start", as_float=True)
    out["window_end"] = get_series("window_end", as_float=True)

    derived = {c: get_series(c, as_float=True) for c in DERIVED_COLS}
    dlist = []
    for i in range(len(df_raw)):
        d = {c: float(derived[c].iloc[i]) for c in DERIVED_COLS if pd.notna(derived[c].iloc[i])}
        dlist.append(d or None)
    out["_derived"] = dlist
    return out


def file_stem(name) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    return os.path.splitext(os.path.basename(str(name).strip()))[0].lower()


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _txt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v)
    return s if s.strip() and s.lower() != "nan" else None


def frame_to_event_dicts(internal: pd.DataFrame):
    """INTERNAL frame -> list of (file_stem, event_dict) ready for annotations.insert_events."""
    out = []
    for i, r in internal.iterrows():
        ev = {
            "event_no": _intish(r.get("event_id")),
            "sensor": _txt(r.get("sensor")), "analytes": _txt(r.get("analytes")),
            "solution": _txt(r.get("solution")), "notes": _txt(r.get("notes")),
        }
        for k in ["window_start", "window_end", "event_start", "event_plateau", "event_end",
                  "optical_base", "optical_rise", "optical_end",
                  "opticalre_base", "opticalre_rise", "opticalre_end",
                  "entry_base", "entry_peak", "exit_peak", "exit_base",
                  "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t"]:
            ev[k] = _num(r.get(k))
        ev["derived"] = r.get("_derived")
        out.append((file_stem(r.get("file_name")), ev))
    return out


def _intish(v):
    f = _num(v)
    if f is None:
        return None
    return int(f) if float(f).is_integer() else None


# ----------------------------------------------------------------- derived values
def tagger_derived(e):
    """Tagger_GUI.internal_to_legacy_saveframe formulas (signed OSC)."""
    base, step = _num(e.get("optical_base")), _num(e.get("optical_rise"))
    rb, rs = _num(e.get("opticalre_base")), _num(e.get("opticalre_rise"))

    def osc(a, b):
        if a is None or b is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            v = ((b - a) / ((b + a) / 2.0)) * 100.0 if (b + a) != 0 else float("nan")
        return v

    return {
        "duration": _sub(e.get("event_end"), e.get("event_start")),
        "OSC": osc(base, step),
        "RefOSC": osc(rb, rs),
        "entry spike": _sub(e.get("entry_peak"), e.get("entry_base")),
        "exit spike": _sub(e.get("exit_peak"), e.get("exit_base")),
    }


def nn_derived(e):
    """NeuralNetwork.run_inference_on_npz formulas (absolute OSC, 0 when undefined)."""
    vb, vs = _num(e.get("optical_base")), _num(e.get("optical_rise"))
    rb, rs = _num(e.get("opticalre_base")), _num(e.get("opticalre_rise"))

    def osc(a, b):
        if a is None or b is None:
            return None
        return abs((b - a) / ((b + a) / 2) * 100) if (b + a) != 0 else 0.0

    return {
        "duration": _sub(e.get("event_end"), e.get("event_start")),
        "OSC": osc(vb, vs),
        "RefOSC": osc(rb, rs),
        "entry spike": _sub(e.get("entry_peak"), e.get("entry_base")),
        "exit spike": _sub(e.get("exit_peak"), e.get("exit_base")),
    }


def _sub(a, b):
    a, b = _num(a), _num(b)
    return None if a is None or b is None else a - b


def derived_for(e, formula):
    stored = jload(e.get("derived_json")) if isinstance(e.get("derived_json"), str) else e.get("derived")
    if stored:
        d = {c: stored.get(c) for c in DERIVED_COLS}
        calc = nn_derived(e) if formula == "nn" else tagger_derived(e)
        for c in DERIVED_COLS:  # fill any gaps (e.g. spikes blank in a source file)
            if d[c] is None:
                d[c] = calc[c]
        return d
    return nn_derived(e) if formula == "nn" else tagger_derived(e)


# ----------------------------------------------------------------- export frames
def legacy_rows(events, file_name_by_rec, formula="tagger", audit=False, rec_label=None):
    out = []
    for e in events:
        d = derived_for(e, formula)
        row = {
            "event_id": e.get("event_no"),
            "file_name": file_name_by_rec.get(e["recording_id"], ""),
            "sensor": e.get("sensor") or "", "analytes": e.get("analytes") or "",
            "solution": e.get("solution") or "",
            "event_start (s)": e.get("event_start"), "event_end (s)": e.get("event_end"),
            "notes": e.get("notes") or "",
            "Base (V)": e.get("optical_base"), "Step (V)": e.get("optical_rise"),
            "RefBase (V)": e.get("opticalre_base"), "RefStep (V)": e.get("opticalre_rise"),
            "entry Base (pA)": e.get("entry_base"), "entry Peak (pA)": e.get("entry_peak"),
            "exit Peak (pA)": e.get("exit_peak"), "exit Base (pA)": e.get("exit_base"),
            "duration": d["duration"], "OSC": d["OSC"], "RefOSC": d["RefOSC"],
            "entry spike": d["entry spike"], "exit spike": d["exit spike"],
            "event_plateau": e.get("event_plateau"),
            "entry_base_t": e.get("entry_base_t"), "entry_peak_t": e.get("entry_peak_t"),
            "exit_peak_t": e.get("exit_peak_t"), "exit_base_t": e.get("exit_base_t"),
            "window_start": e.get("window_start"), "window_end": e.get("window_end"),
        }
        if audit:
            row.update({
                "db_event_id": e.get("id"),
                "recording": (rec_label or {}).get(e["recording_id"], e["recording_id"]),
                "created_by": e.get("created_by"), "created_at": _ts(e.get("created_at")),
                "updated_by": e.get("updated_by"), "updated_at": _ts(e.get("updated_at")),
            })
        out.append(row)
    return out


def legacy_frame(events, file_name_by_rec, formula="tagger", audit=False):
    cols = LEGACY_SAVE_ORDER + (AUDIT_COLS if audit else [])
    return pd.DataFrame(legacy_rows(events, file_name_by_rec, formula, audit), columns=cols)


def nn_frame(events, file_name_by_rec):
    rows = []
    for r in legacy_rows(events, file_name_by_rec, "nn"):
        r["event_plateau_t"] = r["event_plateau"]
        rows.append(r)
    return pd.DataFrame(rows, columns=NN_OUT_COLS)


def _ts(t):
    if not t:
        return ""
    import datetime as _dt
    return _dt.datetime.fromtimestamp(float(t)).strftime("%Y-%m-%d %H:%M:%S")


def frame_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)
    return buf.getvalue()
