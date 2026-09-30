"""Web Tagger — the Tagger_GUI.py workflow served from the lab server.

Same modes as the desktop tool (Zoom/Inspect, Add Event 3 optical -> 4 electrical,
Add Window, Delete Events, Edit Points), same labels (O1-O3, R1-R3, E1-E4) and the
same derived values. Differences:
  * data comes from the Zarr store: only the visible time range is read and sent;
    wide views are min/max envelopes, narrow views are exact 10 kHz samples;
  * every channel of the ABF can be shown (checkboxes); roles default to 2/3/0;
  * edits are saved immediately to the shared database with the author recorded,
    and other users' changes appear automatically (revision polling).
"""
import math
from urllib.parse import parse_qs, urlencode

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from flask_login import current_user

from . import annotations as A
from . import auth, legacy, store
from .db import db, jload, one, rows

# ---- constants mirrored from Tagger_GUI.py ----
WIN_SEC = 30.0          # default window length
MIN_WIN_SEC = 0.1
POINT_LABELS = {"O1": "Event start", "O2": "Event plateau", "O3": "Event end",
                "E1": "Entry base", "E2": "Entry peak", "E3": "Exit peak", "E4": "Exit base"}
AUTHOR_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2",
                 "#17becf", "#bcbd22", "#7f7f7f"]
OPT_KEYS = ["event_start", "event_plateau", "event_end"]
ELEC_KEYS = ["entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t"]
OPT_VAL = {"event_start": ("optical_base", "opticalre_base"),
           "event_plateau": ("optical_rise", "opticalre_rise"),
           "event_end": ("optical_end", "opticalre_end")}
ELEC_VAL = {"entry_base_t": "entry_base", "entry_peak_t": "entry_peak",
            "exit_peak_t": "exit_peak", "exit_base_t": "exit_base"}
TABLE_COLS = ["event_id", "event_start (s)", "event_plateau", "event_end (s)", "duration", "OSC", "RefOSC",
              "entry spike", "exit spike", "window_start", "window_end", "notes", "created_by", "updated_by"]
EDITABLE_COLS = {"notes"}


def label_texts(short, long=False):
    return [f"{s} ({POINT_LABELS.get(s, POINT_LABELS.get('O' + s[1:], ''))})" for s in short] if long else short


def _yrange_from_data(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size:
        q1, q99 = np.percentile(y, [1, 99])
        margin = (q99 - q1) * 0.2 if (q99 - q1) > 0 else 1.0
        return [float(q1 - margin), float(q99 + margin)]
    return [-1.0, 1.0]


def _f(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------- data access
def load_context(rec_id, set_id):
    with db() as con:
        rec = one(con, "SELECT * FROM recordings WHERE id=?", (rec_id,))
        if rec is None:
            return None, "Recording not found."
        exp = one(con, "SELECT * FROM experiments WHERE id=?", (rec["experiment_id"],))
        recs = rows(con, "SELECT id, file_name, status FROM recordings WHERE experiment_id=? ORDER BY file_name",
                    (exp["id"],))
    if not auth.can_view_experiment(current_user, exp):
        return None, "You do not have access to this experiment."
    if rec["status"] != "ready" or not rec["zarr_path"]:
        return None, f"Recording '{rec['file_name']}' is not ready yet (status: {rec['status']})."
    sets = A.list_sets(exp["id"])
    if not sets:
        set_id = A.create_set(exp["id"], "Manual tags", current_user.username)
        sets = A.list_sets(exp["id"])
    s = next((x for x in sets if x["id"] == set_id), None) or sets[0]
    attrs = store.group_attrs(rec["zarr_path"])
    chans = jload(rec["channels_json"], []) or attrs.get("channels", [])
    return {
        "rec_id": rec["id"], "set_id": s["id"], "exp_id": exp["id"], "exp_name": exp["name"],
        "file_name": rec["file_name"], "group": rec["zarr_path"],
        "fs": float(rec["display_fs"]), "n": int(rec["display_n"]), "duration": float(rec["duration"]),
        "channels": chans, "levels": attrs.get("levels", []), "factor": attrs.get("pyramid_factor", 10),
        "roles": {"opt": rec["role_opt"], "optref": rec["role_optref"], "elec": rec["role_elec"]},
        "editable": auth.can_edit_set(current_user, s), "locked": bool(s["locked"]),
        "formula": s["formula"], "set_name": s["name"],
        "rec_options": [{"label": f"{r['file_name']}" + ("" if r["status"] == "ready" else f"  [{r['status']}]"),
                         "value": r["id"], "disabled": r["status"] != "ready"} for r in recs],
        "set_options": [{"label": f"{x['name']}  ({x['n_events']} events, {x['kind']}"
                                  f"{', locked' if x['locked'] else ''})", "value": x["id"]} for x in sets],
    }, None


def fetch_events(ctx):
    return A.list_events(ctx["set_id"], ctx["rec_id"])


def ch_label(ctx, ch):
    c = next((x for x in ctx["channels"] if int(x["index"]) == int(ch)), None)
    return f"{ch}: {c['name']} [{c['units']}]" if c else f"ch{ch}"


def channel_order(ctx, visible):
    r = ctx["roles"]
    order = [ch for ch in (r["opt"], r["optref"], r["elec"]) if ch in visible]
    order += sorted(ch for ch in visible if ch not in order)
    return order


def vals_at(ctx, ch, times):
    return store.values_at(ctx["group"], int(ch), times, ctx["fs"], ctx["n"])


def clamp_window(ctx, x0, x1):
    dur = ctx["duration"]
    span = max(float(x1) - float(x0), MIN_WIN_SEC)
    span = min(span, dur)
    x0 = max(float(x0), 0.0)
    x1 = x0 + span
    if x1 > dur:
        x1 = dur
        x0 = max(0.0, x1 - span)
    return [round(x0, 6), round(x1, 6)]


def default_window(ctx):
    return clamp_window(ctx, 0.0, WIN_SEC)


def _ev_times(e):
    return [e.get(k) for k in OPT_KEYS + ELEC_KEYS + ["window_start", "window_end"] if e.get(k) is not None]


def _ev_span(e):
    ws, we = e.get("window_start"), e.get("window_end")
    if ws is not None and we is not None:
        return ws, we
    t = _ev_times(e)
    return (min(t), max(t)) if t else (None, None)


# ----------------------------------------------------------------- figure
def build_figure(ctx, events, window, zoom, click, visible, mode, sel, long_labels, authors_on,
                 color_by_author, budget):
    order = channel_order(ctx, visible)
    if not order:
        order = [ctx["roles"]["opt"]]
    n = len(order)
    gap = 0.035 if n > 1 else 0.0
    h = (1.0 - gap * (n - 1)) / n
    axis_of = {ch: i + 1 for i, ch in enumerate(order)}
    w0, w1 = window
    fig = go.Figure()
    tmap = []

    detail_y = {}
    info = None
    for kind in ("overview", "detail"):
        for ch in order:
            if kind == "overview":
                x, y, _ = store.read_window(ctx["group"], ch, 0.0, ctx["duration"], ctx["fs"], ctx["n"],
                                           ctx["levels"], ctx["factor"], budget=2000)
                fig.add_trace(go.Scatter(x=x.astype(np.float32), y=y, yaxis=f"y{axis_of[ch]}", mode="lines",
                                           line=dict(width=1, color="#b0b7c3"), hoverinfo="skip",
                                           showlegend=False, name=f"{ch_label(ctx, ch)} overview"))
            else:
                x, y, inf = store.read_window(ctx["group"], ch, w0, w1, ctx["fs"], ctx["n"],
                                              ctx["levels"], ctx["factor"], budget=budget)
                info = info or inf
                detail_y[ch] = y
                fig.add_trace(go.Scatter(x=x.astype(np.float32), y=y, yaxis=f"y{axis_of[ch]}", mode="lines",
                                           line=dict(width=1.2, color="#1f3a8a"), hoverinfo="x+y+name",
                                           showlegend=False, name=ch_label(ctx, ch)))
            tmap.append({"ch": ch, "kind": kind})

    y_ranges = {ch: (zoom.get(str(ch)) or _yrange_from_data(detail_y[ch])) for ch in order}
    opt, optref, elec = ctx["roles"]["opt"], ctx["roles"]["optref"], ctx["roles"]["elec"]

    # ---- event markers (grouped per axis/author for speed) ----
    authors = sorted({e.get("created_by") or "?" for e in events})
    color_of = {a: AUTHOR_COLORS[i % len(AUTHOR_COLORS)] for i, a in enumerate(authors)}
    shown = [e for e in events if (authors_on is None or (e.get("created_by") or "?") in authors_on)]
    groups = {}
    for e in shown:
        key = (e.get("created_by") or "?") if color_by_author else "_all"
        groups.setdefault(key, []).append(e)

    def in_win(t):
        return t is not None and w0 <= t <= w1

    for key, evs in groups.items():
        color = color_of.get(key, "#d62728") if color_by_author else None
        pts = {"opt": ([], [], []), "optref": ([], [], []), "elec": ([], [], [])}
        missing = {"opt": [], "optref": [], "elec": []}
        for e in evs:
            has_plateau = e.get("event_plateau") is not None
            for j, k in enumerate(OPT_KEYS):
                t = e.get(k)
                if t is None or (k == "event_plateau" and not has_plateau):
                    continue
                lab = f"O{j + 1}"
                for role, vkey, pref in (("opt", OPT_VAL[k][0], "O"), ("optref", OPT_VAL[k][1], "R")):
                    v = e.get(vkey)
                    pts[role][0].append(t)
                    pts[role][1].append(v)
                    pts[role][2].append(label_texts([pref + lab[1:]], long_labels)[0] if in_win(t) else "")
                    if v is None:
                        missing[role].append(len(pts[role][1]) - 1)
            if all(e.get(k) is not None for k in ELEC_KEYS):
                for j, k in enumerate(ELEC_KEYS):
                    t = e.get(k)
                    v = e.get(ELEC_VAL[k])
                    pts["elec"][0].append(t)
                    pts["elec"][1].append(v)
                    pts["elec"][2].append(label_texts([f"E{j + 1}"], long_labels)[0] if in_win(t) else "")
                    if v is None:
                        missing["elec"].append(len(pts["elec"][1]) - 1)
        for role, ch, sym, tpos in (("opt", opt, "circle", "top center"), ("optref", optref, "diamond", "top center"),
                                    ("elec", elec, "x", "bottom center")):
            if ch not in axis_of or not pts[role][0]:
                continue
            xs, ys, txt = pts[role]
            if missing[role]:
                fill = vals_at(ctx, ch, [xs[i] for i in missing[role]])
                for i, v in zip(missing[role], fill):
                    ys[i] = v
            fig.add_trace(go.Scatter(x=xs, y=ys, yaxis=f"y{axis_of[ch]}", mode="markers+text", text=txt,
                                     textposition=tpos, marker=dict(size=8, symbol=sym, color=color),
                                     showlegend=bool(color_by_author) and role == "opt",
                                     name=key if color_by_author else "events", hoverinfo="skip"))

    # ---- pending clicks (Add mode) ----
    if click and mode == "add":
        ot = click.get("optical_times") or []
        if ot:
            for role, ch, pref, sym in (("opt", opt, "O", "circle-open"), ("optref", optref, "R", "diamond-open")):
                if ch in axis_of:
                    fig.add_trace(go.Scatter(x=ot, y=list(vals_at(ctx, ch, ot)), yaxis=f"y{axis_of[ch]}",
                                             mode="markers+text", textposition="top center",
                                             text=label_texts([f"{pref}{i + 1}" for i in range(len(ot))], long_labels),
                                             marker=dict(size=10, symbol=sym, color="#ea580c"),
                                             showlegend=False, hoverinfo="skip"))
        et = click.get("electrical_times") or []
        if et and elec in axis_of:
            fig.add_trace(go.Scatter(x=et, y=list(vals_at(ctx, elec, et)), yaxis=f"y{axis_of[elec]}",
                                     mode="markers+text", textposition="bottom center",
                                     text=label_texts([f"E{i + 1}" for i in range(len(et))], long_labels),
                                     marker=dict(size=10, symbol="x-open", color="#ea580c"),
                                     showlegend=False, hoverinfo="skip"))

    # ---- shapes: editable guides first (Edit mode), then window rectangles ----
    shapes, smap = [], []
    sel_ev = next((e for e in events if e["id"] == sel), None) if sel else None
    if mode == "edit" and sel_ev is not None and ctx["editable"]:
        for k in OPT_KEYS:
            t = sel_ev.get(k)
            if t is None:
                continue
            for ch in (opt, optref):
                if ch in axis_of:
                    shapes.append(dict(type="line", x0=t, x1=t, y0=y_ranges[ch][0], y1=y_ranges[ch][1], xref="x",
                                       yref=f"y{axis_of[ch]}", line=dict(width=1.5, dash="dot", color="#ea580c")))
                    smap.append(k)
        for k in ELEC_KEYS:
            t = sel_ev.get(k)
            if t is not None and elec in axis_of:
                shapes.append(dict(type="line", x0=t, x1=t, y0=y_ranges[elec][0], y1=y_ranges[elec][1], xref="x",
                                   yref=f"y{axis_of[elec]}", line=dict(width=1.5, dash="dash", color="#ea580c")))
                smap.append(k)
    for e in shown:
        ws, we = e.get("window_start"), e.get("window_end")
        if ws is not None and we is not None:
            hl = sel_ev is not None and e["id"] == sel_ev["id"]
            shapes.append(dict(type="rect", x0=max(ws, 0.0), x1=max(we, 0.0), xref="x", yref="paper", y0=0, y1=1,
                               fillcolor="#f59e0b" if hl else "purple", opacity=0.18 if hl else 0.10,
                               layer="below", line_width=0, editable=False))

    layout = dict(
        template="plotly_white",
        dragmode="select" if mode in ("delete", "window") else "zoom",
        selectdirection="h",
        clickmode="event",
        uirevision=None,
        showlegend=bool(color_by_author),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=70, r=20, t=40, b=30),
        shapes=shapes,
        xaxis=dict(domain=[0, 1], range=[w0, w1], rangeslider=dict(visible=True, thickness=0.07),
                   title=dict(text=f"Time (s) — window {w0:.3f} to {w1:.3f}  ·  "
                                   + ("exact 10 kHz samples" if info and info["mode"] == "full"
                                      else f"min/max envelope, bin {1000 * (info or {}).get('bin_s', 0):.2f} ms "
                                           "(zoom in for exact samples)"),
                              font=dict(size=11))),
    )
    for i, ch in enumerate(order):
        top = 1.0 - i * (h + gap)
        role = {opt: " (optical)", optref: " (opticalR)", elec: " (electrical)"}.get(ch, "")
        layout[f"yaxis{'' if i == 0 else i + 1}"] = dict(
            domain=[max(0.0, top - h), top], range=y_ranges[ch], fixedrange=False,
            title=dict(text=ch_label(ctx, ch) + role, font=dict(size=11)))
    fig.update_layout(**layout)
    if mode == "add":
        fig.add_annotation(text=_add_mode_text(click), x=0.5, xref="paper", y=1.0, yref="paper", yanchor="bottom",
                           showarrow=False, font=dict(size=12, color="#1f3a8a"), bgcolor="#eef2ff",
                           bordercolor="#c7d2fe", borderwidth=1, borderpad=4)
    return fig, {"traces": tmap, "order": order, "smap": smap}


def _add_mode_text(click):
    click = click or {}
    stage = click.get("stage", "optical")
    n_opt = len(click.get("optical_times") or [])
    n_el = len(click.get("electrical_times") or [])
    if stage == "optical":
        prompts = ["Add Optical Start (O1)", "Add Optical Plateau (O2)", "Add Optical End (O3)"]
        return prompts[min(n_opt, 2)] + f" — {n_opt}/3"
    prompts = ["Add Entry Base (E1)", "Add Entry Peak (E2)", "Add Exit Peak (E3)", "Add Exit Base (E4)"]
    return prompts[min(n_el, 3)] + f" — {n_el}/4"


def table_rows(ctx, events):
    lr = legacy.legacy_rows(events, {ctx["rec_id"]: ctx["file_name"]}, ctx["formula"])
    out = []
    for e, r in zip(events, lr):
        row = {c: r.get(c) for c in TABLE_COLS if c in r}
        for c in ("event_start (s)", "event_plateau", "event_end (s)", "window_start", "window_end", "duration"):
            if row.get(c) is not None:
                row[c] = round(float(row[c]), 4)
        for c in ("OSC", "RefOSC", "entry spike", "exit spike"):
            if row.get(c) is not None and math.isfinite(float(row[c])):
                row[c] = round(float(row[c]), 3)
        row["created_by"] = e.get("created_by")
        row["updated_by"] = e.get("updated_by")
        row["id"] = e["id"]
        out.append(row)
    return out


def _menu(title, options, current, href):
    """A drop-down list of links (links, not a callback input, so navigation cannot loop)."""
    cur = next((o["label"] for o in options if o["value"] == current), "?")
    items = [html.Div(dcc.Link(o["label"], href=href(o["value"])) if not o.get("disabled") and o["value"] != current
                      else html.Span(o["label"], style={"color": "#9ca3af" if o.get("disabled") else "#111",
                                                         "fontWeight": 600 if o["value"] == current else 400}),
                      style={"padding": "3px 8px", "whiteSpace": "nowrap"}) for o in options]
    return html.Details([
        html.Summary([html.Span(f"{title}: ", style={"color": "#6b7280"}), html.B(cur)],
                     style={"cursor": "pointer", "padding": "4px 8px", "border": "1px solid #d1d5db",
                            "borderRadius": "6px", "background": "#fff"}),
        html.Div(items, style={"position": "absolute", "zIndex": 20, "background": "#fff", "border": "1px solid #d1d5db",
                               "borderRadius": "6px", "boxShadow": "0 4px 12px #0002", "maxHeight": "60vh",
                               "overflowY": "auto", "marginTop": "2px"}),
    ], style={"position": "relative", "display": "inline-block"})


# ----------------------------------------------------------------- app
def init_tagger(server):
    app = Dash(__name__, server=server, url_base_pathname="/tagger/", title="Tagger",
               suppress_callback_exceptions=True, update_title=None)

    btn = {"marginRight": "6px"}
    small = {"fontSize": "12px", "color": "#555"}
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="ctx"), dcc.Store(id="events", data=[]), dcc.Store(id="rev"),
        dcc.Store(id="window"), dcc.Store(id="nav", data=0), dcc.Store(id="zoom", data={}),
        dcc.Store(id="click", data={"stage": "idle"}), dcc.Store(id="sel"), dcc.Store(id="tmap"),
        dcc.Interval(id="poll", interval=10000),
        dcc.Download(id="download"),
        html.Div([
            html.A("← Experiment", id="back-link", href="/", style={"marginRight": "14px"}),
            html.Div(id="rec-menu", style={"marginRight": "10px"}),
            html.Div(id="set-menu"),
            html.Span(id="who", style={"marginLeft": "14px", **small}),
        ], style={"display": "flex", "alignItems": "center", "padding": "8px 10px",
                  "borderBottom": "1px solid #e5e7eb"}),
        html.Div([
            html.Div([
                html.Div("Mode", style={"fontWeight": 700}),
                dcc.Dropdown(id="mode", clearable=False, value="zoom", options=[
                    {"label": "Zoom/Inspect", "value": "zoom"},
                    {"label": "Add Event (3 Optical → 4 Electrical)", "value": "add"},
                    {"label": "Add Window (drag rectangle)", "value": "window"},
                    {"label": "Delete Events (drag rectangle)", "value": "delete"},
                    {"label": "Edit Points (select row then drag guides)", "value": "edit"},
                ]),
                html.Div("Add: click 3 points on Optical/OpticalRe (start, plateau, end), then 4 on Electrical "
                         "(entry base, entry peak, exit peak, exit base). Window: drag a rectangle. Delete: drag a "
                         "rectangle to remove events in that range. Edit: select a row, then drag the guides. "
                         "Changes save immediately.", style={**small, "margin": "6px 0"}),
                html.Div([html.Button("Undo Last Point", id="undo", style=btn),
                          html.Button("Discard Pending Event", id="discard")]),
                html.Hr(),
                html.Div("Channels shown", style={"fontWeight": 700}),
                dcc.Checklist(id="vis", inputStyle={"marginRight": "4px"}, labelStyle={"display": "block"}),
                html.Div("Channel roles (this recording)", style={"fontWeight": 700, "marginTop": "8px"}),
                html.Div([html.Span("Optical", style={"width": "70px", "display": "inline-block"}),
                          dcc.Dropdown(id="role-opt", clearable=False, style={"width": "170px", "display": "inline-block"})]),
                html.Div([html.Span("OpticalRe", style={"width": "70px", "display": "inline-block"}),
                          dcc.Dropdown(id="role-optref", clearable=False, style={"width": "170px", "display": "inline-block"})]),
                html.Div([html.Span("Electrical", style={"width": "70px", "display": "inline-block"}),
                          dcc.Dropdown(id="role-elec", clearable=False, style={"width": "170px", "display": "inline-block"})]),
                html.Button("Save roles", id="save-roles", style={"marginTop": "4px"}),
                html.Hr(),
                html.Div("X zoom", style={"fontWeight": 700}),
                html.Div([html.Button("X＋", id="x-in", style=btn), html.Button("X－", id="x-out", style=btn),
                          html.Button("X Auto", id="x-auto", style=btn), html.Button("Whole file", id="x-all")]),
                html.Div([html.Button("◀ Prev event", id="prev-ev", style=btn),
                          html.Button("Next event ▶", id="next-ev")], style={"marginTop": "4px"}),
                html.Div("Y zoom", style={"fontWeight": 700, "marginTop": "8px"}),
                dcc.Dropdown(id="y-ch", clearable=False),
                html.Div([html.Button("＋", id="y-plus", style=btn), html.Button("－", id="y-minus", style=btn),
                          html.Button("Auto", id="y-auto", style=btn), html.Button("All auto", id="y-all")],
                         style={"marginTop": "4px"}),
                dcc.Checklist(id="opts", inputStyle={"marginRight": "4px"}, labelStyle={"display": "block"},
                              style={"marginTop": "8px"}, value=[],
                              options=[{"label": "Fit to screen (auto-Y on pan/zoom)", "value": "fit"},
                                       {"label": "Show long labels", "value": "long"},
                                       {"label": "Color markers by author", "value": "author"}]),
                html.Div("Detail points per channel", style={"fontWeight": 700, "marginTop": "8px"}),
                dcc.Dropdown(id="budget", clearable=False, value=8000,
                             options=[{"label": f"{v:,}", "value": v} for v in (2000, 8000, 20000, 60000)]),
                html.Div("Show events by", style={"fontWeight": 700, "marginTop": "8px"}),
                dcc.Checklist(id="authors", inputStyle={"marginRight": "4px"}, labelStyle={"display": "block"}),
            ], style={"flex": "0 0 260px", "padding": "8px 10px", "borderRight": "1px solid #eee",
                      "fontSize": "13px", "overflowY": "auto", "maxHeight": "92vh"}),
            html.Div([
                html.Div(id="msg", style={"minHeight": "20px", "margin": "4px 8px", "color": "#15803d"}),
                dcc.Graph(id="graph", style={"height": "84vh"}, config={
                    "scrollZoom": True, "doubleClick": "reset", "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "edits": {"shapePosition": True}, "responsive": True}),
            ], style={"flex": "1 1 auto", "minWidth": "600px"}),
            html.Div([
                html.Div("Events in this recording (click a row to navigate)", style={"fontWeight": 600}),
                dash_table.DataTable(
                    id="table", columns=[{"name": c, "id": c, "editable": c in EDITABLE_COLS} for c in TABLE_COLS],
                    data=[], row_selectable="single", page_size=25, sort_action="native", filter_action="native",
                    style_table={"height": "68vh", "overflowY": "auto", "overflowX": "auto"},
                    style_cell={"fontSize": 12, "padding": "4px", "textAlign": "left", "minWidth": "60px",
                                "maxWidth": "180px", "whiteSpace": "nowrap", "overflow": "hidden",
                                "textOverflow": "ellipsis"},
                    style_header={"fontWeight": 600}),
                html.Div([html.Button("Delete selected", id="del-row", style=btn),
                          html.Button("Export this recording (CSV)", id="export")], style={"marginTop": "6px"}),
            ], id="side", style={"flex": "0 0 520px", "padding": "8px", "borderLeft": "1px solid #eee"}),
        ], style={"display": "flex", "alignItems": "stretch"}),
    ], style={"fontFamily": "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"})

    # ------------------------------------------------------------ load / navigation
    @app.callback(
        Output("ctx", "data"), Output("events", "data"), Output("rev", "data"), Output("window", "data"),
        Output("vis", "options"), Output("vis", "value"), Output("rec-menu", "children"),
        Output("set-menu", "children"),
        Output("role-opt", "options"), Output("role-opt", "value"), Output("role-optref", "options"),
        Output("role-optref", "value"), Output("role-elec", "options"), Output("role-elec", "value"),
        Output("y-ch", "options"), Output("y-ch", "value"), Output("authors", "options"), Output("authors", "value"),
        Output("msg", "children"), Output("back-link", "href"), Output("who", "children"),
        Output("zoom", "data"), Output("sel", "data"),
        Input("url", "search"))
    def load(search):
        q = parse_qs((search or "").lstrip("?"))
        try:
            rec_id = int(q.get("rec", [0])[0])
            set_id = int(q.get("set", [0])[0])
        except ValueError:
            rec_id, set_id = 0, 0
        ctx, err = load_context(rec_id, set_id)
        if ctx is None:
            return (None, [], None, None, [], [], None, None, [], None, [], None, [], None, [], None, [], [],
                    html.Span(err, style={"color": "#b91c1c"}), "/", "", {}, None)
        evs = fetch_events(ctx)
        chopts = [{"label": ch_label(ctx, c["index"]), "value": int(c["index"])} for c in ctx["channels"]]
        r = ctx["roles"]
        vis = [ch for ch in (r["opt"], r["optref"], r["elec"]) if any(o["value"] == ch for o in chopts)]
        auth_opts = [{"label": a, "value": a} for a in A.authors(ctx["set_id"])] or []
        who = f"Signed in as {current_user.username}" + ("" if ctx["editable"] else
                                                           "  ·  this set is locked (read-only)")
        return (ctx, evs, A.revision(ctx["set_id"]), default_window(ctx), chopts, vis,
                _menu("Recording", ctx["rec_options"], ctx["rec_id"], lambda v: f"?rec={v}&set={ctx['set_id']}"),
                _menu("Annotation set", ctx["set_options"], ctx["set_id"], lambda v: f"?rec={ctx['rec_id']}&set={v}"),
                chopts, r["opt"], chopts, r["optref"], chopts, r["elec"],
                [{"label": ch_label(ctx, c), "value": c} for c in vis] or chopts, vis[0] if vis else None,
                auth_opts, [o["value"] for o in auth_opts],
                f"{ctx['exp_name']} · {ctx['file_name']} · set '{ctx['set_name']}' · {len(evs)} event(s)",
                f"/experiments/{ctx['exp_id']}", who, {}, None)

    @app.callback(Output("y-ch", "options", allow_duplicate=True), Input("vis", "value"), State("ctx", "data"),
                  prevent_initial_call=True)
    def ych_opts(vis, ctx):
        if not ctx:
            return no_update
        return [{"label": ch_label(ctx, c), "value": c} for c in channel_order(ctx, vis or [])]

    # ------------------------------------------------------------ polling for other users' edits
    @app.callback(Output("events", "data", allow_duplicate=True), Output("rev", "data", allow_duplicate=True),
                  Output("authors", "options", allow_duplicate=True),
                  Input("poll", "n_intervals"), State("ctx", "data"), State("rev", "data"),
                  prevent_initial_call=True)
    def poll(_, ctx, rev):
        if not ctx:
            return no_update, no_update, no_update
        cur = A.revision(ctx["set_id"])
        if cur == rev:
            return no_update, no_update, no_update
        return fetch_events(ctx), cur, [{"label": a, "value": a} for a in A.authors(ctx["set_id"])]

    # ------------------------------------------------------------ render
    @app.callback(Output("graph", "figure"), Output("tmap", "data"),
                  Input("events", "data"), Input("window", "data"), Input("nav", "data"), Input("zoom", "data"),
                  Input("click", "data"), Input("vis", "value"), Input("mode", "value"), Input("sel", "data"),
                  Input("opts", "value"), Input("authors", "value"), Input("budget", "value"),
                  State("ctx", "data"), State("rev", "data"))
    def render(events, window, nav, zoom, click, vis, mode, sel, opts, authors_on, budget, ctx, rev):
        if not ctx or not window:
            return go.Figure(), None
        opts = opts or []
        fig, tmap = build_figure(ctx, events or [], window, zoom or {}, click, vis or [], mode, sel,
                                 "long" in opts, authors_on, "author" in opts, int(budget or 8000))
        fig.update_layout(uirevision=f"{ctx['rec_id']}-{nav}")
        return fig, tmap

    @app.callback(Output("table", "data"), Output("table", "selected_rows"),
                  Input("events", "data"), Input("sel", "data"), State("ctx", "data"))
    def table(events, sel, ctx):
        if not ctx:
            return [], []
        rows_ = table_rows(ctx, events or [])
        idx = [i for i, r in enumerate(rows_) if r["id"] == sel]
        return rows_, idx

    # ------------------------------------------------------------ helpers for writes
    def _guard(ctx):
        if not ctx:
            return "No recording loaded."
        s = A.get_set(ctx["set_id"])
        with db() as con:
            exp = one(con, "SELECT * FROM experiments WHERE id=?", (s["experiment_id"],)) if s else None
            rec = one(con, "SELECT experiment_id FROM recordings WHERE id=?", (ctx["rec_id"],))
        if s is None or exp is None or rec is None or rec["experiment_id"] != s["experiment_id"] \
                or not auth.can_view_experiment(current_user, exp):
            return "Not allowed."
        if not auth.can_edit_set(current_user, s):
            return "This annotation set is locked by an administrator (read-only)."
        return None

    def _reload(ctx):
        return fetch_events(ctx), A.revision(ctx["set_id"])

    # ------------------------------------------------------------ plot interactions (relayout)
    @app.callback(
        Output("window", "data", allow_duplicate=True), Output("zoom", "data", allow_duplicate=True),
        Output("events", "data", allow_duplicate=True), Output("rev", "data", allow_duplicate=True),
        Output("msg", "children", allow_duplicate=True),
        Input("graph", "relayoutData"),
        State("ctx", "data"), State("mode", "value"), State("window", "data"), State("zoom", "data"),
        State("events", "data"), State("sel", "data"), State("opts", "value"), State("tmap", "data"),
        prevent_initial_call=True)
    def relayout(rd, ctx, mode, window, zoom, events, sel, opts, tmap):
        if not ctx or not rd or not tmap:
            return (no_update,) * 5
        window = list(window or default_window(ctx))
        zoom = dict(zoom or {})
        out_events, out_rev, msg = no_update, no_update, no_update
        new_window = no_update
        order = tmap["order"]

        # Delete / Add Window via horizontal selection box
        if mode in ("delete", "window") and rd.get("selections"):
            sel_box = rd["selections"][-1]
            if "x0" in sel_box and "x1" in sel_box:
                a, b = sorted([float(sel_box["x0"]), float(sel_box["x1"])])
                err = _guard(ctx)
                if err:
                    return no_update, no_update, no_update, no_update, html.Span(err, style={"color": "#b91c1c"})
                if mode == "delete":
                    ids = [e["id"] for e in events or [] if any(a <= t <= b for t in _ev_times(e))]
                    n = A.delete_events(ids, current_user.username)
                    msg = f"Removed {n} event(s) from {a:.3f}s–{b:.3f}s" if n else "No events found in selected range"
                else:
                    A.add_event(ctx["set_id"], ctx["rec_id"], {"window_start": a, "window_end": b},
                                current_user.username)
                    msg = f"Window added: {a:.3f}s to {b:.3f}s"
                out_events, out_rev = _reload(ctx)
                return no_update, no_update, out_events, out_rev, msg

        x_changed = False
        if "xaxis.range[0]" in rd and "xaxis.range[1]" in rd:
            new_window = clamp_window(ctx, rd["xaxis.range[0]"], rd["xaxis.range[1]"])
            x_changed = True
        elif isinstance(rd.get("xaxis.range"), (list, tuple)) and len(rd["xaxis.range"]) == 2:
            new_window = clamp_window(ctx, *rd["xaxis.range"])
            x_changed = True
        elif "xaxis.autorange" in rd:
            new_window = default_window(ctx)
            x_changed = True
        for i, ch in enumerate(order):
            ax = "yaxis" if i == 0 else f"yaxis{i + 1}"
            if f"{ax}.range[0]" in rd and f"{ax}.range[1]" in rd:
                zoom[str(ch)] = [rd[f"{ax}.range[0]"], rd[f"{ax}.range[1]"]]
            if f"{ax}.autorange" in rd:
                zoom.pop(str(ch), None)
        if x_changed and "fit" in (opts or []):
            zoom = {}

        # Edit mode: guide drags
        if mode == "edit" and sel:
            keys = [k for k in rd if k.startswith("shapes[") and (k.endswith(".x0") or k.endswith(".x1"))]
            if keys:
                err = _guard(ctx)
                if err:
                    return new_window, zoom, no_update, no_update, html.Span(err, style={"color": "#b91c1c"})
                ev = next((e for e in events or [] if e["id"] == sel), None)
                shape_x = {}
                for k in keys:
                    idx = int(k.split("[")[1].split("]")[0])
                    shape_x.setdefault(idx, {})[k.rsplit(".", 1)[1]] = float(rd[k])
                upd = {}
                for idx, co in shape_x.items():
                    if idx >= len(tmap["smap"]):
                        continue
                    field = tmap["smap"][idx]
                    t = (co["x0"] + co["x1"]) / 2.0 if "x0" in co and "x1" in co else co.get("x0", co.get("x1"))
                    upd[field] = t
                    if field in OPT_VAL:
                        vo, vr = OPT_VAL[field]
                        upd[vo] = float(vals_at(ctx, ctx["roles"]["opt"], [t])[0])
                        upd[vr] = float(vals_at(ctx, ctx["roles"]["optref"], [t])[0])
                    else:
                        upd[ELEC_VAL[field]] = float(vals_at(ctx, ctx["roles"]["elec"], [t])[0])
                if ev and upd:
                    try:
                        A.update_event(ev["id"], upd, ev["version"], current_user.username)
                        msg = "Updated event via guides."
                    except A.ConflictError as ex:
                        msg = html.Span(str(ex), style={"color": "#b45309"})
                    out_events, out_rev = _reload(ctx)
        return new_window, zoom, out_events, out_rev, msg

    # ------------------------------------------------------------ clicks (Add mode)
    @app.callback(
        Output("click", "data"), Output("events", "data", allow_duplicate=True),
        Output("rev", "data", allow_duplicate=True), Output("msg", "children", allow_duplicate=True),
        Input("graph", "clickData"),
        State("click", "data"), State("mode", "value"), State("ctx", "data"), State("tmap", "data"),
        prevent_initial_call=True)
    def on_click(cd, click, mode, ctx, tmap):
        if mode != "add" or not cd or not ctx or not tmap:
            return no_update, no_update, no_update, no_update
        err = _guard(ctx)
        if err:
            return no_update, no_update, no_update, html.Span(err, style={"color": "#b91c1c"})
        pt = cd["points"][0]
        cn = pt.get("curveNumber", -1)
        if cn < 0 or cn >= len(tmap["traces"]):
            return no_update, no_update, no_update, no_update
        ch = tmap["traces"][cn]["ch"]
        t = float(pt["x"])
        r = ctx["roles"]
        click = dict(click or {})
        click.setdefault("stage", "optical")
        if click["stage"] == "idle":
            click["stage"] = "optical"
        click.setdefault("optical_times", [])
        click.setdefault("electrical_times", [])
        if click["stage"] == "optical":
            if ch not in (r["opt"], r["optref"]):
                return no_update, no_update, no_update, f"Please click on {ch_label(ctx, r['opt'])} or " \
                                                        f"{ch_label(ctx, r['optref'])} for the Optical stage."
            click["optical_times"] = click["optical_times"] + [t]
            n = len(click["optical_times"])
            msg = f"Optical point {n}/3 recorded"
            if n >= 3:
                click["stage"] = "electrical"
                msg += "  Optical confirmed. Now select 4 points on Electrical."
            return click, no_update, no_update, msg
        if ch != r["elec"]:
            return no_update, no_update, no_update, f"Please click on {ch_label(ctx, r['elec'])} for the Electrical stage."
        click["electrical_times"] = click["electrical_times"] + [t]
        n = len(click["electrical_times"])
        if n < 4:
            return click, no_update, no_update, f"Electrical point {n}/4 recorded"
        ot = sorted(click["optical_times"])
        vo = vals_at(ctx, r["opt"], ot)
        vr = vals_at(ctx, r["optref"], ot)
        et = click["electrical_times"]
        ve = vals_at(ctx, r["elec"], et)
        ev = {"event_start": ot[0], "event_plateau": ot[1], "event_end": ot[2],
              "optical_base": _f(vo[0]), "optical_rise": _f(vo[1]), "optical_end": _f(vo[2]),
              "opticalre_base": _f(vr[0]), "opticalre_rise": _f(vr[1]), "opticalre_end": _f(vr[2]),
              "entry_base": _f(ve[0]), "entry_peak": _f(ve[1]), "exit_peak": _f(ve[2]), "exit_base": _f(ve[3]),
              "entry_base_t": et[0], "entry_peak_t": et[1], "exit_peak_t": et[2], "exit_base_t": et[3]}
        A.add_event(ctx["set_id"], ctx["rec_id"], ev, current_user.username)
        evs, rev = _reload(ctx)
        return ({"stage": "optical", "optical_times": [], "electrical_times": []}, evs, rev,
                "Electrical point 4/4 recorded  Event saved.")

    @app.callback(Output("click", "data", allow_duplicate=True), Output("sel", "data", allow_duplicate=True),
                  Input("mode", "value"), State("ctx", "data"), State("events", "data"), State("window", "data"),
                  State("sel", "data"), prevent_initial_call=True)
    def mode_change(mode, ctx, events, window, sel):
        click = {"stage": "optical" if mode == "add" else "idle", "optical_times": [], "electrical_times": []}
        if mode == "edit" and not sel and events and window:
            c = (window[0] + window[1]) / 2.0
            best = min(events, key=lambda e: abs(np.mean(_ev_span(e)) - c) if _ev_span(e)[0] is not None else 1e18)
            sel = best["id"]
        return click, sel

    @app.callback(Output("click", "data", allow_duplicate=True), Output("msg", "children", allow_duplicate=True),
                  Input("undo", "n_clicks"), Input("discard", "n_clicks"), State("click", "data"),
                  State("mode", "value"), prevent_initial_call=True)
    def undo_discard(nu, nd, click, mode):
        click = dict(click or {})
        if dash.ctx.triggered_id == "discard":
            return {"stage": "optical" if mode == "add" else "idle", "optical_times": [],
                    "electrical_times": []}, "Discarded pending event."
        if click.get("stage") == "optical" and click.get("optical_times"):
            click["optical_times"] = click["optical_times"][:-1]
            return click, "Removed last Optical point."
        if click.get("stage") == "electrical":
            if click.get("electrical_times"):
                click["electrical_times"] = click["electrical_times"][:-1]
                return click, "Removed last Electrical point."
            click["stage"] = "optical"
            click["optical_times"] = click.get("optical_times", [])[:-1]
            return click, "Back to Optical stage; removed last Optical point."
        return no_update, "Nothing to undo."

    # ------------------------------------------------------------ zoom buttons
    @app.callback(Output("window", "data", allow_duplicate=True), Output("nav", "data", allow_duplicate=True),
                  Output("sel", "data", allow_duplicate=True), Output("zoom", "data", allow_duplicate=True),
                  Input("x-in", "n_clicks"), Input("x-out", "n_clicks"), Input("x-auto", "n_clicks"),
                  Input("x-all", "n_clicks"), Input("prev-ev", "n_clicks"), Input("next-ev", "n_clicks"),
                  State("window", "data"), State("nav", "data"), State("ctx", "data"), State("events", "data"),
                  State("sel", "data"), State("opts", "value"), State("zoom", "data"),
                  prevent_initial_call=True)
    def xbuttons(a, b, c_, d, e_, f_, window, nav, ctx, events, sel, opts, zoom):
        if not ctx:
            return (no_update,) * 4
        trig = dash.ctx.triggered_id
        w0, w1 = window or default_window(ctx)
        span = max(w1 - w0, MIN_WIN_SEC)
        new_sel = no_update
        if trig == "x-in":
            win = clamp_window(ctx, w0, w0 + max(span * 0.8, MIN_WIN_SEC))
        elif trig == "x-out":
            win = clamp_window(ctx, w0, w0 + span * 1.25)
        elif trig == "x-auto":
            win = default_window(ctx)
        elif trig == "x-all":
            win = clamp_window(ctx, 0.0, ctx["duration"])
        else:
            spans = sorted(((s, en, ev["id"]) for ev in (events or []) for s, en in [_ev_span(ev)] if s is not None))
            if not spans:
                return no_update, no_update, no_update, no_update
            cur = next((s for s in spans if s[2] == sel), None)
            ref = cur[0] if cur else (w0 + w1) / 2.0
            if trig == "next-ev":
                cand = [s for s in spans if s[0] > ref + 1e-9] or spans[-1:]
                tgt = cand[0]
            else:
                cand = [s for s in spans if s[0] < ref - 1e-9] or spans[:1]
                tgt = cand[-1]
            pad = max(tgt[1] - tgt[0], MIN_WIN_SEC) * 0.07
            win = clamp_window(ctx, tgt[0] - pad, tgt[1] + pad)
            new_sel = tgt[2]
        z = {} if "fit" in (opts or []) else no_update
        return win, (nav or 0) + 1, new_sel, z

    @app.callback(Output("zoom", "data", allow_duplicate=True), Output("nav", "data", allow_duplicate=True),
                  Input("y-plus", "n_clicks"), Input("y-minus", "n_clicks"), Input("y-auto", "n_clicks"),
                  Input("y-all", "n_clicks"),
                  State("y-ch", "value"), State("zoom", "data"), State("window", "data"), State("ctx", "data"),
                  State("nav", "data"), prevent_initial_call=True)
    def ybuttons(p, m, au, al, ch, zoom, window, ctx, nav):
        if not ctx or ch is None:
            return no_update, no_update
        trig = dash.ctx.triggered_id
        zoom = dict(zoom or {})
        if trig == "y-all":
            return {}, (nav or 0) + 1
        if trig == "y-auto":
            zoom.pop(str(ch), None)
            return zoom, (nav or 0) + 1
        cur = zoom.get(str(ch))
        if cur is None:
            _, y, _ = store.read_window(ctx["group"], ch, window[0], window[1], ctx["fs"], ctx["n"],
                                        ctx["levels"], ctx["factor"], 8000)
            cur = _yrange_from_data(y)
        c0 = (cur[0] + cur[1]) / 2.0
        hh = max((cur[1] - cur[0]) / 2.0 * (0.8 if trig == "y-plus" else 1.25), 1e-9)
        zoom[str(ch)] = [c0 - hh, c0 + hh]
        return zoom, (nav or 0) + 1

    # ------------------------------------------------------------ table
    @app.callback(Output("sel", "data", allow_duplicate=True), Output("window", "data", allow_duplicate=True),
                  Output("nav", "data", allow_duplicate=True),
                  Input("table", "selected_row_ids"), State("events", "data"), State("ctx", "data"),
                  State("sel", "data"), State("nav", "data"), prevent_initial_call=True)
    def select_row(ids, events, ctx, sel, nav):
        if not ids or not ctx:
            return no_update, no_update, no_update
        eid = ids[0]
        if eid == sel:
            return no_update, no_update, no_update
        ev = next((e for e in events or [] if e["id"] == eid), None)
        if ev is None:
            return no_update, no_update, no_update
        s, en = _ev_span(ev)
        if s is None:
            return eid, no_update, no_update
        pad = max(en - s, MIN_WIN_SEC) * 0.07
        return eid, clamp_window(ctx, s - pad, en + pad), (nav or 0) + 1

    @app.callback(Output("events", "data", allow_duplicate=True), Output("rev", "data", allow_duplicate=True),
                  Output("msg", "children", allow_duplicate=True), Output("sel", "data", allow_duplicate=True),
                  Input("del-row", "n_clicks"), State("sel", "data"), State("ctx", "data"),
                  prevent_initial_call=True)
    def delete_row(n, sel, ctx):
        if not sel:
            return no_update, no_update, "No row selected.", no_update
        err = _guard(ctx)
        if err:
            return no_update, no_update, html.Span(err, style={"color": "#b91c1c"}), no_update
        A.delete_events([sel], current_user.username)
        evs, rev = _reload(ctx)
        return evs, rev, "Deleted selected event.", None

    @app.callback(Output("events", "data", allow_duplicate=True), Output("rev", "data", allow_duplicate=True),
                  Output("msg", "children", allow_duplicate=True),
                  Input("table", "data_timestamp"), State("table", "data"), State("events", "data"),
                  State("ctx", "data"), prevent_initial_call=True)
    def edit_cells(ts, data, events, ctx):
        if not ctx or not data:
            return no_update, no_update, no_update
        err = _guard(ctx)
        by_id = {e["id"]: e for e in events or []}
        changed = 0
        for row in data:
            e = by_id.get(row.get("id"))
            if e is None:
                continue
            new = row.get("notes") or None
            if (e.get("notes") or None) != new:
                if err:
                    return no_update, no_update, html.Span(err, style={"color": "#b91c1c"})
                try:
                    A.update_event(e["id"], {"notes": new}, e["version"], current_user.username)
                    changed += 1
                except A.ConflictError as ex:
                    return (*_reload(ctx), html.Span(str(ex), style={"color": "#b45309"}))
        if not changed:
            return no_update, no_update, no_update
        evs, rev = _reload(ctx)
        return evs, rev, f"Saved notes for {changed} event(s)."

    @app.callback(Output("download", "data"), Input("export", "n_clicks"), State("ctx", "data"),
                  prevent_initial_call=True)
    def export(n, ctx):
        if not ctx:
            return no_update
        df = legacy.legacy_frame(fetch_events(ctx), {ctx["rec_id"]: ctx["file_name"]}, ctx["formula"])
        stem = ctx["file_name"].rsplit(".", 1)[0]
        return dict(content=df.to_csv(index=False), filename=f"{stem}_event.csv", type="text/csv")

    @app.callback(Output("ctx", "data", allow_duplicate=True), Output("msg", "children", allow_duplicate=True),
                  Output("vis", "value", allow_duplicate=True),
                  Input("save-roles", "n_clicks"), State("role-opt", "value"), State("role-optref", "value"),
                  State("role-elec", "value"), State("ctx", "data"), State("vis", "value"),
                  prevent_initial_call=True)
    def save_roles(n, ro, rr, re_, ctx, vis):
        if not ctx:
            return no_update, no_update, no_update
        if len({ro, rr, re_}) < 3:
            return no_update, html.Span("Optical, OpticalRe and Electrical must be three different channels.",
                                        style={"color": "#b91c1c"}), no_update
        with db() as con:
            con.execute("UPDATE recordings SET role_opt=?, role_optref=?, role_elec=? WHERE id=?",
                        (ro, rr, re_, ctx["rec_id"]))
        ctx = dict(ctx)
        ctx["roles"] = {"opt": ro, "optref": rr, "elec": re_}
        vis = sorted(set(vis or []) | {ro, rr, re_})
        return ctx, "Channel roles saved for this recording (also used by the NN).", vis

    return app
