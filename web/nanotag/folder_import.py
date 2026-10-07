"""Import a whole folder: every ABF plus its event CSV, grouped into experiments by file name.

    2026_09_14_0001 B3S8-15 100 aM SiO2 DC.abf          ┐
    2026_09_14_0001 B3S8-15 100 aM SiO2 DC_event.csv    ┘ one recording (DC) + its events
    2026_09_14_0004 B3S8-15 100 aM SiO2 AC.abf  + _event.csv   -> same experiment, condition AC
    2026_09_14_0007 B3S8-15 100 aM SiO2 AOM.abf + _event.csv   -> same experiment, condition AOM
    2026_09_14_0000 B3S8-15 100 aM SiO2 baseline.abf            -> offered, not ticked (no events)

How an ABF finds its event file (first match wins):
  1. by name: <abf name>_event.csv / _events.csv / .csv / .xlsx, compared forgivingly (case, doubled or
     non-breaking spaces and '_' vs ' ' are ignored);
  2. by content: a CSV/XLSX whose file_name column names exactly that ABF (e.g. '20260914001B3S8-15100amSiO2.csv').

`scan()` builds the plan shown on the Import-a-folder page (nothing is changed);
`job_import_folder()` carries out the ticked part of the plan in the background worker. One bad file never
stops the others: every file is handled on its own and problems are listed in the job's result.
"""
import os
import shutil
from pathlib import Path

from . import annotations as A
from . import legacy, naming
from .config import cfg
from .db import db, one, rows

TABLE_EXTS = (".csv", ".xlsx", ".xls", ".xlsm")
DEFAULT_SET_NAME = "Event CSVs"
MODES = ("inplace", "copy", "link", "move")


def is_abf(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".abf") or (cfg().ALLOW_FAKE_ABF and n.endswith(".fakeabf.npz"))


def _read_table(path: Path):
    """(DataFrame or None, error or None)"""
    try:
        return legacy.read_table_bytes(path.read_bytes(), path.name), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _table_key(p: Path):
    """Normalised ABF key an event file belongs to by its name, e.g. '..._event.csv' -> '...'."""
    k = legacy.norm_name(p.name)
    for suf in (" events", " event"):
        if k.endswith(suf):
            return k[: -len(suf)].strip()
    return k


def _named_files(df):
    """Normalised ABF keys named in a table's file_name column."""
    if df is None:
        return set()
    col = next((c for c in df.columns if legacy._norm(c) in ("file_name", "filename")), None)
    if col is None:
        return set()
    return {legacy.norm_name(v) for v in df[col].dropna().astype(str) if str(v).strip()}


def list_folder(folder: Path, recursive=False):
    it = folder.rglob("*") if recursive else folder.iterdir()
    out = []
    for p in it:
        if p.name.startswith(".") or not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def pair_event_files(abfs, tables):
    """{abf path: (table path, how, DataFrame, error)} — by name first, then by the file_name column."""
    by_key = {}
    for t in tables:
        by_key.setdefault((str(t.parent), _table_key(t)), []).append(t)
    cache = {}

    def load(t):
        if t not in cache:
            cache[t] = _read_table(t)
        return cache[t]

    out, used = {}, set()
    for a in abfs:
        cands = by_key.get((str(a.parent), legacy.norm_name(a.name)), [])
        # prefer '<name>_event.csv' over other spellings, CSV over XLSX
        cands = sorted(cands, key=lambda t: ("event" not in t.name.lower(), t.suffix.lower() != ".csv", t.name))
        if cands:
            t = cands[0]
            df, err = load(t)
            out[a] = (t, "name", df, err)
            used.add(t)
    for a in abfs:          # content fallback for ABFs still without an event file
        if a in out:
            continue
        key = legacy.norm_name(a.name)
        for t in tables:
            if t in used or t.parent != a.parent:
                continue
            df, err = load(t)
            if err is None and _named_files(df) == {key}:
                out[a] = (t, "content", df, None)
                used.add(t)
                break
    return out, used, cache


def scan(folder, recursive=False, set_name=DEFAULT_SET_NAME):
    """Plan for importing `folder`. Nothing is modified."""
    folder = Path(folder)
    files = list_folder(folder, recursive)
    tables = [p for p in files if p.suffix.lower() in TABLE_EXTS]
    abfs = [p for p in files if is_abf(p)]
    paired, used, cache = pair_event_files(abfs, tables)
    with db() as con:
        exps = {e["name"]: e for e in rows(con, "SELECT id, name, restricted FROM experiments")}
        recs = {}
        for r in rows(con, "SELECT id, experiment_id, stem, file_name, status FROM recordings"):
            recs[(r["experiment_id"], legacy.norm_name(r["file_name"]))] = r
        sets = {(s["experiment_id"], s["name"]): s["id"]
                for s in rows(con, "SELECT id, experiment_id, name FROM annotation_sets")}
        n_events = {(e["set_id"], e["recording_id"]): e["n"] for e in
                    rows(con, "SELECT set_id, recording_id, COUNT(*) AS n FROM events WHERE deleted=0 "
                              "GROUP BY set_id, recording_id")}
    groups = {}
    for p in abfs:
        info = naming.parse_file_name(p.name)
        ev = None
        if p in paired:
            t, how, df, err = paired[p]
            ev = {"path": str(t), "name": t.name, "rows": None if df is None else len(df), "error": err,
                  "matched_by": how}
        exp = exps.get(info["experiment"])
        existing = None
        if exp:
            r = recs.get((exp["id"], legacy.norm_name(p.name)))
            if r:
                sid = sets.get((exp["id"], set_name))
                n_ev = n_events.get((sid, r["id"]), 0) if sid else 0
                existing = {"recording_id": r["id"], "status": r["status"], "n_events": n_ev,
                            "has_events": n_ev > 0}
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        usable = bool(ev and not ev["error"] and ev["rows"])
        if existing:
            existing["differs"] = bool(usable and existing["has_events"] and existing["n_events"] != ev["rows"])
        already = bool(existing and existing["status"] != "missing")
        events_needed = usable and not (existing and existing["has_events"])
        item = {"abf": str(p), "name": p.name, "rel": str(p.relative_to(folder)), "size": size,
                "condition": info["condition"], "run": info["run"], "experiment": info["experiment"],
                "events": ev, "existing": existing,
                "selected": usable and (not already or events_needed)}
        g = groups.setdefault(info["experiment"], {
            "experiment": info["experiment"], "exists": bool(exp), "experiment_id": exp["id"] if exp else None,
            "restricted": bool(exp and exp["restricted"]), "items": []})
        g["items"].append(item)
    for g in groups.values():
        g["items"].sort(key=lambda x: (x["run"] or "", x["name"].lower()))
        conds = {}
        for it in g["items"]:
            conds[it["condition"] or "other"] = conds.get(it["condition"] or "other", 0) + 1
        g["conditions"] = conds
    abf_keys = {legacy.norm_name(a.name): a.name for a in abfs}
    unused = []
    for t in tables:
        if t in used:
            continue
        df, err = cache.get(t) or _read_table(t)
        named = _named_files(df)
        why = (f"could not be read ({err})" if err else
               "no file_name column and its name matches no ABF" if not named else
               f"its rows name {len(named)} different files" if len(named) > 1 else
               f"duplicate: another event file is already paired with {abf_keys[next(iter(named))]}"
               if next(iter(named)) in abf_keys else
               "its file_name column names an ABF that is not in this folder")
        unused.append({"path": str(t), "name": t.name, "why": why})
    return {"folder": str(folder), "recursive": bool(recursive), "set_name": set_name,
            "groups": sorted(groups.values(), key=lambda g: g["experiment"].lower()),
            "unused_tables": sorted(unused, key=lambda x: x["name"].lower()),
            "n_abf": len(abfs)}


# ------------------------------------------------------------------ execution (worker)
def bring_abf(src: Path, exp_id: int, mode: str, log):
    """Put the ABF where NanoTag reads it from. 'inplace' leaves it where it is (never deleted by NanoTag)."""
    from .jobs import abf_dest
    if mode == "inplace":
        return src
    dest = abf_dest(exp_id, src.name)
    if dest.exists():
        log("  a copy is already in the NanoTag store; reusing it")
        return dest
    if mode == "move":
        shutil.move(str(src), str(dest))
    elif mode == "link":
        try:
            os.link(src, dest)
        except OSError:
            log("  hard link not possible -> copying")
            shutil.copy2(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest


def _find_recording(exp_id, file_name):
    key = legacy.norm_name(file_name)
    with db() as con:
        for r in rows(con, "SELECT * FROM recordings WHERE experiment_id=?", (exp_id,)):
            if legacy.norm_name(r["file_name"]) == key:
                return r
    return None


def _import_one(it, exp_id, set_id, set_name, mode, user, ctx, stats, replace=False):
    from .jobs import enqueue, register_recording
    src = Path(it["abf"])
    ex = _find_recording(exp_id, src.name)
    if ex and ex["status"] != "missing":
        rec_id = ex["id"]
        stats["reused"] += 1
        ctx.log(f"  {src.name}: already in this experiment (recording {rec_id}); ABF left as is")
    else:
        if not src.is_file():
            raise FileNotFoundError(f"ABF not found: {src}")
        dest = bring_abf(src, exp_id, mode, ctx.log)
        rec_id, _ = register_recording(exp_id, src.name, dest, user, replace=True, source_path=str(src))
        jid = enqueue("ingest", {"recording_id": rec_id}, user, exp_id)
        stats["recordings"] += 1
        stats["ingest_jobs"] += 1
        ctx.log(f"  {src.name}: recording {rec_id} ({'used in place' if mode == 'inplace' else mode}); "
                f"ingest job {jid} queued")
    evp = it.get("events")
    if not (evp and set_id):
        return
    with db() as con:
        have = one(con, "SELECT COUNT(*) AS n FROM events WHERE set_id=? AND recording_id=? AND deleted=0",
                   (set_id, rec_id))["n"]
    p = Path(evp)
    df = legacy.read_table_bytes(p.read_bytes(), p.name)
    if have and replace and have != len(df):
        with db() as con:
            old = [r["id"] for r in rows(con, "SELECT id FROM events WHERE set_id=? AND recording_id=? AND deleted=0",
                                         (set_id, rec_id))]
        A.delete_events(old, user)      # soft delete: stays in the edit history
        stats["events_replaced"] += len(old)
        ctx.log(f"    events: replacing {have} event(s) in '{set_name}' with the {len(df)} row(s) of {p.name}")
        have = 0
    if have:
        ctx.log(f"    events: '{set_name}' already has {have} event(s) for this recording -> not re-imported"
                + (f" (the file has {len(df)} rows; tick 'replace' to re-import)" if have != len(df) else ""))
        stats["events_existing"] += have
        return
    items_ev = legacy.frame_to_event_dicts(legacy.load_legacy_frame(df))
    key = legacy.norm_name(src.name)
    other = sorted({legacy.norm_name(r) for r in (df.get("file_name") if "file_name" in df else [])
                    if isinstance(r, str) and legacy.norm_name(r) != key})
    if other:
        ctx.log(f"    note: {p.name} names other file(s) in file_name ({', '.join(other[:3])}); "
                f"all rows are attached to {src.name} because the files pair up")
    evs = [ev for _, ev in items_ev]
    A.insert_events(set_id, rec_id, evs, user, action="import")
    stats["events"] += len(evs)
    stats["event_files"] += 1
    ctx.log(f"    events: {len(evs)} from {p.name}")


def job_import_folder(params, ctx, job):
    exp_id = int(params["experiment_id"])
    mode = params.get("mode", "inplace")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    user = job["created_by"]
    items = params["items"]
    set_name = (params.get("set_name") or DEFAULT_SET_NAME).strip()
    with db() as con:
        s = one(con, "SELECT * FROM annotation_sets WHERE experiment_id=? AND name=?", (exp_id, set_name))
    if s:
        set_id = s["id"]
        ctx.log(f"Adding events to existing annotation set '{set_name}' (id {set_id})")
    elif any(it.get("events") for it in items):
        set_id = A.create_set(exp_id, set_name, user, kind="import", formula=params.get("formula") or "tagger",
                              source={"folder": params.get("folder"),
                                      "files": [Path(it["events"]).name for it in items if it.get("events")]})
        ctx.log(f"Created annotation set '{set_name}' (id {set_id})")
    else:
        set_id = None
    stats = {"recordings": 0, "reused": 0, "events": 0, "event_files": 0, "events_existing": 0,
             "events_replaced": 0, "ingest_jobs": 0, "failed": 0}
    failures = []
    for i, it in enumerate(items):
        name = Path(it["abf"]).name
        ctx.progress(i / max(1, len(items)), f"{i + 1}/{len(items)}: {name}")
        try:
            _import_one(it, exp_id, set_id, set_name, mode, user, ctx, stats, replace=bool(params.get("replace")))
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the rest
            stats["failed"] += 1
            failures.append(f"{name}: {type(e).__name__}: {e}")
            ctx.log(f"  FAILED {name}: {type(e).__name__}: {e}")
    ctx.log(f"Done: {stats}")
    msg = (f"{stats['recordings'] + stats['reused']} recording(s), {stats['events']} event(s) from "
           f"{stats['event_files']} file(s)")
    if failures:
        msg += f"; {len(failures)} FAILED — see log"
    ctx.flush(msg)
    with db() as con:
        con.execute("UPDATE jobs SET message=? WHERE id=?", (msg[:500], job["id"]))
    return {"experiment_id": exp_id, "set_id": set_id, **stats, "failures": failures}


def plan_experiment_ids(groups, user, can_view):
    """Create missing experiments for the groups being imported. Returns {group name: exp_id}."""
    out = {}
    for name in groups:
        exp_id, _ = naming.get_or_create_experiment(name, user)
        with db() as con:
            exp = one(con, "SELECT * FROM experiments WHERE id=?", (exp_id,))
        if not can_view(exp):
            raise PermissionError(f"Experiment '{name}' exists but is restricted; ask an admin for access.")
        out[name] = exp_id
    return out


def enqueue_import(items, mode, set_name, user, can_view=lambda e: True, folder=None, formula="tagger",
                   replace=False):
    """items: [{'abf', 'events', 'experiment'}]. Returns [(exp_id, job_id, n_items)]."""
    from .jobs import enqueue
    by_exp = {}
    for it in items:
        by_exp.setdefault(it["experiment"].strip() or naming.guess_experiment_name(it["abf"]), []).append(it)
    ids = plan_experiment_ids(list(by_exp), user, can_view)
    out = []
    for name, lst in by_exp.items():
        jid = enqueue("import_folder", {"experiment_id": ids[name], "items": lst, "mode": mode,
                                        "set_name": set_name, "folder": folder, "formula": formula,
                                        "replace": bool(replace)},
                      user, ids[name])
        out.append((ids[name], jid, len(lst)))
    return out

