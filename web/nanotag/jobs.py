"""Background jobs: ingest, folder import, NN landmark detection, clustering, bundle import.

Jobs are rows in the `jobs` table. The worker (worker.py) claims queued jobs and
runs them in separate processes. Each job function receives a JobContext for
logging and progress; results are written to the database.
"""
import json
import os
import re
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import annotations as A
from . import legacy
from .config import cfg
from .db import db, jdump, jload, now, one, rows, tx


# ------------------------------------------------------------------ queue API
def enqueue(kind, params, user, experiment_id=None):
    with db() as con:
        cur = con.execute("INSERT INTO jobs(kind, experiment_id, params_json, status, created_by, created_at) "
                          "VALUES (?,?,?,?,?,?)", (kind, experiment_id, jdump(params), "queued", user, now()))
        return cur.lastrowid


def get_job(job_id):
    with db() as con:
        return one(con, "SELECT * FROM jobs WHERE id=?", (job_id,))


def list_jobs(experiment_id=None, limit=100):
    with db() as con:
        if experiment_id is None:
            return rows(con, "SELECT id, kind, experiment_id, status, progress, message, created_by, created_at, "
                             "started_at, finished_at FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        return rows(con, "SELECT id, kind, experiment_id, status, progress, message, created_by, created_at, "
                         "started_at, finished_at FROM jobs WHERE experiment_id=? ORDER BY id DESC LIMIT ?",
                    (experiment_id, limit))


def cancel_job(job_id):
    with db() as con:
        con.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                    (now(), job_id))


class JobContext:
    def __init__(self, job_id):
        self.job_id = job_id
        self._buf = []
        self._last = 0.0
        self._progress = 0.0

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._buf.append(line)
        if time.time() - self._last > 1.0:
            self.flush()

    def progress(self, frac, message=None):
        self._progress = float(max(0.0, min(1.0, frac)))
        if message:
            self.log(message)
        if time.time() - self._last > 1.0:
            self.flush(message)

    def flush(self, message=None):
        lines = self._buf
        self._buf = []
        self._last = time.time()
        with db() as con:
            if lines:
                con.execute("UPDATE jobs SET log=COALESCE(log,'') || ?, progress=? WHERE id=?",
                            ("\n".join(lines) + "\n", self._progress, self.job_id))
            else:
                con.execute("UPDATE jobs SET progress=? WHERE id=?", (self._progress, self.job_id))
            if message:
                con.execute("UPDATE jobs SET message=? WHERE id=?", (message[:500], self.job_id))


def run_job(job_id):
    """Entry point executed inside a worker process."""
    job = get_job(job_id)
    ctx = JobContext(job_id)
    params = jload(job["params_json"], {})
    fn = JOB_FUNCS.get(job["kind"])
    try:
        if fn is None:
            raise ValueError(f"Unknown job kind {job['kind']}")
        result = fn(params, ctx, job)
        ctx.progress(1.0)
        ctx.flush()
        with db() as con:
            con.execute("UPDATE jobs SET status='done', progress=1, finished_at=?, result_json=?, "
                        "message=COALESCE(NULLIF(message,''),'done') WHERE id=?",
                        (now(), jdump(result or {}), job_id))
    except Exception as e:
        ctx.log("ERROR: " + "".join(traceback.format_exception(e)))
        ctx.flush()
        with db() as con:
            con.execute("UPDATE jobs SET status='error', finished_at=?, message=? WHERE id=?",
                        (now(), f"{type(e).__name__}: {e}"[:500], job_id))
        _on_job_failed(job, params, e)


def _on_job_failed(job, params, err):
    with db() as con:
        if job["kind"] == "ingest":
            con.execute("UPDATE recordings SET status='error', error=? WHERE id=?",
                        (f"{type(err).__name__}: {err}"[:500], params.get("recording_id")))
        if job["kind"] == "cluster":
            con.execute("UPDATE cluster_runs SET status='error' WHERE id=?", (params.get("run_id"),))


# ------------------------------------------------------------------ ingest
def job_ingest(params, ctx, job):
    from .ingest import ingest_recording
    return ingest_recording(int(params["recording_id"]), log=ctx.log, progress=ctx.progress)


# ------------------------------------------------------------------ recordings
def safe_name(name):
    name = os.path.basename(name)
    name = re.sub(r"[\x00-\x1f/\\]", "_", name).strip()
    return name or "unnamed"


def abf_dest(exp_id, file_name):
    d = cfg().ABF_DIR / str(int(exp_id))
    d.mkdir(parents=True, exist_ok=True)
    return d / safe_name(file_name)


def register_recording(exp_id, file_name, abf_path, user, replace=False):
    """Create (or attach to a 'missing' placeholder) a recording row. Returns (rec_id, created)."""
    stem = legacy.file_stem(file_name)
    with db() as con, tx(con):
        ex = one(con, "SELECT * FROM recordings WHERE experiment_id=? AND stem=?", (exp_id, stem))
        if ex:
            if ex["status"] != "missing" and not replace:
                raise ValueError(f"A recording named '{file_name}' already exists in this experiment.")
            con.execute("UPDATE recordings SET abf_path=?, file_name=?, status='pending', error=NULL WHERE id=?",
                        (str(abf_path), file_name, ex["id"]))
            return ex["id"], False
        cur = con.execute("""INSERT INTO recordings(experiment_id, file_name, stem, abf_path, status,
                             created_by, created_at) VALUES (?,?,?,?, 'pending', ?, ?)""",
                          (exp_id, file_name, stem, str(abf_path), user, now()))
        return cur.lastrowid, True


def job_import_files(params, ctx, job):
    """Bring ABFs from an allowed server folder into the store, then queue ingests."""
    exp_id = int(params["experiment_id"])
    mode = params.get("mode", "copy")
    user = job["created_by"]
    files = params["files"]
    done = []
    for i, src in enumerate(files):
        src = Path(src)
        dest = abf_dest(exp_id, src.name)
        ctx.log(f"{mode}: {src} -> {dest}")
        if dest.exists():
            ctx.log("  destination exists; reusing it")
        elif mode == "move":
            shutil.move(str(src), str(dest))
        elif mode == "link":
            try:
                os.link(src, dest)
            except OSError:
                ctx.log("  hardlink not possible (different filesystem) -> copying")
                shutil.copy2(src, dest)
        else:
            shutil.copy2(src, dest)
        try:
            rec_id, _ = register_recording(exp_id, src.name, dest, user, replace=bool(params.get("replace")))
        except ValueError as e:
            ctx.log(f"  skipped: {e}")
            continue
        jid = enqueue("ingest", {"recording_id": rec_id}, user, exp_id)
        ctx.log(f"  recording {rec_id} registered; ingest job {jid} queued")
        done.append(rec_id)
        ctx.progress((i + 1) / len(files))
    return {"recordings": done}


# ------------------------------------------------------------------ NN
PRED_MAP = {
    "event_start": "event_start (s)", "event_end": "event_end (s)", "event_plateau": "event_plateau",
    "optical_base": "Base (V)", "optical_rise": "Step (V)",
    "opticalre_base": "RefBase (V)", "opticalre_rise": "RefStep (V)",
    "entry_base": "entry Base (pA)", "entry_peak": "entry Peak (pA)",
    "exit_peak": "exit Peak (pA)", "exit_base": "exit Base (pA)",
    "entry_base_t": "entry_base_t", "entry_peak_t": "entry_peak_t",
    "exit_peak_t": "exit_peak_t", "exit_base_t": "exit_base_t",
}


def _fnum(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def job_nn(params, ctx, job):
    import torch
    from . import vendor
    from .abfio import open_abf

    NN = vendor.neuralnetwork()
    NN.set_seed()
    torch.set_num_threads(max(1, cfg().TORCH_THREADS))
    src_set = A.get_set(int(params["source_set_id"]))
    exp_id = src_set["experiment_id"]
    with db() as con:
        model = one(con, "SELECT * FROM models WHERE id=?", (int(params["model_id"]),))
        recs = {r["id"]: r for r in rows(con, "SELECT * FROM recordings WHERE experiment_id=?", (exp_id,))}
    ctx.log(f"Model: {model['name']} ({model['path']}) on device {NN.DEVICE}")
    ck = torch.load(model["path"], map_location="cpu")
    net = NN.StrongOpt1D().to(NN.DEVICE).eval()
    net.load_state_dict(ck["model"])

    evs = A.list_events(src_set["id"])
    only = set(int(x) for x in params.get("recording_ids") or [])
    by_rec = {}
    for e in evs:
        if only and e["recording_id"] not in only:
            continue
        by_rec.setdefault(e["recording_id"], []).append(e)
    total = sum(len(v) for v in by_rec.values())
    if total == 0:
        raise ValueError("The source annotation set has no events to process.")

    out_name = A.unique_set_name(exp_id, params.get("out_name") or f"NN {model['name']} on {src_set['name']}")
    out_id = A.create_set(exp_id, out_name, job["created_by"], kind="nn", formula="nn",
                          source={"job_id": job["id"], "model": model["name"], "model_sha256": model["sha256"],
                                  "source_set_id": src_set["id"], "source_set_name": src_set["name"]})
    ctx.log(f"Output set: '{out_name}' (id {out_id}); {total} event(s) in {len(by_rec)} recording(s)")

    tmp = Path(tempfile.mkdtemp(prefix=f"nn_job{job['id']}_", dir=str(cfg().JOBS_DIR)))
    done = 0
    stats = {"ok": 0, "failed": 0, "no_window": 0}
    try:
        for rec_id, lst in by_rec.items():
            rec = recs[rec_id]
            base = legacy.file_stem(rec["file_name"])
            ctx.log(f"Recording {rec['file_name']}: opening ABF once for {len(lst)} event(s)")
            if not rec["abf_path"] or not os.path.exists(rec["abf_path"]):
                ctx.log("  ABF file missing -> events skipped")
                out = [_nn_blank(e, "abf not found") for e in lst]
                stats["failed"] += len(lst)
                A.insert_events(out_id, rec_id, out, job["created_by"], action="nn")
                done += len(lst)
                continue
            abf = open_abf(rec["abf_path"], load_data=True)
            fs = float(abf.dataRate)
            ranges = NN.resolve_sweep_range(abf)
            NN.USE_FIXED_CHANNELS = True
            NN.CH_ELEC, NN.CH_OPT, NN.CH_OPTREF = int(rec["role_elec"]), int(rec["role_opt"]), int(rec["role_optref"])
            out = []
            for e in lst:
                if e.get("window_start") is None or e.get("window_end") is None:
                    out.append(_nn_blank(e, "no window"))
                    stats["no_window"] += 1
                    continue
                row = {"event_id": e["event_no"], "file_name": rec["file_name"], "sensor": e.get("sensor"),
                       "analytes": e.get("analytes"), "solution": e.get("solution"),
                       "window_start": e["window_start"], "window_end": e["window_end"]}
                npz = NN.npz_path_for(str(tmp), base, e["event_no"])
                try:
                    NN.build_npz_for_row(abf, fs, ranges, base, row, npz)
                except Exception as ex:
                    out.append(_nn_blank(e, f"npz build failed: {ex}"))
                    stats["failed"] += 1
                    continue
                try:
                    pred = NN.run_inference_on_npz(net, npz)
                except Exception as ex:
                    out.append(_nn_blank(e, f"infer fail: {ex}"))
                    stats["failed"] += 1
                    continue
                ev = {k: _fnum(pred.get(v)) for k, v in PRED_MAP.items()}
                ev.update({"event_no": e["event_no"], "window_start": e["window_start"],
                           "window_end": e["window_end"], "sensor": e.get("sensor"),
                           "analytes": e.get("analytes"), "solution": e.get("solution"),
                           "notes": pred.get("notes")})
                ev["derived"] = {c: _fnum(pred.get(c)) for c in legacy.DERIVED_COLS}
                out.append(ev)
                stats["ok"] += 1
                done += 1
                if done % 5 == 0:
                    ctx.progress(done / total, f"{done}/{total} events")
            del abf
            A.insert_events(out_id, rec_id, out, job["created_by"], action="nn")
            ctx.log(f"  saved {len(out)} row(s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ctx.log(f"Done: {stats}")
    return {"set_id": out_id, "set_name": out_name, **stats}


def _nn_blank(e, note):
    return {"event_no": e["event_no"], "window_start": e.get("window_start"), "window_end": e.get("window_end"),
            "sensor": e.get("sensor"), "analytes": e.get("analytes"), "solution": e.get("solution"),
            "notes": note}


# ------------------------------------------------------------------ clustering
def job_cluster(params, ctx, job):
    from . import vendor
    CL = vendor.clustering()
    run_id = int(params["run_id"])
    with db() as con:
        run = one(con, "SELECT * FROM cluster_runs WHERE id=?", (run_id,))
        con.execute("UPDATE cluster_runs SET status='running' WHERE id=?", (run_id,))
    p = jload(run["params_json"], {})
    set_ids = jload(run["set_ids_json"], [])
    out_dir = cfg().JOBS_DIR / f"cluster_run_{run_id}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    in_dir = out_dir / "inputs"
    in_dir.mkdir(parents=True)

    # 1) Write one legacy-format CSV per (set, recording), exactly what clustering.py reads.
    paths, rowmap = [], {}
    multi = len(set_ids) > 1
    for sid in set_ids:
        s = A.get_set(sid)
        with db() as con:
            recs = {r["id"]: r for r in rows(con, "SELECT id, file_name, stem FROM recordings WHERE experiment_id=?",
                                             (s["experiment_id"],))}
            exp = one(con, "SELECT name FROM experiments WHERE id=?", (s["experiment_id"],))
        evs = A.list_events(sid)
        by_rec = {}
        for e in evs:
            by_rec.setdefault(e["recording_id"], []).append(e)
        for rec_id, lst in by_rec.items():
            rec = recs[rec_id]
            prefix = f"{_slug(exp['name'])}__{_slug(s['name'])}__" if multi else ""
            fname = f"{prefix}{rec['stem']}.csv"
            df = legacy.legacy_frame(lst, {rec_id: rec["file_name"]}, s["formula"])
            fp = in_dir / fname
            df.to_csv(fp, index=False)
            paths.append(str(fp))
            for i, e in enumerate(lst):
                rowmap[(fname, i + 2)] = (e["id"], rec_id)
    if not paths:
        raise ValueError("No events in the selected annotation set(s).")
    ctx.log(f"Prepared {len(paths)} input file(s), {len(rowmap)} event(s)")
    ctx.progress(0.05, "Running clustering (GMM sweep + bootstrap LRT) ...")

    # 2) Run the original engine.
    R = CL.run_clustering(
        paths=paths, alpha=float(p.get("alpha", 0.05)), max_k=int(p.get("max_k", 8)),
        duration_min=float(p.get("duration_min", 0.0)), duration_max=float(p.get("duration_max", 0.0)),
        n_boot=int(p.get("n_boot", 25)), impute_missing=bool(p.get("impute", True)),
        selected_features=p.get("features") or None, excluded_points=None,
        min_cluster_pct=float(p.get("min_cluster_pct", 0.0)), log=ctx.log)
    ctx.progress(0.85, f"Final K = {R['K']}; writing outputs ...")

    # 3) Outputs identical to clustering.py's "Save Results" + HTML report.
    R["labeled"].to_csv(out_dir / "pooled_cluster_labels.csv", index=False)
    R["summary"].to_csv(out_dir / "cluster_summary.csv", index=False)
    R["per_file"].to_csv(out_dir / "cluster_counts_by_file.csv", index=False)
    R["lrt"].to_csv(out_dir / "bootstrap_lrt.csv", index=False)
    R["k_breakdown"].to_csv(out_dir / "k_breakdown.csv", index=False)
    excluded_df = pd.DataFrame([], columns=["source_file", "source_row"])
    excluded_df.to_csv(out_dir / "manually_excluded_points.csv", index=False)
    hist_df = pd.DataFrame([])
    hist_df.to_csv(out_dir / "removed_cluster_history.csv", index=False)
    for source, sub in R["labeled"].groupby("source_file"):
        stem = os.path.splitext(os.path.basename(source))[0]
        sub.to_csv(out_dir / f"{stem}_clustered.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / "clustering_results.xlsx", engine="openpyxl") as xw:
            R["labeled"].to_excel(xw, sheet_name="labels", index=False)
            R["summary"].to_excel(xw, sheet_name="cluster_summary", index=False)
            R["per_file"].to_excel(xw, sheet_name="per_file", index=False)
            R["lrt"].to_excel(xw, sheet_name="LRT", index=False)
            R["k_breakdown"].to_excel(xw, sheet_name="k_breakdown", index=False)
            excluded_df.to_excel(xw, sheet_name="excluded_points", index=False)
            hist_df.to_excel(xw, sheet_name="removed_clusters", index=False)
    except Exception as e:
        ctx.log(f"xlsx write failed (CSV outputs are complete): {e}")
    feats = list(R["features"])
    xyz = (p.get("x") or (feats[0] if len(feats) > 0 else None),
           p.get("y") or (feats[1] if len(feats) > 1 else None),
           p.get("z") or (feats[2] if len(feats) > 2 else None))
    CL.build_html_report(R, *xyz, {}, str(out_dir / "cluster_report.html"),
                         run_meta={"sample_name": run["name"]}, names={})

    # 4) Labels back into the database.
    with db() as con, tx(con):
        con.execute("DELETE FROM cluster_labels WHERE run_id=?", (run_id,))
        for r in R["labeled"].itertuples(index=False):
            key = (str(r.source_file), int(r.source_row))
            if key in rowmap:
                eid, rid = rowmap[key]
                con.execute("INSERT OR REPLACE INTO cluster_labels(run_id, event_id, recording_id, cluster) "
                            "VALUES (?,?,?,?)", (run_id, eid, rid, int(r.cluster)))
        con.execute("""UPDATE cluster_runs SET status='done', final_k=?, k_bic=?, n_events=?, features_json=?,
                       out_dir=? WHERE id=?""",
                    (int(R["K"]), int(R["k_bic"]), int(len(R["labels"])), jdump(feats), str(out_dir), run_id))
    return {"run_id": run_id, "K": int(R["K"]), "k_bic": int(R["k_bic"]), "n_events": int(len(R["labels"]))}


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")[:60] or "x"


# ------------------------------------------------------------------ bundles
def job_import_bundle(params, ctx, job):
    """Import an experiment bundle (.zip made by 'Export experiment bundle')."""
    zpath = params["zip_path"]
    user = job["created_by"]
    with zipfile.ZipFile(zpath) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        exp = manifest["experiment"]
        name = exp["name"]
        with db() as con:
            existing = {r["name"] for r in rows(con, "SELECT name FROM experiments")}
        base, i = name, 2
        while name in existing:
            name = f"{base} (imported {i})"
            i += 1
        with db() as con:
            cur = con.execute("""INSERT INTO experiments(name, description, sensor, analytes, solution,
                                 restricted, created_by, created_at) VALUES (?,?,?,?,?,0,?,?)""",
                              (name, exp.get("description", ""), exp.get("sensor", ""), exp.get("analytes", ""),
                               exp.get("solution", ""), user, now()))
            exp_id = cur.lastrowid
        ctx.log(f"Created experiment '{name}' (id {exp_id})")
        names = set(zf.namelist())
        rec_ids = {}
        for r in manifest["recordings"]:
            member = f"abf/{r['file_name']}"
            if member in names:
                dest = abf_dest(exp_id, r["file_name"])
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
                rid, _ = register_recording(exp_id, r["file_name"], dest, user)
                with db() as con:
                    con.execute("UPDATE recordings SET role_elec=?, role_opt=?, role_optref=? WHERE id=?",
                                (r.get("role_elec", 0), r.get("role_opt", 2), r.get("role_optref", 3), rid))
                enqueue("ingest", {"recording_id": rid}, user, exp_id)
                ctx.log(f"  ABF {r['file_name']} extracted; ingest queued")
            else:
                with db() as con:
                    cur = con.execute("""INSERT INTO recordings(experiment_id, file_name, stem, abf_path, status,
                                         role_elec, role_opt, role_optref, created_by, created_at)
                                         VALUES (?,?,?,'', 'missing', ?,?,?,?,?)""",
                                      (exp_id, r["file_name"], legacy.file_stem(r["file_name"]),
                                       r.get("role_elec", 0), r.get("role_opt", 2), r.get("role_optref", 3),
                                       user, now()))
                    rid = cur.lastrowid
                ctx.log(f"  recording {r['file_name']} registered as 'missing' (import the ABF later)")
            rec_ids[r["file_name"]] = rid
        for s in manifest["sets"]:
            sid = A.create_set(exp_id, s["name"], s.get("created_by") or user, kind=s.get("kind", "manual"),
                               formula=s.get("formula", "tagger"), source=s.get("source") or {})
            evs = [json.loads(line) for line in zf.read(s["events_file"]).decode().splitlines() if line.strip()]
            by_rec = {}
            for e in evs:
                rid = rec_ids.get(e.pop("file_name", None))
                if rid is None:
                    continue
                by_rec.setdefault(rid, []).append(e)
            for rid, lst in by_rec.items():
                A.insert_events(sid, rid, lst, user, action="import")
            ctx.log(f"  set '{s['name']}': {sum(len(v) for v in by_rec.values())} event(s)")
    try:
        os.remove(zpath)
    except OSError:
        pass
    return {"experiment_id": exp_id}


JOB_FUNCS = {
    "ingest": job_ingest,
    "import_files": job_import_files,
    "nn": job_nn,
    "cluster": job_cluster,
    "import_bundle": job_import_bundle,
}
