"""Deleting experiments, recordings, annotation sets, clustering runs and finished jobs.

What is removed for an experiment: its database rows (recordings, annotation sets, events, edit history,
clustering runs + labels, jobs), its Zarr display data, its clustering output folders and — if asked —
the ABF copies NanoTag keeps in <data>/abf/<experiment id>/. ABFs that were imported *in place*
(left in their original folder) are never deleted.
"""
import shutil
from pathlib import Path

from .config import cfg
from .db import db, jload, one, rows, tx
from . import store


def owned_abf(path) -> bool:
    """True if the ABF is NanoTag's own copy (inside the ABF store), i.e. safe to delete."""
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(cfg().ABF_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


def _size(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


def experiment_summary(exp_id):
    with db() as con:
        exp = one(con, "SELECT * FROM experiments WHERE id=?", (exp_id,))
        if exp is None:
            return None
        recs = rows(con, "SELECT id, abf_path, zarr_path, size_bytes, status FROM recordings WHERE experiment_id=?",
                    (exp_id,))
        n_sets = one(con, "SELECT COUNT(*) AS n FROM annotation_sets WHERE experiment_id=?", (exp_id,))["n"]
        n_events = one(con, """SELECT COUNT(*) AS n FROM events e JOIN annotation_sets s ON s.id=e.set_id
                               WHERE s.experiment_id=? AND e.deleted=0""", (exp_id,))["n"]
        n_runs = one(con, "SELECT COUNT(*) AS n FROM cluster_runs WHERE experiment_id=?", (exp_id,))["n"]
        active = one(con, "SELECT COUNT(*) AS n FROM jobs WHERE experiment_id=? AND status IN ('queued','running')",
                     (exp_id,))["n"]
    owned = [r for r in recs if owned_abf(r["abf_path"])]
    return {"exp": exp, "n_recordings": len(recs), "n_sets": n_sets, "n_events": n_events, "n_runs": n_runs,
            "active_jobs": active, "owned_abf": len(owned),
            "owned_bytes": sum(_size(r["abf_path"]) for r in owned),
            "inplace_abf": sum(1 for r in recs if r["abf_path"] and not owned_abf(r["abf_path"]))}


def delete_experiment(exp_id, delete_abf=True):
    c = cfg()
    with db() as con:
        con.execute("UPDATE jobs SET status='cancelled' WHERE experiment_id=? AND status='queued'", (exp_id,))
        recs = rows(con, "SELECT abf_path, zarr_path FROM recordings WHERE experiment_id=?", (exp_id,))
        runs = rows(con, "SELECT out_dir FROM cluster_runs WHERE experiment_id=?", (exp_id,))
    for r in recs:
        if r["zarr_path"]:
            store.delete_group(r["zarr_path"])
    for r in runs:
        if r["out_dir"]:
            shutil.rmtree(r["out_dir"], ignore_errors=True)
    with db() as con, tx(con):
        con.execute("DELETE FROM event_history WHERE set_id IN (SELECT id FROM annotation_sets WHERE experiment_id=?)",
                    (exp_id,))
        con.execute("DELETE FROM jobs WHERE experiment_id=? AND status!='running'", (exp_id,))
        con.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
    freed = 0
    if delete_abf:
        d = c.ABF_DIR / str(int(exp_id))
        if d.exists():
            freed = sum(_size(p) for p in d.rglob("*") if p.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


def delete_recording(rec, delete_abf=True):
    if rec["zarr_path"]:
        store.delete_group(rec["zarr_path"])
    with db() as con:
        con.execute("DELETE FROM recordings WHERE id=?", (rec["id"],))
    if delete_abf and owned_abf(rec["abf_path"]):
        Path(rec["abf_path"]).unlink(missing_ok=True)
        return True
    return False


def delete_set(set_id):
    with db() as con, tx(con):
        con.execute("DELETE FROM event_history WHERE set_id=?", (set_id,))
        con.execute("DELETE FROM annotation_sets WHERE id=?", (set_id,))


def runs_using_set(set_id):
    with db() as con:
        return [r["name"] for r in rows(con, "SELECT name, set_ids_json FROM cluster_runs")
                if int(set_id) in (jload(r["set_ids_json"], []) or [])]


def delete_run(run):
    if run.get("out_dir"):
        shutil.rmtree(run["out_dir"], ignore_errors=True)
    with db() as con:
        con.execute("DELETE FROM cluster_runs WHERE id=?", (run["id"],))
        if run.get("job_id"):
            con.execute("UPDATE jobs SET status='cancelled' WHERE id=? AND status='queued'", (run["job_id"],))


def clear_finished_jobs(exp_id=None, user=None):
    sql = "DELETE FROM jobs WHERE status IN ('done','error','cancelled')"
    args = []
    if exp_id is not None:
        sql += " AND experiment_id=?"
        args.append(exp_id)
    if user is not None:
        sql += " AND created_by=?"
        args.append(user)
    with db() as con:
        return con.execute(sql, args).rowcount
