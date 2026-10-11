"""Move a whole NanoTag server: export data + configuration on one machine, import it on another.

A *site bundle* is a plain .tar file:

    manifest.json                 what is inside, the source server's folders and settings, in-place ABF list
    db/nanotag.sqlite3            consistent snapshot of the database: user accounts (password hashes), experiments,
                                  recordings, annotation sets, events + edit history, models, jobs, clustering runs
    config/nanotag.env            the source server's NANOTAG_* settings (secret key left out)
    models/...                    NN checkpoints
    jobs/cluster_run_<id>/...     clustering outputs (CSV/XLSX/HTML report)
    zarr/store.zarr/...           optional: display data (saves re-ingesting every ABF)
    abf/<experiment id>/...       optional: NanoTag's own ABF copies
    abf-inplace/<absolute path>   optional: ABFs that were used in place (outside the data folder)
    incoming/...                  optional (with ABFs): files uploaded but not imported yet

`export_site()` streams it (web download or file); `import_site()` installs it on a new server, replacing that
server's data, and rewrites file paths for the new machine. Large parts (Zarr, ABFs) can instead be copied
with rsync into the staging folder by deploy/migrate-from.sh, which is resumable.
"""
import json
import os
import shutil
import socket
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path

from .config import cfg
from .db import SCHEMA_VERSION, connect, db, migrate, now, one, rows, tx

FORMAT = "nanotag-site"
FORMAT_VERSION = 1
BLOCK = 512
RECORD = 20 * BLOCK
CHUNK = 4 << 20
# settings worth carrying to the new server (paths, port, bind and secret key belong to the new machine)
PORTABLE_KEYS = ("NANOTAG_WORKERS", "NANOTAG_TORCH_THREADS", "NANOTAG_WEB_WORKERS", "NANOTAG_DISPLAY_FS",
                 "NANOTAG_CHUNK_SAMPLES", "NANOTAG_PYRAMID_FACTOR")


# ------------------------------------------------------------------ helpers
def _under(p, root):
    try:
        Path(os.path.abspath(p)).relative_to(os.path.abspath(root))
        return True
    except (ValueError, TypeError):
        return False


def _walk(root):
    root = Path(root)
    if not root.is_dir():
        return
    for d, _dirs, files in os.walk(root):
        _dirs.sort()
        for f in sorted(files):
            p = Path(d) / f
            if p.is_file() and not p.is_symlink():
                yield p, p.relative_to(root).as_posix()


def _size_of(root):
    return sum(p.stat().st_size for p, _ in _walk(root))


def _config_dict():
    """NANOTAG_* settings of this process (systemd loads them from /etc/nanotag/nanotag.env)."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("NANOTAG_") and k != "NANOTAG_SECRET_KEY"}


def _snapshot_db(dest):
    src = connect()
    dst = sqlite3.connect(str(dest))
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
        src.close()


def inplace_abfs():
    """Recordings whose ABF lives outside NanoTag's ABF store (imported 'in place')."""
    c = cfg()
    with db() as con:
        recs = rows(con, "SELECT id, experiment_id, file_name, abf_path, size_bytes FROM recordings "
                         "WHERE abf_path IS NOT NULL AND abf_path != ''")
    return [r for r in recs if not _under(r["abf_path"], c.ABF_DIR)]


def site_summary(sizes=True):
    """What an export would contain (for the admin page and the CLI)."""
    c = cfg()
    with db() as con:
        cnt = {k: one(con, f"SELECT COUNT(*) AS n FROM {t}")["n"] for k, t in (
            ("users", "users"), ("experiments", "experiments"), ("recordings", "recordings"),
            ("annotation_sets", "annotation_sets"), ("events", "events"), ("cluster_runs", "cluster_runs"),
            ("models", "models"), ("jobs", "jobs"))}
        abf_bytes = one(con, "SELECT COALESCE(SUM(size_bytes),0) AS b FROM recordings")["b"]
    out = {"counts": cnt}
    if sizes:
        inpl = inplace_abfs()
        out["bytes"] = {
            "database": c.DB_PATH.stat().st_size if c.DB_PATH.exists() else 0,
            "models": _size_of(c.MODELS_DIR),
            "cluster_outputs": sum(_size_of(d) for d in c.JOBS_DIR.glob("cluster_run_*") if d.is_dir()),
            "zarr": _size_of(c.ZARR_ROOT),
            "abf": _size_of(c.ABF_DIR) + sum(Path(r["abf_path"]).stat().st_size for r in inpl
                                             if Path(r["abf_path"]).is_file()),
            "abf_recorded": abf_bytes,
            "incoming": _size_of(c.INCOMING_DIR),
        }
        out["inplace_abf"] = len(inpl)
    return out


# ------------------------------------------------------------------ streaming tar writer
class _Tar:
    """Writes a POSIX tar as a stream of byte chunks (nothing is buffered beyond one chunk)."""

    def __init__(self):
        self.total = 0

    def _out(self, b):
        self.total += len(b)
        return b

    def bytes_entry(self, name, data, mtime=None):
        ti = tarfile.TarInfo(name)
        ti.size, ti.mtime, ti.mode = len(data), int(mtime or time.time()), 0o644
        yield self._out(ti.tobuf(format=tarfile.PAX_FORMAT))
        yield self._out(data)
        pad = (-len(data)) % BLOCK
        if pad:
            yield self._out(b"\0" * pad)

    def file_entry(self, name, path):
        try:
            st = os.stat(path)
            f = open(path, "rb")
        except OSError:
            return                                       # vanished since it was listed
        with f:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = st.st_size, int(st.st_mtime), 0o644
            yield self._out(ti.tobuf(format=tarfile.PAX_FORMAT))
            left = st.st_size
            while left > 0:
                b = f.read(min(CHUNK, left))
                if not b:
                    break
                left -= len(b)
                yield self._out(b)
            if left > 0:                                 # file shrank while reading: keep the archive valid
                yield self._out(b"\0" * left)
        pad = (-st.st_size) % BLOCK
        if pad:
            yield self._out(b"\0" * pad)

    def end(self):
        yield self._out(b"\0" * (2 * BLOCK))
        pad = (-self.total) % RECORD
        if pad:
            yield self._out(b"\0" * pad)


def export_site(include_zarr=False, include_abf=False, user="?"):
    """Generator of the site bundle's bytes."""
    c = cfg()
    tmpdir = Path(tempfile.mkdtemp(prefix="site_export_", dir=str(c.BACKUP_DIR)))
    try:
        snap = tmpdir / "nanotag.sqlite3"
        _snapshot_db(snap)
        inpl = inplace_abfs() if include_abf else []
        with db() as con:
            models = rows(con, "SELECT id, name, path FROM models")
        manifest = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "nanotag_version": c.VERSION,
            "schema_version": SCHEMA_VERSION, "created": now(), "created_by": user,
            "created_iso": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": socket.gethostname(),
            "includes": {"zarr": bool(include_zarr), "abf": bool(include_abf), "incoming": bool(include_abf)},
            "source": {"DATA": str(c.DATA), "DB_PATH": str(c.DB_PATH), "ZARR_ROOT": str(c.ZARR_ROOT),
                       "ABF_DIR": str(c.ABF_DIR), "JOBS_DIR": str(c.JOBS_DIR), "MODELS_DIR": str(c.MODELS_DIR),
                       "INCOMING_DIR": str(c.INCOMING_DIR), "IMPORT_ROOTS": [str(r) for r in c.IMPORT_ROOTS]},
            "config": _config_dict(),
            "counts": site_summary(sizes=False)["counts"],
            "inplace_abf": [{"recording_id": r["id"], "experiment_id": r["experiment_id"],
                             "file_name": r["file_name"], "path": r["abf_path"], "size": r["size_bytes"]}
                            for r in inplace_abfs()],
        }
        t = _Tar()
        yield from t.bytes_entry("manifest.json", json.dumps(manifest, indent=2).encode())
        env = "".join(f"{k}={v}\n" for k, v in manifest["config"].items())
        yield from t.bytes_entry("config/nanotag.env", ("# NanoTag settings exported from "
                                                        f"{manifest['host']} (secret key not included)\n"
                                                        + env).encode())
        yield from t.file_entry("db/nanotag.sqlite3", snap)
        seen = set()
        for p, rel in _walk(c.MODELS_DIR):
            seen.add(os.path.abspath(p))
            yield from t.file_entry(f"models/{rel}", p)
        for m in models:                                  # checkpoints registered from outside the models folder
            if m["path"] and os.path.abspath(m["path"]) not in seen and os.path.isfile(m["path"]):
                seen.add(os.path.abspath(m["path"]))
                yield from t.file_entry(f"models/{os.path.basename(m['path'])}", m["path"])
        for d in sorted(c.JOBS_DIR.glob("cluster_run_*")):
            if d.is_dir():
                for p, rel in _walk(d):
                    yield from t.file_entry(f"jobs/{d.name}/{rel}", p)
        if include_zarr:
            for p, rel in _walk(c.ZARR_ROOT):
                yield from t.file_entry(f"zarr/store.zarr/{rel}", p)
        if include_abf:
            for p, rel in _walk(c.ABF_DIR):
                yield from t.file_entry(f"abf/{rel}", p)
            for r in inpl:
                if os.path.isfile(r["abf_path"]):
                    yield from t.file_entry("abf-inplace/" + os.path.abspath(r["abf_path"]).lstrip("/"),
                                            r["abf_path"])
            for p, rel in _walk(c.INCOMING_DIR):
                yield from t.file_entry(f"incoming/{rel}", p)
        yield from t.end()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_site_to_file(path, include_zarr=False, include_abf=False, user="cli", log=print):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    n, t0, last = 0, time.time(), 0.0
    with open(tmp, "wb") as f:
        for b in export_site(include_zarr, include_abf, user):
            f.write(b)
            n += len(b)
            if time.time() - last > 5:
                last = time.time()
                log(f"  {n / 1e9:.2f} GB written ...")
    os.replace(tmp, path)
    log(f"Wrote {path} ({n / 1e9:.2f} GB in {time.time() - t0:.0f} s)")
    return path


# ------------------------------------------------------------------ import
def _extract(bundle, dest, log):
    log(f"Extracting {bundle} -> {dest}")
    with tarfile.open(bundle, "r:*") as tf:
        try:
            tf.extractall(dest, filter="data")          # refuses absolute paths, '..', devices, links out
        except TypeError:                               # Python without extraction filters
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                    raise ValueError(f"unsafe entry in bundle: {m.name}")
            tf.extractall(dest)


def _move_tree(src, dst, replace=True):
    """Move every file under src into dst (same relative paths). Returns the number of files moved."""
    n = 0
    for p, rel in list(_walk(src)):
        d = Path(dst) / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            if not replace:
                continue
            d.unlink()
        shutil.move(str(p), str(d))
        n += 1
    return n


def _db_counts(path):
    con = sqlite3.connect(str(path))
    try:
        def n(t):
            try:
                return con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.Error:
                return 0
        return {"experiments": n("experiments"), "users": n("users"),
                "schema": con.execute("PRAGMA user_version").fetchone()[0]}
    finally:
        con.close()


def import_site(source, force=False, keep_inplace=True, path_map=(), log=print):
    """Install a site bundle (a .tar file, or an already-extracted staging folder) on this server.

    This REPLACES this server's database (users, experiments, annotations ...) with the bundle's. The current
    database is backed up first. Services must be stopped while this runs (deploy/migrate-from.sh does that).
      force         also replace a server that already has experiments or extra users
      keep_inplace  ABFs used in place on the old server stay where they are if the same path exists here
      path_map      [(old_prefix, new_prefix)] rewrites for ABF paths (e.g. a data folder mounted elsewhere)
    Returns a report dict."""
    c = cfg()
    c.ensure_dirs()
    ts = time.strftime("%Y%m%d-%H%M%S")
    src = Path(source)
    own_staging = src.is_file()
    if own_staging:
        staging = c.DATA / f"import-staging-{ts}"
        staging.mkdir(parents=True)
        _extract(src, staging, log)
    elif src.is_dir():
        staging = src
    else:
        raise ValueError(f"{source} is neither a bundle file nor a folder")
    report = {"warnings": [], "staging": str(staging)}
    try:
        mf = staging / "manifest.json"
        if not mf.exists():
            raise ValueError("manifest.json not found — this is not a NanoTag site bundle "
                             "(experiment bundles are imported from the Experiments page instead)")
        man = json.loads(mf.read_text())
        if man.get("format") != FORMAT or int(man.get("format_version", 0)) > FORMAT_VERSION:
            raise ValueError(f"unsupported bundle format {man.get('format')} v{man.get('format_version')}")
        db_src = staging / "db" / "nanotag.sqlite3"
        if not db_src.exists():
            raise ValueError("the bundle has no database (db/nanotag.sqlite3)")
        theirs = _db_counts(db_src)
        if theirs["schema"] > SCHEMA_VERSION:
            raise ValueError(f"the bundle comes from a newer NanoTag (database schema v{theirs['schema']}; this "
                             f"server understands v{SCHEMA_VERSION}). Update this server first.")
        log(f"Bundle from {man.get('host')} (NanoTag {man.get('nanotag_version')}, {man.get('created_iso')}): "
            f"{theirs['users']} user(s), {theirs['experiments']} experiment(s)")

        ours = _db_counts(c.DB_PATH) if c.DB_PATH.exists() else {"experiments": 0, "users": 0}
        if (ours["experiments"] > 0 or ours["users"] > 1) and not force:
            raise ValueError(f"this server already has {ours['experiments']} experiment(s) and {ours['users']} "
                             f"user(s). Importing replaces all of it (a backup is made first). Re-run with --force "
                             f"to go ahead, or use experiment bundles to merge single experiments instead.")

        # 1) back up and replace the database
        if c.DB_PATH.exists():
            bk = c.BACKUP_DIR / f"nanotag-pre-import-{ts}.sqlite3"
            _snapshot_db(bk)
            log(f"Current database backed up to {bk}")
            report["backup"] = str(bk)
        tmp = c.DB_PATH.with_name(c.DB_PATH.name + ".importing")
        shutil.copy2(db_src, tmp)
        for suffix in ("", "-wal", "-shm"):
            Path(str(c.DB_PATH) + suffix).unlink(missing_ok=True)
        os.replace(tmp, c.DB_PATH)
        v = migrate()
        log(f"Database installed (schema v{v})")

        # 2) files
        n = _move_tree(staging / "models", c.MODELS_DIR)
        log(f"Models: {n} file(s)")
        n_runs = 0
        for d in sorted((staging / "jobs").glob("cluster_run_*")) if (staging / "jobs").is_dir() else []:
            dest = c.JOBS_DIR / d.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(d), str(dest))
            n_runs += 1
        log(f"Clustering outputs: {n_runs} run folder(s)")
        zsrc = staging / "zarr" / "store.zarr"
        if zsrc.is_dir():
            if c.ZARR_ROOT.exists() and any(c.ZARR_ROOT.iterdir()):
                old = c.ZARR_ROOT.with_name(f"{c.ZARR_ROOT.stem}.pre-import-{ts}{c.ZARR_ROOT.suffix}")
                c.ZARR_ROOT.rename(old)
                report["warnings"].append(f"the previous display store was kept as {old} (delete it when happy)")
            elif c.ZARR_ROOT.exists():
                c.ZARR_ROOT.rmdir()
            c.ZARR_ROOT.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(zsrc), str(c.ZARR_ROOT))
            log("Display data (Zarr) installed")
        n = _move_tree(staging / "abf", c.ABF_DIR)
        if n:
            log(f"ABF copies: {n} file(s)")
        n = _move_tree(staging / "incoming", c.INCOMING_DIR, replace=False)
        if n:
            log(f"Uploads not yet imported: {n} file(s)")

        # 3) rewrite paths in the database for this machine
        old = man.get("source", {})
        old_abf, old_models, old_jobs = old.get("ABF_DIR"), old.get("MODELS_DIR"), old.get("JOBS_DIR")
        from .jobs import abf_dest
        reingest, missing_abf, missing_rec = [], [], []
        with db() as con:
            models = rows(con, "SELECT id, path FROM models")
            runs = rows(con, "SELECT id, out_dir, status FROM cluster_runs")
            recs = rows(con, "SELECT id, experiment_id, file_name, abf_path, zarr_path, status FROM recordings")
        with db() as con, tx(con):
            for m in models:
                cand = c.MODELS_DIR / os.path.basename(m["path"] or "")
                if cand.is_file():
                    con.execute("UPDATE models SET path=? WHERE id=?", (str(cand), m["id"]))
                elif not os.path.isfile(m["path"] or ""):
                    report["warnings"].append(f"model #{m['id']} file missing: {m['path']}")
            for r in runs:
                d = c.JOBS_DIR / f"cluster_run_{r['id']}"
                if d.is_dir():
                    con.execute("UPDATE cluster_runs SET out_dir=? WHERE id=?", (str(d), r["id"]))
                elif r["status"] == "done":
                    report["warnings"].append(f"clustering run #{r['id']}: outputs not in the bundle")
            for r in recs:
                p = r["abf_path"] or ""
                newp = p
                if p and old_abf and _under(p, old_abf):
                    newp = str(c.ABF_DIR / Path(os.path.abspath(p)).relative_to(os.path.abspath(old_abf)))
                else:
                    for a, b in path_map:
                        if p.startswith(a):
                            newp = b + p[len(a):]
                            break
                    copied = staging / "abf-inplace" / os.path.abspath(p).lstrip("/") if p else None
                    if p and not (keep_inplace and os.path.isfile(newp)) and copied is not None and copied.is_file():
                        dest = abf_dest(r["experiment_id"], r["file_name"])
                        shutil.move(str(copied), str(dest))
                        newp = str(dest)
                if newp != p:
                    con.execute("UPDATE recordings SET abf_path=? WHERE id=?", (newp, r["id"]))
                has_abf = bool(newp) and os.path.isfile(newp)
                has_zarr = bool(r["zarr_path"]) and (c.ZARR_ROOT / r["zarr_path"]).is_dir()
                if r["status"] == "missing":
                    continue
                if not has_zarr and r["status"] != "error":
                    # no display data here: rebuild it from the ABF, or wait for the ABF to be imported again
                    if has_abf:
                        con.execute("UPDATE recordings SET zarr_path=NULL, status='pending', error=NULL WHERE id=?",
                                    (r["id"],))
                        reingest.append(r)
                    else:
                        con.execute("UPDATE recordings SET zarr_path=NULL, status='missing', error=? WHERE id=?",
                                    ("ABF not found after server move — import it again", r["id"]))
                        missing_rec.append(r["file_name"])
                if has_zarr and not has_abf:
                    missing_abf.append(r["file_name"])
            # jobs that were waiting or running on the old server are not carried over
            n_old = con.execute("UPDATE jobs SET status='cancelled', finished_at=?, "
                                "message='not carried over (server move)' "
                                "WHERE status IN ('queued','running','cancelling')", (now(),)).rowcount
            con.execute("UPDATE cluster_runs SET status='cancelled' WHERE status IN ('queued','running')")
            con.execute("UPDATE recordings SET status='pending' WHERE status='ingesting'")
        from .jobs import enqueue
        for r in reingest:
            enqueue("ingest", {"recording_id": r["id"]}, "migration", r["experiment_id"])
        if reingest:
            log(f"{len(reingest)} recording(s) queued for ingest (display data was not in the bundle)")
        if missing_rec:
            report["warnings"].append(f"{len(missing_rec)} recording(s) have neither display data nor an ABF here and "
                                      f"are marked 'missing' (import their ABFs again): "
                                      + ", ".join(missing_rec[:10]) + (" ..." if len(missing_rec) > 10 else ""))
        if missing_abf:
            report["warnings"].append(f"{len(missing_abf)} recording(s) can be tagged (display data is here) but "
                                      f"their ABF is not on this machine, so the NN cannot run on them: "
                                      + ", ".join(missing_abf[:10]) + (" ..." if len(missing_abf) > 10 else ""))
        if n_old:
            log(f"{n_old} job(s) that were queued/running on the old server marked cancelled")

        # 4) settings: saved next to the data, applied by migrate-from.sh --apply-config
        cfgfile = c.DATA / f"imported-config-{ts}.env"
        cfgfile.write_text("".join(f"{k}={v}\n" for k, v in (man.get("config") or {}).items()))
        report["config_file"] = str(cfgfile)
        report["portable_config"] = {k: v for k, v in (man.get("config") or {}).items() if k in PORTABLE_KEYS}
        roots = [r for r in (man.get("source", {}).get("IMPORT_ROOTS") or [])
                 if r != man.get("source", {}).get("INCOMING_DIR")]
        if roots:
            report["warnings"].append("the old server could read these folders; allow them here too if you copied "
                                      "them: " + ", ".join(roots) + "  (sudo deploy/add-import-root.sh DIR)")
        report.update({"users": theirs["users"], "experiments": theirs["experiments"],
                       "reingest": len(reingest)})
        log("Import finished.")
        shutil.rmtree(staging, ignore_errors=True)      # everything needed has been moved out of it
        return report
    except BaseException:
        if own_staging:                                 # a folder prepared by migrate-from.sh is kept for a retry
            shutil.rmtree(staging, ignore_errors=True)
        raise
