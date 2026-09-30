"""Flask web application (pages, REST endpoints, uploads, exports) + the Dash Tagger."""
import io
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from functools import wraps
from pathlib import Path

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_file, stream_with_context, url_for)
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from . import annotations as A
from . import auth, jobs, legacy, store
from .config import cfg
from .db import EVENT_FIELDS, db, jdump, jload, migrate, now, one, rows, tx

CHUNK_SIZE = 16 * 1024 * 1024
ABF_EXTS = (".abf",)


def create_app():
    c = cfg()
    c.ensure_dirs()
    migrate()
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
                static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
    app.config.update(
        SECRET_KEY=c.SECRET_KEY,
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,   # forms/CSV; big files use the chunked uploader
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=60 * 60 * 12,
    )
    lm = LoginManager(app)
    lm.login_view = "login"

    @lm.user_loader
    def _load(uid):
        u = auth.get_user_by_id(uid)
        return u if u and u.is_active else None

    @app.before_request
    def _require_login():
        p = request.path
        if p.startswith("/static/") or p in ("/login", "/healthz"):
            return None
        if not current_user.is_authenticated:
            if p.startswith("/api/") or "/_dash-" in p:
                return jsonify({"error": "login required"}), 401
            return redirect(url_for("login", next=request.full_path))
        return None

    @app.context_processor
    def _inject():
        return {
            "version": c.VERSION,
            "default_admin_pw": (current_user.is_authenticated and current_user.is_admin
                                 and auth.default_admin_password_in_use()),
            "fmt_ts": fmt_ts, "fmt_bytes": fmt_bytes, "fmt_dur": fmt_dur,
        }

    _register_routes(app)
    from .tagger import init_tagger
    init_tagger(app)
    return app


# ----------------------------------------------------------------- helpers
def fmt_ts(t):
    if not t:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(t)))


def fmt_bytes(n):
    if not n:
        return ""
    n = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_dur(s):
    if s is None:
        return ""
    s = float(s)
    return f"{int(s // 60)}:{s % 60:06.3f}" if s >= 60 else f"{s:.3f} s"


def admin_required(f):
    @wraps(f)
    def w(*a, **kw):
        if not current_user.is_admin:
            abort(403)
        return f(*a, **kw)
    return w


def get_exp_or_404(exp_id):
    with db() as con:
        exp = one(con, "SELECT * FROM experiments WHERE id=?", (exp_id,))
    if exp is None or not auth.can_view_experiment(current_user, exp):
        abort(404)
    return exp


def get_set_or_404(set_id, edit=False):
    s = A.get_set(set_id)
    if s is None:
        abort(404)
    get_exp_or_404(s["experiment_id"])
    if edit and not auth.can_edit_set(current_user, s):
        abort(403)
    return s


def get_rec_or_404(rec_id):
    with db() as con:
        r = one(con, "SELECT * FROM recordings WHERE id=?", (rec_id,))
    if r is None:
        abort(404)
    get_exp_or_404(r["experiment_id"])
    return r


def recordings_of(exp_id):
    with db() as con:
        return rows(con, "SELECT * FROM recordings WHERE experiment_id=? ORDER BY file_name", (exp_id,))


def back(default="/"):
    return redirect(request.form.get("next") or request.referrer or default)


def _download_name(s):
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", s).strip() or "export"


def _within_roots(p: Path):
    p = p.resolve()
    for r in cfg().IMPORT_ROOTS:
        try:
            p.relative_to(r.resolve())
            return True
        except ValueError:
            continue
    return False


# ----------------------------------------------------------------- zip streaming
class _ZipSink(io.RawIOBase):
    def __init__(self):
        self.parts = []

    def writable(self):
        return True

    def write(self, b):
        self.parts.append(bytes(b))
        return len(b)

    def take(self):
        out = b"".join(self.parts)
        self.parts = []
        return out


def stream_zip(entries):
    """entries: iterable of (arcname, bytes | path). Yields a zip file progressively (no temp file)."""
    sink = _ZipSink()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for arcname, content in entries:
            if isinstance(content, (bytes, bytearray)):
                zf.writestr(arcname, content)
            else:
                zi = zipfile.ZipInfo.from_file(str(content), arcname)
                zi.compress_type = zipfile.ZIP_STORED
                with open(content, "rb") as src, zf.open(zi, "w", force_zip64=True) as dst:
                    while True:
                        b = src.read(8 << 20)
                        if not b:
                            break
                        dst.write(b)
                        data = sink.take()
                        if data:
                            yield data
            data = sink.take()
            if data:
                yield data
    yield sink.take()


# ----------------------------------------------------------------- exports shared
def set_export_frame(s, layout="legacy", audit=False, recording_id=None):
    evs = A.list_events(s["id"], recording_id)
    recs = recordings_of(s["experiment_id"])
    fnames = {r["id"]: r["file_name"] for r in recs}
    if layout == "nn":
        return legacy.nn_frame(evs, fnames)
    return legacy.legacy_frame(evs, fnames, s["formula"], audit=audit)


def history_frame(s):
    import pandas as pd
    recs = {r["id"]: r["file_name"] for r in recordings_of(s["experiment_id"])}
    out = []
    for h in A.history(s["id"]):
        out.append({"when": fmt_ts(h["at"]), "user": h["username"], "action": h["action"],
                    "db_event_id": h["event_id"], "event_id": h["event_no"],
                    "file_name": recs.get(h["recording_id"], ""),
                    "before": h["before_json"] or "", "after": h["after_json"] or ""})
    return pd.DataFrame(out, columns=["when", "user", "action", "db_event_id", "event_id", "file_name",
                                      "before", "after"])


# ================================================================= routes
def _register_routes(app):
    c = cfg()

    @app.get("/healthz")
    def healthz():
        with db() as con:
            con.execute("SELECT 1")
        return jsonify({"ok": True, "version": c.VERSION})

    # ------------------------------------------------------------- auth
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            u = auth.authenticate(request.form.get("username", "").strip(), request.form.get("password", ""))
            if u:
                login_user(u, remember=bool(request.form.get("remember")))
                nxt = request.args.get("next") or "/"
                if not nxt.startswith("/"):
                    nxt = "/"
                return redirect(nxt)
            flash("Invalid username or password.", "error")
        return render_template("login.html")

    @app.get("/logout")
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.route("/account", methods=["GET", "POST"])
    def account():
        if request.method == "POST":
            if not auth.authenticate(current_user.username, request.form.get("current", "")):
                flash("Current password is incorrect.", "error")
            elif request.form.get("new1") != request.form.get("new2"):
                flash("New passwords do not match.", "error")
            else:
                try:
                    auth.set_password(current_user.username, request.form.get("new1", ""))
                    flash("Password changed.", "ok")
                except ValueError as e:
                    flash(str(e), "error")
        return render_template("account.html")

    # ------------------------------------------------------------- experiments
    @app.get("/")
    def index():
        return redirect(url_for("experiments"))

    @app.get("/experiments")
    def experiments():
        exps = auth.visible_experiments(current_user)
        with db() as con:
            stats = {r["experiment_id"]: r for r in rows(con, """SELECT experiment_id, COUNT(*) AS n,
                     SUM(COALESCE(size_bytes,0)) AS bytes, SUM(status='ready') AS ready
                     FROM recordings GROUP BY experiment_id""")}
        return render_template("experiments.html", exps=exps, stats=stats)

    @app.post("/experiments/new")
    def experiment_new():
        name = request.form.get("name", "").strip()
        if not name:
            flash("Experiment name is required.", "error")
            return redirect(url_for("experiments"))
        try:
            with db() as con:
                cur = con.execute("""INSERT INTO experiments(name, description, sensor, analytes, solution,
                                     created_by, created_at) VALUES (?,?,?,?,?,?,?)""",
                                  (name, request.form.get("description", ""), request.form.get("sensor", ""),
                                   request.form.get("analytes", ""), request.form.get("solution", ""),
                                   current_user.username, now()))
                exp_id = cur.lastrowid
        except Exception:
            flash(f"An experiment named '{name}' already exists.", "error")
            return redirect(url_for("experiments"))
        A.create_set(exp_id, "Manual tags", current_user.username)
        return redirect(url_for("experiment", exp_id=exp_id))

    @app.get("/experiments/<int:exp_id>")
    def experiment(exp_id):
        exp = get_exp_or_404(exp_id)
        recs = recordings_of(exp_id)
        for r in recs:
            r["channels"] = jload(r["channels_json"], [])
        sets = A.list_sets(exp_id)
        with db() as con:
            runs = rows(con, "SELECT * FROM cluster_runs WHERE experiment_id=? ORDER BY id DESC", (exp_id,))
            models = rows(con, "SELECT * FROM models ORDER BY is_default DESC, name")
            access = [r["username"] for r in rows(con, "SELECT username FROM experiment_access WHERE experiment_id=?",
                                                  (exp_id,))]
        all_sets = []
        for e in auth.visible_experiments(current_user):
            for s in A.list_sets(e["id"]):
                all_sets.append({"id": s["id"], "label": f"{e['name']} / {s['name']} ({s['n_events']})",
                                 "exp_id": e["id"]})
        for r in runs:
            r["set_names"] = ", ".join(next((x["label"] for x in all_sets if x["id"] == sid), str(sid))
                                       for sid in jload(r["set_ids_json"], []))
        users = auth.list_users() if current_user.is_admin else []
        return render_template("experiment.html", exp=exp, recs=recs, sets=sets, runs=runs, models=models,
                               all_sets=all_sets, access=access, users=users,
                               jobs_list=jobs.list_jobs(exp_id, 30), roots=[str(r) for r in c.IMPORT_ROOTS],
                               features=["duration", "OSC", "RefOSC", "entry_peak", "exit_peak",
                                         "entry_spike", "exit_spike"],
                               default_features=["duration", "OSC", "RefOSC", "entry_spike", "exit_spike"])

    @app.get("/api/experiments")
    def api_experiments():
        return jsonify([{k: e[k] for k in ("id", "name", "description", "sensor", "analytes", "solution",
                                           "restricted", "created_by", "created_at")}
                        for e in auth.visible_experiments(current_user)])

    @app.get("/api/experiments/<int:exp_id>")
    def api_experiment(exp_id):
        exp = get_exp_or_404(exp_id)
        recs = [{k: r[k] for k in ("id", "file_name", "status", "error", "fs", "duration", "size_bytes",
                                   "role_elec", "role_opt", "role_optref")} | {"channels": jload(r["channels_json"], [])}
                for r in recordings_of(exp_id)]
        sets = [{k: s[k] for k in ("id", "name", "kind", "formula", "locked", "n_events", "created_by", "created_at")}
                for s in A.list_sets(exp_id)]
        with db() as con:
            runs = rows(con, "SELECT id, name, status, final_k, k_bic, n_events, created_by, created_at, job_id "
                             "FROM cluster_runs WHERE experiment_id=? ORDER BY id", (exp_id,))
            models = rows(con, "SELECT id, name, is_default FROM models ORDER BY id")
        return jsonify({"experiment": exp, "recordings": recs, "sets": sets, "cluster_runs": runs, "models": models})

    @app.post("/experiments/<int:exp_id>/edit")
    def experiment_edit(exp_id):
        get_exp_or_404(exp_id)
        with db() as con:
            con.execute("UPDATE experiments SET name=?, description=?, sensor=?, analytes=?, solution=? WHERE id=?",
                        (request.form["name"].strip(), request.form.get("description", ""),
                         request.form.get("sensor", ""), request.form.get("analytes", ""),
                         request.form.get("solution", ""), exp_id))
        flash("Experiment updated.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/access")
    @admin_required
    def experiment_access(exp_id):
        restricted = 1 if request.form.get("restricted") else 0
        names = [u.strip() for u in request.form.getlist("users") if u.strip()]
        with db() as con, tx(con):
            con.execute("UPDATE experiments SET restricted=? WHERE id=?", (restricted, exp_id))
            con.execute("DELETE FROM experiment_access WHERE experiment_id=?", (exp_id,))
            for n in names:
                con.execute("INSERT OR IGNORE INTO experiment_access(experiment_id, username) VALUES (?,?)",
                            (exp_id, n))
        flash("Access updated.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/delete")
    @admin_required
    def experiment_delete(exp_id):
        exp = get_exp_or_404(exp_id)
        if request.form.get("confirm") != exp["name"]:
            flash("Type the experiment name exactly to confirm deletion.", "error")
            return back()
        for r in recordings_of(exp_id):
            if r["zarr_path"]:
                store.delete_group(r["zarr_path"])
        with db() as con, tx(con):
            con.execute("DELETE FROM event_history WHERE set_id IN (SELECT id FROM annotation_sets WHERE experiment_id=?)",
                        (exp_id,))
            con.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
        if request.form.get("delete_abf"):
            shutil.rmtree(c.ABF_DIR / str(exp_id), ignore_errors=True)
        flash(f"Experiment '{exp['name']}' deleted.", "ok")
        return redirect(url_for("experiments"))

    # ------------------------------------------------------------- chunked uploads
    def _up_paths(uid):
        if not re.fullmatch(r"[0-9a-f]{32}", uid or ""):
            abort(400)
        return c.UPLOAD_DIR / f"{uid}.json", c.UPLOAD_DIR / f"{uid}.part"

    @app.post("/api/uploads/init")
    def upload_init():
        d = request.get_json(force=True)
        purpose = d.get("purpose")
        name = jobs.safe_name(d.get("filename", ""))
        size = int(d.get("size", 0))
        if purpose == "abf":
            get_exp_or_404(int(d["experiment_id"]))
            if not name.lower().endswith(ABF_EXTS) and not (c.ALLOW_FAKE_ABF and name.endswith(".fakeabf.npz")):
                return jsonify({"error": "Only .abf files can be uploaded as recordings."}), 400
        elif purpose == "bundle":
            if not name.lower().endswith(".zip"):
                return jsonify({"error": "Bundles must be .zip files."}), 400
        elif purpose == "model":
            if not current_user.is_admin:
                abort(403)
            if not name.lower().endswith(".pt"):
                return jsonify({"error": "Models must be .pt files."}), 400
        else:
            return jsonify({"error": "bad purpose"}), 400
        free = shutil.disk_usage(c.UPLOAD_DIR).free
        if size * 1.1 > free:
            return jsonify({"error": f"Not enough disk space ({fmt_bytes(free)} free)."}), 507
        # resume: same user, file, size, purpose, experiment and not yet completed
        for mf in c.UPLOAD_DIR.glob("*.json"):
            try:
                m = json.loads(mf.read_text())
            except Exception:
                continue
            if (m.get("user") == current_user.username and m.get("filename") == name and m.get("size") == size
                    and m.get("purpose") == purpose and m.get("experiment_id") == d.get("experiment_id")):
                return jsonify({"upload_id": m["id"], "chunk_size": m["chunk_size"], "received": m["received"]})
        uid = uuid.uuid4().hex
        meta = {"id": uid, "user": current_user.username, "filename": name, "size": size, "purpose": purpose,
                "experiment_id": d.get("experiment_id"), "chunk_size": CHUNK_SIZE, "received": [],
                "created": now(), "replace": bool(d.get("replace"))}
        mp, pp = _up_paths(uid)
        with open(pp, "wb") as f:
            f.truncate(size)
        mp.write_text(json.dumps(meta))
        return jsonify({"upload_id": uid, "chunk_size": CHUNK_SIZE, "received": []})

    @app.put("/api/uploads/<uid>/chunk")
    def upload_chunk(uid):
        mp, pp = _up_paths(uid)
        if not mp.exists():
            abort(404)
        meta = json.loads(mp.read_text())
        if meta["user"] != current_user.username:
            abort(403)
        idx = int(request.args["index"])
        off = idx * meta["chunk_size"]
        expected = min(meta["chunk_size"], meta["size"] - off)
        if expected <= 0:
            abort(400)
        with open(pp, "r+b") as f:
            f.seek(off)
            got = 0
            while True:
                b = request.stream.read(1 << 20)
                if not b:
                    break
                f.write(b)
                got += len(b)
        if got != expected:
            return jsonify({"error": f"chunk {idx}: got {got} bytes, expected {expected}"}), 400
        if idx not in meta["received"]:
            meta["received"].append(idx)
        tmp = mp.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, mp)
        return jsonify({"ok": True, "received": len(meta["received"])})

    @app.post("/api/uploads/<uid>/complete")
    def upload_complete(uid):
        mp, pp = _up_paths(uid)
        if not mp.exists():
            abort(404)
        meta = json.loads(mp.read_text())
        if meta["user"] != current_user.username:
            abort(403)
        n_chunks = max(1, math.ceil(meta["size"] / meta["chunk_size"]))
        missing = sorted(set(range(n_chunks)) - set(meta["received"]))
        if missing and meta["size"] > 0:
            return jsonify({"error": "incomplete", "missing": missing[:20]}), 409
        purpose = meta["purpose"]
        try:
            if purpose == "abf":
                exp_id = int(meta["experiment_id"])
                dest = jobs.abf_dest(exp_id, meta["filename"])
                if dest.exists() and not meta.get("replace"):
                    return jsonify({"error": f"{meta['filename']} already exists in this experiment."}), 409
                rec_id, _ = jobs.register_recording(exp_id, meta["filename"], dest, current_user.username,
                                                    replace=meta.get("replace", False))
                os.replace(pp, dest)
                jid = jobs.enqueue("ingest", {"recording_id": rec_id}, current_user.username, exp_id)
                result = {"recording_id": rec_id, "job_id": jid}
            elif purpose == "bundle":
                dest = c.UPLOAD_DIR / f"{uid}.zip"
                os.replace(pp, dest)
                jid = jobs.enqueue("import_bundle", {"zip_path": str(dest)}, current_user.username)
                result = {"job_id": jid}
            else:  # model
                result = _register_model(pp, meta["filename"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 409
        mp.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        return jsonify({"ok": True, **result})

    def _register_model(path, filename):
        import hashlib
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        name = Path(filename).stem
        dest = c.MODELS_DIR / f"{name}_{h[:8]}.pt"
        shutil.move(str(path), dest)
        with db() as con:
            base, i = name, 2
            while one(con, "SELECT id FROM models WHERE name=?", (name,)):
                name = f"{base}_{i}"
                i += 1
            first = one(con, "SELECT COUNT(*) AS n FROM models")["n"] == 0
            cur = con.execute("INSERT INTO models(name, path, sha256, is_default, uploaded_by, created_at) "
                              "VALUES (?,?,?,?,?,?)", (name, str(dest), h, 1 if first else 0,
                                                       current_user.username, now()))
        return {"model_id": cur.lastrowid, "name": name}

    # ------------------------------------------------------------- server-folder import
    @app.get("/api/browse")
    def browse():
        p = request.args.get("path") or ""
        if not p:
            return jsonify({"path": "", "dirs": [str(r) for r in c.IMPORT_ROOTS if r.exists()], "files": []})
        path = Path(p)
        if not path.exists() or not _within_roots(path):
            return jsonify({"error": "Path is outside the allowed import folders."}), 403
        dirs, files = [], []
        try:
            for e in sorted(path.iterdir(), key=lambda x: x.name.lower()):
                if e.name.startswith("."):
                    continue
                if e.is_dir():
                    dirs.append(str(e))
                elif e.suffix.lower() in ABF_EXTS or (c.ALLOW_FAKE_ABF and e.name.endswith(".fakeabf.npz")):
                    files.append({"path": str(e), "name": e.name, "size": e.stat().st_size})
        except PermissionError:
            return jsonify({"error": "Permission denied (the nanotag service user cannot read this folder)."}), 403
        parent = str(path.parent) if _within_roots(path.parent) else ""
        return jsonify({"path": str(path), "parent": parent, "dirs": dirs, "files": files})

    @app.post("/experiments/<int:exp_id>/import-folder")
    def import_folder(exp_id):
        get_exp_or_404(exp_id)
        files = request.form.getlist("files")
        bad = [f for f in files if not _within_roots(Path(f))]
        if not files or bad:
            flash("Select at least one ABF inside an allowed import folder.", "error")
            return back()
        jid = jobs.enqueue("import_files", {"experiment_id": exp_id, "files": files,
                                            "mode": request.form.get("mode", "copy"),
                                            "replace": bool(request.form.get("replace"))},
                           current_user.username, exp_id)
        flash(f"Import job #{jid} queued for {len(files)} file(s).", "ok")
        return back()

    # ------------------------------------------------------------- recordings
    @app.post("/recordings/<int:rec_id>/roles")
    def recording_roles(rec_id):
        r = get_rec_or_404(rec_id)
        vals = [int(request.form[k]) for k in ("role_elec", "role_opt", "role_optref")]
        n = len(jload(r["channels_json"], [])) or 6
        if any(v < 0 or v >= n for v in vals):
            flash("Channel index out of range.", "error")
            return back()
        with db() as con:
            con.execute("UPDATE recordings SET role_elec=?, role_opt=?, role_optref=? WHERE id=?", (*vals, rec_id))
        flash(f"Channel roles saved for {r['file_name']}.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/roles-all")
    def recording_roles_all(exp_id):
        get_exp_or_404(exp_id)
        vals = [int(request.form[k]) for k in ("role_elec", "role_opt", "role_optref")]
        with db() as con:
            con.execute("UPDATE recordings SET role_elec=?, role_opt=?, role_optref=? WHERE experiment_id=?",
                        (*vals, exp_id))
        flash("Channel roles applied to all recordings.", "ok")
        return back()

    @app.post("/recordings/<int:rec_id>/reingest")
    def recording_reingest(rec_id):
        r = get_rec_or_404(rec_id)
        jid = jobs.enqueue("ingest", {"recording_id": rec_id}, current_user.username, r["experiment_id"])
        flash(f"Re-ingest job #{jid} queued.", "ok")
        return back()

    @app.post("/recordings/<int:rec_id>/delete")
    @admin_required
    def recording_delete(rec_id):
        r = get_rec_or_404(rec_id)
        if r["zarr_path"]:
            store.delete_group(r["zarr_path"])
        with db() as con:
            con.execute("DELETE FROM recordings WHERE id=?", (rec_id,))
        if request.form.get("delete_abf") and r["abf_path"] and Path(r["abf_path"]).is_relative_to(c.ABF_DIR):
            Path(r["abf_path"]).unlink(missing_ok=True)
        flash(f"Recording {r['file_name']} removed.", "ok")
        return back()

    @app.get("/recordings/<int:rec_id>/abf")
    def recording_abf(rec_id):
        r = get_rec_or_404(rec_id)
        if not r["abf_path"] or not os.path.exists(r["abf_path"]):
            abort(404)
        return send_file(r["abf_path"], as_attachment=True, download_name=r["file_name"], conditional=True)

    # ------------------------------------------------------------- annotation sets
    @app.post("/experiments/<int:exp_id>/sets/new")
    def set_new(exp_id):
        get_exp_or_404(exp_id)
        name = request.form.get("name", "").strip()
        try:
            src = request.form.get("copy_from")
            if src:
                get_set_or_404(int(src))
                A.copy_set(int(src), name, current_user.username)
            else:
                A.create_set(exp_id, name, current_user.username)
            flash(f"Annotation set '{name}' created.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return back()

    @app.post("/sets/<int:set_id>/lock")
    @admin_required
    def set_lock(set_id):
        get_set_or_404(set_id)
        A.set_locked(set_id, request.form.get("locked") == "1")
        return back()

    @app.post("/sets/<int:set_id>/rename")
    def set_rename(set_id):
        get_set_or_404(set_id, edit=True)
        try:
            A.rename_set(set_id, request.form["name"])
        except Exception:
            flash("A set with that name already exists.", "error")
        return back()

    @app.post("/sets/<int:set_id>/delete")
    @admin_required
    def set_delete(set_id):
        s = get_set_or_404(set_id)
        A.delete_set(set_id)
        flash(f"Annotation set '{s['name']}' deleted.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/import-events")
    def import_events(exp_id):
        get_exp_or_404(exp_id)
        files = [f for f in request.files.getlist("files") if f and f.filename]
        if not files:
            flash("Choose one or more CSV/XLSX files.", "error")
            return back()
        target = request.form.get("target_set") or ""
        user = current_user.username
        if target:
            s = get_set_or_404(int(target), edit=True)
            set_id = s["id"]
        else:
            name = request.form.get("new_name", "").strip() or A.unique_set_name(
                exp_id, "Imported " + time.strftime("%Y-%m-%d %H:%M"))
            try:
                set_id = A.create_set(exp_id, name, user, kind="import",
                                      formula=request.form.get("formula", "tagger"),
                                      source={"files": [f.filename for f in files]})
            except ValueError as e:
                flash(str(e), "error")
                return back()
        recs = {r["stem"]: r for r in recordings_of(exp_id)}
        default_rec = request.form.get("default_recording")
        n_ok, unmatched = 0, {}
        for f in files:
            try:
                df = legacy.read_table_bytes(f.read(), f.filename)
            except Exception as e:
                flash(f"{f.filename}: {e}", "error")
                continue
            items = legacy.frame_to_event_dicts(legacy.load_legacy_frame(df))
            by_rec = {}
            for stem, ev in items:
                r = recs.get(stem)
                if r is None and default_rec:
                    r = next((x for x in recs.values() if str(x["id"]) == default_rec), None)
                if r is None:
                    unmatched[stem or "(blank file_name)"] = unmatched.get(stem or "(blank file_name)", 0) + 1
                    continue
                ev["derived"] = ev.get("derived")
                by_rec.setdefault(r["id"], []).append(ev)
            for rid, lst in by_rec.items():
                A.insert_events(set_id, rid, lst, user, action="import")
                n_ok += len(lst)
        msg = f"Imported {n_ok} event(s) into set #{set_id}."
        if unmatched:
            msg += " Unmatched rows (no recording with that file_name in this experiment): " + \
                   "; ".join(f"{k}: {v}" for k, v in unmatched.items())
        flash(msg, "ok" if not unmatched else "warn")
        return back()

    @app.get("/sets/<int:set_id>/export")
    def set_export(set_id):
        s = get_set_or_404(set_id)
        layout = request.args.get("layout", "legacy")
        fmt = request.args.get("fmt", "csv")
        audit = request.args.get("audit") == "1"
        base = _download_name(f"{s['name']}")
        if layout == "zip":
            recs = [r for r in recordings_of(s["experiment_id"])]

            def entries():
                for r in recs:
                    df = set_export_frame(s, "legacy", audit, r["id"])
                    if len(df):
                        yield f"{Path(r['file_name']).stem}_event.csv", legacy.frame_to_bytes(df, "csv")
                yield "edit_history.csv", legacy.frame_to_bytes(history_frame(s), "csv")
            return Response(stream_with_context(stream_zip(entries())), mimetype="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{base}_per_recording.zip"'})
        df = set_export_frame(s, layout, audit, int(request.args["rec"]) if request.args.get("rec") else None)
        suffix = "__predicted" if layout == "nn" else ""
        data = legacy.frame_to_bytes(df, "xlsx" if fmt == "xlsx" else "csv")
        return send_file(io.BytesIO(data), as_attachment=True, download_name=f"{base}{suffix}.{fmt}",
                         mimetype="text/csv" if fmt == "csv" else
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.get("/sets/<int:set_id>/history.csv")
    def set_history(set_id):
        s = get_set_or_404(set_id)
        data = legacy.frame_to_bytes(history_frame(s), "csv")
        return send_file(io.BytesIO(data), as_attachment=True,
                         download_name=f"{_download_name(s['name'])}_history.csv", mimetype="text/csv")

    # ------------------------------------------------------------- NN + clustering
    @app.post("/experiments/<int:exp_id>/nn")
    def run_nn(exp_id):
        get_exp_or_404(exp_id)
        s = get_set_or_404(int(request.form["source_set_id"]))
        jid = jobs.enqueue("nn", {"source_set_id": s["id"], "model_id": int(request.form["model_id"]),
                                  "out_name": request.form.get("out_name", "").strip() or None},
                           current_user.username, exp_id)
        flash(f"NN job #{jid} queued. Results appear as a new annotation set when it finishes.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/cluster")
    def run_cluster(exp_id):
        get_exp_or_404(exp_id)
        set_ids = [int(x) for x in request.form.getlist("set_ids")]
        if not set_ids:
            flash("Select at least one annotation set to cluster.", "error")
            return back()
        for sid in set_ids:
            get_set_or_404(sid)

        def fnum(k, d):
            try:
                return float(request.form.get(k, d))
            except ValueError:
                return d
        params = {"alpha": fnum("alpha", 0.05), "max_k": int(fnum("max_k", 8)),
                  "duration_min": fnum("duration_min", 0.0), "duration_max": fnum("duration_max", 0.0),
                  "n_boot": int(fnum("n_boot", 25)), "min_cluster_pct": fnum("min_cluster_pct", 0.0),
                  "impute": bool(request.form.get("impute")),
                  "features": request.form.getlist("features") or None}
        name = request.form.get("name", "").strip() or f"Clustering {time.strftime('%Y-%m-%d %H:%M')}"
        with db() as con:
            cur = con.execute("""INSERT INTO cluster_runs(experiment_id, name, set_ids_json, params_json, status,
                                 created_by, created_at) VALUES (?,?,?,?, 'queued', ?, ?)""",
                              (exp_id, name, jdump(set_ids), jdump(params), current_user.username, now()))
            run_id = cur.lastrowid
        jid = jobs.enqueue("cluster", {"run_id": run_id}, current_user.username, exp_id)
        with db() as con:
            con.execute("UPDATE cluster_runs SET job_id=? WHERE id=?", (jid, run_id))
        flash(f"Clustering run '{name}' queued (job #{jid}).", "ok")
        return back()

    def _run_or_404(run_id):
        with db() as con:
            run = one(con, "SELECT * FROM cluster_runs WHERE id=?", (run_id,))
        if run is None:
            abort(404)
        get_exp_or_404(run["experiment_id"])
        return run

    @app.get("/cluster/<int:run_id>/report")
    def cluster_report(run_id):
        run = _run_or_404(run_id)
        p = Path(run["out_dir"] or "") / "cluster_report.html"
        if not p.exists():
            abort(404)
        return send_file(p, mimetype="text/html")

    @app.get("/cluster/<int:run_id>/download.zip")
    def cluster_zip(run_id):
        run = _run_or_404(run_id)
        d = Path(run["out_dir"] or "")
        if not d.exists():
            abort(404)

        def entries():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    yield str(p.relative_to(d)), p.read_bytes()
        return Response(stream_with_context(stream_zip(entries())), mimetype="application/zip",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{_download_name(run["name"])}_clustering.zip"'})

    @app.get("/cluster/<int:run_id>/labels.csv")
    def cluster_labels_csv(run_id):
        run = _run_or_404(run_id)
        import pandas as pd
        with db() as con:
            lab = rows(con, """SELECT l.cluster, e.id AS db_event_id, e.event_no AS event_id, r.file_name,
                               s.name AS annotation_set FROM cluster_labels l
                               JOIN events e ON e.id=l.event_id JOIN recordings r ON r.id=e.recording_id
                               JOIN annotation_sets s ON s.id=e.set_id WHERE l.run_id=?
                               ORDER BY s.name, r.file_name, e.event_no""", (run_id,))
        df = pd.DataFrame(lab, columns=["annotation_set", "file_name", "event_id", "db_event_id", "cluster"])
        df["cluster"] = "C" + df["cluster"].astype(str)
        return send_file(io.BytesIO(df.to_csv(index=False).encode()), as_attachment=True,
                         download_name=f"{_download_name(run['name'])}_labels.csv", mimetype="text/csv")

    @app.post("/cluster/<int:run_id>/delete")
    def cluster_delete(run_id):
        run = _run_or_404(run_id)
        if not (current_user.is_admin or run["created_by"] == current_user.username):
            abort(403)
        if run["out_dir"]:
            shutil.rmtree(run["out_dir"], ignore_errors=True)
        with db() as con:
            con.execute("DELETE FROM cluster_runs WHERE id=?", (run_id,))
        return back()

    # ------------------------------------------------------------- experiment bundle
    @app.get("/experiments/<int:exp_id>/bundle.zip")
    def experiment_bundle(exp_id):
        exp = get_exp_or_404(exp_id)
        include_abf = request.args.get("abf") == "1"
        recs = recordings_of(exp_id)
        sets = A.list_sets(exp_id)
        fnames = {r["id"]: r["file_name"] for r in recs}

        def entries():
            manifest = {
                "format": "nanotag-bundle", "version": 1, "exported_at": now(), "exported_by": current_user.username,
                "experiment": {k: exp[k] for k in ("name", "description", "sensor", "analytes", "solution")},
                "recordings": [{k: r[k] for k in ("file_name", "sha256", "size_bytes", "fs", "duration",
                                                  "role_elec", "role_opt", "role_optref")} for r in recs],
                "sets": [],
            }
            for s in sets:
                ev_file = f"sets/{s['id']}_events.jsonl"
                lines = []
                for e in A.list_events(s["id"]):
                    d = {k: e[k] for k in EVENT_FIELDS}
                    d["derived"] = jload(e["derived_json"])
                    d["file_name"] = fnames.get(e["recording_id"])
                    lines.append(json.dumps(d))
                yield ev_file, ("\n".join(lines) + "\n").encode()
                df = legacy.legacy_frame(A.list_events(s["id"]), fnames, s["formula"], audit=True)
                yield f"sets/{_download_name(s['name'])}.csv", legacy.frame_to_bytes(df, "csv")
                yield f"sets/{_download_name(s['name'])}_history.csv", legacy.frame_to_bytes(history_frame(s), "csv")
                manifest["sets"].append({"name": s["name"], "kind": s["kind"], "formula": s["formula"],
                                         "created_by": s["created_by"], "source": jload(s["source_json"], {}),
                                         "events_file": ev_file})
            if include_abf:
                for r in recs:
                    if r["abf_path"] and os.path.exists(r["abf_path"]):
                        yield f"abf/{r['file_name']}", Path(r["abf_path"])
            yield "manifest.json", json.dumps(manifest, indent=2).encode()
        return Response(stream_with_context(stream_zip(entries())), mimetype="application/zip",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{_download_name(exp["name"])}_bundle.zip"'})

    # ------------------------------------------------------------- jobs
    @app.get("/jobs")
    def jobs_page():
        return render_template("jobs.html", jobs_list=jobs.list_jobs(None, 200))

    @app.get("/jobs/<int:job_id>")
    def job_page(job_id):
        j = jobs.get_job(job_id)
        if j is None:
            abort(404)
        if j["experiment_id"]:
            get_exp_or_404(j["experiment_id"])
        return render_template("job.html", j=j, params=jload(j["params_json"], {}),
                               result=jload(j["result_json"], {}))

    @app.get("/api/jobs")
    def api_jobs():
        exp_id = request.args.get("experiment_id")
        lst = jobs.list_jobs(int(exp_id) if exp_id else None, 30)
        with db() as con:
            recs = rows(con, "SELECT id, status, error FROM recordings WHERE experiment_id=?",
                        (int(exp_id),)) if exp_id else []
        return jsonify({"jobs": lst, "recordings": recs})

    @app.get("/api/jobs/<int:job_id>")
    def api_job(job_id):
        j = jobs.get_job(job_id)
        if j is None:
            abort(404)
        return jsonify({k: j[k] for k in ("id", "status", "progress", "message", "log", "result_json")})

    @app.post("/jobs/<int:job_id>/cancel")
    def job_cancel(job_id):
        j = jobs.get_job(job_id)
        if j and (current_user.is_admin or j["created_by"] == current_user.username):
            jobs.cancel_job(job_id)
        return back()

    # ------------------------------------------------------------- models
    @app.get("/models")
    def models_page():
        with db() as con:
            ms = rows(con, "SELECT * FROM models ORDER BY name")
        return render_template("models.html", models=ms)

    @app.post("/models/<int:model_id>/default")
    @admin_required
    def model_default(model_id):
        with db() as con, tx(con):
            con.execute("UPDATE models SET is_default=0")
            con.execute("UPDATE models SET is_default=1 WHERE id=?", (model_id,))
        return back()

    @app.post("/models/<int:model_id>/delete")
    @admin_required
    def model_delete(model_id):
        with db() as con:
            m = one(con, "SELECT * FROM models WHERE id=?", (model_id,))
            con.execute("DELETE FROM models WHERE id=?", (model_id,))
        if m and m["path"].startswith(str(c.MODELS_DIR)):
            Path(m["path"]).unlink(missing_ok=True)
        return back()

    @app.get("/models/<int:model_id>/download")
    def model_download(model_id):
        with db() as con:
            m = one(con, "SELECT * FROM models WHERE id=?", (model_id,))
        if not m:
            abort(404)
        return send_file(m["path"], as_attachment=True, download_name=f"{m['name']}.pt")

    # ------------------------------------------------------------- admin
    @app.get("/admin")
    @admin_required
    def admin():
        du = shutil.disk_usage(c.DATA)
        return render_template("admin.html", users=auth.list_users(),
                               disk={"total": du.total, "used": du.used, "free": du.free},
                               roots=[str(r) for r in c.IMPORT_ROOTS], cfg=c)

    @app.post("/admin/users/add")
    @admin_required
    def admin_user_add():
        try:
            auth.add_user(request.form["username"], request.form["password"], request.form.get("role", "user"),
                          request.form.get("full_name", ""))
            flash(f"User {request.form['username']} added.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return back()

    @app.post("/admin/users/<username>/password")
    @admin_required
    def admin_user_pw(username):
        try:
            auth.set_password(username, request.form["password"])
            flash(f"Password reset for {username}.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return back()

    @app.post("/admin/users/<username>/role")
    @admin_required
    def admin_user_role(username):
        try:
            auth.set_role(username, request.form["role"])
            flash(f"{username} is now {request.form['role']}.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return back()

    @app.post("/admin/users/<username>/active")
    @admin_required
    def admin_user_active(username):
        try:
            auth.set_active(username, request.form.get("active") == "1")
        except ValueError as e:
            flash(str(e), "error")
        return back()
