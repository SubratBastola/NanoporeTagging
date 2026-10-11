"""Flask web application (pages, REST endpoints, uploads, exports) + the Dash Tagger."""
import io
import json
import math
import os
import re
import shutil
import socket
import threading
import time
import uuid
import zipfile
from functools import wraps
from pathlib import Path

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_file, session, stream_with_context, url_for)
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from . import annotations as A
from . import auth, deletion, folder_import, jobs, legacy, naming, store
from .config import cfg
from .db import EVENT_FIELDS, db, jdump, jload, migrate, now, one, rows, tx

CHUNK_SIZE = 16 * 1024 * 1024
ABF_EXTS = (".abf",)
SESSION_MAX_AGE = 12 * 3600          # sign in again after 12 h


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
        SESSION_PERMANENT=False,            # session cookie ends when the browser closes
        SEND_FILE_MAX_AGE_DEFAULT=None,
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
        if current_user.is_authenticated:
            # Sessions from an older release, or older than SESSION_MAX_AGE, start over at the login page.
            if session.get("epoch") != c.VERSION or now() - float(session.get("login_at") or 0) > SESSION_MAX_AGE:
                logout_user()
                session.clear()
                flash("Please sign in again (the server was updated or your session expired).", "warn")
        if not current_user.is_authenticated:
            if p.startswith("/api/") or "/_dash-" in p:
                return jsonify({"error": "login required"}), 401
            return redirect(url_for("login"))
        # A page opened from a typed address, a bookmark or a restored browser tab (no link from this
        # site) never drops you into the middle of an analysis: it starts at the Experiments home.
        if (request.method == "GET" and p not in ("/", "/experiments", "/logout")
                and "text/html" in (request.headers.get("Accept") or "")
                and not p.startswith("/api/") and "/_dash-" not in p and not _came_from_this_site()):
            return redirect(url_for("experiments"))
        return None

    @app.after_request
    def _no_stale(resp):
        """Pages and data are never cached; static files are versioned (?v=<release>) instead."""
        p = request.path
        is_asset = p.startswith("/static/") or "/_dash-component-suites/" in p
        if is_asset and request.args.get("v") == c.VERSION:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif is_asset:
            resp.headers["Cache-Control"] = "no-cache"
        else:
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp

    @app.context_processor
    def _inject():
        return {
            "version": c.VERSION,
            "asset": lambda f: url_for("static", filename=f, v=c.VERSION),
            "default_admin_pw": (current_user.is_authenticated and current_user.is_admin
                                 and auth.default_admin_password_in_use()),
            "fmt_ts": fmt_ts, "fmt_bytes": fmt_bytes, "fmt_dur": fmt_dur,
        }

    _register_routes(app)
    from .tagger import init_tagger
    init_tagger(app)
    return app


# ----------------------------------------------------------------- helpers
def _came_from_this_site():
    from urllib.parse import urlparse
    ref = request.headers.get("Referer") or ""
    return bool(ref) and urlparse(ref).netloc == request.host


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


def back(default="/experiments"):
    nxt = request.form.get("next") or ""
    if nxt.startswith("/") and not nxt.startswith("//"):
        return redirect(nxt)
    return redirect(request.referrer or default)


def _download_name(s):
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", s).strip() or "export"


def browse_roots():
    """Folders shown in the file browser: the import folders plus the NanoTag ABF store."""
    c = cfg()
    out = [{"path": str(r), "label": r.name or str(r), "writable": _is_writable_root(r)} for r in c.IMPORT_ROOTS]
    out.append({"path": str(c.ABF_DIR), "label": "NanoTag ABF store (per experiment)", "writable": False})
    return out


def _is_writable_root(r: Path):
    return r.resolve() == cfg().INCOMING_DIR.resolve()


def _within(p: Path, roots):
    try:
        p = p.resolve()
    except OSError:
        return False
    for r in roots:
        try:
            p.relative_to(Path(r).resolve())
            return True
        except ValueError:
            continue
    return False


def _within_roots(p: Path, include_store=True):
    roots = list(cfg().IMPORT_ROOTS) + ([cfg().ABF_DIR] if include_store else [])
    return _within(p, roots)


def _writable(p: Path):
    return _within(p, [cfg().INCOMING_DIR])


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

    def _auto_experiment(file_name):
        name = naming.guess_experiment_name(file_name)
        exp_id, _ = naming.get_or_create_experiment(name, current_user.username)
        with db() as con:
            exp = one(con, "SELECT * FROM experiments WHERE id=?", (exp_id,))
        if not auth.can_view_experiment(current_user, exp):
            raise ValueError(f"Experiment '{name}' exists but is restricted; ask an admin for access.")
        return exp_id

    @app.get("/api/guess-experiment")
    def api_guess_experiment():
        out = []
        with db() as con:
            for n in request.args.getlist("name"):
                g = naming.guess_experiment_name(n)
                ex = one(con, "SELECT id FROM experiments WHERE name=?", (g,))
                out.append({"file": n, "experiment": g, "exists": bool(ex), "experiment_id": ex["id"] if ex else None})
        return jsonify(out)

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
                session.clear()
                login_user(u, remember=False)
                session["epoch"] = c.VERSION
                session["login_at"] = now()
                return redirect(url_for("experiments"))   # always start at the home page
            flash("Invalid username or password.", "error")
        return render_template("login.html")

    @app.get("/logout")
    def logout():
        logout_user()
        session.clear()
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
        """The server's front door is always the login page."""
        if current_user.is_authenticated:
            logout_user()
        session.clear()
        return redirect(url_for("login"))

    @app.get("/experiments")
    def experiments():
        exps = auth.visible_experiments(current_user)
        with db() as con:
            stats = {r["experiment_id"]: r for r in rows(con, """SELECT experiment_id, COUNT(*) AS n,
                     SUM(COALESCE(size_bytes,0)) AS bytes, SUM(status='ready') AS ready
                     FROM recordings GROUP BY experiment_id""")}
        deletable = {e["id"] for e in exps if auth.can_delete_experiment(current_user, e)}
        return render_template("experiments.html", exps=exps, stats=stats, deletable=deletable,
                               roots=[str(r) for r in c.IMPORT_ROOTS])

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
        flash(f"Experiment '{name}' created — now add ABF files.", "ok")
        return redirect(url_for("experiment", exp_id=exp_id) + "#recordings")

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
        for s_ in sets:
            s_["can_delete"] = auth.can_delete_set(current_user, s_, exp)
        for r in runs:
            r["can_delete"] = auth.can_delete_run(current_user, r, exp)
        with db() as con:
            per = rows(con, """SELECT e.set_id, e.recording_id, COUNT(*) AS n FROM events e
                               JOIN annotation_sets s ON s.id=e.set_id
                               WHERE s.experiment_id=? AND e.deleted=0 GROUP BY e.set_id, e.recording_id""", (exp_id,))
        set_name = {s_["id"]: s_["name"] for s_ in sets}
        for s_ in sets:
            s_["per_rec"] = {p["recording_id"]: p["n"] for p in per if p["set_id"] == s_["id"]}
        for r in recs:
            r["owned"] = deletion.owned_abf(r["abf_path"])
            r["set_counts"] = [(set_name[p["set_id"]], p["n"]) for p in per if p["recording_id"] == r["id"]]
        conditions = sorted({(r["condition"] or "other") for r in recs},
                            key=lambda x: ["DC", "AC", "AOM", "AC-AOM", "baseline", "other"].index(x)
                            if x in ["DC", "AC", "AOM", "AC-AOM", "baseline", "other"] else 99)
        return render_template("experiment.html", exp=exp, recs=recs, sets=sets, runs=runs, models=models,
                               can_delete_exp=auth.can_delete_experiment(current_user, exp),
                               can_delete_recs=auth.can_delete_experiment(current_user, exp),
                               conditions=conditions,
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
                                   "role_elec", "role_opt", "role_optref", "condition", "abf_path", "source_path")} | {"channels": jload(r["channels_json"], [])}
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

    def _deletable_experiments(ids):
        out, refused = [], []
        for i in ids:
            with db() as con:
                exp = one(con, "SELECT * FROM experiments WHERE id=?", (i,))
            if exp is None or not auth.can_view_experiment(current_user, exp):
                continue
            (out if auth.can_delete_experiment(current_user, exp) else refused).append(exp)
        return out, refused

    @app.get("/experiments/delete")
    def experiments_delete_confirm():
        ids = [int(x) for x in request.args.getlist("ids") if str(x).isdigit()]
        exps, refused = _deletable_experiments(ids)
        if not exps and not refused:
            flash("Tick one or more experiments to delete.", "warn")
            return redirect(url_for("experiments"))
        summaries = [deletion.experiment_summary(e["id"]) for e in exps]
        return render_template("delete_confirm.html", summaries=summaries, refused=refused,
                               total_bytes=sum(x["owned_bytes"] for x in summaries))

    @app.post("/experiments/delete")
    def experiments_delete():
        ids = [int(x) for x in request.form.getlist("ids") if str(x).isdigit()]
        exps, refused = _deletable_experiments(ids)
        if refused:
            abort(403)
        freed = 0
        for e in exps:
            freed += deletion.delete_experiment(e["id"], delete_abf=bool(request.form.get("delete_abf")))
        names = ", ".join(e["name"] for e in exps)
        flash(f"Deleted {len(exps)} experiment(s): {names}" + (f" — freed {fmt_bytes(freed)} of ABF copies." if freed
                                                                else "."), "ok")
        return redirect(url_for("experiments"))

    @app.post("/experiments/<int:exp_id>/delete")
    def experiment_delete(exp_id):
        exp = get_exp_or_404(exp_id)
        if not auth.can_delete_experiment(current_user, exp):
            abort(403)
        freed = deletion.delete_experiment(exp_id, delete_abf=bool(request.form.get("delete_abf")))
        flash(f"Experiment '{exp['name']}' deleted" + (f" — freed {fmt_bytes(freed)}." if freed else "."), "ok")
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
            if str(d.get("experiment_id")) != "auto":
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
        elif purpose == "file":
            target = Path(d.get("dir") or "")
            if not target.is_dir() or not _writable(target):
                return jsonify({"error": "You can only upload into the incoming folder (or its sub-folders)."}), 403
            if (target / name).exists() and not d.get("replace"):
                return jsonify({"error": f"{name} already exists in that folder."}), 409
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
                    and m.get("purpose") == purpose and m.get("experiment_id") == d.get("experiment_id")
                    and m.get("dir") == d.get("dir")):
                return jsonify({"upload_id": m["id"], "chunk_size": m["chunk_size"], "received": m["received"]})
        uid = uuid.uuid4().hex
        meta = {"id": uid, "user": current_user.username, "filename": name, "size": size, "purpose": purpose,
                "experiment_id": d.get("experiment_id"), "chunk_size": CHUNK_SIZE, "received": [],
                "created": now(), "replace": bool(d.get("replace")), "dir": d.get("dir")}
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
                if str(meta["experiment_id"]) == "auto":
                    exp_id = _auto_experiment(meta["filename"])
                else:
                    exp_id = int(meta["experiment_id"])
                dest = jobs.abf_dest(exp_id, meta["filename"])
                if dest.exists() and not meta.get("replace"):
                    return jsonify({"error": f"{meta['filename']} already exists in this experiment."}), 409
                rec_id, _ = jobs.register_recording(exp_id, meta["filename"], dest, current_user.username,
                                                    replace=meta.get("replace", False))
                os.replace(pp, dest)
                jid = jobs.enqueue("ingest", {"recording_id": rec_id}, current_user.username, exp_id)
                result = {"recording_id": rec_id, "job_id": jid, "experiment_id": exp_id}
            elif purpose == "bundle":
                dest = c.UPLOAD_DIR / f"{uid}.zip"
                os.replace(pp, dest)
                jid = jobs.enqueue("import_bundle", {"zip_path": str(dest)}, current_user.username)
                result = {"job_id": jid}
            elif purpose == "file":
                target = Path(meta["dir"])
                if not _writable(target):
                    abort(403)
                dest = target / meta["filename"]
                if dest.exists() and not meta.get("replace"):
                    return jsonify({"error": f"{meta['filename']} already exists in that folder."}), 409
                shutil.move(str(pp), dest)
                try:
                    os.chmod(dest, 0o664)
                except OSError:
                    pass
                result = {"path": str(dest)}
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
            return jsonify({"path": "", "dirs": [str(r) for r in c.IMPORT_ROOTS if r.exists()] + [str(c.ABF_DIR)],
                            "files": []})
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
    def recording_delete(rec_id):
        r = get_rec_or_404(rec_id)
        if not auth.can_delete_experiment(current_user, get_exp_or_404(r["experiment_id"])):
            abort(403)
        deletion.delete_recording(r, delete_abf=bool(request.form.get("delete_abf")))
        flash(f"Recording {r['file_name']} removed.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/recordings/delete")
    def recordings_delete(exp_id):
        exp = get_exp_or_404(exp_id)
        if not auth.can_delete_experiment(current_user, exp):
            abort(403)
        ids = {int(x) for x in request.form.getlist("rec_ids") if str(x).isdigit()}
        n = 0
        for r in recordings_of(exp_id):
            if r["id"] in ids:
                deletion.delete_recording(r, delete_abf=bool(request.form.get("delete_abf")))
                n += 1
        flash(f"Removed {n} recording(s) and their events.", "ok" if n else "warn")
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

    def _delete_sets(sets_):
        used = []
        for s_ in sets_:
            u = deletion.runs_using_set(s_["id"])
            if u:
                used.append(f"{s_['name']} (used by clustering run(s) {', '.join(u)}; their outputs are kept)")
            deletion.delete_set(s_["id"])
        return used

    @app.post("/sets/<int:set_id>/delete")
    def set_delete(set_id):
        s_ = get_set_or_404(set_id)
        exp = get_exp_or_404(s_["experiment_id"])
        if not auth.can_delete_set(current_user, s_, exp):
            abort(403)
        used = _delete_sets([s_])
        flash(f"Annotation set '{s_['name']}' deleted." + (" Note: " + "; ".join(used) if used else ""), "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/sets/delete")
    def sets_delete(exp_id):
        exp = get_exp_or_404(exp_id)
        ids = {int(x) for x in request.form.getlist("set_ids") if str(x).isdigit()}
        chosen = [x for x in A.list_sets(exp_id) if x["id"] in ids]
        if not chosen:
            flash("Tick one or more annotation sets to delete.", "warn")
            return back()
        if any(not auth.can_delete_set(current_user, x, exp) for x in chosen):
            flash("You can delete only sets you (or this experiment's creator) made, and not locked ones — "
                  "ask an admin.", "error")
            return back()
        used = _delete_sets(chosen)
        flash(f"Deleted {len(chosen)} annotation set(s): {', '.join(x['name'] for x in chosen)}."
              + (" Note: " + "; ".join(used) if used else ""), "ok")
        return back()

    def _import_tables(exp_id, tables, form):
        """tables: list of (filename, bytes). Returns (set_id or None, message, category)."""
        target = form.get("target_set") or ""
        user = current_user.username
        if target:
            s_ = get_set_or_404(int(target), edit=True)
            if s_["experiment_id"] != exp_id:
                abort(400)
            set_id = s_["id"]
        else:
            name = (form.get("new_name") or "").strip() or A.unique_set_name(
                exp_id, "Imported " + time.strftime("%Y-%m-%d %H:%M"))
            try:
                set_id = A.create_set(exp_id, name, user, kind="import", formula=form.get("formula") or "tagger",
                                      source={"files": [n for n, _ in tables]})
            except ValueError as e:
                return None, str(e), "error"
        recs = {legacy.norm_name(r["file_name"]): r for r in recordings_of(exp_id)}
        default_rec = form.get("default_recording")
        n_ok, unmatched, errors = 0, {}, []
        for fname, data in tables:
            try:
                df = legacy.read_table_bytes(data, fname)
            except Exception as e:
                errors.append(f"{fname}: {e}")
                continue
            frame = legacy.load_legacy_frame(df)
            items = legacy.frame_to_event_dicts(frame)
            keys = [legacy.norm_name(v) for v in frame["file_name"]]
            by_rec = {}
            for (stem, ev), key in zip(items, keys):
                r = recs.get(key)
                if r is None and default_rec:
                    r = next((x for x in recs.values() if str(x["id"]) == str(default_rec)), None)
                if r is None:
                    key = stem or "(blank file_name)"
                    unmatched[key] = unmatched.get(key, 0) + 1
                    continue
                by_rec.setdefault(r["id"], []).append(ev)
            for rid, lst in by_rec.items():
                A.insert_events(set_id, rid, lst, user, action="import")
                n_ok += len(lst)
        msg = f"Imported {n_ok} event(s) into set #{set_id}."
        if unmatched:
            msg += " Unmatched rows (no recording with that file_name in this experiment): " + \
                   "; ".join(f"{k}: {v}" for k, v in unmatched.items())
        if errors:
            msg += " Errors: " + "; ".join(errors)
        return set_id, msg, ("ok" if not unmatched and not errors else "warn")

    @app.post("/experiments/<int:exp_id>/import-events")
    def import_events(exp_id):
        get_exp_or_404(exp_id)
        files = [f for f in request.files.getlist("files") if f and f.filename]
        if not files:
            flash("Choose one or more CSV/XLSX files.", "error")
            return back()
        _, msg, cat = _import_tables(exp_id, [(f.filename, f.read()) for f in files], request.form)
        flash(msg, cat)
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
        conds = [x for x in request.form.getlist("conditions") if x]
        if conds and request.form.get("split"):
            plan = [(f"{name} · {cnd}", [cnd]) for cnd in conds]
        else:
            plan = [(name + (f" · {'+'.join(conds)}" if conds else ""), conds)]
        queued = []
        for run_name, cl in plan:
            p = dict(params, conditions=cl)
            with db() as con:
                cur = con.execute("""INSERT INTO cluster_runs(experiment_id, name, set_ids_json, params_json, status,
                                     created_by, created_at) VALUES (?,?,?,?, 'queued', ?, ?)""",
                                  (exp_id, run_name, jdump(set_ids), jdump(p), current_user.username, now()))
                run_id = cur.lastrowid
            jid = jobs.enqueue("cluster", {"run_id": run_id}, current_user.username, exp_id)
            with db() as con:
                con.execute("UPDATE cluster_runs SET job_id=? WHERE id=?", (jid, run_id))
            queued.append(f"'{run_name}' (job #{jid})")
        flash(f"Clustering queued: {', '.join(queued)}.", "ok")
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
        if not auth.can_delete_run(current_user, run, get_exp_or_404(run["experiment_id"])):
            abort(403)
        deletion.delete_run(run)
        flash(f"Clustering run '{run['name']}' deleted.", "ok")
        return back()

    @app.post("/experiments/<int:exp_id>/cluster/delete")
    def clusters_delete(exp_id):
        exp = get_exp_or_404(exp_id)
        ids = {int(x) for x in request.form.getlist("run_ids") if str(x).isdigit()}
        with db() as con:
            runs = [r for r in rows(con, "SELECT * FROM cluster_runs WHERE experiment_id=?", (exp_id,)) if r["id"] in ids]
        if not runs:
            flash("Tick one or more clustering runs to delete.", "warn")
            return back()
        if any(not auth.can_delete_run(current_user, r, exp) for r in runs):
            flash("You can delete only your own runs (or runs in an experiment you created) — ask an admin.", "error")
            return back()
        for r in runs:
            deletion.delete_run(r)
        flash(f"Deleted {len(runs)} clustering run(s).", "ok")
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

    @app.post("/jobs/clear")
    def jobs_clear():
        exp_id = request.form.get("experiment_id")
        if exp_id:
            exp = get_exp_or_404(int(exp_id))
            mine_only = not auth.can_delete_experiment(current_user, exp)
            n = deletion.clear_finished_jobs(int(exp_id), None if not mine_only else current_user.username)
        else:
            n = deletion.clear_finished_jobs(None, None if current_user.is_admin else current_user.username)
        flash(f"Cleared {n} finished job(s) from the list.", "ok")
        return back()

    @app.post("/jobs/<int:job_id>/cancel")
    def job_cancel(job_id):
        """Cancel a queued job or stop a running one (its creator or an admin)."""
        j = jobs.get_job(job_id)
        if j is None:
            abort(404)
        if not (current_user.is_admin or j["created_by"] == current_user.username):
            flash(f"Only {j['created_by']} or an administrator can stop job #{job_id}.", "error")
            return back()
        st = jobs.cancel_job(job_id, current_user.username)
        if st == "cancelled":
            flash(f"Job #{job_id} ({j['kind']}) cancelled before it started.", "ok")
        elif st == "cancelling":
            flash(f"Stopping job #{job_id} ({j['kind']}) — it will show as cancelled in a few seconds; "
                  f"partial results are removed.", "ok")
        else:
            flash(f"Job #{job_id} is already {j['status']}.", "warn")
        if request.headers.get("Accept", "").startswith("application/json"):
            return jsonify({"status": st})
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

    # ------------------------------------------------------------- whole-server export (move to another machine)
    @app.get("/admin/export-site")
    @admin_required
    def admin_export_site():
        """Download users, experiments, annotations + history, models, clustering outputs and settings as one
        .tar (optionally with display data and ABFs). Import on the new server: deploy/migrate-from.sh FILE."""
        from . import sitemove
        zarr_ = request.args.get("zarr") == "1"
        abf = request.args.get("abf") == "1"
        name = f"nanotag-site-{_download_name(socket.gethostname())}-{time.strftime('%Y%m%d-%H%M')}.tar"
        app.logger.info("site export by %s (zarr=%s abf=%s)", current_user.username, zarr_, abf)
        return Response(stream_with_context(sitemove.export_site(zarr_, abf, current_user.username)),
                        mimetype="application/x-tar",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/admin/site-summary")
    @admin_required
    def api_site_summary():
        from . import sitemove
        return jsonify(sitemove.site_summary())

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

    # ------------------------------------------------------------- general file browser
    @app.get("/files")
    def files_page():
        return render_template("files.html", roots=browse_roots(), exps=auth.visible_experiments(current_user),
                               incoming=str(c.INCOMING_DIR),
                               start=request.args.get("path") or str(c.INCOMING_DIR))

    @app.get("/api/files")
    def api_files():
        path = Path(request.args.get("path") or c.INCOMING_DIR)
        if not path.exists() or not path.is_dir() or not _within_roots(path):
            return jsonify({"error": "That folder is outside the folders NanoTag may read."}), 403
        with db() as con:
            used = {}
            for r in rows(con, """SELECT r.stem, r.abf_path, e.id AS exp_id, e.name AS exp_name
                                  FROM recordings r JOIN experiments e ON e.id=r.experiment_id"""):
                used.setdefault(r["stem"], []).append({"exp_id": r["exp_id"], "exp_name": r["exp_name"]})
        dirs, files = [], []
        try:
            for e in sorted(path.iterdir(), key=lambda x: x.name.lower()):
                if e.name.startswith("."):
                    continue
                try:
                    st = e.stat()
                except OSError:
                    continue
                if e.is_dir():
                    dirs.append({"path": str(e), "name": e.name, "mtime": st.st_mtime})
                    continue
                ext = e.suffix.lower()
                kind = ("abf" if ext == ".abf" or (c.ALLOW_FAKE_ABF and e.name.endswith(".fakeabf.npz"))
                        else "table" if ext in (".csv", ".xlsx", ".xls", ".xlsm")
                        else "zip" if ext == ".zip" else "model" if ext == ".pt" else "other")
                item = {"path": str(e), "name": e.name, "size": st.st_size, "mtime": st.st_mtime, "kind": kind}
                if kind == "abf":
                    item["used_in"] = used.get(legacy.file_stem(e.name), [])
                files.append(item)
        except PermissionError:
            return jsonify({"error": "Permission denied (the nanotag service user cannot read this folder)."}), 403
        crumbs = []
        root = next((r for r in browse_roots() if _within(path, [r["path"]])), None)
        if root:
            q = path.resolve()
            rp = Path(root["path"]).resolve()
            parts = [q] + [x for x in q.parents if _within(x, [rp])]
            crumbs = [{"path": str(x), "name": (x.name if x != rp else root["label"])} for x in reversed(parts)]
        return jsonify({"path": str(path), "writable": _writable(path), "dirs": dirs, "files": files,
                        "crumbs": crumbs, "root": root})

    @app.get("/files/download")
    def files_download():
        path = Path(request.args.get("path") or "")
        if not path.is_file() or not _within_roots(path):
            abort(404)
        return send_file(path, as_attachment=True, download_name=path.name, conditional=True)

    @app.post("/api/files/mkdir")
    def files_mkdir():
        d = request.get_json(force=True)
        parent = Path(d.get("path") or "")
        name = jobs.safe_name(d.get("name") or "")
        if not parent.is_dir() or not _writable(parent) or not name:
            return jsonify({"error": "Folders can only be created inside the incoming folder."}), 403
        (parent / name).mkdir(exist_ok=True)
        try:
            os.chmod(parent / name, 0o2775)
        except OSError:
            pass
        return jsonify({"ok": True, "path": str(parent / name)})

    @app.post("/files/delete")
    def files_delete():
        paths = [Path(x) for x in request.form.getlist("paths")]
        bad = [x for x in paths if not x.is_file() or not _writable(x)]
        if bad or not paths:
            flash("Only files inside the incoming folder can be deleted here.", "error")
            return back()
        if not current_user.is_admin:
            abort(403)
        for x in paths:
            x.unlink(missing_ok=True)
        flash(f"Deleted {len(paths)} file(s).", "ok")
        return back()

    def _target_experiment():
        """Existing experiment from the form, a new one if a name was typed, or 'auto' (from file names)."""
        if (request.form.get("experiment_id") or "") == "auto" and not (request.form.get("new_experiment") or "").strip():
            return "auto"
        new_name = (request.form.get("new_experiment") or "").strip()
        if new_name:
            with db() as con:
                ex = one(con, "SELECT * FROM experiments WHERE name=?", (new_name,))
                if ex:
                    get_exp_or_404(ex["id"])
                    return ex["id"]
                cur = con.execute("INSERT INTO experiments(name, created_by, created_at) VALUES (?,?,?)",
                                  (new_name, current_user.username, now()))
                exp_id = cur.lastrowid
            A.create_set(exp_id, "Manual tags", current_user.username)
            return exp_id
        exp_id = int(request.form.get("experiment_id") or 0)
        get_exp_or_404(exp_id)
        return exp_id

    @app.post("/files/import-abf")
    def files_import_abf():
        files = request.form.getlist("paths")
        if not files or any(not _within_roots(Path(f)) or not Path(f).is_file() for f in files):
            flash("Select one or more ABF files first.", "error")
            return back()
        exp_id = _target_experiment()
        groups = {}
        if exp_id == "auto":
            try:
                for f in files:
                    groups.setdefault(_auto_experiment(Path(f).name), []).append(f)
            except ValueError as e:
                flash(str(e), "error")
                return back()
        else:
            groups[exp_id] = files
        for eid, lst in groups.items():
            jobs.enqueue("import_files", {"experiment_id": eid, "files": lst, "mode": request.form.get("mode", "copy"),
                                          "replace": bool(request.form.get("replace"))}, current_user.username, eid)
        if len(groups) == 1:
            eid = next(iter(groups))
            flash(f"Importing {len(files)} ABF file(s). They appear below as they are processed.", "ok")
            return redirect(url_for("experiment", exp_id=eid) + "#recordings")
        flash(f"Importing {len(files)} ABF file(s) into {len(groups)} experiments (named from the file names).", "ok")
        return redirect(url_for("experiments"))

    @app.post("/files/import-events")
    def files_import_events():
        files = [Path(f) for f in request.form.getlist("paths")]
        if not files or any(not _within_roots(f) or not f.is_file() for f in files):
            flash("Select one or more CSV/XLSX files first.", "error")
            return back()
        exp_id = _target_experiment()
        form = dict(request.form)
        form.pop("target_set", None)
        if exp_id != "auto":
            set_id, msg, cat = _import_tables(exp_id, [(f.name, f.read_bytes()) for f in files], form)
            flash(msg, cat)
            return redirect(url_for("experiment", exp_id=exp_id) + "#annotations")
        # automatic: send each row to the experiment that holds a recording with that file_name
        per_exp, unmatched = {}, {}
        with db() as con:
            stems = {}
            for r in rows(con, "SELECT r.file_name, r.experiment_id FROM recordings r"):
                stems.setdefault(legacy.norm_name(r["file_name"]), r["experiment_id"])
        for f in files:
            try:
                df = legacy.read_table_bytes(f.read_bytes(), f.name)
            except Exception as e:
                flash(f"{f.name}: {e}", "error")
                continue
            fn_col = next((col for col in df.columns if legacy._norm(col) in ("file_name", "filename")), None)
            if fn_col is None:
                flash(f"{f.name}: no file_name column, so the experiment cannot be worked out — choose one.", "error")
                continue
            for i, fn in enumerate(df[fn_col]):
                eid = stems.get(legacy.norm_name(fn))
                if eid is None:
                    unmatched[str(fn)] = unmatched.get(str(fn), 0) + 1
                    continue
                per_exp.setdefault(eid, {}).setdefault(f.name, []).append(i)
            for eid, by_file in per_exp.items():
                if f.name in by_file:
                    by_file[f.name] = df.iloc[by_file[f.name]].to_csv(index=False).encode()
        msgs = []
        for eid, by_file in per_exp.items():
            get_exp_or_404(eid)
            tables = [(n if n.lower().endswith(".csv") else n.rsplit(".", 1)[0] + ".csv", b)
                      for n, b in by_file.items() if isinstance(b, bytes)]
            _, msg, _ = _import_tables(eid, tables, form)
            with db() as con:
                msgs.append(f"{one(con, 'SELECT name FROM experiments WHERE id=?', (eid,))['name']}: {msg}")
        if unmatched:
            msgs.append("Rows whose file_name matches no imported ABF (import the ABFs first): " +
                        "; ".join(f"{k}: {v}" for k, v in list(unmatched.items())[:10]))
        flash(" | ".join(msgs) or "Nothing imported.", "ok" if per_exp and not unmatched else "warn")
        if len(per_exp) == 1:
            return redirect(url_for("experiment", exp_id=next(iter(per_exp))) + "#annotations")
        return redirect(url_for("experiments"))

    # ------------------------------------------------------------- import a whole folder
    def _root_hint(folder):
        return f"sudo /opt/nanotag/current/deploy/add-import-root.sh '{folder}'"

    @app.get("/import")
    def import_page():
        start = request.args.get("path") or ""
        if not start:
            extra = [str(r) for r in c.IMPORT_ROOTS if Path(r).resolve() != c.INCOMING_DIR.resolve()]
            start = extra[0] if extra else str(c.INCOMING_DIR)
        return render_template("import.html", start=start, roots=browse_roots(),
                               default_set=folder_import.DEFAULT_SET_NAME)

    @app.get("/api/import/scan")
    def api_import_scan():
        raw = (request.args.get("path") or "").strip()
        if not raw:
            return jsonify({"error": "Type or pick a folder first."}), 400
        folder = Path(raw).expanduser()
        if not folder.is_absolute():
            return jsonify({"error": "Use the full path of the folder, e.g. /home/you/NanoporeTagging/SiO2_Tagging"}), 400
        if not _within_roots(folder, include_store=False):
            return jsonify({"error": f"NanoTag is not allowed to read {folder} yet. An admin can allow it by "
                                     f"running this once on the server:", "hint": _root_hint(folder)}), 403
        try:
            if not folder.is_dir():
                return jsonify({"error": f"{folder} is not a folder (or the NanoTag service cannot see it).",
                                "hint": _root_hint(folder)}), 404
            plan = folder_import.scan(folder, recursive=request.args.get("recursive") == "1",
                                      set_name=(request.args.get("set_name") or folder_import.DEFAULT_SET_NAME).strip())
        except PermissionError:
            return jsonify({"error": f"The NanoTag service may not read {folder}. Fix with:",
                            "hint": _root_hint(folder)}), 403
        for g in plan["groups"]:
            g["allowed"] = not g["exists"] or auth.can_view_experiment(
                current_user, {"id": g["experiment_id"], "restricted": g["restricted"]})
        return jsonify(plan)

    @app.post("/api/import/run")
    def api_import_run():
        d = request.get_json(force=True) or {}
        mode = d.get("mode") or "inplace"
        if mode not in folder_import.MODES:
            return jsonify({"error": "bad mode"}), 400
        items = []
        for it in d.get("items") or []:
            abf = Path(it.get("abf") or "")
            if not abf.is_file() or not folder_import.is_abf(abf) or not _within_roots(abf, include_store=False):
                return jsonify({"error": f"Not an ABF inside an allowed folder: {abf}"}), 400
            ev = it.get("events") or None
            if ev and (not Path(ev).is_file() or not _within_roots(Path(ev), include_store=False)):
                return jsonify({"error": f"Event file not readable: {ev}"}), 400
            items.append({"abf": str(abf), "events": ev,
                          "experiment": (it.get("experiment") or "").strip() or naming.guess_experiment_name(abf.name)})
        if not items:
            return jsonify({"error": "Nothing ticked to import."}), 400
        set_name = (d.get("set_name") or folder_import.DEFAULT_SET_NAME).strip()
        try:
            done = folder_import.enqueue_import(items, mode, set_name, current_user.username,
                                                can_view=lambda e: auth.can_view_experiment(current_user, e),
                                                folder=d.get("folder"), formula=d.get("formula") or "tagger",
                                                replace=bool(d.get("replace")))
        except PermissionError as e:
            return jsonify({"error": str(e)}), 403
        n_ev = sum(1 for it in items if it["events"])
        flash(f"Importing {len(items)} recording(s) and {n_ev} event file(s) into {len(done)} experiment(s). "
              f"Events are available right away; each recording shows 'ready' once its signal is processed "
              f"(about a minute per 1.8 GB file, several in parallel).", "ok")
        target = (url_for("experiment", exp_id=done[0][0]) + "#recordings") if len(done) == 1 else url_for("experiments")
        return jsonify({"ok": True, "jobs": [{"experiment_id": e, "job_id": j, "n": n} for e, j, n in done],
                        "redirect": target})

    @app.get("/help")
    def help_page():
        return render_template("help.html", incoming=str(c.INCOMING_DIR), roots=[str(r) for r in c.IMPORT_ROOTS])

    @app.post("/admin/users/<username>/active")
    @admin_required
    def admin_user_active(username):
        try:
            auth.set_active(username, request.form.get("active") == "1")
        except ValueError as e:
            flash(str(e), "error")
        return back()
