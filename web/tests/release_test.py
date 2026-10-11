"""Tests for the 0.1.11 changes, on throw-away servers with synthetic data (nothing touches /srv/nanotag).

    python tests/release_test.py          (from the source/release folder, inside the venv)

  1. Tagger: the events table scrolls through every event (no 25-row pages), and events added on a set
     that had no events yet are drawn right away (the "Show events by" author filter follows new authors).
  2. Jobs: a queued job can be cancelled and a *running* job can be stopped — its process and the processes
     it started are killed, partial results are cleaned up; only its creator or an admin may stop it.
  3. Moving a server: Admin -> Export data and configuration, then `nanotag-admin import-site` on a fresh
     server: users sign in with their old passwords, experiments / annotations / clustering outputs are there,
     display data is used as is (or rebuilt from the ABFs when it was left out), a server with data is not
     overwritten without --force.
"""
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import synthetic as S  # noqa: E402
from smoke_test import _outputs_spec, validate_dash_graph  # noqa: E402

PY = sys.executable
GUN = shutil.which("gunicorn", path=str(Path(PY).parent)) or "gunicorn"
PORT0 = int(os.environ.get("RELEASE_TEST_PORT", "8121"))


# ------------------------------------------------------------------ servers
class Server:
    def __init__(self, name, port, data=None, init=True):
        self.name, self.port = name, port
        self.base = f"http://127.0.0.1:{port}"
        self.data = Path(data or tempfile.mkdtemp(prefix=f"nanotag_rt_{name}_"))
        self.env = dict(os.environ, NANOTAG_DATA=str(self.data), NANOTAG_ALLOW_FAKE_ABF="1",
                        NANOTAG_VENDOR=str(ROOT / "vendor"), NANOTAG_SECRET_KEY=f"secret-{name}",
                        NANOTAG_WORKERS="2", NANOTAG_IMPORT_ROOTS=str(self.data / "incoming"),
                        PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
        self.procs = []
        if init:
            self.cli("init")
            self.cli("ensure-admin")
            self.cli("register-model", str(ROOT / "vendor" / "best_opt_strict.pt"), "--default")

    def cli(self, *args, check=True):
        r = subprocess.run([PY, "-m", "nanotag.cli", *args], env=self.env, cwd=ROOT, capture_output=True, text=True)
        if check and r.returncode:
            raise RuntimeError(f"nanotag-admin {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r

    def py(self, code):
        """Run Python inside this server's configuration (same database and folders)."""
        r = subprocess.run([PY, "-c", code], env=self.env, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        return r.stdout.strip()

    def start(self):
        self.procs = [subprocess.Popen([GUN, "-b", f"127.0.0.1:{self.port}", "-w", "2", "--threads", "4",
                                        "--timeout", "300", "nanotag.wsgi:app"], env=self.env, cwd=ROOT,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                      subprocess.Popen([PY, "-m", "nanotag.worker"], env=self.env, cwd=ROOT,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
        for _ in range(80):
            try:
                if requests.get(self.base + "/healthz", timeout=2).ok:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"server {self.name} did not start")

    def stop(self):
        for p in self.procs:
            p.send_signal(signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(40)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []

    def session(self, user="admin", pw="admin2025!!"):
        s = requests.Session()
        s.headers["Referer"] = self.base + "/experiments"
        r = s.post(self.base + "/login", data={"username": user, "password": pw})
        assert r.ok and "Invalid username" not in r.text, f"{user} cannot sign in on {self.name}"
        return s

    def wait_jobs(self, s, exp_id=None, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = s.get(self.base + "/api/jobs", params={"experiment_id": exp_id} if exp_id else {}).json()
            if not [j for j in d["jobs"] if j["status"] in ("queued", "running", "cancelling")]:
                return d
            time.sleep(1)
        raise TimeoutError("jobs did not finish")

    def dash(self, s, output, inputs, state=()):
        body = {"output": output, "outputs": _outputs_spec(output),
                "inputs": [dict(id=i, property=p, value=v) for i, p, v in inputs],
                "state": [dict(id=i, property=p, value=v) for i, p, v in state],
                "changedPropIds": [f"{inputs[0][0]}.{inputs[0][1]}"]}
        r = s.post(self.base + "/tagger/_dash-update-component", json=body)
        assert r.status_code < 400, f"dash {output}: HTTP {r.status_code} {r.text[:300]}"
        return r.json()["response"] if r.status_code == 200 else {}


def ok(r, what):
    assert r.status_code < 400, f"{what}: HTTP {r.status_code} {r.text[:300]}"
    return r


def upload(srv, s, path, exp_id):
    size = path.stat().st_size
    init = ok(s.post(srv.base + "/api/uploads/init", json=dict(filename=path.name, size=size, purpose="abf",
                                                                experiment_id=exp_id)), "upload init").json()
    cs = init["chunk_size"]
    with open(path, "rb") as f:
        for i in range(max(1, -(-size // cs))):
            f.seek(i * cs)
            ok(s.put(f"{srv.base}/api/uploads/{init['upload_id']}/chunk", params={"index": i}, data=f.read(cs)), "chunk")
    ok(s.post(f"{srv.base}/api/uploads/{init['upload_id']}/complete"), "upload complete")


def job_status(srv, jid):
    return json.loads(srv.py(f"from nanotag import jobs; import json; print(json.dumps(jobs.get_job({jid})))"))


def wait_status(srv, jid, want, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = job_status(srv, jid)
        if j["status"] in want:
            return j
        time.sleep(0.3)
    raise TimeoutError(f"job {jid} still {j['status']}, wanted {want}")


def pid_alive(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return st != "Z"
    except OSError:
        return False


# ------------------------------------------------------------------ 1. tagger
def test_tagger(srv, s, exp_id, rid):
    r = ok(s.post(f"{srv.base}/experiments/{exp_id}/sets/new", data={"name": "Fresh manual"}), "new set")
    info = s.get(f"{srv.base}/api/experiments/{exp_id}").json()
    sid = next(x["id"] for x in info["sets"] if x["name"] == "Fresh manual")
    s.get(f"{srv.base}/tagger/?rec={rid}&set={sid}")
    layout = ok(s.get(srv.base + "/tagger/_dash-layout"), "layout").text
    deps = ok(s.get(srv.base + "/tagger/_dash-dependencies"), "deps").json()
    validate_dash_graph(deps)

    # table: no 25-row pages; header fixed; scrolls
    lay = json.loads(layout)

    def find(node, cid):
        if isinstance(node, dict):
            if node.get("props", {}).get("id") == cid:
                return node
            for v in node.values():
                f = find(v, cid)
                if f:
                    return f
        elif isinstance(node, list):
            for v in node:
                f = find(v, cid)
                if f:
                    return f
        return None
    tp = find(lay, "table")["props"]
    assert tp["page_size"] >= 2000 and tp["fixed_rows"] == {"headers": True}, tp.get("page_size")
    assert tp["style_table"]["overflowY"] == "scroll", tp["style_table"]
    assert "nt-mpan" in ok(s.get(f"{srv.base}/tagger/?rec={rid}&set={sid}"), "tagger page").text  # middle-button pan
    print("  table: one scrolling list (page size", tp["page_size"], "), fixed header; middle-button pan script present")

    load = next(d for d in deps if d["inputs"] == [{"id": "url", "property": "search"}])
    resp = srv.dash(s, load["output"], [("url", "search", f"?rec={rid}&set={sid}")], [("allfiles", "value", [])])
    ctx, events, window = resp["ctx"]["data"], resp["events"]["data"], resp["window"]["data"]
    assert events == [] and resp["authors"]["value"] == [], "a fresh set starts with no events and no authors"

    # draw a window (Add Window mode) on the empty set
    render = next(d for d in deps if d["output"].startswith("..graph.figure"))
    ins = {"events": events, "window": window, "nav": 0, "zoom": {}, "click": {"stage": "idle"}, "vis": [2, 3, 0],
           "mode": "zoom", "sel": None, "opts": [], "authors": [], "budget": 2000, "drag": "zoom"}
    st = {"ctx": ctx, "rev": resp["rev"]["data"], "allfiles": [], "panst": {"prev": None, "skip": None}}
    fig = srv.dash(s, render["output"], [(i["id"], i["property"], ins[i["id"]]) for i in render["inputs"]],
                   [(x["id"], x["property"], st[x["id"]]) for x in render["state"]])
    tmap = fig["tmap"]["data"]
    rel = next(d for d in deps if any(i["property"] == "relayoutData" for i in d["inputs"]))
    rst = {"ctx": ctx, "mode": "window", "window": window, "zoom": {}, "events": events, "sel": None, "opts": [],
           "tmap": tmap, "allfiles": []}
    res = srv.dash(s, rel["output"], [("graph", "relayoutData", {"selections": [{"x0": 3.0, "x1": 4.0}]})],
                   [(x["id"], x["property"], rst[x["id"]]) for x in rel["state"]])
    events = res["events"]["data"]
    assert len(events) == 1, events

    # before the fix the author filter stayed [] here and the new window was not drawn
    sync = next(d for d in deps if d["inputs"] == [{"id": "events", "property": "data"}]
                and "authors.value" in d["output"])

    def sync_call(evs, options, value):
        sv = {("ctx", "data"): ctx, ("authors", "options"): options, ("authors", "value"): value}
        return srv.dash(s, sync["output"], [("events", "data", evs)],
                        [(x["id"], x["property"], sv[(x["id"], x["property"])]) for x in sync["state"]])
    sres = sync_call(events, [], [])
    assert sres["authors"]["value"] == ["admin"], sres
    fig = srv.dash(s, render["output"],
                   [(i["id"], i["property"], dict(ins, events=events, authors=sres["authors"]["value"])[i["id"]])
                    for i in render["inputs"]],
                   [(x["id"], x["property"], st[x["id"]]) for x in render["state"]])
    rects = [sh for sh in fig["graph"]["figure"]["layout"].get("shapes", []) if sh.get("type") == "rect"]
    assert len(rects) == 1, rects
    # an author the user unticked stays unticked when someone else adds an event
    sres2 = sync_call(events, [{"label": "zed", "value": "zed"}, {"label": "admin", "value": "admin"}], ["zed"])
    # 'zed' no longer has events and 'admin' was known (unticked by the user): admin stays unticked
    assert sres2["authors"]["value"] == [], sres2
    print("  new events on an empty set are drawn at once (author filter follows new authors)")


# ------------------------------------------------------------------ 2. jobs
def test_jobs(srv, s, alice):
    enq = ("from nanotag import jobs; print(jobs.enqueue('_test_sleep', {'seconds': %d}, 'admin', None))")
    j1 = int(srv.py(enq % 120))
    j2 = int(srv.py(enq % 120))
    j3 = int(srv.py(enq % 120))      # 2 worker slots: this one waits in the queue
    wait_status(srv, j1, ("running",))
    wait_status(srv, j2, ("running",))
    time.sleep(2.0)
    log = job_status(srv, j1)["log"]
    child = int(re.search(r"child pid (\d+)", log).group(1))
    assert pid_alive(child)

    # alice may not stop admin's job
    alice.post(f"{srv.base}/jobs/{j1}/cancel")
    assert job_status(srv, j1)["status"] == "running"
    # queued -> cancelled at once
    ok(s.post(f"{srv.base}/jobs/{j3}/cancel"), "cancel queued")
    assert job_status(srv, j3)["status"] == "cancelled"
    # running -> cancelling -> cancelled, process group killed
    t0 = time.time()
    r = ok(s.post(f"{srv.base}/jobs/{j1}/cancel"), "stop running")
    assert "Stopping job" in r.text, r.text[:500]
    j = wait_status(srv, j1, ("cancelled",), timeout=20)
    assert "Stop requested by admin" in j["log"] and "stopped by user" in j["log"], j["log"]
    for _ in range(20):
        if not pid_alive(child):
            break
        time.sleep(0.25)
    assert not pid_alive(child), "the job's child process was not killed"
    print(f"  running job stopped in {time.time() - t0:.1f} s (child process killed); queued job cancelled; "
          f"other users cannot stop it")
    # the other running job is untouched
    assert job_status(srv, j2)["status"] == "running"
    ok(s.post(f"{srv.base}/jobs/{j2}/cancel"), "stop 2")
    wait_status(srv, j2, ("cancelled",), timeout=20)
    # pages show Stop for running jobs
    j4 = int(srv.py(enq % 60))
    wait_status(srv, j4, ("running",))
    page = s.get(srv.base + "/jobs").text
    assert "⏹ Stop" in page
    assert "Stop job" in s.get(f"{srv.base}/jobs/{j4}").text
    ok(s.post(f"{srv.base}/jobs/{j4}/cancel"), "stop 4")
    wait_status(srv, j4, ("cancelled",), timeout=20)

    # clean-up of a stopped clustering run (status + partial outputs)
    out = srv.py("""
from nanotag import jobs
from nanotag.config import cfg
from nanotag.db import db, now, one
with db() as con:
    exp = one(con, "SELECT id FROM experiments LIMIT 1")["id"]
    rid = con.execute("INSERT INTO cluster_runs(experiment_id, name, set_ids_json, params_json, status, created_by, "
                      "created_at) VALUES (?,?,?,?,?,?,?)", (exp, 'x', '[]', '{}', 'running', 'admin', now())).lastrowid
    jid = con.execute("INSERT INTO jobs(kind, experiment_id, params_json, status, created_by, created_at) "
                      "VALUES ('cluster', ?, ?, 'cancelling', 'admin', ?)", (exp, '{"run_id": %d}' % rid, now())).lastrowid
d = cfg().JOBS_DIR / f"cluster_run_{rid}"; d.mkdir(parents=True); (d / "partial.csv").write_text("x")
assert jobs.finish_cancel(jid)
with db() as con:
    print(one(con, "SELECT status FROM cluster_runs WHERE id=?", (rid,))["status"], d.exists())
""")
    assert out == "cancelled False", out
    print("  stopped clustering run marked cancelled, partial outputs removed")


# ------------------------------------------------------------------ 3. moving a server
def test_move(src, s, exp_id, n_events_by_set, tmp):
    # a clustering output folder to carry over
    run_id = int(src.py(f"""
from nanotag.config import cfg
from nanotag.db import db, now
with db() as con:
    rid = con.execute("INSERT INTO cluster_runs(experiment_id, name, set_ids_json, params_json, status, final_k, "
                      "created_by, created_at) VALUES ({exp_id}, 'carried', '[]', '{{}}', 'done', 2, 'admin', ?)",
                      (now(),)).lastrowid
    d = cfg().JOBS_DIR / f"cluster_run_{{rid}}"; d.mkdir(parents=True, exist_ok=True)
    (d / "cluster_report.html").write_text("<html>report</html>")
    con.execute("UPDATE cluster_runs SET out_dir=? WHERE id=?", (str(d), rid))
print(rid)
"""))
    adm = s.get(src.base + "/admin").text
    assert "Export data and configuration" in adm and "migrate-from.sh" in adm
    summ = s.get(src.base + "/api/admin/site-summary").json()
    assert summ["counts"]["users"] == 2 and summ["bytes"]["zarr"] > 0, summ

    full = tmp / "site_full.tar"
    with s.get(src.base + "/admin/export-site", params={"zarr": "1", "abf": "1"}, stream=True) as r:
        ok(r, "export")
        assert r.headers["Content-Type"].startswith("application/x-tar")
        with open(full, "wb") as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
    import tarfile
    with tarfile.open(full) as tf:
        names = tf.getnames()
    assert {"manifest.json", "db/nanotag.sqlite3", "config/nanotag.env"} <= set(names), names[:10]
    assert any(n.startswith("zarr/store.zarr/") for n in names) and any(n.startswith("abf/") for n in names)
    assert any(n.startswith(f"jobs/cluster_run_{run_id}/") for n in names) and any(n.startswith("models/") for n in names)
    print(f"  web export: {full.stat().st_size / 1e6:.1f} MB, {len(names)} entries")
    lite = tmp / "site_lite.tar"
    out = src.cli("export-site", "--out", str(lite)).stdout
    assert f"BUNDLE={lite}" in out, out

    # A) fresh server, full bundle: works at once, old passwords, same content
    dst = Server("dst", PORT0 + 1)
    rep = dst.cli("import-site", str(full)).stdout
    assert "Imported 2 user(s) and 1 experiment(s)" in rep, rep
    dst.start()
    try:
        a = dst.session("alice", "alicepass1")
        sa = dst.session()
        info = sa.get(f"{dst.base}/api/experiments/{exp_id}").json()
        assert all(r["status"] == "ready" for r in info["recordings"]), info["recordings"]
        assert {x["name"]: x["n_events"] for x in info["sets"]} == n_events_by_set, info["sets"]
        assert a.get(f"{dst.base}/experiments/{exp_id}").ok
        assert sa.get(f"{dst.base}/cluster/{run_id}/report").text == "<html>report</html>"
        # paths point into the new data folder
        con = sqlite3.connect(dst.data / "db" / "nanotag.sqlite3")
        paths = [r[0] for r in con.execute("SELECT abf_path FROM recordings")]
        mpaths = [r[0] for r in con.execute("SELECT path FROM models")]
        con.close()
        assert all(p.startswith(str(dst.data)) and Path(p).exists() for p in paths + mpaths), (paths, mpaths)
        # the tagger reads the moved display data
        rid = info["recordings"][0]["id"]
        sid = info["sets"][0]["id"]
        sa.get(f"{dst.base}/tagger/?rec={rid}&set={sid}")
        deps = sa.get(dst.base + "/tagger/_dash-dependencies").json()
        load = next(d for d in deps if d["inputs"] == [{"id": "url", "property": "search"}])
        res = dst.dash(sa, load["output"], [("url", "search", f"?rec={rid}&set={sid}")], [("allfiles", "value", [])])
        assert res["ctx"]["data"]["rec_id"] == rid
        print("  imported on a fresh server: users sign in with old passwords, experiments, annotations, "
              "clustering outputs, display data and ABFs in place")

        # B) a server with data is not replaced without --force
        r = dst.cli("import-site", str(lite), check=False)
        assert r.returncode and "--force" in (r.stdout + r.stderr), r.stdout + r.stderr
        print("  a server that already has data is not overwritten without --force")
    finally:
        dst.stop()

    # C) bundle without display data but with ABFs -> recordings are re-ingested on the new server
    abf_only = tmp / "site_abf.tar"
    with s.get(src.base + "/admin/export-site", params={"abf": "1"}, stream=True) as r:
        with open(abf_only, "wb") as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
    d3 = Server("d3", PORT0 + 2)
    out = d3.cli("import-site", str(abf_only)).stdout
    assert "queued for ingest" in out, out
    d3.start()
    try:
        s3 = d3.session("alice", "alicepass1")
        d3.wait_jobs(s3, exp_id)
        recs = s3.get(f"{d3.base}/api/jobs", params={"experiment_id": exp_id}).json()["recordings"]
        assert recs and all(r["status"] == "ready" for r in recs), recs
        print("  bundle without display data: recordings re-ingested from the ABFs on the new server")
    finally:
        d3.stop()

    # D) data + configuration only (no Zarr, no ABF) with --force over an existing server
    d4 = Server("d4", PORT0 + 3)
    d4.cli("add-user", "bob", "--password", "bobpass123")
    r = d4.cli("import-site", str(lite), check=False)
    assert r.returncode, "must refuse"
    out = d4.cli("import-site", str(lite), "--force").stdout
    assert "Previous database" in out and "marked 'missing'" in out, out
    con = sqlite3.connect(d4.data / "db" / "nanotag.sqlite3")
    st = {r[0] for r in con.execute("SELECT status FROM recordings")}
    users = {r[0] for r in con.execute("SELECT username FROM users")}
    n_ev = con.execute("SELECT COUNT(*) FROM events WHERE deleted=0").fetchone()[0]
    con.close()
    assert st == {"missing"} and users == {"admin", "alice"} and n_ev == sum(n_events_by_set.values()), (st, users)
    assert list((d4.data / "backups").glob("nanotag-pre-import-*.sqlite3"))
    print("  data-only bundle with --force: annotations kept, recordings wait for their ABFs, old DB backed up")
    for d in (dst, d3, d4):
        shutil.rmtree(d.data, ignore_errors=True)


# ------------------------------------------------------------------ main
def main():
    tmp = Path(tempfile.mkdtemp(prefix="nanotag_rt_"))
    src = Server("src", PORT0)
    src.start()
    try:
        s = src.session()
        ok(s.post(src.base + "/admin/users/add", data={"username": "alice", "password": "alicepass1",
                                                       "role": "user"}), "add alice")
        alice = src.session("alice", "alicepass1")
        r = ok(s.post(src.base + "/experiments/new", data={"name": "Release SiO2", "sensor": "B3S8"}), "new exp")
        exp_id = int(re.search(r"/experiments/(\d+)", r.url).group(1))
        from nanotag.abfio import write_fake_abf
        d0, _ = S.make_recording(0, duration=40.0, n_events=8)
        f0 = tmp / "rel_a.fakeabf.npz"
        write_fake_abf(f0, d0, 50000, S.NAMES, S.UNITS)
        upload(src, s, f0, exp_id)
        src.wait_jobs(s, exp_id)
        info = s.get(f"{src.base}/api/experiments/{exp_id}").json()
        rid = info["recordings"][0]["id"]
        assert info["recordings"][0]["status"] == "ready", info
        print("1. tagger")
        test_tagger(src, s, exp_id, rid)
        print("2. stopping jobs")
        test_jobs(src, s, alice)
        print("3. moving the server")
        info = s.get(f"{src.base}/api/experiments/{exp_id}").json()
        test_move(src, s, exp_id, {x["name"]: x["n_events"] for x in info["sets"]}, tmp)
        print("\nRELEASE TEST PASSED")
    finally:
        src.stop()
        shutil.rmtree(src.data, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
