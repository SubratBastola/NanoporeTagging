"""End-to-end test: starts a throw-away NanoTag (gunicorn + worker) on synthetic data and
drives it over HTTP exactly like a browser would.

    python tests/smoke_test.py            (from the source/release folder, inside the venv)

Uses NANOTAG_ALLOW_FAKE_ABF=1 so no real ABF is needed. Nothing touches /srv/nanotag.
"""
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import synthetic as S  # noqa: E402

PORT = int(os.environ.get("SMOKE_PORT", "8099"))
BASE = f"http://127.0.0.1:{PORT}"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nanotag_smoke_"))
    env = dict(os.environ, NANOTAG_DATA=str(tmp), NANOTAG_ALLOW_FAKE_ABF="1", NANOTAG_VENDOR=str(ROOT / "vendor"),
               NANOTAG_SECRET_KEY="smoke", NANOTAG_WORKERS="2", NANOTAG_IMPORT_ROOTS=str(tmp / "incoming"),
               PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    py = sys.executable
    subprocess.run([py, "-m", "nanotag.cli", "init"], env=env, check=True, cwd=ROOT)
    subprocess.run([py, "-m", "nanotag.cli", "ensure-admin"], env=env, check=True, cwd=ROOT)
    subprocess.run([py, "-m", "nanotag.cli", "register-model", str(ROOT / "vendor" / "best_opt_strict.pt"),
                    "--default"], env=env, check=True, cwd=ROOT)
    gun = shutil.which("gunicorn", path=str(Path(py).parent)) or "gunicorn"
    web = subprocess.Popen([gun, "-b", f"127.0.0.1:{PORT}", "-w", "2", "--threads", "4", "--timeout", "300",
                            "nanotag.wsgi:app"], env=env, cwd=ROOT)
    worker = subprocess.Popen([py, "-m", "nanotag.worker"], env=env, cwd=ROOT)
    try:
        run_checks(tmp)
        print("\nSMOKE TEST PASSED")
    finally:
        for p in (web, worker):
            p.send_signal(signal.SIGTERM)
        for p in (web, worker):
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def wait_up():
    for _ in range(60):
        try:
            if requests.get(BASE + "/healthz", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not start")


def wait_jobs(s, exp_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = s.get(f"{BASE}/api/jobs", params={"experiment_id": exp_id}).json()
        active = [j for j in d["jobs"] if j["status"] in ("queued", "running")]
        if not active:
            bad = [j for j in d["jobs"] if j["status"] == "error"]
            if bad:
                for j in bad:
                    print(s.get(f"{BASE}/api/jobs/{j['id']}").json()["log"][-3000:])
                raise AssertionError(f"job(s) failed: {[(j['id'], j['kind'], j['message']) for j in bad]}")
            return d
        time.sleep(1)
    raise TimeoutError("jobs did not finish")


def ok(r, what):
    assert r.status_code < 400, f"{what}: HTTP {r.status_code} {r.text[:400]}"
    return r


def dash_call(s, output, inputs, state=(), changed=None):
    """Invoke a Dash callback the way the browser does."""
    body = {"output": output, "outputs": _outputs_spec(output),
            "inputs": [dict(id=i, property=p, value=v) for i, p, v in inputs],
            "state": [dict(id=i, property=p, value=v) for i, p, v in state],
            "changedPropIds": changed or [f"{inputs[0][0]}.{inputs[0][1]}"]}
    r = s.post(f"{BASE}/tagger/_dash-update-component", json=body)
    ok(r, f"dash {output}")
    return r.json()["response"]


def _outputs_spec(output):
    parts = output.strip(".").split("...") if output.startswith("..") else [output]
    specs = []
    for p in parts:
        if not p:
            continue
        cid, prop = p.rsplit(".", 1)
        prop = prop.split("@")[0]
        specs.append({"id": cid, "property": prop})
    return specs if output.startswith("..") else specs[0]


def run_checks(tmp):
    wait_up()
    s = requests.Session()
    r = s.post(BASE + "/login", data={"username": "admin", "password": "wrong"})
    assert "Invalid username" in r.text
    ok(s.post(BASE + "/login", data={"username": "admin", "password": "admin2025!!"}), "login")
    assert "default password" in s.get(BASE + "/experiments").text
    print("login ok")

    # users
    ok(s.post(BASE + "/admin/users/add", data={"username": "alice", "password": "alicepass1", "role": "user"}), "add user")
    alice = requests.Session()
    ok(alice.post(BASE + "/login", data={"username": "alice", "password": "alicepass1"}), "alice login")
    assert alice.get(BASE + "/admin").status_code == 403

    # experiment
    r = ok(s.post(BASE + "/experiments/new", data={"name": "Smoke SiO2", "sensor": "B3S8"}), "new exp")
    exp_id = int(re.search(r"/experiments/(\d+)", r.url).group(1))
    print("experiment", exp_id)

    # recording 0 via chunked upload, recording 1 via server-folder import
    truth = {}
    d0, ev0 = S.make_recording(0)
    d1, ev1 = S.make_recording(1)
    f0 = tmp / "rec_a.fakeabf.npz"
    np_save(f0, d0)
    inc = tmp / "incoming"
    inc.mkdir(exist_ok=True)
    f1 = inc / "rec_b.fakeabf.npz"
    np_save(f1, d1)
    truth["rec_a.fakeabf.npz"], truth["rec_b.fakeabf.npz"] = ev0, ev1
    upload(s, f0, {"purpose": "abf", "experiment_id": exp_id}, chunk_resume_test=True)
    b = s.get(BASE + "/api/browse", params={"path": str(inc)}).json()
    assert any(f["name"] == "rec_b.fakeabf.npz" for f in b["files"]), b
    ok(s.post(f"{BASE}/experiments/{exp_id}/import-folder", data={"files": [str(f1)], "mode": "link"}), "folder import")
    assert s.get(BASE + "/api/browse", params={"path": "/etc"}).status_code == 403
    wait_jobs(s, exp_id)
    recs = s.get(f"{BASE}/api/jobs", params={"experiment_id": exp_id}).json()["recordings"]
    assert len(recs) == 2 and all(r["status"] == "ready" for r in recs), recs
    print("ingest ok", recs)

    # import window CSVs (one file for both recordings)
    df = pd.concat([S.windows_csv("rec_a.fakeabf.npz", ev0), S.windows_csv("rec_b.fakeabf.npz", ev1),
                    S.windows_csv("unknown_file.abf", ev1[:2])])
    r = ok(s.post(f"{BASE}/experiments/{exp_id}/import-events",
                  files={"files": ("windows.csv", df.to_csv(index=False).encode(), "text/csv")},
                  data={"new_name": "Windows"}), "import events")
    assert "Unmatched rows" in r.text and "unknown_file" in r.text
    info = s.get(f"{BASE}/api/experiments/{exp_id}").json()
    sid = next(x["id"] for x in info["sets"] if x["name"] == "Windows")
    rid = next(r["id"] for r in info["recordings"] if r["file_name"] == "rec_a.fakeabf.npz")
    assert next(x for x in info["sets"] if x["name"] == "Windows")["n_events"] == len(ev0) + len(ev1)
    print("windows set", sid, "rec", rid)

    # ---- Tagger via Dash callbacks
    s.get(f"{BASE}/tagger/?rec={rid}&set={sid}")
    layout = ok(s.get(f"{BASE}/tagger/_dash-layout"), "dash layout").json()
    deps = ok(s.get(f"{BASE}/tagger/_dash-dependencies"), "dash deps").json()
    validate_dash_graph(deps)
    load_out = next(d["output"] for d in deps if d["inputs"] == [{"id": "url", "property": "search"}])
    resp = dash_call(s, load_out, [("url", "search", f"?rec={rid}&set={sid}")])
    ctx = resp["ctx"]["data"]
    events = resp["events"]["data"]
    window = resp["window"]["data"]
    vis = resp["vis"]["value"]
    assert ctx["roles"] == {"opt": 2, "optref": 3, "elec": 0} and vis == [2, 3, 0], (ctx["roles"], vis)
    assert len(events) == len(ev0 if rid == 1 else ev1)
    print("tagger load ok:", len(events), "events, window", window)
    render_dep = next(d for d in deps if d["output"].startswith("..graph.figure"))
    ins = {"events": events, "window": window, "nav": 0, "zoom": {}, "click": {"stage": "idle"}, "vis": vis,
           "mode": "zoom", "sel": None, "opts": [], "authors": None, "budget": 8000,
           "pan": {"on": False, "prev": None, "skip": None}}
    props = {"events": "data", "window": "data", "nav": "data", "zoom": "data", "click": "data", "vis": "value",
             "mode": "value", "sel": "data", "opts": "value", "authors": "value", "budget": "value"}
    rr = dash_call(s, render_dep["output"], [(i["id"], i["property"], ins[i["id"]]) for i in render_dep["inputs"]],
                   [(st["id"], st["property"], {"ctx": ctx, "rev": resp["rev"]["data"]}[st["id"]])
                    for st in render_dep["state"]])
    fig = rr["graph"]["figure"]
    tmap = rr["tmap"]["data"]
    import base64
    import numpy as np

    def npts(v):
        if isinstance(v, dict) and "bdata" in v:
            return len(np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"]))
        return len(v or [])
    n_pts = [npts(t.get("x")) for t in fig["data"][:6]]
    size_kb = len(json.dumps(rr)) / 1024
    assert all(n > 100 for n in n_pts), n_pts
    print("render ok: points per trace", n_pts, f"payload {size_kb:.0f} KB |", fig["layout"]["xaxis"]["title"]["text"])
    assert len(tmap["traces"]) == 6
    assert abs(fig["layout"]["xaxis"]["rangeslider"]["thickness"] - 0.063) < 1e-9
    assert fig["layout"]["dragmode"] == "zoom"

    # add an event by clicking 3 optical + 4 electrical points
    click_dep = next(d for d in deps if d["output"].startswith("..click.data"))
    click = {"stage": "optical", "optical_times": [], "electrical_times": []}
    ev = ev0[0] if rid == 1 else ev1[0]
    t_s, t_e = ev["start"], ev["end"]
    seq = [(0, t_s), (0, (t_s + t_e) / 2), (0, t_e), (4, t_s - 0.01), (4, t_s + 0.001), (4, t_e + 0.001), (4, t_e + 0.02)]
    for curve, t in seq:  # curve 0..2 = overview traces (opt, optref, elec); 3..5 detail; use detail
        cn = 3 if curve == 0 else 5
        res = dash_call(s, click_dep["output"], [("graph", "clickData", {"points": [{"curveNumber": cn, "x": t}]})],
                        [("click", "data", click), ("mode", "value", "add"), ("ctx", "data", ctx), ("tmap", "data", tmap)])
        if "click" in res:
            click = res["click"]["data"]
    assert "events" in res, res
    new_events = res["events"]["data"]
    added = [e for e in new_events if e["event_start"] is not None and e["created_by"] == "admin" and e["entry_base_t"]]
    assert added and abs(added[-1]["event_start"] - t_s) < 1e-6, added
    assert added[-1]["optical_rise"] is not None and added[-1]["entry_peak"] is not None
    print("add-event via clicks ok: event_no", added[-1]["event_no"])

    # alice edits the event via Edit-mode guide drag (version check), admin sees it via poll
    s_al = alice
    s_al.get(f"{BASE}/tagger/?rec={rid}&set={sid}")
    resp_a = dash_call(s_al, load_out, [("url", "search", f"?rec={rid}&set={sid}")])
    ctx_a = resp_a["ctx"]["data"]
    target = next(e for e in resp_a["events"]["data"] if e["id"] == added[-1]["id"])
    rel_dep = next(d for d in deps if any(i["property"] == "relayoutData" for i in d["inputs"]))
    smap_state = {"traces": tmap["traces"], "order": tmap["order"], "smap": ["event_start", "event_start",
                                                                             "event_plateau", "event_plateau",
                                                                             "event_end", "event_end",
                                                                             "entry_base_t", "entry_peak_t",
                                                                             "exit_peak_t", "exit_base_t"]}
    stv = {"ctx": ctx_a, "mode": "edit", "window": window, "zoom": {}, "events": resp_a["events"]["data"],
           "sel": target["id"], "opts": [], "tmap": smap_state}
    res = dash_call(s_al, rel_dep["output"], [("graph", "relayoutData", {"shapes[0].x0": t_s + 0.005,
                                                                         "shapes[0].x1": t_s + 0.005})],
                    [(st["id"], st["property"], stv[st["id"]]) for st in rel_dep["state"]])
    upd = next(e for e in res["events"]["data"] if e["id"] == target["id"])
    assert abs(upd["event_start"] - (t_s + 0.005)) < 1e-6 and upd["updated_by"] == "alice", upd
    # stale version from admin must be rejected
    stv2 = dict(stv, ctx=ctx, events=new_events)
    res2 = dash_call(s, rel_dep["output"], [("graph", "relayoutData", {"shapes[0].x0": t_s, "shapes[0].x1": t_s})],
                     [(st["id"], st["property"], stv2[st["id"]]) for st in rel_dep["state"]])
    assert "changed by alice" in json.dumps(res2["msg"]), res2["msg"]
    print("multi-user edit + conflict detection ok")

    # window + delete via selection
    stv3 = dict(stv, ctx=ctx, mode="window")
    res = dash_call(s, rel_dep["output"], [("graph", "relayoutData", {"selections": [{"x0": 1.0, "x1": 1.5}]})],
                    [(st["id"], st["property"], stv3[st["id"]]) for st in rel_dep["state"]])
    assert "Window added" in json.dumps(res["msg"])
    stv4 = dict(stv, ctx=ctx, mode="delete", events=res["events"]["data"])
    res = dash_call(s, rel_dep["output"], [("graph", "relayoutData", {"selections": [{"x0": 0.9, "x1": 1.6}]})],
                    [(st["id"], st["property"], stv4[st["id"]]) for st in rel_dep["state"]])
    assert "Removed 1 event" in json.dumps(res["msg"]), res["msg"]
    print("window add + delete ok")

    # ---- Pan toggle: pauses Add Event (mode -> zoom), keeps pending clicks, restores the mode
    pan_dep = next(d for d in deps if d["inputs"] == [{"id": "pan-btn", "property": "n_clicks"}])
    mode_dep = next(d for d in deps if d["inputs"] == [{"id": "mode", "property": "value"}])
    ui_dep = next(d for d in deps if d["output"].startswith("..pan-btn.className"))
    pend = {"stage": "optical", "optical_times": [t_s], "electrical_times": []}
    r = dash_call(s, pan_dep["output"], [("pan-btn", "n_clicks", 1)],
                  [("pan", "data", {"on": False, "prev": None, "skip": None}), ("mode", "value", "add")])
    pan_on = r["pan"]["data"]
    assert pan_on == {"on": True, "prev": "add", "skip": "zoom"} and r["mode"]["value"] == "zoom", r
    mst = {"ctx": ctx, "events": new_events, "window": window, "sel": None, "pan": pan_on}
    r = dash_call(s, mode_dep["output"], [("mode", "value", "zoom")],
                  [(st["id"], st["property"], mst[st["id"]]) for st in mode_dep["state"]])
    assert "click" not in r and r["pan"]["data"]["skip"] is None, r   # pending clicks untouched
    pan_on = r["pan"]["data"]
    ins_p = dict(ins, mode="zoom", click=pend, pan=pan_on)
    rr = dash_call(s, render_dep["output"], [(i["id"], i["property"], ins_p[i["id"]]) for i in render_dep["inputs"]],
                   [(st["id"], st["property"], {"ctx": ctx, "rev": resp["rev"]["data"]}[st["id"]])
                    for st in render_dep["state"]])
    lay = rr["graph"]["figure"]["layout"]
    assert lay["dragmode"] == "pan", lay["dragmode"]
    assert any("Pan" in a.get("text", "") and "Add Event paused" in a.get("text", "") for a in lay["annotations"])
    assert any(t.get("marker", {}).get("color") == "#ea580c" for t in rr["graph"]["figure"]["data"])  # pending O1
    r = dash_call(s, ui_dep["output"], [("pan", "data", pan_on)], [("pan-badge", "style", {"display": "none"})])
    assert r["pan-btn"]["className"] == "on" and r["pan-badge"]["style"]["display"] == "block", r
    r = dash_call(s, pan_dep["output"], [("pan-btn", "n_clicks", 2)], [("pan", "data", pan_on), ("mode", "value", "zoom")])
    assert r["mode"]["value"] == "add" and r["pan"]["data"] == {"on": False, "prev": None, "skip": "add"}, r
    r = dash_call(s, mode_dep["output"], [("mode", "value", "add")],
                  [(st["id"], st["property"], dict(mst, pan=r["pan"]["data"])[st["id"]]) for st in mode_dep["state"]])
    assert "click" not in r and r["pan"]["data"] == {"on": False, "prev": None, "skip": None}, r
    # picking a click/drag mode by hand while panning turns Pan off and starts that mode fresh
    r = dash_call(s, mode_dep["output"], [("mode", "value", "window")],
                  [(st["id"], st["property"], dict(mst, pan={"on": True, "prev": "add", "skip": None})[st["id"]])
                   for st in mode_dep["state"]])
    assert r["pan"]["data"]["on"] is False and r["click"]["data"]["optical_times"] == [], r
    # Pan from Zoom/Inspect: mode stays, nothing to restore
    r = dash_call(s, pan_dep["output"], [("pan-btn", "n_clicks", 3)],
                  [("pan", "data", {"on": False, "prev": None, "skip": None}), ("mode", "value", "zoom")])
    assert "mode" not in r and r["pan"]["data"] == {"on": True, "prev": None, "skip": None}, r
    print("pan toggle ok")

    # lock the set -> alice cannot edit
    ok(s.post(f"{BASE}/sets/{sid}/lock", data={"locked": "1"}), "lock")
    res = dash_call(s_al, rel_dep["output"], [("graph", "relayoutData", {"selections": [{"x0": 2.0, "x1": 2.5}]})],
                    [(st["id"], st["property"], dict(stv, mode="window")[st["id"]]) for st in rel_dep["state"]])
    assert "locked" in json.dumps(res["msg"]), res
    ok(s.post(f"{BASE}/sets/{sid}/lock", data={"locked": "0"}), "unlock")
    print("lock ok")

    # ---- NN
    model_id = s.get(f"{BASE}/api/experiments/{exp_id}").json()["models"][0]["id"]
    ok(s.post(f"{BASE}/experiments/{exp_id}/nn", data={"source_set_id": sid, "model_id": model_id,
                                                        "out_name": "NN run"}), "nn")
    wait_jobs(s, exp_id)
    nn_sid = next(x["id"] for x in s.get(f"{BASE}/api/experiments/{exp_id}").json()["sets"] if x["name"] == "NN run")
    x = s.get(f"{BASE}/sets/{nn_sid}/export", params={"layout": "nn", "fmt": "xlsx"})
    ok(x, "nn export")
    nn_df = pd.read_excel(io.BytesIO(x.content))
    assert list(nn_df.columns)[:5] == ["event_id", "file_name", "sensor", "analytes", "solution"]
    assert nn_df["event_start (s)"].notna().sum() >= len(ev0) + len(ev1) - 2, nn_df.head()
    errs = []
    for fname, evs in truth.items():
        sub = nn_df[nn_df["file_name"] == fname].sort_values("event_id")
        for (_, row), t in zip(sub.iterrows(), evs):
            errs.append(abs(row["event_start (s)"] - t["start"]))
    print(f"NN ok: {len(nn_df)} rows, median |start error| = {1000 * pd.Series(errs).median():.2f} ms")

    # ---- clustering on the NN set + the manual windows set
    ok(s.post(f"{BASE}/experiments/{exp_id}/cluster",
              data={"name": "smoke run", "set_ids": [nn_sid], "alpha": "0.05", "max_k": "4", "n_boot": "25",
                    "impute": "1", "features": ["duration", "entry_spike", "exit_spike"]}), "cluster")
    wait_jobs(s, exp_id)
    run = s.get(f"{BASE}/api/experiments/{exp_id}").json()["cluster_runs"][-1]
    assert run["status"] == "done", run
    run_id = run["id"]
    rep = ok(s.get(f"{BASE}/cluster/{run_id}/report"), "report")
    assert "Cluster Viewer Report" in rep.text
    z = zipfile.ZipFile(io.BytesIO(ok(s.get(f"{BASE}/cluster/{run_id}/download.zip"), "cluster zip").content))
    names = z.namelist()
    for need in ("pooled_cluster_labels.csv", "cluster_summary.csv", "clustering_results.xlsx", "cluster_report.html",
                 "bootstrap_lrt.csv", "k_breakdown.csv"):
        assert need in names, (need, names)
    labels = pd.read_csv(io.BytesIO(ok(s.get(f"{BASE}/cluster/{run_id}/labels.csv"), "labels").content))
    print(f"clustering ok: K={run['final_k']} (BIC-best {run['k_bic']}), labelled events={len(labels)}; "
          f"cluster sizes {labels['cluster'].value_counts().to_dict()}")
    # population check: duration-based clusters should match synthetic populations
    if run['final_k'] >= 2:
        lab = labels.merge(nn_df[["file_name", "event_id", "duration"]], on=["file_name", "event_id"])
        print("  mean duration per cluster:", lab.groupby("cluster")["duration"].mean().round(3).to_dict())

    # ---- exports + bundle round trip
    for q in ({"fmt": "csv"}, {"fmt": "xlsx"}, {"fmt": "csv", "audit": 1}, {"layout": "zip"}):
        ok(s.get(f"{BASE}/sets/{sid}/export", params=q), f"export {q}")
    hist = pd.read_csv(io.BytesIO(ok(s.get(f"{BASE}/sets/{sid}/history.csv"), "history").content))
    assert {"admin", "alice"} <= set(hist["user"]), hist["user"].unique()
    csv_audit = pd.read_csv(io.BytesIO(s.get(f"{BASE}/sets/{sid}/export", params={"fmt": "csv", "audit": 1}).content))
    assert "alice" in set(csv_audit["updated_by"])
    bundle = ok(s.get(f"{BASE}/experiments/{exp_id}/bundle.zip", params={"abf": "1"}), "bundle").content
    bz = zipfile.ZipFile(io.BytesIO(bundle))
    assert "manifest.json" in bz.namelist() and any(n.startswith("abf/") for n in bz.namelist())
    bpath = tmp / "bundle.zip"
    bpath.write_bytes(bundle)
    upload(s, bpath, {"purpose": "bundle"})
    for _ in range(300):
        exps = s.get(BASE + "/experiments").text
        if "Smoke SiO2 (imported 2)" in exps:
            break
        time.sleep(1)
    assert "Smoke SiO2 (imported 2)" in exps
    new_id = next(e["id"] for e in s.get(BASE + "/api/experiments").json() if e["name"] == "Smoke SiO2 (imported 2)")
    wait_jobs(s, new_id)
    info2 = s.get(f"{BASE}/api/experiments/{new_id}").json()
    names2 = {x["name"]: x["n_events"] for x in info2["sets"]}
    assert names2.get("NN run") and names2.get("Windows"), names2
    assert all(r["status"] == "ready" for r in info2["recordings"]), info2["recordings"]
    print("bundle export/import round-trip ok (experiment", new_id, ")")

    # ---- every HTML page renders
    for path in ["/experiments", f"/experiments/{exp_id}", f"/experiments/{new_id}", "/jobs", "/jobs/1", "/models",
                 "/admin", "/account", f"/tagger/?rec={rid}&set={sid}"]:
        r = ok(s.get(BASE + path), path)
        assert "Traceback" not in r.text
    print("all pages render")

    # ---- restricted experiment is hidden from alice
    ok(s.post(f"{BASE}/experiments/{exp_id}/access", data={"restricted": "1", "users": []}), "restrict")
    assert alice.get(f"{BASE}/experiments/{exp_id}").status_code == 404
    ok(s.post(f"{BASE}/experiments/{exp_id}/access", data={"restricted": "1", "users": ["alice"]}), "grant")
    assert alice.get(f"{BASE}/experiments/{exp_id}").status_code == 200
    print("access control ok")

    # ---- password change
    ok(alice.post(BASE + "/account", data={"current": "alicepass1", "new1": "alicepass2", "new2": "alicepass2"}), "pw")
    a2 = requests.Session()
    assert "Invalid username" not in a2.post(BASE + "/login", data={"username": "alice", "password": "alicepass2"}).text
    print("password change ok")


def validate_dash_graph(deps):
    """The checks Dash's browser renderer performs: no un-flagged duplicate outputs, no cycles."""
    def outs(d):
        o = d["output"]
        parts = o[2:-2].split("...") if o.startswith("..") else [o]
        return [p for p in parts if p]
    seen = {}
    for i, d in enumerate(deps):
        for o in outs(d):
            base = o.split("@")[0]
            seen.setdefault(base, []).append(o)
    for base, lst in seen.items():
        plain = [o for o in lst if "@" not in o]
        assert len(plain) <= 1, f"duplicate output without allow_duplicate: {base}"
    prod = {}
    for i, d in enumerate(deps):
        for o in outs(d):
            prod.setdefault(o.split("@")[0], set()).add(i)
    edges = {i: set() for i in range(len(deps))}
    for j, d in enumerate(deps):
        for inp in d["inputs"]:
            for i in prod.get(f"{inp['id']}.{inp['property']}", ()):
                edges[i].add(j)
    color = {}

    def dfs(u, path):
        color[u] = 1
        for v in edges[u]:
            if color.get(v) == 1:
                raise AssertionError("callback cycle: " + " -> ".join(deps[k]["output"][:60] for k in path + [v]))
            if not color.get(v):
                dfs(v, path + [v])
        color[u] = 2
    for u in edges:
        if not color.get(u):
            dfs(u, [u])
    print(f"dash graph ok: {len(deps)} callbacks, no cycles, duplicates flagged")


def np_save(path, data):
    from nanotag.abfio import write_fake_abf
    write_fake_abf(path, data, 50000, S.NAMES, S.UNITS)


def upload(s, path, opts, chunk_resume_test=False):
    size = path.stat().st_size
    init = ok(s.post(BASE + "/api/uploads/init", json=dict(filename=path.name, size=size, **opts)), "upload init").json()
    cs = init["chunk_size"]
    n = max(1, -(-size // cs))
    with open(path, "rb") as f:
        for i in range(n):
            if chunk_resume_test and i == 1:
                # simulate a dropped connection: re-init must report chunk 0 as received
                again = s.post(BASE + "/api/uploads/init", json=dict(filename=path.name, size=size, **opts)).json()
                assert again["upload_id"] == init["upload_id"] and 0 in again["received"], again
            f.seek(i * cs)
            ok(s.put(f"{BASE}/api/uploads/{init['upload_id']}/chunk", params={"index": i}, data=f.read(cs)), "chunk")
    return ok(s.post(f"{BASE}/api/uploads/{init['upload_id']}/complete"), "complete").json()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
