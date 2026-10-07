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
    s.headers["Referer"] = BASE + "/experiments"
    r = s.post(BASE + "/login", data={"username": "admin", "password": "wrong"})
    assert "Invalid username" in r.text
    ok(s.post(BASE + "/login", data={"username": "admin", "password": "admin2025!!"}), "login")
    assert "default password" in s.get(BASE + "/experiments").text
    print("login ok")

    # users
    ok(s.post(BASE + "/admin/users/add", data={"username": "alice", "password": "alicepass1", "role": "user"}), "add user")
    alice = requests.Session()
    alice.headers["Referer"] = BASE + "/experiments"
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
    resp = dash_call(s, load_out, [("url", "search", f"?rec={rid}&set={sid}")], [("allfiles", "value", [])])
    ctx = resp["ctx"]["data"]
    events = resp["events"]["data"]
    window = resp["window"]["data"]
    vis = resp["vis"]["value"]
    assert ctx["roles"] == {"opt": 2, "optref": 3, "elec": 0} and vis == [2, 3, 0], (ctx["roles"], vis)
    assert len(events) == len(ev0 if rid == 1 else ev1)
    print("tagger load ok:", len(events), "events, window", window)
    render_dep = next(d for d in deps if d["output"].startswith("..graph.figure"))
    ins = {"events": events, "window": window, "nav": 0, "zoom": {}, "click": {"stage": "idle"}, "vis": vis,
           "mode": "zoom", "sel": None, "opts": [], "authors": None, "budget": 8000, "drag": "zoom"}
    props = {"events": "data", "window": "data", "nav": "data", "zoom": "data", "click": "data", "vis": "value",
             "mode": "value", "sel": "data", "opts": "value", "authors": "value", "budget": "value",
             "drag": "data"}
    rr = dash_call(s, render_dep["output"], [(i["id"], i["property"], ins[i["id"]]) for i in render_dep["inputs"]],
                   [(st["id"], st["property"], {"ctx": ctx, "rev": resp["rev"]["data"], "allfiles": [],
                                                         "panst": {"prev": None, "skip": None}}[st["id"]])
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

    # events table: a window-only set shows its windows (not blank landmark columns) and a status
    tdep = next(d for d in deps if d["output"].startswith("..table.data"))
    tv = {"events": events, "sel": None, "allfiles": [], "ctx": ctx}
    tr = dash_call(s, tdep["output"], [(i["id"], i["property"], tv[i["id"]]) for i in tdep["inputs"]],
                   [(st["id"], st["property"], tv[st["id"]]) for st in tdep["state"]])
    cols = [c["id"] for c in tr["table"]["columns"]]
    assert cols[:4] == ["event_id", "status", "window_start", "window_end"] and "event_start (s)" not in cols, cols
    assert tr["table"]["data"][0]["window_start"] is not None and tr["table"]["data"][0]["status"] == "window only"
    assert "window(s) not tagged yet" in json.dumps(tr["table-title"]), tr["table-title"]
    # all files of the experiment: one overview figure, table rows from every recording with a file column
    tv["allfiles"] = ["on"]
    tr = dash_call(s, tdep["output"], [(i["id"], i["property"], tv[i["id"]]) for i in tdep["inputs"]],
                   [(st["id"], st["property"], tv[st["id"]]) for st in tdep["state"]])
    assert "file" in [c["id"] for c in tr["table"]["columns"]] and len(tr["table"]["data"]) == len(ev0) + len(ev1)
    tog = next(d for d in deps if any(i["id"] == "allfiles" for i in d["inputs"]) and "..window.data" in d["output"])
    tst = {"window": window, "ctx": ctx, "nav": 0}
    tr_on = dash_call(s, tog["output"], [("allfiles", "value", ["on"])],
                      [(st["id"], st["property"], tst[st["id"]]) for st in tog["state"]])
    gwin = tr_on["window"]["data"]
    rstate = {"ctx": ctx, "rev": resp["rev"]["data"], "allfiles": ["on"], "panst": {"prev": None, "skip": None}}
    ra = dash_call(s, render_dep["output"], [(i["id"], i["property"], dict(ins, window=gwin)[i["id"]])
                                             for i in render_dep["inputs"]],
                   [(st["id"], st["property"], rstate[st["id"]]) for st in render_dep["state"]])
    atm = ra["tmap"]["data"]
    assert atm["all"] and len(atm["segs"]) == 2 and atm["segs"][1]["offset"] > 0, atm["segs"]
    other = next(t for t in atm["segs"] if t["rec_id"] != rid)
    # editing in the all-files view: a window drawn over the other file lands in that file, in its own time
    rel = next(d for d in deps if any(i["property"] == "relayoutData" for i in d["inputs"]))
    ast = {"ctx": ctx, "mode": "window", "window": gwin, "zoom": {}, "events": events, "sel": None, "opts": [],
           "tmap": atm, "allfiles": ["on"]}
    a0 = other["offset"] + 1.0
    res = dash_call(s, rel["output"], [("graph", "relayoutData", {"selections": [{"x0": a0, "x1": a0 + 0.5}]})],
                    [(st["id"], st["property"], ast[st["id"]]) for st in rel["state"]])
    assert "Window added: 1.000s to 1.500s" in json.dumps(res["msg"]), res["msg"]
    oth = dash_call(s, load_out, [("url", "search", f"?rec={other['rec_id']}&set={sid}")],
                    [("allfiles", "value", [])])["events"]["data"]
    assert any(abs((e["window_start"] or -1) - 1.0) < 1e-6 and e["recording_id"] == other["rec_id"] for e in oth)
    res = dash_call(s, rel["output"], [("graph", "relayoutData", {"selections": [{"x0": a0 - 0.01, "x1": a0 + 0.51}]})],
                    [(st["id"], st["property"], dict(ast, mode="delete")[st["id"]]) for st in rel["state"]])
    assert "Removed 1 event" in json.dumps(res["msg"]), res["msg"]
    # unticking the box opens the file in the middle of the view
    tr_off = dash_call(s, tog["output"], [("allfiles", "value", [])],
                       [(st["id"], st["property"], dict(tst, window=[a0 - 2, a0 + 8])[st["id"]]) for st in tog["state"]])
    assert tr_off["url"]["search"].startswith(f"?rec={other['rec_id']}&set={sid}&t="), tr_off
    print("events table columns + all-files view (editable, maps to the right file) ok")

    # add an event by clicking 3 optical + 4 electrical points
    click_dep = next(d for d in deps if d["output"].startswith("..click.data"))
    click = {"stage": "optical", "optical_times": [], "electrical_times": []}
    ev = ev0[0] if rid == 1 else ev1[0]
    t_s, t_e = ev["start"], ev["end"]
    seq = [(0, t_s), (0, (t_s + t_e) / 2), (0, t_e), (4, t_s - 0.01), (4, t_s + 0.001), (4, t_e + 0.001), (4, t_e + 0.02)]
    for curve, t in seq:  # curve 0..2 = overview traces (opt, optref, elec); 3..5 detail; use detail
        cn = 3 if curve == 0 else 5
        res = dash_call(s, click_dep["output"], [("graph", "clickData", {"points": [{"curveNumber": cn, "x": t}]})],
                        [("click", "data", click), ("mode", "value", "add"), ("ctx", "data", ctx), ("tmap", "data", tmap),
                         ("allfiles", "value", [])])
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
    resp_a = dash_call(s_al, load_out, [("url", "search", f"?rec={rid}&set={sid}")], [("allfiles", "value", [])])
    ctx_a = resp_a["ctx"]["data"]
    target = next(e for e in resp_a["events"]["data"] if e["id"] == added[-1]["id"])
    rel_dep = next(d for d in deps if any(i["property"] == "relayoutData" for i in d["inputs"]))
    smap_state = {"traces": tmap["traces"], "order": tmap["order"], "smap": ["event_start", "event_start",
                                                                             "event_plateau", "event_plateau",
                                                                             "event_end", "event_end",
                                                                             "entry_base_t", "entry_peak_t",
                                                                             "exit_peak_t", "exit_base_t"]}
    stv = {"ctx": ctx_a, "mode": "edit", "window": window, "zoom": {}, "events": resp_a["events"]["data"],
           "sel": target["id"], "opts": [], "tmap": smap_state, "allfiles": []}
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

    # ---- Pan: pauses Add Event (mode -> Zoom/Inspect), keeps pending clicks, restores the mode
    pan_dep = next(d for d in deps if d["inputs"] == [{"id": "pan-mode", "property": "n_clicks"}])
    mode_dep = next(d for d in deps if d["inputs"] == [{"id": "mode", "property": "value"}])
    off = {"prev": None, "skip": None}
    r = dash_call(s, pan_dep["output"], [("pan-mode", "n_clicks", 1)],
                  [("drag", "data", "zoom"), ("mode", "value", "add"), ("panst", "data", off)])
    assert r["drag"]["data"] == "pan" and r["mode"]["value"] == "zoom", r
    assert r["panst"]["data"] == {"prev": "add", "skip": "zoom"} and r["pan-mode"]["className"] == "nt-tbtn on", r
    mst = {"ctx": ctx, "events": new_events, "window": window, "sel": None, "allfiles": [],
           "panst": r["panst"]["data"], "drag": "pan"}
    r = dash_call(s, mode_dep["output"], [("mode", "value", "zoom")],
                  [(st["id"], st["property"], mst[st["id"]]) for st in mode_dep["state"]])
    assert "click" not in r and r["panst"]["data"] == {"prev": "add", "skip": None}, r   # clicks untouched
    pend = {"stage": "optical", "optical_times": [t_s], "electrical_times": []}
    ins_p = dict(ins, mode="zoom", click=pend, drag="pan")
    rr = dash_call(s, render_dep["output"], [(i["id"], i["property"], ins_p[i["id"]]) for i in render_dep["inputs"]],
                   [(st["id"], st["property"], {"ctx": ctx, "rev": resp["rev"]["data"], "allfiles": [],
                                                "panst": {"prev": "add", "skip": None}}[st["id"]])
                    for st in render_dep["state"]])
    lay = rr["graph"]["figure"]["layout"]
    assert lay["dragmode"] == "pan", lay["dragmode"]
    assert any("Add Event paused" in a.get("text", "") for a in lay.get("annotations", [])), lay.get("annotations")
    assert any(t.get("marker", {}).get("color") == "#ea580c" for t in rr["graph"]["figure"]["data"])  # pending O1
    r = dash_call(s, pan_dep["output"], [("pan-mode", "n_clicks", 2)],
                  [("drag", "data", "pan"), ("mode", "value", "zoom"), ("panst", "data", {"prev": "add", "skip": None})])
    assert r["drag"]["data"] == "zoom" and r["mode"]["value"] == "add", r
    assert r["panst"]["data"] == {"prev": None, "skip": "add"} and r["pan-mode"]["className"] == "nt-tbtn", r
    r = dash_call(s, mode_dep["output"], [("mode", "value", "add")],
                  [(st["id"], st["property"], dict(mst, panst={"prev": None, "skip": "add"}, drag="zoom")[st["id"]])
                   for st in mode_dep["state"]])
    assert "click" not in r and r["panst"]["data"] == off, r                  # resumed with clicks kept
    # Add Window is paused and restored the same way
    r = dash_call(s, pan_dep["output"], [("pan-mode", "n_clicks", 3)],
                  [("drag", "data", "zoom"), ("mode", "value", "window"), ("panst", "data", off)])
    assert r["mode"]["value"] == "zoom" and r["panst"]["data"]["prev"] == "window", r
    # picking a click/drag mode by hand while panning turns Pan off and starts that mode fresh
    r = dash_call(s, mode_dep["output"], [("mode", "value", "window")],
                  [(st["id"], st["property"], dict(mst, panst={"prev": "add", "skip": None})[st["id"]])
                   for st in mode_dep["state"]])
    assert r["drag"]["data"] == "zoom" and r["panst"]["data"] == off and r["click"]["data"]["optical_times"] == [], r
    # Pan from Zoom/Inspect: the mode stays as it is
    r = dash_call(s, pan_dep["output"], [("pan-mode", "n_clicks", 4)],
                  [("drag", "data", "zoom"), ("mode", "value", "zoom"), ("panst", "data", off)])
    assert "mode" not in r and r["drag"]["data"] == "pan" and r["panst"]["data"] == off, r
    print("pan pause/resume ok")

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

    # ---- Files page: browse, upload into incoming, import ABF + CSV into a NEW experiment
    fl = ok(s.get(BASE + "/api/files", params={"path": str(inc)}), "api files").json()
    assert fl["writable"] and any(f["name"] == "rec_b.fakeabf.npz" and f["used_in"] for f in fl["files"]), fl
    assert s.get(BASE + "/api/files", params={"path": "/etc"}).status_code == 403
    csv_path = tmp / "win_b.csv"
    S.windows_csv("rec_b.fakeabf.npz", ev1).to_csv(csv_path, index=False)
    res = upload(s, csv_path, {"purpose": "file", "dir": str(inc)})
    assert Path(res["path"]).exists(), res
    assert s.post(BASE + "/api/uploads/init", json={"purpose": "file", "dir": "/etc", "filename": "x.csv",
                                                     "size": 3}).status_code == 403
    ok(s.post(BASE + "/api/files/mkdir", json={"path": str(inc), "name": "sub"}), "mkdir")
    assert (inc / "sub").is_dir()
    r = ok(s.post(BASE + "/files/import-abf", data={"paths": [str(f1)], "new_experiment": "From files page",
                                                     "mode": "copy"}), "files import abf")
    fexp = int(re.search(r"/experiments/(\d+)", r.url).group(1))
    wait_jobs(s, fexp)
    r = ok(s.post(BASE + "/files/import-events", data={"paths": [res["path"]], "experiment_id": fexp,
                                                        "new_name": "CSV via files"}), "files import csv")
    finfo = s.get(f"{BASE}/api/experiments/{fexp}").json()
    assert {x["name"]: x["n_events"] for x in finfo["sets"]}.get("CSV via files") == len(ev1), finfo["sets"]
    dl = ok(s.get(BASE + "/files/download", params={"path": res["path"]}), "download")
    assert dl.content == csv_path.read_bytes()
    print("files page: browse, upload, mkdir, import ABF + CSV, download ok")

    # ---- automatic experiment from the file name
    g = s.get(BASE + "/api/guess-experiment", params={"name": ["2026_09_14_0001 B3S8-15 100 aM SiO2 DC.abf",
                                                               "2026_09_14_0007 B3S8-15 100 aM SiO2 AC-AOM.abf"]}).json()
    assert {x["experiment"] for x in g} == {"2026_09_14 B3S8-15 100 aM SiO2"}, g
    auto_dc = tmp / "2026_09_14_0001 B3S8-15 100 aM SiO2 DC.fakeabf.npz"
    np_save(auto_dc, d0)
    res = upload(s, auto_dc, {"purpose": "abf", "experiment_id": "auto"})
    auto_ac = inc / "2026_09_14_0002 B3S8-15 100 aM SiO2 AC.fakeabf.npz"
    np_save(auto_ac, d1)
    ok(s.post(BASE + "/files/import-abf", data={"paths": [str(auto_ac)], "experiment_id": "auto", "mode": "copy"}),
       "auto import")
    exps = {e["name"]: e["id"] for e in s.get(BASE + "/api/experiments").json()}
    aid = exps["2026_09_14 B3S8-15 100 aM SiO2"]
    assert res["experiment_id"] == aid
    wait_jobs(s, aid)
    arecs = s.get(f"{BASE}/api/experiments/{aid}").json()["recordings"]
    assert len(arecs) == 2 and all(r["status"] == "ready" for r in arecs), arecs
    auto_csv = inc / "auto_windows.csv"
    S.windows_csv(auto_dc.name, ev0).to_csv(auto_csv, index=False)
    ok(s.post(BASE + "/files/import-events", data={"paths": [str(auto_csv)], "experiment_id": "auto",
                                                    "new_name": "auto CSV"}), "auto csv")
    asets = {x["name"]: x["n_events"] for x in s.get(f"{BASE}/api/experiments/{aid}").json()["sets"]}
    assert asets.get("auto CSV") == len(ev0), asets
    print("automatic experiment from file name ok (upload, server-folder import, CSV routing)")

    # ---- import a whole folder: ABF + _event.csv pairs, grouped by file name, conditions kept
    fold = inc / "SiO2_Tagging"
    fold.mkdir()
    base_name = "2026_09_15_{:04d} S1 10 aM SiO2 {}"
    fold_truth = {}
    for run_no, cond, seed in [(0, "baseline", 5), (1, "DC", 2), (2, "AC", 3), (3, "AOM", 4)]:
        dd, evs = S.make_recording(seed)
        stem = base_name.format(run_no, cond)
        np_save(fold / (stem + ".fakeabf.npz"), dd)
        if cond == "DC":      # plain '<name>_event.csv'
            S.windows_csv(stem + ".abf", evs).to_csv(fold / (stem + "_event.csv"), index=False)
        elif cond == "AC":    # doubled space + Windows-1252 text, as saved by Excel on a lab PC
            df_ = S.windows_csv(stem + ".abf", evs)
            df_["notes"] = "5 µm"
            (fold / (stem.replace(" 10 aM", "  10 aM") + "_Event.csv")).write_bytes(
                df_.to_csv(index=False).encode("cp1252"))
        elif cond == "AOM":   # arbitrary file name: paired through its file_name column
            S.windows_csv(stem + ".abf", evs).to_csv(fold / "20260915003S110amSiO2.csv", index=False)
        if cond != "baseline":
            fold_truth[stem] = evs
    (fold / "notes.csv").write_text("a,b\n1,2\n")
    plan = ok(s.get(BASE + "/api/import/scan", params={"path": str(fold)}), "scan").json()
    assert len(plan["groups"]) == 1 and plan["groups"][0]["experiment"] == "2026_09_15 S1 10 aM SiO2", plan
    items = plan["groups"][0]["items"]
    assert [i["condition"] for i in items] == ["baseline", "DC", "AC", "AOM"], items
    assert [i["selected"] for i in items] == [False, True, True, True], items
    assert [t["name"] for t in plan["unused_tables"]] == ["notes.csv"], plan["unused_tables"]
    assert [i["events"]["matched_by"] for i in items[1:]] == ["name", "name", "content"], items
    bad = s.get(BASE + "/api/import/scan", params={"path": "/etc"})
    assert bad.status_code == 403 and "add-import-root.sh" in bad.json()["hint"], bad.text
    sel = [{"abf": i["abf"], "events": i["events"]["path"] if i["events"] else None, "experiment": plan["groups"][0]["experiment"]}
           for i in items if i["selected"]]
    r = ok(s.post(BASE + "/api/import/run", json={"folder": plan["folder"], "mode": "inplace", "items": sel}), "run import").json()
    fid = r["jobs"][0]["experiment_id"]
    assert r["redirect"].startswith(f"/experiments/{fid}")
    wait_jobs(s, fid)
    finfo = s.get(f"{BASE}/api/experiments/{fid}").json()
    assert sorted(x["condition"] for x in finfo["recordings"]) == ["AC", "AOM", "DC"], finfo["recordings"]
    assert all(x["status"] == "ready" and x["abf_path"] == x["source_path"] and x["abf_path"].startswith(str(fold))
               for x in finfo["recordings"]), finfo["recordings"]
    ev_set = next(x for x in finfo["sets"] if x["name"] == "Event CSVs")
    assert ev_set["n_events"] == sum(len(v) for v in fold_truth.values()), ev_set
    # scanning again: already imported, nothing ticked; importing again adds no duplicates
    plan2 = s.get(BASE + "/api/import/scan", params={"path": str(fold)}).json()
    assert [i["selected"] for i in plan2["groups"][0]["items"]] == [False] * 4, plan2
    ok(s.post(BASE + "/api/import/run", json={"folder": str(fold), "mode": "inplace", "items": sel}), "re-import")
    wait_jobs(s, fid)
    assert next(x for x in s.get(f"{BASE}/api/experiments/{fid}").json()["sets"]
                if x["name"] == "Event CSVs")["n_events"] == ev_set["n_events"]
    base_item = items[0]
    ok(s.post(BASE + "/api/import/run", json={"folder": str(fold), "mode": "copy",
                                              "items": [{"abf": base_item["abf"], "events": None, "experiment": ""}]}), "baseline")
    wait_jobs(s, fid)
    finfo = s.get(f"{BASE}/api/experiments/{fid}").json()
    assert len(finfo["recordings"]) == 4, finfo["recordings"]
    assert "Import a folder" in s.get(BASE + "/import", params={"path": str(fold)}).text
    print("folder import ok: 3 ABF+CSV pairs -> 1 experiment (DC/AC/AOM), in place, no duplicates on re-import")

    # ---- conditions: NN on the imported windows, then one clustering run per condition
    ok(s.post(f"{BASE}/experiments/{fid}/nn", data={"source_set_id": ev_set["id"], "model_id": model_id,
                                                     "out_name": "NN folder"}), "nn folder")
    wait_jobs(s, fid)
    nn_f = next(x for x in s.get(f"{BASE}/api/experiments/{fid}").json()["sets"] if x["name"] == "NN folder")
    ok(s.post(f"{BASE}/experiments/{fid}/cluster",
              data={"name": "by cond", "set_ids": [nn_f["id"]], "conditions": ["DC", "AC"], "split": "1", "max_k": "3",
                    "n_boot": "5", "impute": "1", "features": ["duration", "entry_spike", "exit_spike"]}), "cluster split")
    wait_jobs(s, fid)
    fruns = s.get(f"{BASE}/api/experiments/{fid}").json()["cluster_runs"]
    assert sorted(r_["name"] for r_ in fruns) == ["by cond · AC", "by cond · DC"] and all(r_["status"] == "done" for r_ in fruns), fruns
    assert all(r_["n_events"] <= len(max(fold_truth.values(), key=len)) for r_ in fruns), fruns
    print("clustering per condition ok:", [(r_["name"], r_["n_events"], r_["final_k"]) for r_ in fruns])

    # ---- deleting analyses, recordings and experiments
    r = alice.post(f"{BASE}/experiments/{fid}/sets/delete", data={"set_ids": [nn_f["id"]]})
    assert "You can delete only" in r.text, r.text[:300]             # alice did not create it
    ok(s.post(f"{BASE}/experiments/{fid}/sets/delete", data={"set_ids": [nn_f["id"]]}), "delete set")
    assert all(x["name"] != "NN folder" for x in s.get(f"{BASE}/api/experiments/{fid}").json()["sets"])
    ok(s.post(f"{BASE}/experiments/{fid}/cluster/delete", data={"run_ids": [x["id"] for x in fruns]}), "delete runs")
    assert not s.get(f"{BASE}/api/experiments/{fid}").json()["cluster_runs"]
    assert not list((tmp / "jobs").glob("cluster_run_*")) or all(
        int(p_.name.split("_")[-1]) not in [x["id"] for x in fruns] for p_ in (tmp / "jobs").glob("cluster_run_*"))
    base_rec = next(x for x in finfo["recordings"] if x["condition"] == "baseline")
    assert (tmp / "abf" / str(fid)).exists()                           # the baseline was copied
    ok(s.post(f"{BASE}/experiments/{fid}/recordings/delete", data={"rec_ids": [base_rec["id"]], "delete_abf": "1"}), "rm rec")
    assert not list((tmp / "abf" / str(fid)).glob("*")) and Path(base_item["abf"]).exists()
    page = ok(s.get(f"{BASE}/experiments/delete", params={"ids": [fid, fexp]}), "delete confirm").text
    assert "Delete 2 experiments" in page and "2026_09_15 S1 10 aM SiO2" in page, page[:500]
    assert alice.post(f"{BASE}/experiments/delete", data={"ids": [fexp]}).status_code == 403
    assert (tmp / "abf" / str(fexp)).exists()
    ok(s.post(f"{BASE}/experiments/delete", data={"ids": [fid, fexp], "delete_abf": "1"}), "delete experiments")
    names_left = {e["name"] for e in s.get(BASE + "/api/experiments").json()}
    assert "2026_09_15 S1 10 aM SiO2" not in names_left and "From files page" not in names_left, names_left
    assert not (tmp / "abf" / str(fexp)).exists(), "copied ABFs of the deleted experiment remain"
    assert all((fold / (st + ".fakeabf.npz")).exists() for st in fold_truth), "in-place ABFs must never be deleted"
    ok(s.post(BASE + "/jobs/clear", data={}), "clear jobs")
    assert not [j for j in s.get(BASE + "/api/jobs").json()["jobs"] if j["status"] in ("done", "error", "cancelled")]
    print("delete ok: sets, clustering runs, recordings, experiments (bulk, permissions, in-place ABFs kept), clear jobs")

    # ---- no stale pages; logged-out users always get the login page
    r = s.get(BASE + "/experiments")
    assert "no-store" in r.headers.get("Cache-Control", ""), r.headers
    css = re.search(r'href="(/static/style.css\?v=[^"]+)"', r.text).group(1)
    assert "immutable" in s.get(BASE + css).headers.get("Cache-Control", "")
    assert "no-store" in s.get(BASE + "/tagger/_dash-layout").headers.get("Cache-Control", "")
    html_hdr = {"Accept": "text/html,application/xhtml+xml"}
    # a bookmarked / typed deep link lands on the home page, not in the middle of an analysis
    for path in [f"/tagger/?rec={rid}&set={sid}", f"/experiments/{exp_id}", "/files", "/jobs"]:
        r = requests.get(BASE + path, headers=html_hdr, cookies=s.cookies)
        assert r.url.split("#")[0].endswith("/experiments"), (path, r.url)
    # ...while the same link clicked inside the app opens normally
    r = s.get(f"{BASE}/tagger/?rec={rid}&set={sid}", headers=html_hdr)
    assert "/tagger/" in r.url, r.url
    # every page shows the version
    for path in ["/experiments", f"/experiments/{exp_id}", "/files", "/help"]:
        assert f"v{s.get(BASE + '/healthz').json()['version']}" in s.get(BASE + path).text, path
    anon = requests.Session()
    assert "v" + s.get(BASE + "/healthz").json()["version"] in anon.get(BASE + "/login").text
    for path in ["/", "/experiments", f"/experiments/{exp_id}#nn", f"/tagger/?rec={rid}&set={sid}", "/files"]:
        r = anon.get(BASE + path)
        assert r.url.split("#")[0].endswith("/login") and "Sign in" in r.text, (path, r.url)
    r = anon.post(BASE + "/login", data={"username": "admin", "password": "admin2025!!"})
    assert r.url.endswith("/experiments"), r.url
    assert anon.post(BASE + "/tagger/_dash-update-component", json={}).status_code in (200, 400, 500)
    anon2 = requests.Session()
    assert anon2.post(BASE + "/tagger/_dash-update-component", json={}).status_code == 401
    s2 = requests.Session()
    s2.post(BASE + "/login", data={"username": "admin", "password": "admin2025!!"})
    r = s2.get(BASE + "/", headers=html_hdr)
    assert r.url.endswith("/login") and "Sign in" in r.text, r.url
    assert s2.get(BASE + "/api/experiments").status_code == 401     # visiting / signed out
    print("no-cache headers, versioned assets, login-first, deep links -> home, version on pages ok")

    # ---- every HTML page renders
    jid_ = ok(s.post(f"{BASE}/experiments/{exp_id}/nn", data={"source_set_id": sid, "model_id": model_id,
                                                                "out_name": "NN again"}), "nn again")
    wait_jobs(s, exp_id)
    last_job = s.get(BASE + "/api/jobs").json()["jobs"][0]["id"]
    for path in ["/experiments", f"/experiments/{exp_id}", f"/experiments/{new_id}", "/jobs", f"/jobs/{last_job}", "/models",
                 "/admin", "/account", "/files", f"/files?path={inc}", "/help", "/import", f"/experiments/delete?ids={exp_id}", f"/tagger/?rec={rid}&set={sid}"]:
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
