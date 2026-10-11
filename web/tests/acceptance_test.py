"""Post-install acceptance test against a RUNNING NanoTag server (production mode, real pyabf path).

    sudo -u nanotag /opt/nanotag/venv/bin/python /opt/nanotag/current/tests/acceptance_test.py \
         --url http://127.0.0.1:3389 --user admin --password '...'

It writes two synthetic 6-channel ABF files (500 kHz layout scaled down to 50 kHz, same channel
names as the lab rigs) into the import folder, then: creates an experiment, imports them from the
server folder, waits for ingest, imports window CSVs, loads the Tagger, runs the NN and clustering,
downloads every export — and finally deletes the test experiment and files (unless --keep).
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abf1_writer as W  # noqa: E402
import smoke_test as T  # noqa: E402
import synthetic as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:3389")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--folder", default="/srv/nanotag/incoming")
    ap.add_argument("--keep", action="store_true", help="keep the test experiment and files")
    a = ap.parse_args()
    T.BASE = a.url.rstrip("/")
    B = T.BASE
    s = requests.Session()
    r = s.post(B + "/login", data={"username": a.user, "password": a.password})
    assert "Invalid username" not in r.text, "login failed"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(a.folder)
    files, truth = [], {}
    for k in range(2):
        d, ev = S.make_recording(10 + k)
        f = folder / f"acceptance_{stamp}_{k}.abf"
        W.write_abf1(f, d, 50000, S.NAMES, S.UNITS)
        os.chmod(f, 0o664)
        files.append(f)
        truth[f.name] = ev
    print(f"wrote {len(files)} test ABFs to {folder}")
    name = f"Acceptance test {stamp}"
    r = T.ok(s.post(B + "/experiments/new", data={"name": name}), "new experiment")
    exp_id = int(re.search(r"/experiments/(\d+)", r.url).group(1))
    T.ok(s.post(f"{B}/experiments/{exp_id}/import-folder", data={"files": [str(f) for f in files], "mode": "copy"}),
         "import folder")
    T.wait_jobs(s, exp_id)
    info = s.get(f"{B}/api/experiments/{exp_id}").json()
    assert all(r["status"] == "ready" for r in info["recordings"]), info["recordings"]
    ch = info["recordings"][0]["channels"]
    assert [c["name"] for c in ch] == S.NAMES, ch
    print("ingest ok:", [(r["file_name"], r["duration"]) for r in info["recordings"]])

    df = pd.concat([S.windows_csv(fn, ev) for fn, ev in truth.items()])
    T.ok(s.post(f"{B}/experiments/{exp_id}/import-events",
                files={"files": ("windows.csv", df.to_csv(index=False).encode(), "text/csv")},
                data={"new_name": "Windows"}), "import csv")
    info = s.get(f"{B}/api/experiments/{exp_id}").json()
    sid = next(x["id"] for x in info["sets"] if x["name"] == "Windows")
    rid = info["recordings"][0]["id"]

    deps = s.get(f"{B}/tagger/_dash-dependencies").json()
    load_out = next(d["output"] for d in deps if d["inputs"] == [{"id": "url", "property": "search"}])
    t0 = time.time()
    resp = T.dash_call(s, load_out, [("url", "search", f"?rec={rid}&set={sid}")], [("allfiles", "value", [])])
    ctx = resp["ctx"]["data"]
    render_dep = next(d for d in deps if d["output"].startswith("..graph.figure"))
    ins = {"events": resp["events"]["data"], "window": resp["window"]["data"], "nav": 0, "zoom": {},
           "click": {"stage": "idle"}, "vis": resp["vis"]["value"], "mode": "zoom", "sel": None, "opts": [],
           "authors": None, "budget": 8000, "drag": "zoom"}
    rstate = {"ctx": ctx, "rev": resp["rev"]["data"], "allfiles": [], "panst": {"prev": None, "skip": None}}
    T.dash_call(s, render_dep["output"], [(i["id"], i["property"], ins[i["id"]]) for i in render_dep["inputs"]],
                [(st["id"], st["property"], rstate[st["id"]]) for st in render_dep["state"]])
    print(f"tagger load + render ok ({time.time() - t0:.2f} s)")

    model_id = info["models"][0]["id"]
    T.ok(s.post(f"{B}/experiments/{exp_id}/nn", data={"source_set_id": sid, "model_id": model_id,
                                                        "out_name": "NN"}), "nn")
    T.wait_jobs(s, exp_id)
    info = s.get(f"{B}/api/experiments/{exp_id}").json()
    nn_sid = next(x["id"] for x in info["sets"] if x["name"] == "NN")
    x = T.ok(s.get(f"{B}/sets/{nn_sid}/export", params={"layout": "nn", "fmt": "xlsx"}), "nn export")
    nn_df = pd.read_excel(io.BytesIO(x.content))
    print(f"NN ok: {nn_df['event_start (s)'].notna().sum()}/{len(nn_df)} events with landmarks")

    T.ok(s.post(f"{B}/experiments/{exp_id}/cluster", data={"name": "acceptance", "set_ids": [nn_sid],
                                                            "max_k": "4", "n_boot": "25", "impute": "1",
                                                            "features": ["duration", "entry_spike", "exit_spike"]}),
         "cluster")
    T.wait_jobs(s, exp_id)
    run = s.get(f"{B}/api/experiments/{exp_id}").json()["cluster_runs"][-1]
    assert run["status"] == "done", run
    z = zipfile.ZipFile(io.BytesIO(T.ok(s.get(f"{B}/cluster/{run['id']}/download.zip"), "zip").content))
    assert "cluster_report.html" in z.namelist()
    print(f"clustering ok: final K={run['final_k']}, {run['n_events']} events")
    for q in ({"fmt": "csv"}, {"fmt": "xlsx"}, {"layout": "zip"}):
        T.ok(s.get(f"{B}/sets/{sid}/export", params=q), f"export {q}")
    T.ok(s.get(f"{B}/experiments/{exp_id}/bundle.zip"), "bundle")
    print("exports ok")

    if not a.keep:
        page = s.get(f"{B}/admin")
        if page.status_code == 200:
            T.ok(s.post(f"{B}/experiments/{exp_id}/delete", data={"confirm": name, "delete_abf": "1"}), "delete")
            print("test experiment deleted")
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass
    print("\nACCEPTANCE TEST PASSED")


if __name__ == "__main__":
    main()
