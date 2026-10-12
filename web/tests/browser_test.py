"""Tagger in a real browser (headless Chromium via Playwright), on a throw-away server with synthetic data.

    python tests/browser_test.py [--chromium /path/to/chrome]

Checks what the HTTP-level tests cannot see: that the page actually draws the plot and fills the events table,
that it loads only ONCE (0.1.11 re-ran its load callback in a loop in some browsers, so the table and plot never
appeared), that the all-files view lists every event, that the middle mouse button pans, and that the browser
reports no errors. Needs `pip install playwright` and a Chromium; skipped (exit 0) if Playwright is missing.
Runs at 100 % and at 125 % display scaling.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import release_test as R  # noqa: E402
import synthetic as S  # noqa: E402

PORT = 8151


def build(srv, tmp):
    from nanotag.abfio import write_fake_abf
    import pandas as pd
    s = srv.session()
    ok = R.ok
    ok(s.post(srv.base + "/admin/users/add", data={"username": "oguz", "password": "oguzpass1", "role": "admin"}), "user")
    r = ok(s.post(srv.base + "/experiments/new", data={"name": "Browser"}), "exp")
    exp = int(re.search(r"/experiments/(\d+)", r.url).group(1))
    tables = []
    for k in range(4):
        d, ev = S.make_recording(k, duration=30.0, n_events=20)
        f = tmp / f"2026_10_09_000{k + 1} browser DC.fakeabf.npz"
        write_fake_abf(f, d, 50000, S.NAMES, S.UNITS)
        R.upload(srv, s, f, exp)
        tables.append(S.windows_csv(f.name, ev))
    srv.wait_jobs(s, exp)
    df = pd.concat(tables)
    ok(s.post(f"{srv.base}/experiments/{exp}/import-events",
              files={"files": ("w.csv", df.to_csv(index=False).encode(), "text/csv")}, data={"new_name": "Tags"}), "csv")
    info = s.get(f"{srv.base}/api/experiments/{exp}").json()
    sid = next(x["id"] for x in info["sets"] if x["name"] == "Tags")
    # two authors, some without an author, half the events fully tagged (all table columns in use)
    srv.py(f"""
from nanotag.db import db
with db() as con:
    con.execute("UPDATE events SET created_by='oguz' WHERE set_id={sid} AND id % 3 = 0")
    con.execute("UPDATE events SET created_by=NULL WHERE set_id={sid} AND id % 7 = 0")
    con.execute("UPDATE events SET event_start=window_start+0.05, event_plateau=window_start+0.1, "
                "event_end=window_end-0.05, entry_base_t=window_start+0.05, entry_peak_t=window_start+0.06, "
                "exit_peak_t=window_end-0.06, exit_base_t=window_end-0.05 WHERE set_id={sid} AND id % 2 = 0")
""")
    n_total = next(x["n_events"] for x in info["sets"] if x["name"] == "Tags")
    return exp, info["recordings"], sid, n_total


def run_view(p, chromium, srv, exp, recs, sid, n_total, scale):
    b = p.chromium.launch(executable_path=chromium) if chromium else p.chromium.launch()
    vw = {"width": int(1900 / scale), "height": int(950 / scale)}
    pg = b.new_page(viewport=vw, device_scale_factor=scale)
    errors, loads = [], []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}")
          if m.type == "error" and "favicon" not in m.text and "404" not in m.text else None)

    def on_request(rq):
        if "_dash-update-component" in rq.url:
            try:
                out = json.loads(rq.post_data or "{}").get("output", "")
            except ValueError:
                out = ""
            if out.startswith("..ctx.data"):
                loads.append(time.time())
    pg.on("request", on_request)
    pg.goto(srv.base + "/login")
    pg.fill("input[name=username]", "oguz")
    pg.fill("input[name=password]", "oguzpass1")
    pg.click("button[type=submit], input[type=submit]")
    rid = recs[-1]["id"]
    pg.goto(f"{srv.base}/tagger/?rec={rid}&set={sid}", referer=f"{srv.base}/experiments/{exp}")
    pg.wait_for_function("document.querySelectorAll('#table tbody tr').length > 2", timeout=30000)
    pg.wait_for_function("document.querySelectorAll('#graph .scatterlayer .trace').length > 5", timeout=30000)
    time.sleep(6)                                       # long enough to see a reload loop
    n_rows = pg.locator("#table tbody tr").count() - 2  # minus the header and filter rows
    title = pg.locator("#table-title").inner_text()
    n_here = int(re.match(r"(\d+) event", title).group(1))
    assert n_rows == n_here > 0, (n_rows, title)
    assert len(loads) == 1, f"the Tagger loaded {len(loads)} times (reload loop)"
    print(f"  [{scale:.2f}x] one recording: plot drawn, {n_here} events in the table, page loaded once")

    # middle button pans and leaves the mode alone
    pg.evaluate("Plotly.relayout(document.querySelector('#graph .js-plotly-plot'), "
                "{'xaxis.range[0]': 5, 'xaxis.range[1]': 15})")
    pg.wait_for_function("(() => { const r = document.querySelector('#graph .js-plotly-plot').layout.xaxis.range;"
                         " return Math.abs(r[0] - 5) < 1e-6 && Math.abs(r[1] - 15) < 1e-6; })()", timeout=15000)
    time.sleep(2)
    x0 = pg.evaluate("document.querySelector('#graph .js-plotly-plot').layout.xaxis.range.map(Number)")
    box = pg.locator("#graph .nsewdrag").first.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    pg.mouse.move(cx, cy)
    pg.mouse.down(button="middle")
    pg.mouse.move(cx - box["width"] / 4, cy, steps=8)
    pg.mouse.up(button="middle")
    time.sleep(2.5)
    x1 = pg.evaluate("document.querySelector('#graph .js-plotly-plot').layout.xaxis.range.map(Number)")
    span = x0[1] - x0[0]
    # a pan: both ends move by the same amount and the span stays (a zoom box would change the span)
    assert abs((x1[0] - x0[0]) - span / 4) < span * 0.05 and abs((x1[1] - x1[0]) - span) < span * 0.01, (x0, x1)
    print(f"  [{scale:.2f}x] middle-button drag panned the view {x1[0] - x0[0]:.2f} s")

    # all files: every event of the set in one scrolling list
    loads.clear()
    pg.check("#allfiles input")
    pg.wait_for_function(f"document.querySelectorAll('#table tbody tr').length > {n_total}", timeout=30000)
    time.sleep(3)
    n_rows = pg.locator("#table tbody tr").count() - 2
    assert n_rows == n_total, (n_rows, n_total)
    assert not loads, f"ticking all files reloaded the page {len(loads)} times"
    sc = pg.evaluate("""(() => { let el = document.querySelector('#table tbody');
        while (el && !(el.scrollHeight > el.clientHeight + 5 && /(auto|scroll)/.test(getComputedStyle(el).overflowY)))
          el = el.parentElement;
        if (!el) return null; el.scrollTop = el.scrollHeight; return [el.scrollTop, el.scrollHeight]; })()""")
    assert sc and sc[0] > 0, "the events table does not scroll"
    print(f"  [{scale:.2f}x] all files: all {n_total} events in one scrolling list")
    assert not errors, "browser errors:\n" + "\n".join(errors)
    b.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", default=None, help="Chromium/Chrome executable (default: Playwright's)")
    args = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — browser test skipped")
        return
    tmp = Path(tempfile.mkdtemp(prefix="nanotag_bt_"))
    srv = R.Server("bt", PORT).start()
    try:
        exp, recs, sid, n_total = build(srv, tmp)
        with sync_playwright() as p:
            for scale in (1.0, 1.25):
                run_view(p, args.chromium, srv, exp, recs, sid, n_total, scale)
        # the page's error reporter reaches the server log
        s = srv.session()
        assert s.post(srv.base + "/api/client-log", json={"kind": "test", "msg": "hello"}).status_code == 204
        print("\nBROWSER TEST PASSED")
    finally:
        srv.stop()
        shutil.rmtree(srv.data, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
