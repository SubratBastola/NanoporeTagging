"""ABF -> Zarr ingestion.

The display signal is produced with *exactly* the Tagger_GUI.py filter:
    wn  = FILTER_CUTOFF_HZ / (fs / 2)
    sos = scipy.signal.bessel(FILTER_ORDER, wn, output="sos")
    y   = scipy.signal.sosfilt(sos, raw)          (falls back to raw on error)
applied over the whole recording from a zero initial state, then sampled onto
the Tagger's 10 kHz detail grid (OPTICAL_FS / ELECTRICAL_FS). Nothing about the
filter is changed; see README "Known issues" for recommendations parked for later.
"""
import hashlib
import json
import os

import numpy as np
from scipy import signal

from . import store
from .abfio import full_channel, header_info, open_abf
from .config import cfg
from .db import db, jdump, now, one, tx

# ---- copied verbatim from Tagger_GUI.py ----
FILTER_ORDER = 8
FILTER_CUTOFF_HZ = 100.0
OPTICAL_FS = 10000.0
ELECTRICAL_FS = 10000.0


def tagger_filter(raw, fs):
    """Tagger_GUI.py: 8-pole Bessel @ 100 Hz, causal sosfilt; fallback to raw if filter fails."""
    try:
        wn = FILTER_CUTOFF_HZ / (fs / 2.0)
        sos = signal.bessel(FILTER_ORDER, wn, output="sos")
        return signal.sosfilt(sos, raw)
    except Exception:
        return raw


def to_display_grid(y, fs, target_fs):
    """Sample the filtered full-rate signal onto the Tagger detail grid."""
    if target_fs >= fs:
        return np.asarray(y, dtype=np.float32), fs
    ratio = fs / target_fs
    if abs(ratio - round(ratio)) < 1e-9:
        return np.asarray(y[:: int(round(ratio))], dtype=np.float32), fs / round(ratio)
    t_src = np.arange(len(y)) / fs
    t_dst = np.arange(0.0, t_src[-1], 1.0 / target_fs)
    return np.interp(t_dst, t_src, y).astype(np.float32), target_fs


def sha256_file(path, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ingest_recording(rec_id, log=print, progress=lambda f: None):
    with db() as con:
        rec = one(con, "SELECT * FROM recordings WHERE id=?", (rec_id,))
    if rec is None:
        raise ValueError(f"Recording {rec_id} not found")
    path = rec["abf_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with db() as con:
        con.execute("UPDATE recordings SET status='ingesting', error=NULL WHERE id=?", (rec_id,))

    log(f"Reading header: {os.path.basename(path)}")
    info = header_info(path)
    log(f"  fs={info['fs']:.0f} Hz, channels={len(info['channels'])}, "
        f"samples/channel={info['n_samples']}, duration={info['duration']:.2f} s, sweeps={info['n_sweeps']}")
    progress(0.02)
    log("Computing SHA-256 ...")
    digest = sha256_file(path)
    progress(0.08)

    log("Loading ABF (pyabf) ...")
    abf = open_abf(path, load_data=True)
    fs = float(abf.dataRate)
    n_ch = int(abf.channelCount)
    target = min(OPTICAL_FS, fs)

    n_src = len(full_channel(abf, 0))
    probe, fs_disp = to_display_grid(np.zeros(min(n_src, 10), dtype=np.float32), fs, target)
    if fs_disp == fs:
        n_disp = n_src
    elif abs(fs / fs_disp - round(fs / fs_disp)) < 1e-9:
        n_disp = -(-n_src // int(round(fs / fs_disp)))
    else:
        n_disp = len(np.arange(0.0, (n_src - 1) / fs, 1.0 / target))
    del probe

    gpath = store.new_group_path(rec["experiment_id"], rec_id)
    attrs = {
        "fs_source": fs, "fs_display": fs_disp, "n_source": n_src, "n_display": n_disp,
        "channels": info["channels"], "file_name": rec["file_name"], "sha256": digest,
        "filter": f"Tagger_GUI: bessel order {FILTER_ORDER} @ {FILTER_CUTOFF_HZ} Hz, sosfilt (causal)",
        "created": now(),
    }
    w = store.DisplayWriter(gpath, n_ch, n_disp, attrs)
    for ch in range(n_ch):
        raw = full_channel(abf, ch).astype(float)
        filtered = tagger_filter(raw, fs)
        disp, _ = to_display_grid(filtered, fs, target)
        w.write_channel(ch, disp[:n_disp])
        del raw, filtered, disp
        log(f"  channel {ch} ({info['channels'][ch]['name']}) written")
        progress(0.1 + 0.88 * (ch + 1) / n_ch)
    del abf

    old = rec.get("zarr_path")
    roles = [rec["role_elec"], rec["role_opt"], rec["role_optref"]]
    if any(r >= n_ch for r in roles):          # e.g. a 1- or 2-channel file: keep roles in range
        roles = [min(r, n_ch - 1) for r in roles]
        log(f"  channel roles adjusted to {roles} (file has {n_ch} channel(s))")
    with db() as con, tx(con):
        con.execute("UPDATE recordings SET role_elec=?, role_opt=?, role_optref=? WHERE id=?", (*roles, rec_id))
        con.execute("""UPDATE recordings SET sha256=?, size_bytes=?, fs=?, n_samples=?, duration=?,
                       n_sweeps=?, channels_json=?, status='ready', error=NULL, zarr_path=?,
                       display_fs=?, display_n=? WHERE id=?""",
                    (digest, os.path.getsize(path), fs, n_src, n_src / fs, info["n_sweeps"],
                     jdump(info["channels"]), gpath, fs_disp, n_disp, rec_id))
    if old and old != gpath:
        store.delete_group(old)
    progress(1.0)
    log("Ingest complete.")
    return {"recording_id": rec_id, "zarr_path": gpath, "fs": fs, "display_fs": fs_disp, "n_display": n_disp}
