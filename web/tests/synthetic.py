"""Synthetic 6-channel recordings for automated tests (no real ABF needed).

Channel layout mirrors the lab's ABFs: 0 Ipatch [pA], 1 CommandV [mV], 2 Optical [V],
3 OpticalRe [V], 4 Edge [V], 5 Pulse [V]. Two particle populations with different
optical step sizes and dwell times give the clustering something to find.
"""
import numpy as np
import pandas as pd

NAMES = ["Ipatch", "CommandV", "Optical", "OpticalRe", "Edge", "Pulse"]
UNITS = ["pA", "mV", "V", "V", "V", "V"]


def make_recording(seed, fs=50000, duration=90.0, n_events=24):
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    data = np.zeros((6, n), dtype=np.float32)
    data[0] = 100 + rng.normal(0, 2.0, n)
    data[1] = 100.0
    data[2] = 1.0 + rng.normal(0, 0.004, n) + 0.01 * np.sin(2 * np.pi * 0.05 * t)
    data[3] = 0.8 + rng.normal(0, 0.003, n)
    data[4] = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)
    data[5] = rng.normal(0, 0.01, n)
    events = []
    starts = np.sort(rng.uniform(2.0, duration - 4.0, n_events))
    last_end = 0.0
    for i, s in enumerate(starts):
        pop = i % 2
        dur = rng.uniform(0.3, 0.6) if pop == 0 else rng.uniform(1.0, 1.6)
        amp = 0.05 if pop == 0 else 0.12
        if s < last_end + 0.8:
            s = last_end + 0.8
        e = s + dur
        if e > duration - 1.0:
            break
        i0, i1 = int(s * fs), int(e * fs)
        data[2, i0:i1] += amp
        data[3, i0:i1] += amp * 0.4
        k = int(0.004 * fs)
        data[0, i0:i0 + k] += 40 if pop == 0 else 70
        data[0, i1:i1 + k] -= 30 if pop == 0 else 60
        events.append({"start": s, "end": e, "pop": pop})
        last_end = e
    return data, events


def windows_csv(file_name, events, pad=0.3):
    rows = []
    for i, ev in enumerate(events, start=1):
        rows.append({"event_id": i, "file_name": file_name, "sensor": "", "analytes": "", "solution": "",
                     "window_start": max(0.0, ev["start"] - pad), "window_end": ev["end"] + pad})
    cols = ["event_id", "file_name", "sensor", "analytes", "solution", "event_start (s)", "event_end (s)",
            "notes", "Base (V)", "Step (V)", "RefBase (V)", "RefStep (V)", "entry Base (pA)", "entry Peak (pA)",
            "exit Peak (pA)", "exit Base (pA)", "duration", "OSC", "RefOSC", "entry spike", "exit spike",
            "event_plateau", "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
            "window_start", "window_end"]
    return pd.DataFrame(rows).reindex(columns=cols)
