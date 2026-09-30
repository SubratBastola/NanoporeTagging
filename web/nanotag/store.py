"""Zarr signal store.

Layout (one group per recording, versioned so a re-ingest never races readers):

    store.zarr/
      experiments/<exp_id>/recordings/<rec_id>/v<gen>/
          display        float32 (channels, n)      Tagger-filtered signal at DISPLAY_FS
          pyr1, pyr2 ... float32 (channels, 2, n/F^L) min/max envelopes (F = PYRAMID_FACTOR)
      attrs: fs_display, fs_source, decimation, channels, filter, created

Readers only ever touch the chunks that cover the requested time range.
"""
import math
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import zarr

from .config import cfg

MAX_RAW_READ = 4_000_000       # samples read from `display` before switching to a pyramid level


def new_group_path(exp_id, rec_id):
    gen = int(time.time() * 1000)
    return f"experiments/{int(exp_id)}/recordings/{int(rec_id)}/v{gen}"


def _root(mode="a"):
    return zarr.open_group(store=str(cfg().ZARR_ROOT), mode=mode)


class DisplayWriter:
    """Create the display array and write it one channel at a time."""

    def __init__(self, group_path, n_channels, n_samples, attrs):
        self.group_path = group_path
        root = _root("a")
        self.g = root.require_group(group_path)
        c = cfg()
        self.chunk = min(c.CHUNK_SAMPLES, max(1, n_samples))
        self.n_channels, self.n = n_channels, n_samples
        self.display = self.g.create_array(
            "display", shape=(n_channels, n_samples), chunks=(1, self.chunk),
            dtype="float32", fill_value=float("nan"), overwrite=True)
        self.levels = []
        f = c.PYRAMID_FACTOR
        L, n = 1, n_samples
        while True:
            bins = math.ceil(n_samples / f ** L)
            if bins < 64 or L > 8:
                break
            arr = self.g.create_array(
                f"pyr{L}", shape=(n_channels, 2, bins), chunks=(1, 2, min(self.chunk, bins)),
                dtype="float32", fill_value=float("nan"), overwrite=True)
            self.levels.append((L, f ** L, arr))
            L += 1
        self.g.attrs.update(attrs | {"pyramid_factor": f, "levels": [lv for lv, _, _ in self.levels]})

    def write_channel(self, ch, y):
        y = np.asarray(y, dtype=np.float32)
        assert y.shape[0] == self.n, (y.shape, self.n)
        self.display[ch, :] = y
        mn, mx = y, y
        prev_factor = 1
        for L, factor, arr in self.levels:
            step = factor // prev_factor
            mn = _block_reduce(mn, step, np.fmin)
            mx = _block_reduce(mx, step, np.fmax)
            arr[ch, 0, :] = mn[: arr.shape[2]]
            arr[ch, 1, :] = mx[: arr.shape[2]]
            prev_factor = factor


def _block_reduce(x, step, fn):
    n = len(x)
    nb = math.ceil(n / step)
    pad = nb * step - n
    if pad:
        x = np.concatenate([x, np.full(pad, np.nan, dtype=x.dtype)])
    x = x.reshape(nb, step)
    with np.errstate(all="ignore"):
        return fn.reduce(x, axis=1)


# ------------------------------------------------------------------ reading
class _LRU:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.d = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key, make):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
        val = make()
        with self.lock:
            self.d[key] = val
            self.d.move_to_end(key)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)
        return val


_arrays = _LRU(256)
_chunks = _LRU(1024)   # decoded chunks, ~0.5 MB each at the default chunk size


def _array(group_path, name):
    return _arrays.get((group_path, name),
                       lambda: zarr.open_array(store=str(Path(cfg().ZARR_ROOT) / group_path / name), mode="r"))


def group_attrs(group_path):
    return dict(zarr.open_group(store=str(Path(cfg().ZARR_ROOT) / group_path), mode="r").attrs)


def _read_1d(group_path, name, ch, i0, i1, sub=None):
    """Read arr[ch, (sub,) i0:i1] via a decoded-chunk cache."""
    arr = _array(group_path, name)
    n = arr.shape[-1]
    i0, i1 = max(0, int(i0)), min(n, int(i1))
    if i1 <= i0:
        return np.zeros(0, dtype=np.float32)
    cs = arr.chunks[-1]
    out = []
    for ci in range(i0 // cs, (i1 - 1) // cs + 1):
        def make(ci=ci):
            a, b = ci * cs, min(n, (ci + 1) * cs)
            return np.asarray(arr[ch, a:b] if sub is None else arr[ch, sub, a:b])
        chunk = _chunks.get((group_path, name, ch, sub, ci), make)
        a = ci * cs
        out.append(chunk[max(i0, a) - a: min(i1, a + len(chunk)) - a])
    return np.concatenate(out) if len(out) > 1 else out[0]


def read_window(group_path, ch, t0, t1, fs, n_total, levels, factor, budget=8000):
    """Return (x, y, info) for channel `ch` between t0 and t1 seconds.

    If the window holds <= budget samples the exact display samples are returned;
    otherwise a min/max envelope with ~budget points is returned (spikes are preserved).
    """
    i0 = max(0, int(math.floor(t0 * fs)))
    i1 = min(n_total, int(math.ceil(t1 * fs)) + 1)
    n = max(0, i1 - i0)
    if n == 0:
        return np.array([t0, t1]), np.array([np.nan, np.nan]), {"mode": "empty", "bin_s": 0.0}
    if n <= budget:
        y = _read_1d(group_path, "display", ch, i0, i1)
        x = (i0 + np.arange(len(y))) / fs
        return x, y, {"mode": "full", "bin_s": 1.0 / fs}
    # choose a source resolution
    level, lf = 0, 1
    for L in levels:
        if n / lf <= MAX_RAW_READ:
            break
        level, lf = L, factor ** L
    if level == 0:
        y = _read_1d(group_path, "display", ch, i0, i1)
        mn = mx = y
        src_t0 = i0 / fs
        src_dt = 1.0 / fs
    else:
        j0, j1 = i0 // lf, math.ceil(i1 / lf)
        mn = _read_1d(group_path, f"pyr{level}", ch, j0, j1, sub=0)
        mx = _read_1d(group_path, f"pyr{level}", ch, j0, j1, sub=1)
        src_t0 = j0 * lf / fs
        src_dt = lf / fs
    nb = max(1, min(budget // 2, len(mn)))
    edges = np.linspace(0, len(mn), nb + 1).astype(int)
    starts = edges[:-1]
    with np.errstate(all="ignore"):
        bmn = np.fmin.reduceat(mn, starts)
        bmx = np.fmax.reduceat(mx, starts)
    tb = src_t0 + starts * src_dt
    tm = src_t0 + (starts + np.diff(edges) / 2.0) * src_dt
    x = np.empty(2 * nb)
    y = np.empty(2 * nb, dtype=np.float32)
    x[0::2], x[1::2] = tb, tm
    y[0::2], y[1::2] = bmn, bmx
    return x, y, {"mode": "envelope", "bin_s": float((len(mn) / nb) * src_dt)}


def values_at(group_path, ch, times, fs, n_total):
    """Linear interpolation of the display signal at arbitrary times."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    out = np.full(times.shape, np.nan)
    for k, t in enumerate(times):
        if not np.isfinite(t):
            continue
        f = t * fs
        i = int(math.floor(f))
        if i < 0 or i >= n_total:
            continue
        seg = _read_1d(group_path, "display", ch, i, min(n_total, i + 2))
        if len(seg) == 1:
            out[k] = float(seg[0])
        else:
            w = f - i
            out[k] = float(seg[0] * (1 - w) + seg[1] * w)
    return out


def delete_group(group_path):
    import shutil
    p = Path(cfg().ZARR_ROOT) / group_path
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def clear_caches():
    _arrays.d.clear()
    _chunks.d.clear()
