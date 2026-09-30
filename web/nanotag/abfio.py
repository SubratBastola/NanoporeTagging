"""ABF access. Production uses pyabf exactly as the original scripts do.

For automated tests only (NANOTAG_ALLOW_FAKE_ABF=1) a '*.fakeabf.npz' file can
stand in for an ABF; it mimics the small subset of the pyabf.ABF API used by
Tagger_GUI.py and NeuralNetwork.py.
"""
import numpy as np

from .config import cfg


class FakeABF:
    def __init__(self, path, loadData=True):
        z = np.load(path, allow_pickle=False)
        self.data = z["data"].astype(np.float32)
        self.dataRate = int(z["fs"])
        self.adcNames = [str(s) for s in z["names"]]
        self.adcUnits = [str(s) for s in z["units"]]
        self.channelCount = self.data.shape[0]
        self.sweepCount = 1
        self.dataPointCount = int(self.data.size)
        self.dataSecPerPoint = 1.0 / self.dataRate
        self.abfVersionString = "fake"
        self.sweepLengthSec = self.data.shape[1] / self.dataRate
        self.setSweep(0)

    def setSweep(self, sweepNumber=0, channel=0, **kw):
        self.sweepY = self.data[channel]
        self.sweepX = np.arange(len(self.sweepY)) * self.dataSecPerPoint


def _is_fake(path):
    return str(path).endswith(".fakeabf.npz")


def open_abf(path, load_data=True):
    if _is_fake(path):
        if not cfg().ALLOW_FAKE_ABF:
            raise ValueError("Fake ABF files are only allowed in test mode.")
        return FakeABF(path, loadData=load_data)
    import pyabf
    return pyabf.ABF(str(path), loadData=load_data)


def header_info(path):
    """Cheap metadata read (no signal data loaded)."""
    a = open_abf(path, load_data=False)
    n_ch = int(a.channelCount)
    fs = float(a.dataRate)
    n = int(a.dataPointCount // max(1, n_ch))
    chans = [{"index": i, "name": str(a.adcNames[i]), "units": str(a.adcUnits[i])} for i in range(n_ch)]
    return {
        "fs": fs,
        "n_samples": n,
        "duration": n / fs if fs else 0.0,
        "n_sweeps": int(getattr(a, "sweepCount", 1)),
        "channels": chans,
        "abf_version": str(getattr(a, "abfVersionString", "")),
    }


def full_channel(abf, ch):
    """The complete signal of one channel in physical units.

    For the single-sweep gap-free recordings this is identical to
    `abf.setSweep(0, channel=ch); abf.sweepY` (what Tagger_GUI.py reads);
    for multi-sweep files the sweeps are concatenated end-to-end, matching the
    absolute timeline NeuralNetwork.py builds in resolve_sweep_range().
    """
    if getattr(abf, "sweepCount", 1) <= 1:
        abf.setSweep(0, channel=ch)
        return np.asarray(abf.sweepY)
    parts = []
    for s in range(abf.sweepCount):
        abf.setSweep(sweepNumber=s, channel=ch)
        parts.append(np.asarray(abf.sweepY))
    return np.concatenate(parts)


def write_fake_abf(path, data, fs, names, units):
    np.savez(path, data=np.asarray(data, dtype=np.float32), fs=int(fs),
             names=np.array(names), units=np.array(units))
