"""Minimal multi-channel ABF1 (gap-free) writer, used only to create realistic test files
that pyabf reads through its normal code path (the lab's real files are ABF2, which pyabf
cannot write). Layout follows pyabf/abf1/headerV1.py.
"""
import struct

import numpy as np

BLOCK = 512
HEADER_BLOCKS = 12


def write_abf1(path, data, fs, names, units):
    data = np.asarray(data, dtype=np.float64)
    n_ch, n = data.shape
    total = n_ch * n
    hdr = bytearray(HEADER_BLOCKS * BLOCK)
    struct.pack_into("4s", hdr, 0, b"ABF ")
    struct.pack_into("f", hdr, 4, 1.83)            # fFileVersionNumber
    struct.pack_into("h", hdr, 8, 3)               # nOperationMode = gap-free
    struct.pack_into("i", hdr, 10, total)          # lActualAcqLength
    struct.pack_into("i", hdr, 16, 1)              # lActualEpisodes
    struct.pack_into("i", hdr, 40, HEADER_BLOCKS)  # lDataSectionPtr
    struct.pack_into("h", hdr, 100, 0)             # nDataFormat int16
    struct.pack_into("h", hdr, 120, n_ch)          # nADCNumChannels
    struct.pack_into("f", hdr, 122, 1e6 / (fs * n_ch))  # fADCSampleInterval (per interleaved sample)
    struct.pack_into("i", hdr, 138, total)         # lNumSamplesPerEpisode
    adc_range, resolution = 10.0, 32768
    struct.pack_into("f", hdr, 244, adc_range)
    struct.pack_into("i", hdr, 252, resolution)
    raw = np.empty((n, n_ch), dtype=np.int16)
    for i in range(16):
        struct.pack_into("h", hdr, 378 + 2 * i, i)                       # nADCPtoLChannelMap
        struct.pack_into("h", hdr, 410 + 2 * i, i if i < n_ch else -1)   # nADCSamplingSeq
        struct.pack_into("f", hdr, 1050 + 4 * i, 1.0)                    # fSignalGain
        struct.pack_into("f", hdr, 730 + 4 * i, 1.0)                     # fADCProgrammableGain
        struct.pack_into("f", hdr, 922 + 4 * i, 1.0)                     # fInstrumentScaleFactor
    for c in range(n_ch):
        y = data[c]
        off = float(np.mean(y))
        dev = float(np.max(np.abs(y - off))) or 1.0
        scale = adc_range * 0.95 / dev            # value (V at ADC) per physical unit
        struct.pack_into("f", hdr, 922 + 4 * c, scale)
        struct.pack_into("f", hdr, 986 + 4 * c, off)                     # fInstrumentOffset
        struct.pack_into("10s", hdr, 442 + 10 * c, names[c].ljust(10)[:10].encode())
        struct.pack_into("8s", hdr, 602 + 8 * c, units[c].ljust(8)[:8].encode())
        raw[:, c] = np.round((y - off) * scale * resolution / adc_range).astype(np.int16)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(raw.tobytes())
