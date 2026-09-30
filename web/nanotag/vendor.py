"""Load the original NeuralNetwork.py / clustering.py unmodified.

Both scripts import tkinter for their desktop GUIs. The server never calls those
GUI functions, so if tkinter is not installed a harmless stub module is inserted
before import. All algorithm code runs exactly as written in the originals.
"""
import importlib.util
import os
import sys
import types

from .config import cfg

_cache = {}


def _ensure_tk_stub():
    try:
        import tkinter  # noqa: F401
        return
    except Exception:
        pass
    tk = types.ModuleType("tkinter")
    for sub in ("filedialog", "ttk", "simpledialog", "messagebox", "scrolledtext"):
        m = types.ModuleType(f"tkinter.{sub}")
        setattr(tk, sub, m)
        sys.modules[f"tkinter.{sub}"] = m
    sys.modules["tkinter"] = tk


def _load(name, filename):
    if name in _cache:
        return _cache[name]
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_tk_stub()
    path = cfg().VENDOR_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Original script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"nanotag_vendor_{name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cache[name] = mod
    return mod


def neuralnetwork():
    return _load("nn", "NeuralNetwork.py")


def clustering():
    import matplotlib
    matplotlib.use("Agg", force=True)
    return _load("clustering", "clustering.py")
