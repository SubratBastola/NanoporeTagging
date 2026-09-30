"""Runtime configuration, read from environment variables.

All paths default to the layout created by deploy/install.sh:
    /srv/nanotag/{db,zarr,abf,incoming,uploads,jobs,models,backups}
The environment file is /etc/nanotag/nanotag.env (loaded by systemd).
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
APP_DIR = PKG_DIR.parent


def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


class Config:
    def __init__(self):
        self.DATA = Path(_env("NANOTAG_DATA", "/srv/nanotag"))
        self.DB_PATH = Path(_env("NANOTAG_DB", str(self.DATA / "db" / "nanotag.sqlite3")))
        self.ZARR_ROOT = Path(_env("NANOTAG_ZARR", str(self.DATA / "zarr" / "store.zarr")))
        self.ABF_DIR = Path(_env("NANOTAG_ABF_DIR", str(self.DATA / "abf")))
        self.INCOMING_DIR = Path(_env("NANOTAG_INCOMING", str(self.DATA / "incoming")))
        self.UPLOAD_DIR = Path(_env("NANOTAG_UPLOADS", str(self.DATA / "uploads")))
        self.JOBS_DIR = Path(_env("NANOTAG_JOBS", str(self.DATA / "jobs")))
        self.MODELS_DIR = Path(_env("NANOTAG_MODELS", str(self.DATA / "models")))
        self.BACKUP_DIR = Path(_env("NANOTAG_BACKUPS", str(self.DATA / "backups")))
        self.VENDOR_DIR = Path(_env("NANOTAG_VENDOR", str(APP_DIR / "vendor")))
        roots = _env("NANOTAG_IMPORT_ROOTS", str(self.INCOMING_DIR))
        self.IMPORT_ROOTS = [Path(p) for p in roots.split(":") if p.strip()]
        self.SECRET_KEY = _env("NANOTAG_SECRET_KEY", "dev-insecure-change-me")
        self.WORKERS = int(_env("NANOTAG_WORKERS", "4"))
        self.TORCH_THREADS = int(_env("NANOTAG_TORCH_THREADS", "4"))
        # Tagger display settings (mirror Tagger_GUI.py constants)
        self.DISPLAY_FS = float(_env("NANOTAG_DISPLAY_FS", "10000"))
        self.CHUNK_SAMPLES = int(_env("NANOTAG_CHUNK_SAMPLES", str(1 << 17)))
        self.PYRAMID_FACTOR = int(_env("NANOTAG_PYRAMID_FACTOR", "10"))
        self.ALLOW_FAKE_ABF = _env("NANOTAG_ALLOW_FAKE_ABF", "0") == "1"
        self.VERSION = _read_version()

    def ensure_dirs(self):
        for p in (self.DB_PATH.parent, self.ZARR_ROOT.parent, self.ABF_DIR, self.INCOMING_DIR,
                  self.UPLOAD_DIR, self.JOBS_DIR, self.MODELS_DIR, self.BACKUP_DIR):
            p.mkdir(parents=True, exist_ok=True)


def _read_version():
    f = APP_DIR / "VERSION"
    try:
        return f.read_text().strip()
    except Exception:
        return "dev"


_cfg = None


def cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg


def reset_cfg():
    """For tests: re-read the environment."""
    global _cfg
    _cfg = None
