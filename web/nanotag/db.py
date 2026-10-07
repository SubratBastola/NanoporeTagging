"""SQLite database: schema, migrations and small helpers.

The database holds everything except signal data:
users, experiments, recordings (metadata), annotation sets, events,
event history (audit trail), models, jobs and clustering runs.
Signals live in the Zarr store (see store.py); original ABFs stay on disk.
"""
import json
import sqlite3
import time
from contextlib import contextmanager

from .config import cfg

SCHEMA_VERSION = 2

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name     TEXT DEFAULT '',
    pw_hash       TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL,
    last_login    REAL
);

CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT DEFAULT '',
    sensor        TEXT DEFAULT '',
    analytes      TEXT DEFAULT '',
    solution      TEXT DEFAULT '',
    restricted    INTEGER NOT NULL DEFAULT 0,
    created_by    TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_access (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    username      TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (experiment_id, username)
);

CREATE TABLE IF NOT EXISTS recordings (
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    file_name     TEXT NOT NULL,
    stem          TEXT NOT NULL,
    abf_path      TEXT NOT NULL,
    sha256        TEXT,
    size_bytes    INTEGER,
    fs            REAL,
    n_samples     INTEGER,
    duration      REAL,
    n_sweeps      INTEGER,
    channels_json TEXT DEFAULT '[]',
    role_elec     INTEGER NOT NULL DEFAULT 0,
    role_opt      INTEGER NOT NULL DEFAULT 2,
    role_optref   INTEGER NOT NULL DEFAULT 3,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    zarr_path     TEXT,
    display_fs    REAL,
    display_n     INTEGER,
    created_by    TEXT,
    created_at    REAL NOT NULL,
    UNIQUE (experiment_id, stem)
);

CREATE TABLE IF NOT EXISTS annotation_sets (
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'manual',
    formula       TEXT NOT NULL DEFAULT 'tagger',
    source_json   TEXT DEFAULT '{}',
    locked        INTEGER NOT NULL DEFAULT 0,
    revision      INTEGER NOT NULL DEFAULT 0,
    created_by    TEXT,
    created_at    REAL NOT NULL,
    UNIQUE (experiment_id, name)
);

CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY,
    set_id         INTEGER NOT NULL REFERENCES annotation_sets(id) ON DELETE CASCADE,
    recording_id   INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    event_no       INTEGER,
    window_start   REAL, window_end REAL,
    event_start    REAL, event_plateau REAL, event_end REAL,
    optical_base   REAL, optical_rise REAL, optical_end REAL,
    opticalre_base REAL, opticalre_rise REAL, opticalre_end REAL,
    entry_base     REAL, entry_peak REAL, exit_peak REAL, exit_base REAL,
    entry_base_t   REAL, entry_peak_t REAL, exit_peak_t REAL, exit_base_t REAL,
    sensor         TEXT, analytes TEXT, solution TEXT, notes TEXT,
    derived_json   TEXT,
    version        INTEGER NOT NULL DEFAULT 1,
    deleted        INTEGER NOT NULL DEFAULT 0,
    created_by     TEXT, created_at REAL,
    updated_by     TEXT, updated_at REAL
);
CREATE INDEX IF NOT EXISTS ix_events_set_rec ON events(set_id, recording_id, deleted);

CREATE TABLE IF NOT EXISTS event_history (
    id           INTEGER PRIMARY KEY,
    event_id     INTEGER NOT NULL,
    set_id       INTEGER NOT NULL,
    action       TEXT NOT NULL,
    username     TEXT,
    at           REAL NOT NULL,
    before_json  TEXT,
    after_json   TEXT
);
CREATE INDEX IF NOT EXISTS ix_hist_set ON event_history(set_id);

CREATE TABLE IF NOT EXISTS models (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    path         TEXT NOT NULL,
    sha256       TEXT,
    is_default   INTEGER NOT NULL DEFAULT 0,
    uploaded_by  TEXT,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL,
    experiment_id INTEGER,
    params_json   TEXT DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'queued',
    progress      REAL DEFAULT 0,
    message       TEXT DEFAULT '',
    log           TEXT DEFAULT '',
    result_json   TEXT DEFAULT '{}',
    created_by    TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS cluster_runs (
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    set_ids_json  TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    job_id        INTEGER,
    final_k       INTEGER,
    k_bic         INTEGER,
    n_events      INTEGER,
    features_json TEXT,
    out_dir       TEXT,
    created_by    TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_labels (
    run_id        INTEGER NOT NULL REFERENCES cluster_runs(id) ON DELETE CASCADE,
    event_id      INTEGER NOT NULL,
    recording_id  INTEGER,
    cluster       INTEGER,
    PRIMARY KEY (run_id, event_id)
);
"""

EVENT_FIELDS = [
    "event_no", "window_start", "window_end",
    "event_start", "event_plateau", "event_end",
    "optical_base", "optical_rise", "optical_end",
    "opticalre_base", "opticalre_rise", "opticalre_end",
    "entry_base", "entry_peak", "exit_peak", "exit_base",
    "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
    "sensor", "analytes", "solution", "notes",
]
EVENT_NUMERIC = [f for f in EVENT_FIELDS if f not in ("sensor", "analytes", "solution", "notes", "event_no")]


def now():
    return time.time()


def connect(path=None):
    p = str(path or cfg().DB_PATH)
    con = sqlite3.connect(p, timeout=30.0, isolation_level=None)  # autocommit; explicit BEGIN
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


@contextmanager
def db():
    con = connect()
    try:
        yield con
    finally:
        con.close()


@contextmanager
def tx(con):
    """Write transaction (BEGIN IMMEDIATE so writers serialize cleanly)."""
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def migrate(con=None):
    own = con is None
    con = con or connect()
    try:
        v = con.execute("PRAGMA user_version").fetchone()[0]
        if v < 1:
            con.executescript(SCHEMA_V1)
            con.execute("PRAGMA user_version=1")
        if v < 2:
            # v2: per-recording acquisition condition (DC / AC / AOM / baseline ...) and the folder the ABF
            # came from (for ABFs used in place, abf_path == source_path and NanoTag never deletes them).
            cols = {r[1] for r in con.execute("PRAGMA table_info(recordings)")}
            if "condition" not in cols:
                con.execute("ALTER TABLE recordings ADD COLUMN condition TEXT DEFAULT ''")
            if "source_path" not in cols:
                con.execute("ALTER TABLE recordings ADD COLUMN source_path TEXT")
            con.execute("PRAGMA user_version=2")
        # future migrations: if v < 3: ...
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        if own:
            con.close()


def row_to_dict(r):
    return None if r is None else {k: r[k] for k in r.keys()}


def rows(con, sql, args=()):
    return [row_to_dict(r) for r in con.execute(sql, args).fetchall()]


def one(con, sql, args=()):
    return row_to_dict(con.execute(sql, args).fetchone())


def jdump(x):
    return json.dumps(x, default=_json_default)


def jload(s, default=None):
    if s in (None, ""):
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
