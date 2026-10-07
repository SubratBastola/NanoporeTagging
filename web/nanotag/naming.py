"""Experiment names and acquisition conditions guessed from ABF file names.

Default rule: take the file name without its extension, drop the per-file run number that follows
the date, and drop the trailing acquisition-condition word (DC, AC, AOM, AC-AOM, baseline ...):

    "2026_09_14_0001 B3S8-15 100 aM SiO2 DC.abf"        ->  "2026_09_14 B3S8-15 100 aM SiO2"   (DC)
    "2026_09_14_0004 B3S8-15 100 aM SiO2 AC.abf"        ->  "2026_09_14 B3S8-15 100 aM SiO2"   (AC)
    "2026_09_14_0007 B3S8-15 100 aM SiO2 AOM.abf"       ->  "2026_09_14 B3S8-15 100 aM SiO2"   (AOM)
    "2026_09_14_0000 B3S8-15 100 aM SiO2 baseline.abf"  ->  "2026_09_14 B3S8-15 100 aM SiO2"   (baseline)

so all recordings of one day/sensor/sample land in the same experiment, and each recording remembers
its condition (used to filter or split clustering). The rules can be changed without code edits in
/etc/nanotag/nanotag.env (then restart):

    NANOTAG_EXPERIMENT_RUN     regex matching the run number to drop; group 1 is kept
                               (default: ^(\\d{4}_\\d{2}_\\d{2})_\\d{3,5}(?=[\\s_-]|$) -> keeps the date)
    NANOTAG_EXPERIMENT_STRIP   regex removed from the end of the name; group 1 is the condition
                               (default: (?i)[\\s_-]+(DC|AC[\\s_-]*AOM|AC|AOM|baseline)$)
    NANOTAG_EXPERIMENT_PREFIX  optional text put in front of every guessed name
"""
import os
import re
from pathlib import Path

from . import annotations as A
from .db import db, now, one

DEFAULT_RUN = r"^(\d{4}_\d{2}_\d{2})_\d{3,5}(?=[\s_-]|$)"
DEFAULT_STRIP = r"(?i)[\s_-]+(DC|AC[\s_-]*AOM|AC|AOM|baseline)$"
RUN_NO = re.compile(r"^\d{4}_\d{2}_\d{2}_(\d{3,5})(?=[\s_-]|$)")
KNOWN_EXTS = (".fakeabf.npz", ".abf", ".csv", ".xlsx", ".xls")


def strip_ext(file_name: str) -> str:
    stem = Path(str(file_name)).name
    for ext in KNOWN_EXTS:
        if stem.lower().endswith(ext):
            return stem[: -len(ext)]
    return stem


def normalize_condition(word: str) -> str:
    w = re.sub(r"[\s_-]+", "", (word or "")).upper()
    if not w:
        return ""
    if w == "BASELINE":
        return "baseline"
    if w == "ACAOM":
        return "AC-AOM"
    return w


def parse_file_name(file_name: str) -> dict:
    """{'experiment': ..., 'condition': 'DC'|'AC'|'AOM'|'AC-AOM'|'baseline'|'', 'run': '0001'|None}"""
    import unicodedata
    # non-breaking / odd Unicode spaces (Windows copy-paste, Excel) and doubled spaces count as one space
    stem = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", strip_ext(file_name))).strip()
    name = stem
    run_pat = os.environ.get("NANOTAG_EXPERIMENT_RUN") or DEFAULT_RUN
    strip_pat = os.environ.get("NANOTAG_EXPERIMENT_STRIP") or DEFAULT_STRIP
    try:
        name = re.sub(run_pat, lambda m: m.group(1) if m.groups() else "", name)
    except (re.error, IndexError):
        pass
    condition = ""
    try:
        m = re.search(strip_pat, name)
        if m:
            condition = normalize_condition(m.group(1) if m.groups() else m.group(0))
            name = name[: m.start()]
    except (re.error, IndexError):
        pass
    name = re.sub(r"\s{2,}", " ", name).strip(" _-")
    prefix = os.environ.get("NANOTAG_EXPERIMENT_PREFIX", "")
    rm = RUN_NO.match(stem)
    return {"experiment": (prefix + (name or stem)).strip() or "Unnamed experiment",
            "condition": condition, "run": rm.group(1) if rm else None}


def guess_experiment_name(file_name: str) -> str:
    return parse_file_name(file_name)["experiment"]


def guess_condition(file_name: str) -> str:
    return parse_file_name(file_name)["condition"]


def get_or_create_experiment(name: str, user: str):
    """Returns (experiment_id, created)."""
    with db() as con:
        ex = one(con, "SELECT id FROM experiments WHERE name=?", (name,))
        if ex:
            return ex["id"], False
        cur = con.execute("INSERT INTO experiments(name, created_by, created_at) VALUES (?,?,?)",
                          (name, user, now()))
        exp_id = cur.lastrowid
    A.create_set(exp_id, "Manual tags", user)
    return exp_id, True
