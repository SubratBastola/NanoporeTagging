"""Annotation sets and events, with a full audit trail.

Every create/update/delete writes an event_history row (who, when, before, after)
and bumps the set's `revision`, which the Tagger polls to pick up other users' edits.
Updates use optimistic concurrency: the caller passes the version it last saw.
"""
from .db import EVENT_FIELDS, db, jdump, jload, now, one, rows, tx

LANDMARK_FIELDS = {
    "event_start", "event_plateau", "event_end",
    "optical_base", "optical_rise", "optical_end",
    "opticalre_base", "opticalre_rise", "opticalre_end",
    "entry_base", "entry_peak", "exit_peak", "exit_base",
    "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
}


class ConflictError(Exception):
    pass


class LockedError(PermissionError):
    pass


# ------------------------------------------------------------------ sets
def get_set(set_id):
    with db() as con:
        return one(con, "SELECT * FROM annotation_sets WHERE id=?", (int(set_id),))


def list_sets(exp_id):
    with db() as con:
        return rows(con, """SELECT s.*,
                   (SELECT COUNT(*) FROM events e WHERE e.set_id=s.id AND e.deleted=0) AS n_events,
                   (SELECT MAX(at) FROM event_history h WHERE h.set_id=s.id) AS last_edit
                   FROM annotation_sets s WHERE s.experiment_id=? ORDER BY s.created_at""", (exp_id,))


def create_set(exp_id, name, user, kind="manual", formula="tagger", source=None):
    name = (name or "").strip()
    if not name:
        raise ValueError("Annotation set name is required.")
    with db() as con, tx(con):
        if one(con, "SELECT id FROM annotation_sets WHERE experiment_id=? AND name=?", (exp_id, name)):
            raise ValueError(f"An annotation set named '{name}' already exists in this experiment.")
        cur = con.execute("""INSERT INTO annotation_sets(experiment_id, name, kind, formula, source_json,
                              created_by, created_at) VALUES (?,?,?,?,?,?,?)""",
                          (exp_id, name, kind, formula, jdump(source or {}), user, now()))
        return cur.lastrowid


def unique_set_name(exp_id, base):
    with db() as con:
        names = {r["name"] for r in rows(con, "SELECT name FROM annotation_sets WHERE experiment_id=?", (exp_id,))}
    if base not in names:
        return base
    i = 2
    while f"{base} ({i})" in names:
        i += 1
    return f"{base} ({i})"


def copy_set(set_id, new_name, user):
    src = get_set(set_id)
    new_id = create_set(src["experiment_id"], new_name, user, kind="manual", formula=src["formula"],
                        source={"copied_from": set_id, "copied_from_name": src["name"]})
    evs = list_events(set_id)
    by_rec = {}
    for e in evs:
        by_rec.setdefault(e["recording_id"], []).append(
            {k: e.get(k) for k in EVENT_FIELDS} | {"derived": jload(e.get("derived_json"))})
    for rec_id, lst in by_rec.items():
        insert_events(new_id, rec_id, lst, user, action="copy")
    return new_id


def delete_set(set_id):
    with db() as con, tx(con):
        con.execute("DELETE FROM event_history WHERE set_id=?", (set_id,))
        con.execute("DELETE FROM annotation_sets WHERE id=?", (set_id,))


def set_locked(set_id, locked):
    with db() as con:
        con.execute("UPDATE annotation_sets SET locked=? WHERE id=?", (1 if locked else 0, set_id))


def rename_set(set_id, name):
    with db() as con:
        con.execute("UPDATE annotation_sets SET name=? WHERE id=?", (name.strip(), set_id))


def revision(set_id):
    with db() as con:
        r = one(con, "SELECT revision FROM annotation_sets WHERE id=?", (set_id,))
    return r["revision"] if r else None


def _bump(con, set_id):
    con.execute("UPDATE annotation_sets SET revision=revision+1 WHERE id=?", (set_id,))


# ------------------------------------------------------------------ events
def list_events(set_id, recording_id=None, include_deleted=False):
    sql = "SELECT * FROM events WHERE set_id=?"
    args = [set_id]
    if recording_id is not None:
        sql += " AND recording_id=?"
        args.append(recording_id)
    if not include_deleted:
        sql += " AND deleted=0"
    sql += " ORDER BY recording_id, COALESCE(event_no, 1e18), id"
    with db() as con:
        return rows(con, sql, args)


def get_event(event_id):
    with db() as con:
        return one(con, "SELECT * FROM events WHERE id=?", (event_id,))


def _next_event_no(con, set_id, rec_id):
    r = one(con, "SELECT MAX(event_no) AS m FROM events WHERE set_id=? AND recording_id=?", (set_id, rec_id))
    return int(r["m"] or 0) + 1


def insert_events(set_id, rec_id, evs, user, action="import"):
    """Bulk insert. Each ev is a dict of EVENT_FIELDS (+ optional 'derived')."""
    t = now()
    ids = []
    with db() as con, tx(con):
        nxt = _next_event_no(con, set_id, rec_id)
        for ev in evs:
            ev = dict(ev)
            if ev.get("event_no") is None:
                ev["event_no"] = nxt
            nxt = max(nxt, int(ev["event_no"]) + 1)
            derived = ev.pop("derived", None)
            vals = [ev.get(k) for k in EVENT_FIELDS]
            cur = con.execute(
                f"""INSERT INTO events(set_id, recording_id, {', '.join(EVENT_FIELDS)}, derived_json,
                     version, deleted, created_by, created_at, updated_by, updated_at)
                    VALUES (?,?,{','.join('?' * len(EVENT_FIELDS))},?,1,0,?,?,?,?)""",
                [set_id, rec_id] + vals + [jdump(derived) if derived else None, user, t, user, t])
            eid = cur.lastrowid
            ids.append(eid)
            con.execute("INSERT INTO event_history(event_id, set_id, action, username, at, after_json) "
                        "VALUES (?,?,?,?,?,?)", (eid, set_id, action, user, t,
                                                 jdump({k: ev.get(k) for k in EVENT_FIELDS})))
        _bump(con, set_id)
    return ids


def add_event(set_id, rec_id, fields, user):
    return insert_events(set_id, rec_id, [fields], user, action="create")[0]


def update_event(event_id, fields, expected_version, user):
    fields = {k: v for k, v in fields.items() if k in EVENT_FIELDS}
    if not fields:
        return get_event(event_id)
    t = now()
    with db() as con, tx(con):
        cur = one(con, "SELECT * FROM events WHERE id=?", (event_id,))
        if cur is None or cur["deleted"]:
            raise ConflictError("This event was deleted by someone else.")
        if expected_version is not None and int(cur["version"]) != int(expected_version):
            raise ConflictError(f"This event was changed by {cur['updated_by']} — reloaded the latest version.")
        sets = ", ".join(f"{k}=?" for k in fields)
        extra = ", derived_json=NULL" if LANDMARK_FIELDS & set(fields) else ""
        con.execute(f"UPDATE events SET {sets}{extra}, version=version+1, updated_by=?, updated_at=? WHERE id=?",
                    list(fields.values()) + [user, t, event_id])
        con.execute("INSERT INTO event_history(event_id, set_id, action, username, at, before_json, after_json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (event_id, cur["set_id"], "update", user, t,
                     jdump({k: cur[k] for k in fields}), jdump(fields)))
        _bump(con, cur["set_id"])
        return one(con, "SELECT * FROM events WHERE id=?", (event_id,))


def delete_events(event_ids, user):
    t = now()
    n = 0
    with db() as con, tx(con):
        for eid in event_ids:
            cur = one(con, "SELECT * FROM events WHERE id=? AND deleted=0", (eid,))
            if cur is None:
                continue
            con.execute("UPDATE events SET deleted=1, version=version+1, updated_by=?, updated_at=? WHERE id=?",
                        (user, t, eid))
            con.execute("INSERT INTO event_history(event_id, set_id, action, username, at, before_json) "
                        "VALUES (?,?,?,?,?,?)",
                        (eid, cur["set_id"], "delete", user, t, jdump({k: cur[k] for k in EVENT_FIELDS})))
            _bump(con, cur["set_id"])
            n += 1
    return n


def history(set_id):
    with db() as con:
        return rows(con, """SELECT h.*, e.recording_id, e.event_no FROM event_history h
                            LEFT JOIN events e ON e.id=h.event_id
                            WHERE h.set_id=? ORDER BY h.at""", (set_id,))


def authors(set_id):
    with db() as con:
        return [r["u"] for r in rows(con, """SELECT DISTINCT COALESCE(created_by,'?') AS u FROM events
                                              WHERE set_id=? AND deleted=0 ORDER BY u""", (set_id,))]
