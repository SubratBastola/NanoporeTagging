"""nanotag-admin: command-line administration.

    nanotag-admin init                              create folders + database (idempotent)
    nanotag-admin ensure-admin [--password PW]      create 'admin' if it doesn't exist (default admin2025!!)
    nanotag-admin add-user NAME [--admin] [--password PW] [--full-name "..."]
    nanotag-admin passwd NAME [--password PW]       (prompts if --password is omitted)
    nanotag-admin set-role NAME admin|user
    nanotag-admin disable NAME | enable NAME
    nanotag-admin list-users
    nanotag-admin register-model PATH [--name N] [--default]
    nanotag-admin backup [--keep-days 14]
    nanotag-admin import-folder DIR [--mode inplace|copy|link|move] [--all] [--recursive] [--set-name N]
                                [--user NAME] [--dry-run]
                                                    import every ABF + its _event.csv (like the web page)
    nanotag-admin check                             self-test used by install/update scripts
"""
import argparse
import getpass
import hashlib
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from . import auth
from .config import cfg
from .db import connect, db, migrate, now, one

DEFAULT_ADMIN_PASSWORD = "admin2025!!"


def _pw(args, confirm=True):
    if args.password:
        return args.password
    p1 = getpass.getpass("New password: ")
    if confirm and p1 != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return p1


def cmd_init(args):
    c = cfg()
    c.ensure_dirs()
    v = migrate()
    print(f"Data folder: {c.DATA}\nDatabase: {c.DB_PATH} (schema v{v})")


def cmd_ensure_admin(args):
    migrate()
    if auth.get_user_row("admin"):
        print("User 'admin' already exists (password unchanged).")
        return
    auth.add_user("admin", args.password or DEFAULT_ADMIN_PASSWORD, "admin", "Administrator", enforce_policy=False)
    print("Created user 'admin'.")


def cmd_add_user(args):
    migrate()
    auth.add_user(args.name, _pw(args), "admin" if args.admin else "user", args.full_name or "")
    print(f"Added {'admin' if args.admin else 'user'} '{args.name}'.")


def cmd_passwd(args):
    auth.set_password(args.name, _pw(args))
    print(f"Password updated for '{args.name}'.")


def cmd_set_role(args):
    auth.set_role(args.name, args.role)
    print(f"'{args.name}' is now {args.role}.")


def cmd_enable(args, active=True):
    auth.set_active(args.name, active)
    print(f"'{args.name}' {'enabled' if active else 'disabled'}.")


def cmd_list(args):
    for u in auth.list_users():
        last = time.strftime("%Y-%m-%d %H:%M", time.localtime(u["last_login"])) if u["last_login"] else "never"
        print(f"{u['username']:20s} {u['role']:6s} {'active' if u['active'] else 'DISABLED':9s} last login: {last}")


def cmd_register_model(args):
    c = cfg()
    c.ensure_dirs()
    migrate()
    src = Path(args.path)
    data = src.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    name = args.name or src.stem
    with db() as con:
        ex = one(con, "SELECT * FROM models WHERE sha256=?", (h,))
        if ex:
            print(f"Model already registered as '{ex['name']}'.")
            if args.default:
                con.execute("UPDATE models SET is_default=(id=?)", (ex["id"],))
            return
        dest = c.MODELS_DIR / f"{name}_{h[:8]}.pt"
        shutil.copy2(src, dest)
        first = one(con, "SELECT COUNT(*) AS n FROM models")["n"] == 0
        cur = con.execute("INSERT INTO models(name, path, sha256, is_default, uploaded_by, created_at) "
                          "VALUES (?,?,?,?,?,?)", (name, str(dest), h, 1 if (first or args.default) else 0,
                                                   "install", now()))
        if args.default:
            con.execute("UPDATE models SET is_default=(id=?)", (cur.lastrowid,))
    print(f"Registered model '{name}' -> {dest}")


def cmd_backup(args):
    c = cfg()
    c.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = c.BACKUP_DIR / f"nanotag-{time.strftime('%Y%m%d-%H%M%S')}.sqlite3"
    src = connect()
    dst = sqlite3.connect(str(dest))
    with dst:
        src.backup(dst)
    dst.close()
    src.close()
    print(f"Backup written: {dest}")
    cutoff = time.time() - args.keep_days * 86400
    for f in c.BACKUP_DIR.glob("nanotag-*.sqlite3"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            print(f"Pruned {f.name}")


def cmd_import_folder(args):
    from . import folder_import
    migrate()
    plan = folder_import.scan(Path(args.dir).resolve(), recursive=args.recursive, set_name=args.set_name)
    items = []
    for g in plan["groups"]:
        print(f"\n{g['experiment']}  ({'existing' if g['exists'] else 'new'} experiment)")
        for it in g["items"]:
            take = it["selected"] or (args.all and not (it["existing"] and it["existing"]["status"] != "missing"))
            e = it["events"]
            ev = (f"{e['name']} ({e['rows']} rows)" if e and not e["error"] else
                  f"{e['name']} UNREADABLE: {e['error']}" if e else "-")
            x = it["existing"]
            st = ("new" if not x else f"in NanoTag, {x['n_events']} events" + (" (DIFFERS)" if x.get("differs") else "")
                  if x["has_events"] else
                  "in NanoTag, EVENTS MISSING" if e else "in NanoTag")
            print(f"  [{'x' if take else ' '}] {it['name']:<50s} {it['condition'] or '-':<9s} {st:<28s} {ev}")
            if take:
                items.append({"abf": it["abf"], "events": it["events"]["path"] if it["events"] and not
                              it["events"]["error"] else None, "experiment": g["experiment"]})
    if plan["unused_tables"]:
        print("\nNot used:")
        for t in plan["unused_tables"]:
            print(f"  {t['name']}: {t['why']}")
    print(f"\n{len(items)} of {plan['n_abf']} ABF file(s) selected.")
    if args.dry_run or not items:
        return
    for exp_id, jid, n in folder_import.enqueue_import(items, args.mode, args.set_name, args.user,
                                                       folder=str(Path(args.dir).resolve())):
        print(f"Queued import job #{jid}: {n} recording(s) -> experiment #{exp_id}")


def cmd_check(args):
    c = cfg()
    ok = True
    print(f"NanoTag {c.VERSION}")
    try:
        v = migrate()
        with db() as con:
            n = one(con, "SELECT COUNT(*) AS n FROM users")["n"]
        print(f"  database ok (schema v{v}, {n} users)")
    except Exception as e:
        ok = False
        print(f"  DATABASE ERROR: {e}")
    for f in ("NeuralNetwork.py", "clustering.py"):
        p = c.VENDOR_DIR / f
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p}")
        ok &= p.exists()
    for mod in ("numpy", "scipy", "pandas", "zarr", "pyabf", "torch", "sklearn", "dash", "flask", "matplotlib"):
        try:
            __import__(mod)
            from importlib.metadata import version as _v
            dist = {"sklearn": "scikit-learn"}.get(mod, mod)
            print(f"  ok  {mod} {_v(dist)}")
        except Exception as e:
            ok = False
            print(f"  IMPORT ERROR {mod}: {e}")
    try:
        from . import vendor
        vendor.neuralnetwork()
        vendor.clustering()
        print("  ok  original NeuralNetwork.py / clustering.py import")
    except Exception as e:
        ok = False
        print(f"  ERROR loading original scripts: {e}")
    du = shutil.disk_usage(c.DATA)
    print(f"  disk free under {c.DATA}: {du.free / 1e12:.2f} TB")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nanotag-admin", description="NanoTag administration")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init").set_defaults(fn=cmd_init)
    p = sp.add_parser("ensure-admin"); p.add_argument("--password"); p.set_defaults(fn=cmd_ensure_admin)
    p = sp.add_parser("add-user"); p.add_argument("name"); p.add_argument("--admin", action="store_true")
    p.add_argument("--password"); p.add_argument("--full-name"); p.set_defaults(fn=cmd_add_user)
    p = sp.add_parser("passwd"); p.add_argument("name"); p.add_argument("--password"); p.set_defaults(fn=cmd_passwd)
    p = sp.add_parser("set-role"); p.add_argument("name"); p.add_argument("role", choices=["admin", "user"])
    p.set_defaults(fn=cmd_set_role)
    p = sp.add_parser("disable"); p.add_argument("name"); p.set_defaults(fn=lambda a: cmd_enable(a, False))
    p = sp.add_parser("enable"); p.add_argument("name"); p.set_defaults(fn=lambda a: cmd_enable(a, True))
    sp.add_parser("list-users").set_defaults(fn=cmd_list)
    p = sp.add_parser("register-model"); p.add_argument("path"); p.add_argument("--name")
    p.add_argument("--default", action="store_true"); p.set_defaults(fn=cmd_register_model)
    p = sp.add_parser("backup"); p.add_argument("--keep-days", type=int, default=14); p.set_defaults(fn=cmd_backup)
    sp.add_parser("check").set_defaults(fn=cmd_check)
    p = sp.add_parser("import-folder"); p.add_argument("dir")
    p.add_argument("--mode", choices=["inplace", "copy", "link", "move"], default="inplace")
    p.add_argument("--all", action="store_true", help="also ABFs without an event file (e.g. baseline)")
    p.add_argument("--recursive", action="store_true"); p.add_argument("--set-name", default="Event CSVs")
    p.add_argument("--user", default="admin"); p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_import_folder)
    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
