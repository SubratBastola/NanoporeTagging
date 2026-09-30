"""Users, passwords and permissions."""
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .db import db, now, one, rows, tx

MIN_PASSWORD_LEN = 8


class User(UserMixin):
    def __init__(self, d):
        self.id = d["id"]
        self.username = d["username"]
        self.full_name = d.get("full_name") or ""
        self.role = d["role"]
        self.active_flag = bool(d["active"])

    @property
    def is_admin(self):
        return self.role == "admin"

    @property
    def is_active(self):
        return self.active_flag

    def get_id(self):
        return str(self.id)


def hash_pw(pw):
    return generate_password_hash(pw, method="scrypt")


def check_pw_policy(pw):
    if not pw or len(pw) < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")


def get_user_by_id(uid):
    with db() as con:
        d = one(con, "SELECT * FROM users WHERE id=?", (int(uid),))
    return User(d) if d else None


def get_user_row(username):
    with db() as con:
        return one(con, "SELECT * FROM users WHERE username=?", (username,))


def authenticate(username, password):
    d = get_user_row(username)
    if not d or not d["active"]:
        return None
    if not check_password_hash(d["pw_hash"], password or ""):
        return None
    with db() as con:
        con.execute("UPDATE users SET last_login=? WHERE id=?", (now(), d["id"]))
    return User(d)


def list_users():
    with db() as con:
        return rows(con, "SELECT id, username, full_name, role, active, created_at, last_login "
                         "FROM users ORDER BY username")


def add_user(username, password, role="user", full_name="", enforce_policy=True):
    username = (username or "").strip()
    if not username or any(c.isspace() for c in username):
        raise ValueError("Username must be non-empty and contain no spaces.")
    if role not in ("admin", "user"):
        raise ValueError("Role must be 'admin' or 'user'.")
    if enforce_policy:
        check_pw_policy(password)
    with db() as con, tx(con):
        if one(con, "SELECT id FROM users WHERE username=?", (username,)):
            raise ValueError(f"User '{username}' already exists.")
        con.execute("INSERT INTO users(username, full_name, pw_hash, role, active, created_at) "
                    "VALUES (?,?,?,?,1,?)", (username, full_name, hash_pw(password), role, now()))


def set_password(username, password, enforce_policy=True):
    if enforce_policy:
        check_pw_policy(password)
    with db() as con, tx(con):
        cur = con.execute("UPDATE users SET pw_hash=? WHERE username=?", (hash_pw(password), username))
        if cur.rowcount == 0:
            raise ValueError(f"No such user '{username}'.")


def set_role(username, role):
    if role not in ("admin", "user"):
        raise ValueError("Role must be 'admin' or 'user'.")
    with db() as con, tx(con):
        if role == "user":
            _guard_last_admin(con, username)
        cur = con.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        if cur.rowcount == 0:
            raise ValueError(f"No such user '{username}'.")


def set_active(username, active):
    with db() as con, tx(con):
        if not active:
            _guard_last_admin(con, username)
        cur = con.execute("UPDATE users SET active=? WHERE username=?", (1 if active else 0, username))
        if cur.rowcount == 0:
            raise ValueError(f"No such user '{username}'.")


def _guard_last_admin(con, username):
    d = one(con, "SELECT role, active FROM users WHERE username=?", (username,))
    if d and d["role"] == "admin" and d["active"]:
        n = one(con, "SELECT COUNT(*) AS n FROM users WHERE role='admin' AND active=1")["n"]
        if n <= 1:
            raise ValueError("Refusing to remove the last active admin.")


def default_admin_password_in_use():
    d = get_user_row("admin")
    return bool(d and d["active"] and check_password_hash(d["pw_hash"], "admin2025!!"))


# ----------------------------------------------------------------- permissions
def can_view_experiment(user, exp):
    if exp is None:
        return False
    if user.is_admin or not exp.get("restricted"):
        return True
    with db() as con:
        return one(con, "SELECT 1 AS ok FROM experiment_access WHERE experiment_id=? AND username=?",
                   (exp["id"], user.username)) is not None


def can_edit_set(user, aset):
    """All users may edit any set they can see, unless an admin locked it."""
    if aset is None:
        return False
    return user.is_admin or not aset.get("locked")


def visible_experiments(user):
    with db() as con:
        if user.is_admin:
            return rows(con, "SELECT * FROM experiments ORDER BY created_at DESC")
        return rows(con, """SELECT e.* FROM experiments e
                             WHERE e.restricted=0
                                OR EXISTS (SELECT 1 FROM experiment_access a
                                           WHERE a.experiment_id=e.id AND a.username=?)
                             ORDER BY e.created_at DESC""", (user.username,))
