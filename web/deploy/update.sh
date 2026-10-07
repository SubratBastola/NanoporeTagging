#!/usr/bin/env bash
# =============================================================================
# NanoTag updater
#
#   sudo ./deploy/update.sh                 run from the NEW extracted release folder
#   sudo ./deploy/update.sh nanotag-X.tar.gz   or point it at a release tarball
#   sudo ./deploy/update.sh --rollback      switch back to the previous release
#
# Steps: backs up the database -> installs the new code as a new release folder ->
# installs/updates Python packages -> waits for running jobs (optional) -> switches
# /opt/nanotag/current -> migrates the database -> restarts services -> health check.
# If the health check fails it automatically rolls back to the previous release.
#
# Options:
#   --scripts-dir DIR   use NeuralNetwork.py / clustering.py from DIR
#                       (if omitted, custom scripts from the previous release are carried over)
#   --wait MIN          wait up to MIN minutes for running jobs to finish before restarting (default 0)
#   --keep N            keep the N most recent releases (default 5)
#   --yes               don't ask questions
# =============================================================================
set -euo pipefail
die()  { echo -e "\e[31mERROR:\e[0m $*" >&2; exit 1; }
info() { echo -e "\e[36m==>\e[0m $*"; }
warn() { echo -e "\e[33mWARNING:\e[0m $*"; }

[[ $EUID -eq 0 ]] || die "Run with sudo."
[[ -L /opt/nanotag/current ]] || die "NanoTag is not installed (run deploy/install.sh first)."

SRC=""; SCRIPTS_DIR=""; WAIT=0; KEEP=5; YES=0; ROLLBACK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scripts-dir) SCRIPTS_DIR="$(realpath "$2")"; shift 2;;
    --wait) WAIT="$2"; shift 2;;
    --keep) KEEP="$2"; shift 2;;
    --yes|-y) YES=1; shift;;
    --rollback) ROLLBACK=1; shift;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) SRC="$1"; shift;;
  esac
done

set -a; . /etc/nanotag/nanotag.env; set +a
PORT="${NANOTAG_PORT:-3389}"
PREV="$(readlink -f /opt/nanotag/current)"

health() {
  for i in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

switch_to() {
  ln -sfn "$1" /opt/nanotag/current.new && mv -Tf /opt/nanotag/current.new /opt/nanotag/current
  install -o root -g root -m 755 "$1/deploy/nanotag-admin" /usr/local/bin/nanotag-admin
  for u in nanotag-web.service nanotag-worker.service nanotag-backup.service nanotag-backup.timer; do
    install -o root -g root -m 644 "$1/deploy/systemd/$u" /etc/systemd/system/$u
  done
  systemctl daemon-reload
}

# ---------------------------------------------------------------- rollback
if [[ $ROLLBACK -eq 1 ]]; then
  OLD=$(ls -1dt /opt/nanotag/releases/*/ | sed 's#/$##' | grep -vx "$PREV" | head -1)
  [[ -n "$OLD" ]] || die "No previous release to roll back to."
  info "Rolling back: $PREV -> $OLD"
  switch_to "$OLD"
  systemctl restart nanotag-web nanotag-worker
  health && info "Rolled back to $(basename "$OLD")." || die "Service unhealthy after rollback — check journalctl -u nanotag-web"
  echo "Note: database schema changes are not reverted; restore a backup from ${NANOTAG_DATA}/backups if needed."
  exit 0
fi

# ---------------------------------------------------------------- locate new code
TMPX=""
if [[ -z "$SRC" ]]; then
  SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elif [[ -f "$SRC" ]]; then
  TMPX=$(mktemp -d); tar -xzf "$SRC" -C "$TMPX"
  SRC="$(dirname "$(find "$TMPX" -maxdepth 3 -name requirements.txt | head -1)")"
fi
trap 'if [[ -n "$TMPX" ]]; then rm -rf "$TMPX"; fi' EXIT
[[ -f "$SRC/nanotag/web.py" ]] || die "No NanoTag release found at $SRC"
NEWVER="$(cat "$SRC/VERSION")"; OLDVER="$(cat "$PREV/VERSION" 2>/dev/null || echo '?')"
[[ "$(readlink -f "$SRC")" != "$PREV" ]] || die "That is the release already running."
info "Updating NanoTag $OLDVER -> $NEWVER"

# ---------------------------------------------------------------- backup
info "Backing up the database ..."
nanotag-admin backup --keep-days 60

# ---------------------------------------------------------------- new release folder
REL="/opt/nanotag/releases/${NEWVER}-$(date +%Y%m%d%H%M%S)"
mkdir -p "$REL"
rsync -a --exclude '.venv*' --exclude '__pycache__' --exclude '*.pyc' \
  "$SRC/nanotag" "$SRC/vendor" "$SRC/tests" "$SRC/deploy" "$SRC/requirements.txt" "$SRC/requirements.lock" "$SRC/VERSION" "$REL/"
[[ -f "$SRC/README.md" ]] && cp "$SRC/README.md" "$REL/"
if [[ -n "$SCRIPTS_DIR" ]]; then
  for f in NeuralNetwork.py clustering.py; do cp "$SCRIPTS_DIR/$f" "$REL/vendor/$f"; done
  echo "$SCRIPTS_DIR" > "$REL/vendor/.custom"
  info "Using NeuralNetwork.py / clustering.py from $SCRIPTS_DIR"
elif [[ -f "$PREV/vendor/.custom" ]]; then
  for f in NeuralNetwork.py clustering.py; do cp "$PREV/vendor/$f" "$REL/vendor/$f"; done
  cp "$PREV/vendor/.custom" "$REL/vendor/.custom"
  info "Carried over your custom NeuralNetwork.py / clustering.py from the previous release"
fi
chown -R root:nanotag "$REL"; chmod -R g+rX,o-rwx "$REL"

# ---------------------------------------------------------------- packages
info "Updating Python packages ..."
REQ="$REL/requirements.txt"
[[ "$(/opt/nanotag/venv/bin/python -c 'import sys;print(sys.version_info[:2]==(3,12))')" == "True" && -f "$REL/requirements.lock" ]] && REQ="$REL/requirements.lock"
/opt/nanotag/venv/bin/pip install -q -r "$REQ"

# ---------------------------------------------------------------- running jobs
RUNNING=$(sqlite3 "${NANOTAG_DATA}/db/nanotag.sqlite3" "SELECT COUNT(*) FROM jobs WHERE status='running'" 2>/dev/null || echo 0)
if [[ "$RUNNING" -gt 0 ]]; then
  if [[ "$WAIT" -gt 0 ]]; then
    info "$RUNNING job(s) running; waiting up to $WAIT min ..."
    for i in $(seq 1 $((WAIT*6))); do
      RUNNING=$(sqlite3 "${NANOTAG_DATA}/db/nanotag.sqlite3" "SELECT COUNT(*) FROM jobs WHERE status='running'")
      [[ "$RUNNING" -eq 0 ]] && break
      sleep 10
    done
  fi
  if [[ "$RUNNING" -gt 0 ]]; then
    warn "$RUNNING job(s) still running. Restarting will interrupt them (ingest/import jobs are re-queued"
    warn "automatically; NN/clustering jobs must be re-submitted)."
    if [[ $YES -eq 0 ]]; then
      read -r -p "Continue? (y/N) " a </dev/tty; [[ "$a" =~ ^[Yy] ]] || die "Update aborted; nothing changed (new code left in $REL)."
    fi
  fi
fi

# ---------------------------------------------------------------- switch + migrate + restart
info "Switching to $(basename "$REL") ..."
systemctl stop nanotag-worker
switch_to "$REL"
nanotag-admin init
systemctl restart nanotag-web
systemctl start nanotag-worker
if health && nanotag-admin check >/dev/null; then
  info "NanoTag $NEWVER is up."
else
  warn "Health check failed — rolling back to $(basename "$PREV")"
  journalctl -u nanotag-web -n 40 --no-pager || true
  switch_to "$PREV"
  systemctl restart nanotag-web nanotag-worker
  health && die "Update failed; rolled back to $OLDVER (the database backup is in ${NANOTAG_DATA}/backups)." \
         || die "Update failed and rollback is unhealthy too — check journalctl -u nanotag-web"
fi

# ---------------------------------------------------------------- prune old releases
ls -1dt /opt/nanotag/releases/*/ | sed 's#/$##' | tail -n +$((KEEP+1)) | while read -r d; do
  [[ "$d" == "$(readlink -f /opt/nanotag/current)" ]] && continue
  rm -rf "$d"; info "Removed old release $(basename "$d")"
done
echo "Done. Roll back any time with:  sudo $REL/deploy/update.sh --rollback"
