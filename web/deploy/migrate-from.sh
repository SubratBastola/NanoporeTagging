#!/usr/bin/env bash
# =============================================================================
# NanoTag: move a whole server here (users, experiments, annotations + history,
# NN models, clustering outputs, settings — and optionally display data and ABFs).
#
# Run on the NEW server, after deploy/install.sh, from the release folder:
#
#   sudo ./deploy/migrate-from.sh nanotag-site-....tar        import an export downloaded from
#                                                             Admin -> Export data and configuration
#   sudo ./deploy/migrate-from.sh you@oldserver [options]     fetch everything straight from the old
#                                                             server over SSH and import it
# Options:
#   --with-abf        also copy the ABF files (NanoTag's copies, ABFs used in place, pending uploads)
#   --no-zarr         don't copy display data; the new server re-ingests every recording from its ABF
#   --apply-config    also take the old server's worker / thread / display settings
#   --force           replace this server's data even if it already has experiments or users
#   --map OLD=NEW     rewrite an ABF path prefix (repeatable), e.g. --map /home/oguz=/data/oguz
#   --copy-inplace    put ABFs that were used in place into NanoTag's ABF store even if the same
#                     path exists here
#   --yes             don't ask for confirmation
#
# What happens: (SSH mode) the old server writes a site bundle (you type your sudo password there
# once); bundle, display data and ABFs are copied with rsync into a staging folder here (re-run to
# resume an interrupted copy); then this server's services are stopped, the data is installed
# (the current database is backed up first), paths are rewritten for this machine, and the services
# are started again. The old server is not changed. This server keeps its own secret key, port and
# firewall settings. Users sign in with their old passwords.
#
# The SSH user must be able to read the old server's data folder (installers add the installing
# user to the 'nanotag' group) and run sudo there. The old server must run NanoTag 0.1.11 or later.
# =============================================================================
set -euo pipefail
die()  { echo -e "\e[31mERROR:\e[0m $*" >&2; exit 1; }
info() { echo -e "\e[36m==>\e[0m $*"; }
warn() { echo -e "\e[33mWARNING:\e[0m $*"; }

[[ $EUID -eq 0 ]] || die "Run with sudo."
[[ -L /opt/nanotag/current ]] || die "NanoTag is not installed here (run deploy/install.sh first)."
[[ -f /etc/nanotag/nanotag.env ]] || die "/etc/nanotag/nanotag.env not found."

SRC=""; WITH_ABF=0; ZARR=1; APPLY_CFG=0; FORCE=0; YES=0; COPY_INPLACE=0; MAPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-abf) WITH_ABF=1; shift;;
    --no-zarr) ZARR=0; shift;;
    --apply-config) APPLY_CFG=1; shift;;
    --force) FORCE=1; shift;;
    --copy-inplace) COPY_INPLACE=1; shift;;
    --map) MAPS+=("$2"); shift 2;;
    --yes|-y) YES=1; shift;;
    -h|--help) sed -n '2,36p' "$0"; exit 0;;
    -*) die "unknown option $1";;
    *) [[ -z "$SRC" ]] || die "only one source"; SRC="$1"; shift;;
  esac
done
[[ -n "$SRC" ]] || die "give a bundle file or user@oldserver (see --help)"

set -a; . /etc/nanotag/nanotag.env; set +a
DATA="${NANOTAG_DATA:-/srv/nanotag}"
PORT="${NANOTAG_PORT:-3389}"
ENVF=/etc/nanotag/nanotag.env
TS="$(date +%Y%m%d-%H%M%S)"
RUN_AS="${SUDO_USER:-root}"          # SSH/rsync run as you (your keys, your access on the old server)
as_user() { if [[ "$RUN_AS" == root ]]; then "$@"; else sudo -u "$RUN_AS" -H "$@"; fi; }
admin() { /opt/nanotag/current/deploy/nanotag-admin "$@"; }    # runs as the nanotag user with its settings

# staging folder on the data disk (same filesystem, so files are moved, not copied twice).
# A previous, interrupted run's folder is reused so rsync can resume.
STAGE="$(ls -1d "$DATA"/import-staging-* 2>/dev/null | tail -1 || true)"
if [[ -z "$STAGE" ]]; then STAGE="$DATA/import-staging-$TS"; fi
mkdir -p "$STAGE"
chown "$RUN_AS" "$STAGE"
info "Staging folder: $STAGE"

if [[ -f "$SRC" ]]; then
  # ------------------------------------------------------------- bundle file
  info "Unpacking $(basename "$SRC") ($(du -h "$SRC" | cut -f1)) ..."
  tar -xf "$SRC" -C "$STAGE" --no-same-owner
else
  # ------------------------------------------------------------- straight from the old server
  REMOTE="$SRC"
  command -v rsync >/dev/null || apt-get install -y rsync >/dev/null
  info "Checking $REMOTE ..."
  as_user ssh -o BatchMode=no "$REMOTE" "grep -q export-site /opt/nanotag/current/nanotag/cli.py" \
    || die "$REMOTE does not run NanoTag 0.1.11+ (or $RUN_AS cannot read /opt/nanotag there). Update it first."
  if [[ ! -f "$STAGE/manifest.json" ]]; then
    info "Writing the site bundle on $REMOTE (sudo will ask for your password there) ..."
    as_user ssh -t "$REMOTE" "sudo nanotag-admin export-site" 2>&1 | tee "$STAGE/.export.log"
    RB="$(grep -a 'BUNDLE=' "$STAGE/.export.log" | tail -1 | cut -d= -f2- | tr -d '\r\n')"
    [[ -n "$RB" ]] || die "the export on $REMOTE failed (see above)"
    info "Copying the bundle ..."
    as_user rsync -a --info=progress2 "$REMOTE:$RB" "$STAGE/site.tar"
    tar -xf "$STAGE/site.tar" -C "$STAGE" --no-same-owner
    rm -f "$STAGE/site.tar"
    echo "$RB" > "$STAGE/.remote-bundle"
  else
    info "Bundle already fetched (resuming)."
  fi
  rpath() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['source'][sys.argv[2]])" "$STAGE/manifest.json" "$1"; }
  if [[ $ZARR -eq 1 ]]; then
    info "Copying display data (Zarr) — resumable ..."
    mkdir -p "$STAGE/zarr/store.zarr"; chown -R "$RUN_AS" "$STAGE/zarr"
    as_user rsync -a --info=progress2 "$REMOTE:$(rpath ZARR_ROOT)/" "$STAGE/zarr/store.zarr/" \
      || warn "copying display data failed; those recordings will be re-ingested from their ABFs"
  fi
  if [[ $WITH_ABF -eq 1 ]]; then
    info "Copying ABF files — resumable ..."
    mkdir -p "$STAGE/abf" "$STAGE/abf-inplace" "$STAGE/incoming"; chown "$RUN_AS" "$STAGE"/{abf,abf-inplace,incoming}
    as_user rsync -a --info=progress2 "$REMOTE:$(rpath ABF_DIR)/" "$STAGE/abf/"
    python3 -c "import json,sys; [print(r['path']) for r in json.load(open(sys.argv[1]))['inplace_abf']]" \
      "$STAGE/manifest.json" > "$STAGE/.inplace.txt"
    if [[ -s "$STAGE/.inplace.txt" ]]; then
      info "Copying $(wc -l < "$STAGE/.inplace.txt") ABF(s) that were used in place ..."
      as_user rsync -a -R --info=progress2 --files-from="$STAGE/.inplace.txt" "$REMOTE:/" "$STAGE/abf-inplace/" \
        || warn "some in-place ABFs could not be copied (they keep their old paths)"
    fi
    as_user rsync -a "$REMOTE:$(rpath INCOMING_DIR)/" "$STAGE/incoming/" || warn "pending uploads not copied"
  fi
fi
[[ -f "$STAGE/manifest.json" ]] || die "no manifest.json — not a NanoTag site bundle"

python3 - "$STAGE/manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
c = m.get("counts", {})
print(f"   from {m.get('host')} (NanoTag {m.get('nanotag_version')}, exported {m.get('created_iso')})")
print("   " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in c.items()))
EOF
if [[ $YES -eq 0 ]]; then
  read -r -p "This REPLACES the data on this server (a backup is made). Continue? [y/N] " a
  [[ "$a" == [yY]* ]] || die "stopped (the staging folder is kept; re-run to continue)"
fi

REMOTE_BUNDLE="$(cat "$STAGE/.remote-bundle" 2>/dev/null || true)"

# ------------------------------------------------------------- install
info "Stopping NanoTag services ..."
systemctl stop nanotag-web nanotag-worker
chown -R nanotag:nanotag "$STAGE"
ARGS=("$STAGE")
[[ $FORCE -eq 1 ]] && ARGS+=(--force)
[[ $COPY_INPLACE -eq 1 ]] && ARGS+=(--copy-inplace)
for m in "${MAPS[@]+"${MAPS[@]}"}"; do ARGS+=(--map "$m"); done
OK=1
admin import-site "${ARGS[@]}" | tee "$DATA/import-$TS.log" || OK=0

if [[ $OK -eq 1 && $APPLY_CFG -eq 1 ]]; then
  CF="$(ls -1t "$DATA"/imported-config-*.env 2>/dev/null | head -1 || true)"
  if [[ -n "$CF" ]]; then
    cp -a "$ENVF" "$ENVF.pre-import-$TS"
    for k in NANOTAG_WORKERS NANOTAG_TORCH_THREADS NANOTAG_WEB_WORKERS NANOTAG_DISPLAY_FS NANOTAG_CHUNK_SAMPLES NANOTAG_PYRAMID_FACTOR; do
      v="$(grep -E "^$k=" "$CF" | head -1 | cut -d= -f2- || true)"
      [[ -z "$v" ]] && continue
      if grep -qE "^$k=" "$ENVF"; then sed -i "s|^$k=.*|$k=$v|" "$ENVF"; else echo "$k=$v" >> "$ENVF"; fi
      info "setting $k=$v"
    done
  fi
fi

info "Starting NanoTag services ..."
systemctl start nanotag-web nanotag-worker
for i in $(seq 1 60); do curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || warn "NanoTag does not answer on port $PORT yet — journalctl -u nanotag-web"

if [[ $OK -eq 1 ]]; then
  echo
  info "Done. Users sign in with their old passwords at http://$(hostname -I | awk '{print $1}'):$PORT/"
  echo "   Log: $DATA/import-$TS.log"
  if [[ -n "$REMOTE_BUNDLE" ]]; then
    echo "   The bundle written on the old server can be deleted there: sudo rm $REMOTE_BUNDLE"
  fi
else
  die "the import failed (see above). Nothing was lost: the staging folder $STAGE is kept, fix the problem and re-run."
fi
