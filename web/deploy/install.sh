#!/usr/bin/env bash
# =============================================================================
# NanoTag installer — Ubuntu 22.04 / 24.04
#
# Run from the extracted release folder:
#     sudo ./deploy/install.sh --allow-all            (any address that can reach the server)
#     sudo ./deploy/install.sh --allow 10.8.0.0/24    (or only specific networks)
#
# What it does (idempotent — safe to re-run):
#   * installs OS packages (python3-venv, sqlite3, ufw, acl ...)
#   * creates the 'nanotag' service account and group (adds you to the group)
#   * lays out  /opt/nanotag  (code + Python venv)  and  /srv/nanotag  (data)
#   * installs PyTorch (CPU build unless an NVIDIA GPU is found) + Python deps
#   * writes /etc/nanotag/nanotag.env (keeps an existing one)
#   * creates the database and the default admin account  (admin / admin2025!!)
#   * registers the bundled NN checkpoint best_opt_strict.pt
#   * installs systemd services: nanotag-web, nanotag-worker, nanotag-backup.timer
#   * opens the web port in ufw (to everyone with --allow-all, else only the --allow networks)
#   * runs a health check and prints the URL
#
# Options:
#   --allow-all          open the web port to every IP (the university firewall blocks outside traffic)
#   --allow CIDR         only allow this network (repeatable): VPN subnet, lab LAN ...
#   --port N             web port (default 8050)
#   --data DIR           data folder (default /srv/nanotag)
#   --import-root DIR    extra server folder users may import ABFs from (repeatable), e.g.
#                        /home/oguz/NanoporeTagging — read access is granted to the service via ACLs
#   --workers N          parallel background jobs (default 4)
#   --web-workers N      gunicorn processes (default 4)
#   --scripts-dir DIR    use NeuralNetwork.py / clustering.py from DIR instead of the bundled copies
#   --torch cpu|cuda|auto   PyTorch build (default auto)
#   --enable-ufw         enable ufw if it is currently inactive (SSH is always allowed first)
#   --no-firewall        don't touch the firewall at all
#   --yes                non-interactive (accept defaults)
# =============================================================================
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8050
DATA=/srv/nanotag
WORKERS=4
WEB_WORKERS=4
TORCH=auto
ALLOW=()
ALLOW_ALL=0
IMPORT_ROOTS=()
SCRIPTS_DIR=""
FIREWALL=auto
ENABLE_UFW=0
YES=0
DEFAULT_ADMIN_PW='admin2025!!'

die()  { echo -e "\e[31mERROR:\e[0m $*" >&2; exit 1; }
info() { echo -e "\e[36m==>\e[0m $*"; }
warn() { echo -e "\e[33mWARNING:\e[0m $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow) ALLOW+=("$2"); shift 2;;
    --allow-all) ALLOW_ALL=1; shift;;
    --port) PORT="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --import-root) IMPORT_ROOTS+=("$(realpath -m "$2")"); shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --web-workers) WEB_WORKERS="$2"; shift 2;;
    --scripts-dir) SCRIPTS_DIR="$(realpath "$2")"; shift 2;;
    --torch) TORCH="$2"; shift 2;;
    --enable-ufw) ENABLE_UFW=1; shift;;
    --no-firewall) FIREWALL=skip; shift;;
    --yes|-y) YES=1; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) die "Unknown option: $1 (see --help)";;
  esac
done

[[ $EUID -eq 0 ]] || die "Run with sudo:  sudo $0 $*"
[[ -f "$SRC/nanotag/web.py" && -f "$SRC/requirements.txt" ]] || die "Run this from the extracted NanoTag release folder."
. /etc/os-release || true
[[ "${ID:-}" == "ubuntu" ]] || warn "This script is tested on Ubuntu; detected '${PRETTY_NAME:-unknown}'."
VERSION="$(cat "$SRC/VERSION" 2>/dev/null || echo dev)"
INSTALLER_USER="${SUDO_USER:-}"
info "Installing NanoTag $VERSION from $SRC"

ask() {  # ask "question" default -> echoes answer
  local q="$1" d="${2:-}" a
  if [[ $YES -eq 1 ]]; then echo "$d"; return; fi
  read -r -p "$q ${d:+[$d] }" a </dev/tty || true
  echo "${a:-$d}"
}

# ---------------------------------------------------------------- 1. OS packages
info "Installing OS packages ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-dev build-essential sqlite3 ufw acl curl rsync iproute2 ca-certificates >/dev/null
PY=$(command -v python3.12 || command -v python3)
PYVER=$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
info "System Python: $PYVER"
[[ "$PYVER" == "3.11" || "$PYVER" == "3.12" || "$PYVER" == "3.13" ]] || warn "Tested with Python 3.12; found $PYVER."

# ---------------------------------------------------------------- 2. service account
if ! getent group nanotag >/dev/null; then groupadd --system nanotag; fi
if ! id nanotag >/dev/null 2>&1; then
  useradd --system --gid nanotag --home-dir "$DATA" --no-create-home --shell /usr/sbin/nologin nanotag
  info "Created service user 'nanotag'"
fi
if [[ -n "$INSTALLER_USER" && "$INSTALLER_USER" != "root" ]]; then
  usermod -aG nanotag "$INSTALLER_USER" && info "Added $INSTALLER_USER to group 'nanotag' (log out/in to use $DATA/incoming)"
fi

# ---------------------------------------------------------------- 3. folders
info "Creating folders under /opt/nanotag and $DATA ..."
install -d -o root -g root -m 755 /opt/nanotag /opt/nanotag/releases
install -d -o root -g nanotag -m 750 /etc/nanotag
install -d -o nanotag -g nanotag -m 750 "$DATA"
for d in db zarr abf uploads jobs models backups; do install -d -o nanotag -g nanotag -m 750 "$DATA/$d"; done
install -d -o nanotag -g nanotag -m 2775 "$DATA/incoming"   # group-writable drop folder (setgid)
setfacl -m g:nanotag:rx "$DATA" || true

# ---------------------------------------------------------------- 4. code release
REL="/opt/nanotag/releases/${VERSION}-$(date +%Y%m%d%H%M%S)"
info "Installing code to $REL"
mkdir -p "$REL"
rsync -a --delete --exclude '.venv*' --exclude '__pycache__' --exclude '*.pyc' \
  "$SRC/nanotag" "$SRC/vendor" "$SRC/tests" "$SRC/deploy" "$SRC/requirements.txt" "$SRC/requirements.lock" "$SRC/VERSION" "$REL/"
[[ -f "$SRC/README.md" ]] && cp "$SRC/README.md" "$REL/"
if [[ -n "$SCRIPTS_DIR" ]]; then
  for f in NeuralNetwork.py clustering.py; do
    [[ -f "$SCRIPTS_DIR/$f" ]] || die "$SCRIPTS_DIR/$f not found"
    cp "$SCRIPTS_DIR/$f" "$REL/vendor/$f"
  done
  echo "$SCRIPTS_DIR" > "$REL/vendor/.custom"
  info "Using NeuralNetwork.py / clustering.py from $SCRIPTS_DIR"
fi
chown -R root:nanotag "$REL"; chmod -R g+rX,o-rwx "$REL"
ln -sfn "$REL" /opt/nanotag/current.new && mv -Tf /opt/nanotag/current.new /opt/nanotag/current

# ---------------------------------------------------------------- 5. Python environment
if [[ ! -x /opt/nanotag/venv/bin/python ]]; then
  info "Creating Python virtual environment ..."
  $PY -m venv /opt/nanotag/venv
fi
VPIP=/opt/nanotag/venv/bin/pip
$VPIP install -q --upgrade pip wheel
if ! /opt/nanotag/venv/bin/python -c "import torch" 2>/dev/null; then
  if [[ "$TORCH" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then TORCH=cuda; else TORCH=cpu; fi
  fi
  info "Installing PyTorch ($TORCH build) — this can take a few minutes ..."
  if [[ "$TORCH" == "cpu" ]]; then
    $VPIP install -q torch --index-url https://download.pytorch.org/whl/cpu \
      || { warn "CPU wheel index unreachable; installing torch from PyPI instead"; $VPIP install -q torch; }
  else
    $VPIP install -q torch
  fi
fi
info "Installing Python packages ..."
REQ="$REL/requirements.txt"
[[ "$(/opt/nanotag/venv/bin/python -c 'import sys;print(sys.version_info[:2]==(3,12))')" == "True" ]] && REQ="$REL/requirements.lock"
$VPIP install -q -r "$REQ"
chown -R root:root /opt/nanotag/venv; chmod -R a+rX /opt/nanotag/venv

# ---------------------------------------------------------------- 6. configuration
ENVF=/etc/nanotag/nanotag.env
ROOTS="$DATA/incoming"
for r in "${IMPORT_ROOTS[@]}"; do ROOTS="$ROOTS:$r"; done
if [[ -f "$ENVF" ]]; then
  info "Keeping existing $ENVF (secret key preserved); updating port/workers/import folders"
  sed -i "s|^NANOTAG_PORT=.*|NANOTAG_PORT=$PORT|; s|^NANOTAG_WORKERS=.*|NANOTAG_WORKERS=$WORKERS|; s|^NANOTAG_WEB_WORKERS=.*|NANOTAG_WEB_WORKERS=$WEB_WORKERS|" "$ENVF"
  if [[ ${#IMPORT_ROOTS[@]} -gt 0 ]]; then sed -i "s|^NANOTAG_IMPORT_ROOTS=.*|NANOTAG_IMPORT_ROOTS=$ROOTS|" "$ENVF"; fi
else
  SECRET=$($PY -c 'import secrets; print(secrets.token_hex(32))')
  cat > "$ENVF" <<EOF
# NanoTag configuration — edit, then:  sudo systemctl restart nanotag-web nanotag-worker
NANOTAG_DATA=$DATA
NANOTAG_SECRET_KEY=$SECRET
NANOTAG_BIND=0.0.0.0
NANOTAG_PORT=$PORT
NANOTAG_WEB_WORKERS=$WEB_WORKERS
# parallel background jobs (an ingest of a 1.8 GB / 5-min ABF peaks at ~15 GB RAM and takes ~1 min)
NANOTAG_WORKERS=$WORKERS
NANOTAG_TORCH_THREADS=4
# colon-separated folders users may import ABFs from (the service needs read access)
NANOTAG_IMPORT_ROOTS=$ROOTS
NANOTAG_VENDOR=/opt/nanotag/current/vendor
MPLBACKEND=Agg
EOF
  info "Wrote $ENVF"
fi
chown root:nanotag "$ENVF"; chmod 640 "$ENVF"

# read access for extra import folders (e.g. inside a home directory)
for r in "${IMPORT_ROOTS[@]}"; do
  if [[ -d "$r" ]]; then
    info "Granting the service read access to $r (ACL)"
    p="$r"
    while [[ "$p" != "/" ]]; do p="$(dirname "$p")"; setfacl -m u:nanotag:x "$p" 2>/dev/null || true; done
    setfacl -R -m u:nanotag:rX "$r" && setfacl -R -d -m u:nanotag:rX "$r" || warn "setfacl failed on $r"
  else
    warn "Import folder $r does not exist (yet)."
  fi
done

# ---------------------------------------------------------------- 7. admin CLI + database
install -o root -g root -m 755 "$REL/deploy/nanotag-admin" /usr/local/bin/nanotag-admin
info "Initialising database ..."
nanotag-admin init
nanotag-admin ensure-admin --password "$DEFAULT_ADMIN_PW"
nanotag-admin register-model "$REL/vendor/best_opt_strict.pt" --name best_opt_strict || warn "model registration failed"

# ---------------------------------------------------------------- 8. systemd
info "Installing systemd services ..."
for u in nanotag-web.service nanotag-worker.service nanotag-backup.service nanotag-backup.timer; do
  install -o root -g root -m 644 "$REL/deploy/systemd/$u" /etc/systemd/system/$u
done
systemctl daemon-reload
systemctl enable -q nanotag-web nanotag-worker nanotag-backup.timer
systemctl restart nanotag-web nanotag-worker
systemctl start nanotag-backup.timer

# ---------------------------------------------------------------- 9. firewall
if [[ "$FIREWALL" != "skip" ]]; then
  info "Firewall (ufw) ..."
  if [[ ${#ALLOW[@]} -eq 0 && $ALLOW_ALL -eq 0 ]]; then
    SSHSRC="${SSH_CLIENT%% *}"
    echo "Which networks may reach NanoTag on port $PORT? Give your VPN subnet (ask IT, e.g. 10.8.0.0/24)"
    echo "and optionally the lab LAN, separated by spaces.${SSHSRC:+ (You are connected from $SSHSRC.)}"
    ip -4 -o addr show scope global 2>/dev/null | awk '{print "   this server: " $2 " " $4}' || true
    ans=$(ask "Allowed networks (blank = any address that can reach this machine):" "")
    read -r -a ALLOW <<< "$ans"
  fi
  # remove previous NanoTag rules so re-runs don't accumulate
  while ufw status numbered 2>/dev/null | grep -q "nanotag"; do
    n=$(ufw status numbered | grep "nanotag" | head -1 | sed -E 's/^\[ *([0-9]+)\].*/\1/')
    yes | ufw delete "$n" >/dev/null
  done
  if [[ $ALLOW_ALL -eq 1 || ${#ALLOW[@]} -eq 0 ]]; then
    ufw allow "$PORT/tcp" comment 'nanotag' >/dev/null
    info "  port $PORT open to all addresses (outside traffic is blocked by the university firewall)"
  else
    for c in "${ALLOW[@]}"; do ufw allow from "$c" to any port "$PORT" proto tcp comment 'nanotag' >/dev/null; info "  allowed $c -> port $PORT"; done
  fi
  if ufw status | grep -q "Status: active"; then
    info "ufw is active; NanoTag rules added."
  elif [[ $ALLOW_ALL -eq 1 && $ENABLE_UFW -eq 0 ]]; then
    info "ufw is inactive, so port $PORT is already reachable; the rule is stored in case ufw is enabled later."
  else
    echo
    warn "ufw is currently INACTIVE, so the rules above are not enforced yet."
    echo "Enabling ufw blocks every other incoming port except SSH. Services currently listening:"
    { ss -tlnH 2>/dev/null | awk '{print "   " $4}' | sort -u | head -30; } || true
    doit=$ENABLE_UFW
    if [[ $doit -eq 0 && $YES -eq 0 ]]; then
      [[ "$(ask 'Enable ufw now? (y/N)' N)" =~ ^[Yy] ]] && doit=1
    fi
    if [[ $doit -eq 1 ]]; then
      ufw allow OpenSSH >/dev/null
      ufw --force enable >/dev/null
      info "ufw enabled (SSH + NanoTag rules)."
    else
      echo "   Left ufw inactive. To enforce later:  sudo ufw allow OpenSSH && sudo ufw enable"
    fi
  fi
fi

# ---------------------------------------------------------------- 10. health check
info "Waiting for the web service ..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null || { journalctl -u nanotag-web -n 50 --no-pager; die "Web service did not start (see log above)."; }
nanotag-admin check || warn "Self-check reported problems (see above)."

IP=$( { ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1; } || true)
IP=${IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
cat <<EOF

$(echo -e "\e[32m")NanoTag $VERSION is running.$(echo -e "\e[0m")

  URL (campus or VPN): http://$IP:$PORT
  First login:          admin / $DEFAULT_ADMIN_PW      <-- change it: top-right user menu > Change password
  Add users:            web: Admin page    or    sudo nanotag-admin add-user <name>
  Reset a password:     sudo nanotag-admin passwd <name>
  Drop big ABFs here:   $DATA/incoming   then use "Import from a server folder" in the experiment page
  Logs:                 journalctl -u nanotag-web -f      journalctl -u nanotag-worker -f
  Config:               $ENVF
  Update later:         extract the new release, then  sudo ./deploy/update.sh
EOF
