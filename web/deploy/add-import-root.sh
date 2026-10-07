#!/usr/bin/env bash
# =============================================================================
# Let NanoTag read (and import from) a folder on this server, e.g. your data folder:
#
#     sudo /opt/nanotag/current/deploy/add-import-root.sh /home/oguz/NanoporeTagging
#     sudo /opt/nanotag/current/deploy/add-import-root.sh --list
#     sudo /opt/nanotag/current/deploy/add-import-root.sh --remove /home/oguz/NanoporeTagging
#
# What it does:
#   * gives the 'nanotag' service read access with ACLs (the folder, everything in it, and files
#     added later); parent folders only get "pass-through" (x) access, so nothing else in your home
#     directory becomes readable
#   * adds the folder to NANOTAG_IMPORT_ROOTS in /etc/nanotag/nanotag.env
#   * restarts NanoTag (signed-in users stay signed in) and checks the service can list the folder
#
# Sub-folders are included, so allowing /home/oguz/NanoporeTagging also allows
# /home/oguz/NanoporeTagging/SiO2_Tagging. The "move" import mode also needs write access, which
# this script does not grant; "use in place" and "copy" only need read access.
# =============================================================================
set -euo pipefail
die()  { echo -e "\e[31mERROR:\e[0m $*" >&2; exit 1; }
info() { echo -e "\e[36m==>\e[0m $*"; }
warn() { echo -e "\e[33mWARNING:\e[0m $*"; }

[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"
ENVF=/etc/nanotag/nanotag.env
[[ -f "$ENVF" ]] || die "NanoTag is not installed ($ENVF missing)."
id nanotag >/dev/null 2>&1 || die "service user 'nanotag' not found."
command -v setfacl >/dev/null || { apt-get install -y -qq acl >/dev/null || die "install the 'acl' package"; }

current_roots() { grep -E '^NANOTAG_IMPORT_ROOTS=' "$ENVF" | head -1 | cut -d= -f2- ; }
set_roots() {
  if grep -qE '^NANOTAG_IMPORT_ROOTS=' "$ENVF"; then
    sed -i "s|^NANOTAG_IMPORT_ROOTS=.*|NANOTAG_IMPORT_ROOTS=$1|" "$ENVF"
  else
    echo "NANOTAG_IMPORT_ROOTS=$1" >> "$ENVF"
  fi
}

ADD=(); REMOVE=(); RESTART=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) echo "Folders NanoTag may read:"; current_roots | tr ':' '\n' | sed 's/^/  /'; exit 0;;
    --remove) REMOVE+=("$(realpath -m "$2")"); shift 2;;
    --no-restart) RESTART=0; shift;;
    -h|--help) sed -n '2,22p' "$0"; exit 0;;
    -*) die "unknown option $1";;
    *) ADD+=("$(realpath -m "$1")"); shift;;
  esac
done
[[ ${#ADD[@]} -gt 0 || ${#REMOVE[@]} -gt 0 ]] || die "usage: $0 /path/to/folder   (or --list, --remove DIR)"

ROOTS="$(current_roots)"
for r in "${ADD[@]}"; do
  [[ -d "$r" ]] || die "$r is not a folder."
  [[ "$r" != "/" ]] || die "refusing to allow the whole disk."
  info "Granting the NanoTag service read access to $r"
  p="$r"
  while [[ "$p" != "/" ]]; do p="$(dirname "$p")"; setfacl -m u:nanotag:x "$p" 2>/dev/null || true; done
  setfacl -R -m u:nanotag:rX "$r"
  find "$r" -type d -exec setfacl -d -m u:nanotag:rX {} + 2>/dev/null || true
  if [[ ":$ROOTS:" != *":$r:"* ]]; then ROOTS="${ROOTS:+$ROOTS:}$r"; fi
  if runuser -u nanotag -- ls "$r" >/dev/null 2>&1; then
    n=$(runuser -u nanotag -- find "$r" -maxdepth 2 -iname '*.abf' 2>/dev/null | wc -l)
    info "  ok — the service can read it ($n .abf file(s) within two levels)"
  else
    warn "  the service still cannot list $r (network or FUSE mount without ACL support?)"
  fi
done
for r in "${REMOVE[@]}"; do
  ROOTS="$(echo ":$ROOTS:" | sed "s|:$r:|:|g; s|^:||; s|:$||")"
  setfacl -R -x u:nanotag "$r" 2>/dev/null || true
  setfacl -R -d -x u:nanotag "$r" 2>/dev/null || true
  info "Removed $r"
done
set_roots "$ROOTS"
info "NANOTAG_IMPORT_ROOTS=$ROOTS"
if [[ $RESTART -eq 1 ]]; then
  systemctl restart nanotag-web nanotag-worker
  info "NanoTag restarted. In the web page: 📂 Import folder -> type the folder path -> Scan."
fi
