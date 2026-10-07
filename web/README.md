# NanoTag — NanoporeTagging on a lab server

NanoTag serves the NanoporeTagging workflow (Tagger_GUI → NeuralNetwork → clustering) from one lab
machine to any browser on the VPN. Every experiment holds 12–16 ABF recordings. Several people tag
the same annotation sets, and every edit records who made it. The NN and clustering run as
background jobs, and clustering produces the same outputs and HTML report as `clustering.py`.

```
 browser (VPN) ──HTTP :3389──▶ nanotag-web (gunicorn: Flask pages + Dash Tagger)
                                   │  reads only the visible time range
                                   ▼
                      /srv/nanotag/zarr   Tagger-filtered 10 kHz signal + min/max zoom levels (~2 % of ABF size)
                      /srv/nanotag/abf    original ABFs (the only raw copy)
                      /srv/nanotag/db     SQLite: users, experiments, annotation sets, events, edit history,
                                          jobs, clustering runs
                                   ▲
                      nanotag-worker ─── ingest · NN landmarks · clustering  (original .py code, unchanged)
```

---

## 1. Deploy

Tested on Ubuntu 24.04 (the ANIServer kernel 6.8 is 24.04). The server needs internet access during
install for apt and pip.

**Network model.** The web port is **3389** by default and is open to **all IP addresses**. Over the
VPN only SSH (22) and RDP (3389) reach the lab machine and IT will not open other ports, so NanoTag
takes over the RDP port and SSH stays for administration. That is safe here because the university
firewall already blocks outside traffic, so only people on campus or on the VPN can reach the server,
and everyone still has to log in. Nothing needs to be opened on the university firewall.

**Turn RDP off first.** The installer stops (before installing anything) if another program holds
port 3389, and prints the commands. On ANIServer the RDP server is GNOME's remote desktop
(Wayland), so, from an **SSH** session (this ends any RDP session):

```bash
sudo grdctl --system rdp disable
sudo systemctl disable --now gnome-remote-desktop
```

If remote desktop was turned on as per-user Desktop Sharing instead, switch it off under
Settings → System → Remote Desktop. To keep RDP, install on another port with `--port N`.

**Moving an existing 8050 install to 3389:** turn RDP off as above, then rerun
`sudo ./deploy/install.sh --allow-all` from the release folder. It is idempotent: it keeps the
database, users and secret key, rewrites `NANOTAG_PORT` in `/etc/nanotag/nanotag.env` and replaces
the old ufw rule. (`update.sh` alone keeps whatever port the env file already has.)

```bash
# on your laptop
scp nanotag-0.1.0.tar.gz oguz@ANIServer:~

# on ANIServer
tar xzf nanotag-0.1.0.tar.gz
cd nanotag-0.1.0
sudo ./deploy/install.sh --allow-all --import-root /home/oguz/NanoporeTagging
```

`--import-root` is optional; it lets users import ABFs straight from that folder. To restrict access
to particular networks later, rerun the installer with `--allow <CIDR>` (repeatable) instead of
`--allow-all`; it replaces the previous NanoTag rule.

The script is idempotent, so running it again is safe. It:

- installs the OS packages it needs;
- creates the `nanotag` service account and adds you to its group;
- installs the code in `/opt/nanotag` and the data folders in `/srv/nanotag`;
- installs PyTorch (a CUDA build if an NVIDIA GPU is present, otherwise CPU) and the Python packages;
- writes `/etc/nanotag/nanotag.env` with a random secret key;
- creates the database and the **admin / admin2025!!** account;
- registers `best_opt_strict.pt` as the default model;
- installs the `nanotag-web`, `nanotag-worker` and nightly backup systemd services;
- opens port 3389 in ufw to all addresses (`--allow-all`) or only to the `--allow` networks;
- runs a health check.

Firewall behaviour:

- If `ufw` is already active, the port-3389 rule is added and nothing else changes.
- If `ufw` is inactive (the Ubuntu default), the port is already reachable. The rule is saved for
  later, and the script does not turn ufw on unless you add `--enable-ufw` (SSH is always allowed
  first). Use `--no-firewall` to leave the firewall untouched.

Then:

1. Open `http://<ANIServer-IP>:3389` from any computer on campus or on the VPN.
2. Log in as **admin / admin2025!!**, then change the password under the user name at the top right
   (a banner reminds you until you do).
3. Add people on the **Admin** page, or with `sudo nanotag-admin add-user alice`.
4. Optionally, run the acceptance test. It writes two synthetic 6-channel ABFs, runs ingest → Tagger →
   NN → clustering → exports, then cleans up (about 30 seconds):

   ```bash
   sudo -u nanotag /opt/nanotag/venv/bin/python /opt/nanotag/current/tests/acceptance_test.py --password '<admin password>'
   ```

Traffic is plain HTTP. Off campus it travels inside the encrypted VPN tunnel. On the campus network
itself it is unencrypted, so passwords cross the LAN in the clear. If that matters to you, put Caddy
or nginx in front with a certificate for HTTPS.

---

## 2. Everyday use

**Experiments → New experiment**, then open it. The page walks through four steps.

### Step 1: Recordings

There are two ways to add ABFs:

- **Upload** in the browser. Files go up in 16 MB chunks. If the VPN drops, pick the same file again
  and the upload resumes where it stopped.
- **Import from a server folder.** This is fastest for 1.8 GB files. Copy them onto the server first,
  for example:
  `rsync -P *.abf oguz@ANIServer:/srv/nanotag/incoming/`
  (log out and back in once after install so the `nanotag` group applies). Then tick the files in the
  folder browser and choose copy, hard-link or move.

Each file is ingested in the background, taking about 1 minute and about 15 GB of RAM per 5-minute
file; 4 files run in parallel by default.

Channel roles default to **electrical = 0 (Ipatch), optical = 2 (Optical), opt-ref = 3 (OpticalRe)**.
You can change them per recording or for all recordings at once. The NN uses them too.

### Step 2: Annotation sets

An annotation set is a named collection of events across all recordings of the experiment.

- You can import the legacy CSV/XLSX files: Tagger event files, window lists, or NN `__predicted.xlsx`.
  Rows are matched to recordings by `file_name`, and unmatched rows are reported.
- You can create empty sets or copies of existing sets.
- **Tagger** has the same five modes and O1–O3 / R1–R3 / E1–E4 labels as the desktop tool.
  - Every channel can be shown (checkboxes).
  - Wide views are min/max envelopes that preserve spikes; zoomed-in views show the exact 10 kHz
    samples. Zoom in before placing landmarks precisely.
  - Edits save immediately.
  - Other people's edits appear within about 10 seconds.
  - If two people edit the same event, the second save is refused with "changed by X".
  - "Color markers by author" and the author filter show who tagged what.
- **Exports** available for each set:
  - legacy CSV/XLSX (identical columns to Tagger_GUI);
  - CSV with author columns;
  - one CSV per recording (ZIP);
  - NN `__predicted.xlsx` layout;
  - full edit history (who / when / before / after).

### Step 3: NN

Pick the set that holds the windows and a model. This runs NeuralNetwork.py's own
`build_npz_for_row` and `run_inference_on_npz` on every window, reading each ABF once. The result is
a new annotation set that you review in the Tagger. Admins can upload other `.pt` checkpoints on the
**Models** page.

### Step 4: Clustering

Pick one or more sets (sets from other experiments can be pooled too) and the same parameters as the
desktop app: α, Max K, bootstraps, duration limits, minimum cluster %, NA imputation and features.
This runs clustering.py's engine.

Outputs:

- the interactive **HTML report**;
- a ZIP with every file the desktop "Save Results" writes;
- a labels CSV that maps each event to its cluster.

### Bundles

**Experiment bundle** exports an experiment with its annotation sets and edit history, optionally
including the ABFs. **Import bundle** on the Experiments page recreates it on any NanoTag server.

### Permissions

All users can see and edit everything. An admin can:

- **lock** an annotation set, making it read-only for non-admins;
- **restrict** an experiment to named users.

---

## 3. Administration

| Task | Command |
|---|---|
| Add user / admin | `sudo nanotag-admin add-user alice` · `sudo nanotag-admin add-user bob --admin` |
| Reset password | `sudo nanotag-admin passwd alice` (or on the web Admin page) |
| Promote / demote | `sudo nanotag-admin set-role alice admin` |
| Disable / enable | `sudo nanotag-admin disable alice` |
| List users | `sudo nanotag-admin list-users` |
| Register a model | `sudo nanotag-admin register-model /path/model.pt --default` |
| Backup now | `sudo nanotag-admin backup` (runs nightly at 02:30; kept 30 days in `/srv/nanotag/backups`) |
| Self-check | `sudo nanotag-admin check` |
| Logs | `journalctl -u nanotag-web -f` · `journalctl -u nanotag-worker -f` |
| Restart | `sudo systemctl restart nanotag-web nanotag-worker` |
| Settings | `/etc/nanotag/nanotag.env` (port, parallel jobs `NANOTAG_WORKERS`, import folders), then restart |

**Disk.** Each experiment needs about 1.02× its ABF size: 16 × 1.8 GB ABF plus about 0.6 GB of
Zarr. With 3.3 TB free that is roughly 100 experiments. The Admin page shows the free space.

**What to back up:**

- `/srv/nanotag/db`, the annotations (nightly copies are kept in `/srv/nanotag/backups`);
- `/srv/nanotag/abf`, the raw data.

The Zarr store can always be rebuilt with **Re-ingest**.

---

## 4. Updates

```bash
tar xzf nanotag-0.2.0.tar.gz && cd nanotag-0.2.0
sudo ./deploy/update.sh                 # or: sudo ./deploy/update.sh ~/nanotag-0.2.0.tar.gz
sudo ./deploy/update.sh --wait 30       # first wait up to 30 min for running jobs to finish
sudo /opt/nanotag/current/deploy/update.sh --rollback   # go back to the previous release
```

The updater does the following:

1. Backs up the database.
2. Installs the new code as a new release folder and updates the Python packages.
3. Switches `/opt/nanotag/current` and migrates the database.
4. Restarts the services and health-checks them, rolling back automatically if that fails.
5. Keeps the last 5 releases.

**Using your own versions of the scripts.** The server runs *unmodified* copies of `NeuralNetwork.py`
and `clustering.py` from `vendor/`. To use your own edited versions:

```bash
sudo ./deploy/update.sh --scripts-dir ~/NanoporeTagging
```

They are remembered and carried over in later updates.

---

## 5. Known issues / recommendations parked for later

The filter logic is deliberately **unchanged** from the originals. These are the points noted during
the design review, kept here for a later decision.

1. **Tagger display filter delay.** `Tagger_GUI.py` shows a *causal* 8-pole Bessel (`sosfilt`) at
   100 Hz, which delays edges by roughly 10 ms. The NN path uses zero-phase `filtfilt`. Hand-placed
   and NN landmarks are therefore measured on differently delayed signals. A zero-phase display
   filter would remove the offset; switching is a one-line change in `nanotag/ingest.py` plus a
   re-ingest.
2. `NeuralNetwork.py` names its window filter `BESSEL_ORDER`, but `design_sos` builds a
   **Butterworth** filter.
3. **Sign conventions differ between the scripts:**
   - OSC is signed in the Tagger export but absolute in the NN output.
   - Spikes are signed in both, but clustering takes |peak − base|.

   NanoTag keeps each set's own convention: Tagger formulas for manual/imported sets, NN formulas for
   NN sets, and values from the source file when a file provides them.
4. In the web Tagger, amplitudes at clicked or dragged landmarks are read from the stored 10 kHz
   filtered signal. The desktop tool interpolated the full-rate filtered signal. The difference is
   negligible at a 100 Hz bandwidth.
5. Not yet in the web version: the clustering GUI's interactive Point Inspector (manual exclusions),
   folder comparison and custom cluster names. The HTML report's interactive 3D view and "Explore
   other K" tab are included.

## Uninstall

```bash
sudo systemctl disable --now nanotag-web nanotag-worker nanotag-backup.timer
sudo rm -f /etc/systemd/system/nanotag-* /usr/local/bin/nanotag-admin
sudo rm -rf /opt/nanotag /etc/nanotag           # code + config
# data (ABFs, annotations): /srv/nanotag — delete only if you are sure
```
