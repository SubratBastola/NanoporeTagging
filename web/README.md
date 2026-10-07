# NanoTag — NanoporeTagging on a lab server

NanoTag serves the NanoporeTagging workflow (Tagger_GUI → NeuralNetwork → clustering) from one lab
machine to any browser on the VPN. Every experiment holds 12–16 ABF recordings. Several people tag
the same annotation sets, and every edit records who made it. The NN and clustering run as
background jobs, and clustering produces the same outputs and HTML report as `clustering.py`.

```
 browser (VPN) ──HTTP :8050──▶ nanotag-web (gunicorn: Flask pages + Dash Tagger)
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

**Network model.** The web port (8050) is open to **all IP addresses**. That is safe here because
the university firewall already blocks outside traffic, so only people on campus or on the VPN can
reach the server, and everyone still has to log in. Nothing needs to be opened on the university
firewall.

```bash
# on your laptop
scp nanotag-0.1.9.tar.gz oguz@ANIServer:~

# on ANIServer
tar xzf nanotag-0.1.9.tar.gz
cd nanotag-0.1.9
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
- opens port 8050 in ufw to all addresses (`--allow-all`) or only to the `--allow` networks;
- runs a health check.

Firewall behaviour:

- If `ufw` is already active, the port-8050 rule is added and nothing else changes.
- If `ufw` is inactive (the Ubuntu default), the port is already reachable. The rule is saved for
  later, and the script does not turn ufw on unless you add `--enable-ufw` (SSH is always allowed
  first). Use `--no-firewall` to leave the firewall untouched.

Then:

1. Open `http://<ANIServer-IP>:8050` from any computer on campus or on the VPN.
2. Log in as **admin / admin2025!!**, then change the password with **🔑 Change password** in the
   upper-left menu (a banner reminds you until you do).
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

Everything starts from the **menu in the upper-left corner** of every page, including the Tagger:
🧪 Experiments · 📂 Import folder · 📁 Files · ⏱ Jobs · 🧠 Models · ⚙ Admin · users (admins only) · ❓ Help ·
🔑 Change password · 🚪 Log out. Below it is a list of **tools for the page you are on**. On an experiment
page, for example: Add ABF files, Import event CSV, Open Tagger, Run neural network, Run clustering,
Export, Jobs, Settings.

- **Experiments (home):** drop ABF files straight onto the page. The experiment is **worked out from the
  file name** and created if needed (see below). There are also one-click buttons on each experiment row.
- **Experiment pages** have tabs:
  **Recordings · Annotations · Neural network · Clustering · Export · Jobs · Settings & access**.
- **📁 Files** is a browser for the server folders NanoTag can read. From it you can:
  - see which experiment each ABF is already in;
  - upload files and create folders in the incoming folder;
  - download files;
  - import ABFs or event CSV/XLSX files, either into a chosen experiment or **automatically**.

**📂 Import a folder (the fast way in).** Type or browse to a folder, e.g.
`/home/oguz/NanoporeTagging/SiO2_Tagging`, and press **Scan**. NanoTag shows a plan before changing
anything:

- every `.abf` is paired with its event file `<name>_event.csv` (also `_events.csv`, `.csv`, `.xlsx`);
- recordings are grouped into experiments by file name, so the DC, AC and AOM runs of one sample go
  into **one experiment**; each recording keeps its **condition** (DC / AC / AOM / AC-AOM / baseline);
- ABFs with an event file are ticked; ABFs without one (the `baseline` runs) are listed but not ticked.
  Buttons tick or untick a whole condition. The experiment name can be edited;
- ABFs are **used in place** by default: nothing is copied, and NanoTag never deletes files in your
  folders. (Copy, hard-link and move are also offered.)
- all event files go into one annotation set, **Event CSVs**, in that experiment.

Event files are matched forgivingly: case, doubled or non-breaking spaces, `_event`/`_events` and
Windows (Excel) encodings don't matter, and a CSV with any name is used if its `file_name` column names
exactly one ABF in the folder. Files that are not used are listed with the reason (e.g. a duplicate).
One unreadable file never stops the others; failures are listed in the job's log.

Scanning again later shows, per file, how many events NanoTag holds. **events missing** files are
ticked automatically; if the count differs from the file's rows, tick **replace events that differ
from the file** to re-read them (the old events stay in the edit history). Re-importing never
duplicates recordings or events, so you can add new runs to the same folder and import again. The same works from a terminal:
`sudo nanotag-admin import-folder /home/oguz/NanoporeTagging/SiO2_Tagging` (add `--dry-run` to only
show the plan, `--all` to include files without events).

NanoTag can only read folders an admin has allowed. To allow your data folder (once):

```bash
sudo /opt/nanotag/current/deploy/add-import-root.sh /home/oguz/NanoporeTagging
```

This gives the `nanotag` service read-only access to that folder and its sub-folders (the rest of
your home directory stays private) and restarts NanoTag. `--list` shows the allowed folders.

**Deleting.**
- **Experiments:** on Experiments, tick one or more and press **🗑 Delete selected…**, or use 🗑 on a
  row or "Delete experiment…" in an experiment's sidebar. A confirmation page shows exactly what goes
  (recordings, sets, events, clustering runs, ABF copies). NanoTag's own ABF copies are deleted too
  unless you untick that; files used in place are never deleted.
- **Analyses:** in an experiment, tick annotation sets (NN results are sets), clustering runs or
  recordings and press **🗑 Delete ticked …**, or use 🗑 on a row. **🧹 Clear finished jobs** tidies the
  job lists.
- Admins can delete anything. Other users can delete experiments they created and sets or runs they
  created (or anything in an experiment they created); locked sets only by admins.

**Automatic experiments.** The experiment name is the file name without its extension, without the
run number after the date, and without the trailing acquisition word (DC, AC, AC-AOM, AOM, baseline),
which is kept as the recording's condition. All 12–16
files of one day and sample therefore land in the same experiment:

```
2026_09_14_0001 B3S8-15 100 aM SiO2 DC.abf  ->  experiment "2026_09_14 B3S8-15 100 aM SiO2"
2026_09_14_0012 B3S8-15 100 aM SiO2 AC.abf  ->  experiment "2026_09_14 B3S8-15 100 aM SiO2"
```

The import screens show the guessed experiment before you confirm. Event CSVs imported automatically
send each row to the experiment that holds the ABF named in its `file_name` column. To change the
rule, set these in `/etc/nanotag/nanotag.env` and restart:

- `NANOTAG_EXPERIMENT_RUN` — a regex for the run number to drop (group 1 is kept);
- `NANOTAG_EXPERIMENT_STRIP` — a regex removed from the end of the name (group 1 is the condition);
- `NANOTAG_EXPERIMENT_PREFIX` — text added in front of the name.

**Sign-in and freshness.**
- Opening the server always shows the login page if you are not signed in. After signing in you land
  on the Experiments home, never in the middle of an earlier analysis.
- Sessions end when the browser closes, after 12 hours, and **whenever the server is updated**.
- Pages are never cached. Styles and scripts carry the release number, so browsers pick up an update
  immediately and no hard refresh is needed.

The steps below follow the experiment page's tabs.

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
  - The events table shows only columns that have values, so imported window lists show their
    window start/end, and a **status** column says *window only*, *partial* or *tagged*.
  - **Show all files of the experiment** (tick box above the graph, off by default) lays every
    recording end to end, labelled by run and condition, with the table listing every file's events.
    Everything works there too — add events and windows, delete, drag guides, edit notes; each change
    is saved to the recording under it (an event's 7 points must be in one file). Unticking returns to
    the file in the middle of the view.
  - **Navigation:** ✋ **Pan** (top) drags all channels together in time with Y held; ◀ ▶ step half a
    window. **Thumbwheels** (IRIX style): the vertical wheel left of the plot zooms Y (pick the channel,
    or *all*, under it), the horizontal wheel below zooms X. Drag, or scroll over a wheel;
    double-click resets. **Toggle Side Panel** hides the event table to widen the plot.
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

**Conditions.** Tick DC, AC, AOM … to cluster only those recordings; tick "one run per ticked
condition" to get a separate run (and report) for each.

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
| Allow a data folder | `sudo /opt/nanotag/current/deploy/add-import-root.sh /home/oguz/NanoporeTagging` |
| Import a folder (terminal) | `sudo nanotag-admin import-folder DIR [--dry-run] [--all] [--mode copy]` |
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
sudo ./deploy/update.sh --import-root /home/oguz/NanoporeTagging   # also allow a data folder
sudo /opt/nanotag/current/deploy/update.sh --rollback   # go back to the previous release
```

The updater does the following:

1. Backs up the database.
2. Installs the new code as a new release folder and updates the Python packages.
3. Switches `/opt/nanotag/current` and migrates the database.
4. Restarts the services and checks that NanoTag itself, with the new version, answers on the port,
   rolling back automatically if not. If another program (for example a `python Tagger_GUI.py` left
   running) holds the port, it stops before changing anything and names that program.
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
