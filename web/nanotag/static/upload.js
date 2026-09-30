// Chunked, resumable uploader for large files (ABFs up to several GB).
// Each file is split into chunks (size chosen by the server, 16 MB by default) sent with PUT;
// if the connection drops, re-selecting the same file resumes from the chunks already received.
async function nanotagUpload(file, opts, onProgress) {
  const init = await fetch('/api/uploads/init', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({filename: file.name, size: file.size}, opts))
  });
  const meta = await init.json();
  if (!init.ok) throw new Error(meta.error || ('init failed: ' + init.status));
  const cs = meta.chunk_size;
  const n = Math.max(1, Math.ceil(file.size / cs));
  const have = new Set(meta.received || []);
  let done = have.size;
  onProgress(done / n, have.size ? `resuming (${have.size}/${n} chunks already on server)` : '');
  for (let i = 0; i < n; i++) {
    if (have.has(i)) continue;
    const blob = file.slice(i * cs, Math.min(file.size, (i + 1) * cs));
    let attempt = 0;
    while (true) {
      try {
        const r = await fetch(`/api/uploads/${meta.upload_id}/chunk?index=${i}`, {method: 'PUT', body: blob});
        if (!r.ok) throw new Error((await r.text()).slice(0, 200));
        break;
      } catch (e) {
        attempt++;
        if (attempt > 8) throw e;
        onProgress(done / n, `retrying chunk ${i} (${attempt})…`);
        await new Promise(res => setTimeout(res, 1000 * attempt));
      }
    }
    done++;
    onProgress(done / n, '');
  }
  const fin = await fetch(`/api/uploads/${meta.upload_id}/complete`, {method: 'POST'});
  const res = await fin.json();
  if (!fin.ok) throw new Error(res.error || 'complete failed');
  return res;
}

function fmtBytes(n) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + ' ' + u[i];
}

// Wire a drop zone + file input to the uploader.
function setupUploader(zoneId, inputId, listId, opts, onAllDone) {
  const zone = document.getElementById(zoneId), input = document.getElementById(inputId),
        list = document.getElementById(listId);
  if (!zone || !input) return;
  async function handle(files) {
    let ok = 0;
    window._uploading = (window._uploading || 0) + 1;
    for (const f of files) {
      const row = document.createElement('div'); row.className = 'u';
      row.innerHTML = `<b>${f.name}</b> (${fmtBytes(f.size)}) <span class="bar"><i style="width:0%"></i></span> <span class="st"></span>`;
      list.appendChild(row);
      const bar = row.querySelector('i'), st = row.querySelector('.st');
      const t0 = Date.now();
      try {
        await nanotagUpload(f, opts, (frac, note) => {
          bar.style.width = (100 * frac).toFixed(1) + '%';
          const secs = (Date.now() - t0) / 1000;
          const rate = secs > 1 ? fmtBytes(f.size * frac / secs) + '/s' : '';
          st.textContent = `${(100 * frac).toFixed(1)}% ${rate} ${note || ''}`;
        });
        st.textContent = 'uploaded ✓ — processing queued';
        ok++;
      } catch (e) {
        st.textContent = 'FAILED: ' + e.message + ' (select the file again to resume)';
        st.style.color = '#b91c1c';
      }
    }
    window._uploading -= 1;
    if (onAllDone && ok) onAllDone();
  }
  input.addEventListener('change', () => handle(Array.from(input.files)));
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('hover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('hover'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('hover'); handle(Array.from(e.dataTransfer.files)); });
}
