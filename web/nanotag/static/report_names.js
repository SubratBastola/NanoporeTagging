// Friendly cluster names in a clustering HTML report: "C0" -> "C0 (Full)".
// Loaded only when NanoTag serves a report; the report file on disk is not changed.
// Names are kept per (K, cluster) on the server and shown in the report's tables (K Breakdown and
// the "Explore other K" table), in the legends of the two interactive 3D plots, and in the figures
// the server redraws with the names (K Selection, Feature Relationships, the per-K pies). The page
// that is printed / saved as PDF is this same page, so the PDF carries the names too.
// The Sankey plot shows no cluster names and is left as it is.
(function () {
  var R = window.NANOTAG_REPORT;
  if (!R) return;
  var names = R.names || {};
  var figs = R.figs || {};

  var css = document.createElement('style');
  css.textContent =
    '.nt-edit{border:0;background:none;cursor:pointer;color:#9ca3af;font-size:12px;padding:0 0 0 6px;line-height:1}' +
    '.nt-edit:hover{color:#2563eb}' +
    '.nt-name-input{font:inherit;font-size:12px;width:120px;padding:1px 5px;border:1px solid #2563eb;border-radius:5px}' +
    '@media print{.nt-edit{display:none !important}}';
  document.head.appendChild(css);

  function nameOf(k, c) { return (names[String(k)] || {})[String(c)] || ''; }
  function label(k, c) { var n = nameOf(k, c); return 'C' + c + (n ? ' (' + n + ')' : ''); }

  // ---- K Breakdown: one column per K, each cell "C0 · 45.0%"
  function kbreakCells() {
    var out = [], t = document.querySelector('table.kbreak-table');
    if (!t) return out;
    var ks = Array.prototype.map.call(t.querySelectorAll('thead th'), function (th) {
      var m = /K\s*=\s*(\d+)/.exec(th.textContent); return m ? parseInt(m[1], 10) : null;
    });
    Array.prototype.forEach.call(t.querySelectorAll('tbody tr'), function (tr) {
      Array.prototype.forEach.call(tr.children, function (td, i) {
        if (ks[i] == null) return;
        if (td.dataset.ntC === undefined) {
          var m = /^\s*C(\d+)\s*·\s*(.*\S)\s*$/.exec(td.textContent);
          if (!m) return;
          td.dataset.ntC = m[1]; td.dataset.ntK = ks[i]; td.dataset.ntRest = m[2];
        }
        out.push(td);
      });
    });
    return out;
  }

  function drawCell(td) {
    var k = +td.dataset.ntK, c = +td.dataset.ntC;
    td.textContent = label(k, c) + ' · ' + td.dataset.ntRest;
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'nt-edit'; b.textContent = '✎';
    b.title = 'Name cluster C' + c + ' (at K=' + k + ')';
    b.onclick = function () { editCell(td); };
    td.appendChild(b);
  }

  function editCell(td) {
    var k = +td.dataset.ntK, c = +td.dataset.ntC, done = false;
    td.textContent = 'C' + c + ' ';
    var inp = document.createElement('input');
    inp.className = 'nt-name-input'; inp.maxLength = 60; inp.value = nameOf(k, c);
    inp.placeholder = 'name, e.g. Full';
    td.appendChild(inp);
    inp.focus(); inp.select();
    function finish(save) {
      if (done) return;
      done = true;
      var v = inp.value.replace(/\s+/g, ' ').trim();
      if (!save || v === nameOf(k, c)) { drawAll(); return; }
      fetch('/api/cluster/' + R.run_id + '/names', {
        method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({k: k, cluster: c, name: v})
      }).then(function (r) {
        if (!r.ok) throw new Error(r.status === 401 ? 'Please sign in to NanoTag again.' : 'HTTP ' + r.status);
        return r.json();
      }).then(function (d) { names = d.names || {}; figs = d.figs || {}; drawAll(); })
        .catch(function (e) { drawAll(); window.alert('The name was not saved. ' + e.message); });
    }
    inp.onkeydown = function (e) {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      else if (e.key === 'Escape') { finish(false); }
    };
    inp.onblur = function () { finish(true); };
  }

  // ---- "Explore other K" table: first column is the cluster ("C0"), rebuilt whenever K is changed
  function drawExplore() {
    var sel = document.getElementById('k-select'), tb = document.querySelector('#k-summary-table tbody');
    if (!sel || !tb) return;
    var k = parseInt(sel.value, 10);
    Array.prototype.forEach.call(tb.rows, function (tr) {
      var td = tr.cells[0];
      if (!td) return;
      if (td.dataset.ntC === undefined) {
        var m = /^\s*C(\d+)\s*$/.exec(td.textContent);
        if (!m) return;
        td.dataset.ntC = m[1];
      }
      td.textContent = label(k, +td.dataset.ntC);
    });
  }

  // ---- figures redrawn by the server with the names (PNG pictures)
  function figImg(title) {
    var hs = document.querySelectorAll('#tab-report h2');
    for (var i = 0; i < hs.length; i++) {
      if (hs[i].textContent.trim() === title) {
        var card = hs[i].nextElementSibling;
        return card ? card.querySelector('img.report-fig') : null;
      }
    }
    return null;
  }
  function setImg(img, b64) {
    if (!img) return;
    if (img.dataset.ntOrig === undefined) img.dataset.ntOrig = img.getAttribute('src') || '';
    img.src = b64 ? 'data:image/png;base64,' + b64 : img.dataset.ntOrig;
  }
  var PIES0 = null;
  function drawFigs() {
    setImg(figImg('K Selection'), figs.ksel);
    setImg(figImg('Feature Relationships'), figs.pairs);
    var note = document.getElementById('nt-fig-note'), ks = figImg('K Selection');
    if (!note && ks) {
      note = document.createElement('div'); note.id = 'nt-fig-note'; note.className = 'legend-note';
      ks.parentNode.appendChild(note);
    }
    if (note) {
      note.textContent = (figs.skipped || []).indexOf('ksel') >= 0
        ? 'The cluster names could not be drawn into this figure (the run\u2019s BIC curve is no longer available); it keeps C0, C1, \u2026' : '';
    }
    try {
      if (typeof K_PIES === 'object' && K_PIES) {
        if (!PIES0) { PIES0 = {}; Object.keys(K_PIES).forEach(function (k) { PIES0[k] = K_PIES[k]; }); }
        Object.keys(PIES0).forEach(function (k) { K_PIES[k] = (figs.pies || {})[k] || PIES0[k]; });
        var sel = document.getElementById('k-select'), img = document.getElementById('k-pie-img');
        if (sel && img && K_PIES[sel.value]) img.src = 'data:image/png;base64,' + K_PIES[sel.value];
      }
    } catch (e) {}
  }

  // ---- legends of the interactive 3D plots (trace names "C0 (n=24, 50%)")
  function legend(id, kOf) {
    var gd = document.getElementById(id);
    if (!gd || !gd.data || !window.Plotly) return false;
    if (!gd._ntNames) gd._ntNames = gd.data.map(function (t) { return String(t.name || ''); });
    var changed = false, out = gd._ntNames.map(function (orig, i) {
      var m = /^C(\d+)\b(.*)$/.exec(orig), k = kOf(i);
      var nm = (m && k != null) ? label(k, +m[1]) + m[2] : orig;
      if (nm !== gd.data[i].name) changed = true;
      return nm;
    });
    if (changed) window.Plotly.restyle(gd, {name: out});
    return changed;
  }
  function drawLegends() {
    try {
      var fk = R.final_k;
      if (legend('explorer3d', function () { return fk; }) && typeof refreshExplorer3DPrintImage === 'function') {
        refreshExplorer3DPrintImage();          // the snapshot that is printed / saved as PDF
      }
      legend('exploreK3d', function (i) { return (typeof TRACE_K === 'object' && TRACE_K) ? TRACE_K[i] : null; });
    } catch (e) {}
  }

  function drawAll() { kbreakCells().forEach(drawCell); drawExplore(); drawFigs(); drawLegends(); }

  var orig = window.updateExploreK;
  if (typeof orig === 'function') {
    window.updateExploreK = function () { var r = orig.apply(this, arguments); drawExplore(); return r; };
  }
  function start() { drawAll(); }
  if (document.readyState === 'loading') {
    // after the report's own DOMContentLoaded handler has filled the Explore table
    window.addEventListener('DOMContentLoaded', function () { setTimeout(start, 0); });
  } else { start(); }
})();
