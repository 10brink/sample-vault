// Sample Vault — Frontend Application Logic (Zero-build ES Module)

const state = {
  samples: [],
  mode: "browse", // browse | semantic_text | reference_audio | similar_sample
  query: "",
  sampleType: "",
  bpmMin: "",
  bpmMax: "",
  key: "",
  category: "",
  folder: "",
  favoritesOnly: false,
  sort: "recent",
  selectedIndex: -1,
  playingId: null,
  loopEnabled: false,
  lastManifestId: null,
  status: null,
  inspectingId: null,
};

const $ = (sel) => document.querySelector(sel);
const audio = $("#audio-player");

// ── Utilities ────────────────────────────────────────────────────────────────

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) detail = err.detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function fmtDuration(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  if (sec < 10) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtTimeClock(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function syncSlider(slider) {
  if (!slider) return;
  slider.style.setProperty("--attr-min", slider.min || 0);
  slider.style.setProperty("--attr-max", slider.max || 100);
  slider.style.setProperty("--control-value", slider.value);
}

// ── Dialog Light-Dismiss Fallback & ReturnValue Cleanup ─────────────────────

function initDialogs() {
  const dialogs = document.querySelectorAll("dialog");
  for (const dlg of dialogs) {
    dlg.addEventListener("close", () => {
      dlg.returnValue = "";
    });
  }
  // Fallback for browsers without <dialog closedby="any"> (e.g. Safari)
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e, dialog) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    for (const dialog of document.querySelectorAll('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dialog.addEventListener("pointerdown", (e) => {
        pressedOutside = outside(e, dialog);
      });
      dialog.addEventListener("pointerup", (e) => {
        if (pressedOutside && outside(e, dialog)) dialog.close();
      });
    }
  }
}

// ── Theme Switcher ──────────────────────────────────────────────────────────

function initTheme() {
  const modes = ["auto", "dark", "light"];
  let current = localStorage.getItem("sv-theme") || "auto";
  const apply = (mode) => {
    current = mode;
    if (mode === "auto") {
      document.documentElement.removeAttribute("data-theme");
      $("#theme-label").textContent = "Auto";
    } else {
      document.documentElement.setAttribute("data-theme", mode);
      $("#theme-label").textContent = mode === "dark" ? "Dark" : "Light";
    }
    localStorage.setItem("sv-theme", mode);
    redrawAllWaveforms();
  };
  apply(current);

  $("#btn-theme").addEventListener("click", () => {
    const next = modes[(modes.indexOf(current) + 1) % modes.length];
    apply(next);
  });

  // React to system theme changes for canvas waveform redraws
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", () => redrawAllWaveforms());
}

// ── Status & Facets ─────────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    state.status = data;
    const stats = data.stats;
    const pill = $("#status-pill");
    const statusText = $("#status-text");

    if (data.model_loading) {
      pill.classList.add("loading");
      statusText.textContent = `Loading CLAP (${data.device.toUpperCase()})...`;
    } else {
      pill.classList.remove("loading");
      const dev = data.device.toUpperCase();
      statusText.textContent = `CLAP · ${dev} · ${stats.vec_backend} · ${stats.total_samples} samples`;
    }

    $("#cnt-all").textContent = stats.total_samples;
    $("#cnt-oneshot").textContent = stats.one_shots;
    $("#cnt-loop").textContent = stats.loops;
    $("#cnt-fav").textContent = stats.favorites;

    renderKeyOptions(stats.keys);
    renderCategoryFacets(stats.categories);
    renderSidebarFolders(data.folders);
    renderModalFolders(data.folders);
    updateIndexBanner(data.indexing);
  } catch (err) {
    $("#status-text").textContent = `Offline: ${err.message}`;
  }
}

function renderKeyOptions(keys) {
  const sel = $("#filter-key");
  const prev = state.key;
  sel.innerHTML = '<option value="">All keys</option>';
  for (const k of keys) {
    const opt = document.createElement("option");
    opt.value = k.name;
    opt.textContent = `${k.name} (${k.count})`;
    sel.appendChild(opt);
  }
  sel.value = prev;
}

function renderCategoryFacets(categories) {
  const ul = $("#category-list");
  ul.innerHTML = "";
  $("#btn-clear-cat").hidden = !state.category;
  if (!categories.length) {
    ul.innerHTML = '<li class="muted" style="font-size:12px;padding:4px 8px;">No categories yet</li>';
    return;
  }
  for (const c of categories) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `facet-btn${state.category === c.name ? " active" : ""}`;
    btn.innerHTML = `<span class="facet-name"></span><span class="facet-cnt">${c.count}</span>`;
    btn.querySelector(".facet-name").textContent = c.name;
    btn.addEventListener("click", () => {
      state.category = state.category === c.name ? "" : c.name;
      renderCategoryFacets(categories);
      runSearch();
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

function renderSidebarFolders(folders) {
  const ul = $("#sidebar-folder-list");
  ul.innerHTML = "";
  if (!folders.length) {
    ul.innerHTML = '<li class="muted" style="font-size:12px;padding:4px 8px;">No folders added</li>';
    return;
  }
  for (const f of folders) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `facet-btn${state.folder === f.path ? " active" : ""}`;
    btn.title = f.path;
    btn.innerHTML = `<span class="facet-name"></span><span class="facet-cnt">${f.sample_count}</span>`;
    btn.querySelector(".facet-name").textContent = f.label;
    btn.addEventListener("click", () => {
      state.folder = state.folder === f.path ? "" : f.path;
      renderSidebarFolders(folders);
      runSearch();
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

function renderModalFolders(folders) {
  const ul = $("#modal-folder-list");
  ul.innerHTML = "";
  if (!folders.length) {
    ul.innerHTML = '<li class="muted" style="font-size:13px;">No watched folders yet. Add one above or pick from detected studio locations below.</li>';
    return;
  }
  for (const f of folders) {
    const li = document.createElement("li");
    li.className = "folder-manage-item";
    li.innerHTML = `
      <div class="folder-manage-meta">
        <strong></strong>
        <span class="mono"></span>
      </div>
      <div class="btn-row">
        <span class="tag-pill mono">${f.sample_count} samples</span>
        <button type="button" class="btn btn-ghost btn-sm" data-remove="${f.id}">Remove</button>
      </div>
    `;
    li.querySelector("strong").textContent = f.label;
    li.querySelector(".mono").textContent = f.path;
    li.querySelector("[data-remove]").addEventListener("click", async () => {
      await api(`/api/folders/${f.id}`, { method: "DELETE" });
      if (state.folder === f.path) state.folder = "";
      await refreshStatus();
      await runSearch();
    });
    ul.appendChild(li);
  }
}

async function loadCommonFolders() {
  const grid = $("#common-folders-list");
  grid.innerHTML = '<span class="muted">Scanning local paths...</span>';
  try {
    const items = await api("/api/folders/common");
    grid.innerHTML = "";
    if (!items.length) {
      grid.innerHTML = '<span class="muted">No common folders auto-detected. Use Browse… or enter any path.</span>';
      return;
    }
    for (const item of items) {
      const card = document.createElement("div");
      card.className = "detected-card";
      card.innerHTML = `
        <div class="det-info">
          <span class="det-label"></span>
          <span class="det-path mono"></span>
        </div>
        <button type="button" class="btn btn-secondary btn-sm">Add</button>
      `;
      card.querySelector(".det-label").textContent = item.label;
      card.querySelector(".det-path").textContent = item.path;
      card.querySelector("button").addEventListener("click", async () => {
        await addAndIndexFolder(item.path, item.label);
      });
      grid.appendChild(card);
    }
  } catch (err) {
    grid.innerHTML = `<span class="muted">${err.message}</span>`;
  }
}

async function addAndIndexFolder(path, label = null) {
  const errEl = $("#folder-error");
  errEl.hidden = true;
  try {
    await api("/api/folders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, label, auto_index: true }),
    });
    $("#folder-path-input").value = "";
    await refreshStatus();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.hidden = false;
  }
}

// ── Live Indexing SSE Stream ────────────────────────────────────────────────

function connectIndexEvents() {
  const es = new EventSource("/api/index/events");
  let wasRunning = false;
  es.onmessage = (ev) => {
    try {
      const prog = JSON.parse(ev.data);
      updateIndexBanner(prog);
      if (wasRunning && !prog.running) {
        refreshStatus();
        runSearch();
      }
      wasRunning = Boolean(prog.running);
    } catch (_) {}
  };
}

function updateIndexBanner(prog) {
  const banner = $("#index-banner");
  if (!prog) return;
  if (prog.running) {
    banner.hidden = false;
    $("#index-banner-msg").textContent = prog.message || "Indexing...";
    const pct = prog.total_files > 0
      ? Math.round((prog.processed_files / prog.total_files) * 100)
      : (prog.phase === "loading_model" ? 35 : 10);
    $("#index-progress-fill").style.width = `${Math.max(5, Math.min(100, pct))}%`;
    $("#index-banner-count").textContent = prog.total_files > 0
      ? `${prog.processed_files} / ${prog.total_files}`
      : prog.phase.replace("_", " ");
    $("#btn-cancel-index").hidden = false;
  } else if (prog.phase === "done" && prog.indexed_files > 0 && prog.finished_at && Date.now() / 1000 - prog.finished_at < 6) {
    banner.hidden = false;
    $("#index-banner-msg").textContent = prog.message;
    $("#index-progress-fill").style.width = "100%";
    $("#index-banner-count").textContent = "Complete";
    $("#btn-cancel-index").hidden = true;
    setTimeout(() => {
      if (!state.status?.indexing?.running) banner.hidden = true;
    }, 4500);
  } else if (prog.phase === "error") {
    banner.hidden = false;
    $("#index-banner-msg").textContent = prog.message;
    $("#index-progress-fill").style.width = "100%";
    $("#index-banner-count").textContent = "Error";
    $("#btn-cancel-index").hidden = true;
  } else {
    banner.hidden = true;
  }
}

// ── Search & Reference Matching ─────────────────────────────────────────────

async function runSearch() {
  hideContextBanner();
  const params = new URLSearchParams();
  if (state.query.trim()) params.set("q", state.query.trim());
  if (state.sampleType) params.set("sample_type", state.sampleType);
  if (state.bpmMin) params.set("bpm_min", state.bpmMin);
  if (state.bpmMax) params.set("bpm_max", state.bpmMax);
  if (state.key) params.set("key", state.key);
  if (state.category) params.set("category", state.category);
  if (state.folder) params.set("folder", state.folder);
  if (state.favoritesOnly) params.set("favorites_only", "true");
  params.set("sort", state.sort);
  params.set("limit", "120");

  $("#results-summary").textContent = state.query.trim()
    ? `Searching CLAP embeddings for "${state.query.trim()}"...`
    : "Loading samples...";

  try {
    const data = await api(`/api/search?${params.toString()}`);
    state.mode = data.mode;
    state.samples = data.results;
    state.selectedIndex = state.samples.length ? 0 : -1;
    renderSamples();
    updateSummary(data);
    refreshStatus();
  } catch (err) {
    $("#results-summary").textContent = `Search error: ${err.message}`;
  }
}

async function runReferenceAudioSearch(file) {
  const form = new FormData();
  form.append("file", file);
  if (state.sampleType) form.append("sample_type", state.sampleType);
  if (state.bpmMin) form.append("bpm_min", state.bpmMin);
  if (state.bpmMax) form.append("bpm_max", state.bpmMax);
  if (state.key) form.append("key", state.key);
  if (state.category) form.append("category", state.category);
  if (state.favoritesOnly) form.append("favorites_only", "true");
  form.append("limit", "40");

  $("#results-summary").textContent = `Analyzing reference sound "${file.name}" with CLAP...`;

  try {
    const data = await api("/api/search/audio", { method: "POST", body: form });
    state.mode = "reference_audio";
    state.samples = data.results;
    state.selectedIndex = state.samples.length ? 0 : -1;

    const ref = data.reference;
    const metaParts = [
      ref.sample_type,
      ref.auto_category,
      ref.bpm ? `${ref.bpm} BPM` : null,
      ref.key_name || null,
      fmtDuration(ref.duration_sec),
    ].filter(Boolean);
    showContextBanner("REFERENCE AUDIO", ref.filename, metaParts.join(" · "));
    renderSamples();
    $("#results-summary").textContent = `Found ${data.count} closest matches to "${ref.filename}"`;
  } catch (err) {
    $("#results-summary").textContent = `Reference search failed: ${err.message}`;
  }
}

async function runFindSimilar(sampleId) {
  $("#results-summary").textContent = "Finding sonically similar samples...";
  try {
    const params = new URLSearchParams({ limit: "35" });
    if (state.sampleType) params.set("sample_type", state.sampleType);
    const data = await api(`/api/samples/${sampleId}/similar?${params.toString()}`);
    state.mode = "similar_sample";
    state.samples = data.results;
    state.selectedIndex = state.samples.length ? 0 : -1;

    const src = data.source_sample;
    const metaParts = [
      src.sample_type,
      src.auto_category,
      src.bpm ? `${src.bpm} BPM` : null,
      src.key_name || null,
    ].filter(Boolean);
    showContextBanner("SIMILAR TO", src.filename, metaParts.join(" · "));
    renderSamples();
    $("#results-summary").textContent = `${data.count} samples sonically closest to "${src.filename}"`;
  } catch (err) {
    $("#results-summary").textContent = `Similarity search failed: ${err.message}`;
  }
}

function showContextBanner(badge, title, meta) {
  $("#context-badge").textContent = badge;
  $("#context-title").textContent = title;
  $("#context-meta").textContent = meta;
  $("#context-banner").hidden = false;
}

function hideContextBanner() {
  $("#context-banner").hidden = true;
}

function updateSummary(data) {
  if (data.mode === "semantic_text") {
    $("#results-summary").textContent = `Top ${data.count} semantic matches for "${data.query}"`;
  } else {
    $("#results-summary").textContent = `Showing ${data.count} sample${data.count === 1 ? "" : "s"}`;
  }
}

// ── Rendering Sample List & Waveforms ───────────────────────────────────────

function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawWaveform(canvas, peaks, progress = 0, isPlaying = false) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 240;
  const height = canvas.clientHeight || 44;
  if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const bars = peaks && peaks.length ? peaks : [0.2, 0.5, 0.3];
  const barCount = bars.length;
  const step = width / barCount;
  const barW = Math.max(1.2, step * 0.68);
  const midY = height / 2;

  const idleColor = getCssVar("--wave-idle") || "#4b5568";
  const activeColor = getCssVar("--wave-active") || "#ff6b2b";
  const cyanColor = getCssVar("--cyan") || "#38bdf8";

  for (let i = 0; i < barCount; i++) {
    const amp = Math.max(0.06, bars[i]);
    const h = amp * (height - 8);
    const x = i * step;
    const pct = i / barCount;
    if (isPlaying && pct <= progress) {
      ctx.fillStyle = activeColor;
    } else if (isPlaying) {
      ctx.fillStyle = cyanColor;
      ctx.globalAlpha = 0.55;
    } else {
      ctx.fillStyle = idleColor;
      ctx.globalAlpha = 0.9;
    }
    ctx.fillRect(x, midY - h / 2, barW, h);
    ctx.globalAlpha = 1;
  }

  if (isPlaying && progress > 0 && progress <= 1) {
    const px = progress * width;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(px - 0.75, 0, 1.5, height);
  }
  ctx.restore();
}

function redrawAllWaveforms() {
  const canvases = document.querySelectorAll(".waveform-canvas");
  for (const canvas of canvases) {
    const id = Number(canvas.dataset.id);
    const sample = state.samples.find((s) => s.id === id);
    if (!sample) continue;
    const isPlaying = state.playingId === id;
    const prog = isPlaying && audio.duration ? audio.currentTime / audio.duration : 0;
    drawWaveform(canvas, sample.waveform_peaks, prog, isPlaying);
  }
}

function renderSamples() {
  const list = $("#sample-list");
  const empty = $("#empty-state");
  list.innerHTML = "";

  if (!state.samples.length) {
    empty.hidden = false;
    const total = state.status?.stats?.total_samples || 0;
    if (total > 0) {
      $("#empty-title").textContent = "No samples match these filters";
      $("#empty-desc").textContent = "Try clearing BPM, key, or category filters, or enter a broader sound description.";
      $("#btn-empty-demo").hidden = true;
    } else {
      $("#empty-title").textContent = "Your sample vault is empty";
      $("#empty-desc").textContent = "Add your Bitwig Studio library, OneKey, Splice, or local sample folders — or index the built-in demo pack to try CLAP semantic search right away.";
      $("#btn-empty-demo").hidden = !state.status?.demo_available;
    }
    return;
  }

  empty.hidden = true;

  state.samples.forEach((s, idx) => {
    const row = document.createElement("div");
    const isPlaying = state.playingId === s.id && !audio.paused;
    const isSelected = idx === state.selectedIndex;
    row.className = `sample-row${isSelected ? " selected" : ""}${state.playingId === s.id ? " playing" : ""}`;
    row.dataset.id = String(s.id);
    row.dataset.index = String(idx);
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(isSelected));

    const simHtml = typeof s.similarity === "number"
      ? `<span class="sim-badge">${Math.round(Math.max(0, s.similarity) * 100)}%</span>`
      : "";

    const typeClass = s.sample_type === "loop" ? "type-loop" : "type-oneshot";
    const typeLabel = s.sample_type === "loop" ? "LOOP" : "ONE-SHOT";
    const tags = [...(s.auto_tags || []), ...(s.custom_tags || [])].slice(0, 4);
    const keyDisplay = s.key_name ? (s.camelot ? `${s.key_name} · ${s.camelot}` : s.key_name) : "—";

    row.innerHTML = `
      <button type="button" class="row-play-btn" aria-label="Play ${s.filename}">
        ${isPlaying
          ? '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>'
          : '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"></polygon></svg>'
        }
      </button>

      <div class="row-info">
        <div class="row-title-line">
          <span class="row-filename"></span>
          ${simHtml}
        </div>
        <div class="row-subline">
          <span class="tag-pill ${typeClass}">${typeLabel}</span>
          <span class="tag-pill cat"></span>
          ${tags.map(() => '<span class="tag-pill auto-tag"></span>').join("")}
          ${s.rel_dir ? '<span class="row-dir mono"></span>' : ""}
        </div>
      </div>

      <div class="waveform-wrap" title="Click waveform to audition &amp; seek">
        <canvas class="waveform-canvas" data-id="${s.id}"></canvas>
      </div>

      <div class="row-specs">
        <div class="spec-box">
          <span class="spec-label">BPM</span>
          <span class="spec-val">${s.bpm ? Math.round(s.bpm) : "—"}</span>
        </div>
        <div class="spec-box" style="min-width:86px;">
          <span class="spec-label">KEY</span>
          <span class="spec-val">${keyDisplay}</span>
        </div>
        <div class="spec-box">
          <span class="spec-label">LEN</span>
          <span class="spec-val">${fmtDuration(s.duration_sec)}</span>
        </div>
      </div>

      <div class="row-actions">
        <button type="button" class="icon-act act-similar" title="Find sonically similar samples" aria-label="Find similar">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a8 8 0 0 0 0-6"></path>
            <path d="M4.6 9a8 8 0 0 0 0 6"></path>
          </svg>
        </button>
        <button type="button" class="icon-act act-fav${s.favorite ? " fav-active" : ""}" title="Favorite (F)" aria-label="Toggle favorite">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="${s.favorite ? "currentColor" : "none"}" stroke="currentColor" stroke-width="2">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
          </svg>
        </button>
        <button type="button" class="icon-act act-finder" title="Reveal in Finder (drag into Bitwig)" aria-label="Reveal in Finder">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
          </svg>
        </button>
        <button type="button" class="icon-act act-edit" title="Inspect &amp; edit tags" aria-label="Inspect sample">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="1"></circle>
            <circle cx="19" cy="12" r="1"></circle>
            <circle cx="5" cy="12" r="1"></circle>
          </svg>
        </button>
      </div>
    `;

    row.querySelector(".row-filename").textContent = s.filename;
    row.querySelector(".tag-pill.cat").textContent = s.auto_category;
    const tagEls = row.querySelectorAll(".auto-tag");
    tags.forEach((t, ti) => {
      if (tagEls[ti]) tagEls[ti].textContent = t;
    });
    const dirEl = row.querySelector(".row-dir");
    if (dirEl) dirEl.textContent = s.rel_dir;

    // Events
    row.querySelector(".row-play-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      selectAndTogglePlay(idx);
    });

    row.querySelector(".waveform-wrap").addEventListener("click", (e) => {
      e.stopPropagation();
      const rect = e.currentTarget.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      playSampleAtIndex(idx, ratio);
    });

    row.querySelector(".act-similar").addEventListener("click", (e) => {
      e.stopPropagation();
      runFindSimilar(s.id);
    });

    row.querySelector(".act-fav").addEventListener("click", async (e) => {
      e.stopPropagation();
      await toggleFavorite(s);
    });

    row.querySelector(".act-finder").addEventListener("click", async (e) => {
      e.stopPropagation();
      await api(`/api/samples/${s.id}/reveal`, { method: "POST" });
    });

    row.querySelector(".act-edit").addEventListener("click", (e) => {
      e.stopPropagation();
      openInspector(s);
    });

    row.addEventListener("click", () => {
      state.selectedIndex = idx;
      updateSelectionHighlight();
    });

    row.addEventListener("dblclick", () => {
      playSampleAtIndex(idx, 0);
    });

    list.appendChild(row);
  });

  // Draw waveforms once laid out
  requestAnimationFrame(() => redrawAllWaveforms());
}

function updateSelectionHighlight() {
  const rows = document.querySelectorAll(".sample-row");
  rows.forEach((r, idx) => {
    const sel = idx === state.selectedIndex;
    r.classList.toggle("selected", sel);
    r.setAttribute("aria-selected", String(sel));
  });
}

// ── Audio Playback & Transport ──────────────────────────────────────────────

function playSampleAtIndex(idx, startRatio = 0) {
  const s = state.samples[idx];
  if (!s) return;
  state.selectedIndex = idx;
  const sameSample = state.playingId === s.id;
  state.playingId = s.id;

  if (!sameSample) {
    audio.src = `/api/audio/${s.id}`;
    audio.load();
  }
  audio.loop = state.loopEnabled;

  const seekAndPlay = () => {
    if (startRatio > 0 && Number.isFinite(audio.duration)) {
      audio.currentTime = startRatio * audio.duration;
    } else if (!sameSample) {
      audio.currentTime = 0;
    }
    audio.play().catch(() => {});
  };

  if (sameSample && audio.readyState >= 1) {
    seekAndPlay();
  } else {
    audio.addEventListener("loadedmetadata", seekAndPlay, { once: true });
    audio.play().catch(() => {});
  }

  updateTransportInfo(s);
  renderSamples();
}

function selectAndTogglePlay(idx) {
  const s = state.samples[idx];
  if (!s) return;
  if (state.playingId === s.id) {
    if (audio.paused) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
    updatePlayIcons();
    renderSamples();
    return;
  }
  playSampleAtIndex(idx, 0);
}

function updateTransportInfo(s) {
  $("#np-title").textContent = s.filename;
  const parts = [
    s.sample_type.toUpperCase(),
    s.auto_category,
    s.bpm ? `${Math.round(s.bpm)} BPM` : null,
    s.key_name ? (s.camelot ? `${s.key_name} (${s.camelot})` : s.key_name) : null,
    `${(s.sample_rate / 1000).toFixed(1)}kHz`,
  ].filter(Boolean);
  $("#np-meta").textContent = parts.join(" · ");
  $("#btn-play-pause").disabled = false;
  $("#seek-slider").disabled = false;
  $("#btn-np-similar").disabled = false;
  $("#btn-np-finder").disabled = false;
  $("#time-total").textContent = fmtTimeClock(s.duration_sec);
}

function updatePlayIcons() {
  const playing = !audio.paused && Boolean(state.playingId);
  $("#icon-play").hidden = playing;
  $("#icon-pause").hidden = !playing;
}

function animationLoop() {
  if (state.playingId && !audio.paused && Number.isFinite(audio.duration) && audio.duration > 0) {
    const ratio = audio.currentTime / audio.duration;
    const seek = $("#seek-slider");
    seek.value = String(Math.round(ratio * 1000));
    syncSlider(seek);
    $("#time-current").textContent = fmtTimeClock(audio.currentTime);
    $("#time-total").textContent = fmtTimeClock(audio.duration);

    const canvas = document.querySelector(`.waveform-canvas[data-id="${state.playingId}"]`);
    const sample = state.samples.find((s) => s.id === state.playingId);
    if (canvas && sample) {
      drawWaveform(canvas, sample.waveform_peaks, ratio, true);
    }
  }
  requestAnimationFrame(animationLoop);
}

async function toggleFavorite(sample) {
  const updated = await api(`/api/samples/${sample.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite: !sample.favorite }),
  });
  const idx = state.samples.findIndex((s) => s.id === sample.id);
  if (idx !== -1) state.samples[idx] = { ...state.samples[idx], ...updated };
  renderSamples();
  refreshStatus();
}

// ── Inspector Dialog ────────────────────────────────────────────────────────

function openInspector(sample) {
  state.inspectingId = sample.id;
  $("#insp-filename").textContent = sample.filename;
  $("#insp-path").textContent = sample.path;
  $("#insp-type").value = sample.sample_type;
  $("#insp-category").value = sample.auto_category;
  $("#insp-bpm").value = sample.bpm ?? "";
  $("#insp-key").value = sample.key_name ?? "";
  $("#insp-custom-tags").value = (sample.custom_tags || []).join(", ");

  $("#insp-metrics").innerHTML = `
    <div class="metric-card"><span class="m-label">Duration</span><span class="m-val">${fmtDuration(sample.duration_sec)}</span></div>
    <div class="metric-card"><span class="m-label">Sample Rate</span><span class="m-val">${(sample.sample_rate / 1000).toFixed(1)} kHz</span></div>
    <div class="metric-card"><span class="m-label">Punch / Energy</span><span class="m-val">${Math.round(sample.energy * 100)}%</span></div>
    <div class="metric-card"><span class="m-label">Brightness</span><span class="m-val">${Math.round(sample.brightness * 100)}%</span></div>
  `;
  $("#inspector-dialog").showModal();
}

// ── Auto-Organizer Dialog ───────────────────────────────────────────────────

async function previewOrganize() {
  const targetDir = $("#org-target-dir").value.trim();
  if (!targetDir) return;
  const template = $("#org-template").value;
  const mode = $("#org-mode").value;
  const statusEl = $("#org-status");
  const box = $("#org-preview-box");
  statusEl.hidden = false;
  statusEl.textContent = "Computing organization preview...";
  try {
    const plan = await api("/api/organize/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_dir: targetDir, template, mode }),
    });
    statusEl.textContent = `Ready: ${plan.total_operations} samples into ${plan.folder_summary.length} folders at ${plan.target_dir} (${mode}).`;
    box.hidden = false;
    box.innerHTML = plan.folder_summary
      .map(
        (row) =>
          `<div class="org-tree-row"><span>${row.folder}/</span><span>${row.count} files</span></div>`
      )
      .join("");
    $("#btn-org-apply").disabled = plan.total_operations === 0;
  } catch (err) {
    statusEl.textContent = `Preview failed: ${err.message}`;
  }
}

async function applyOrganize() {
  const targetDir = $("#org-target-dir").value.trim();
  const template = $("#org-template").value;
  const mode = $("#org-mode").value;
  const statusEl = $("#org-status");
  statusEl.hidden = false;
  statusEl.textContent = "Building organized folder tree...";
  try {
    const res = await api("/api/organize/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_dir: targetDir, template, mode }),
    });
    state.lastManifestId = res.manifest_id;
    statusEl.textContent = `Done! Created ${res.created_count} ${res.mode}s (${res.skipped_count} already existed) in ${res.target_dir}.`;
    $("#btn-org-undo").hidden = false;
  } catch (err) {
    statusEl.textContent = `Apply failed: ${err.message}`;
  }
}

async function undoOrganize() {
  if (!state.lastManifestId) return;
  const statusEl = $("#org-status");
  try {
    const res = await api(`/api/organize/undo/${state.lastManifestId}`, { method: "POST" });
    statusEl.textContent = `Undid run ${res.manifest_id}: removed ${res.removed_count} items.`;
    state.lastManifestId = null;
    $("#btn-org-undo").hidden = true;
  } catch (err) {
    statusEl.textContent = `Undo failed: ${err.message}`;
  }
}

// ── Event Wiring ────────────────────────────────────────────────────────────

function bindEvents() {
  // Search form
  $("#search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    state.query = $("#search-input").value;
    runSearch();
  });

  $("#search-input").addEventListener("search", () => {
    state.query = $("#search-input").value;
    if (!state.query.trim()) runSearch();
  });

  // Example chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.query = chip.dataset.query;
      $("#search-input").value = state.query;
      runSearch();
    });
  });

  // Reference file input
  $("#ref-file-input").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) runReferenceAudioSearch(file);
    e.target.value = "";
  });

  // Global drag & drop for reference sound matching
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    if (e.dataTransfer?.types?.includes("Files")) {
      dragDepth++;
      $("#drop-overlay").hidden = false;
    }
  });
  window.addEventListener("dragover", (e) => {
    if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
  });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) $("#drop-overlay").hidden = true;
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    $("#drop-overlay").hidden = true;
    const file = e.dataTransfer?.files?.[0];
    if (file) runReferenceAudioSearch(file);
  });

  // Sidebar filters
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.sampleType = btn.dataset.type || "";
      runSearch();
    });
  });

  $("#filter-favorites").addEventListener("change", (e) => {
    state.favoritesOnly = e.target.checked;
    runSearch();
  });

  let bpmTimer = null;
  const onBpmInput = () => {
    clearTimeout(bpmTimer);
    bpmTimer = setTimeout(() => {
      state.bpmMin = $("#filter-bpm-min").value.trim();
      state.bpmMax = $("#filter-bpm-max").value.trim();
      runSearch();
    }, 260);
  };
  $("#filter-bpm-min").addEventListener("input", onBpmInput);
  $("#filter-bpm-max").addEventListener("input", onBpmInput);

  $("#btn-clear-bpm").addEventListener("click", () => {
    $("#filter-bpm-min").value = "";
    $("#filter-bpm-max").value = "";
    state.bpmMin = "";
    state.bpmMax = "";
    runSearch();
  });

  document.querySelectorAll(".mini-pill[data-bpm]").forEach((pill) => {
    pill.addEventListener("click", () => {
      const [lo, hi] = pill.dataset.bpm.split(",");
      $("#filter-bpm-min").value = lo;
      $("#filter-bpm-max").value = hi;
      state.bpmMin = lo;
      state.bpmMax = hi;
      runSearch();
    });
  });

  $("#filter-key").addEventListener("change", (e) => {
    state.key = e.target.value;
    runSearch();
  });

  $("#btn-clear-cat").addEventListener("click", () => {
    state.category = "";
    runSearch();
  });

  $("#sort-select").addEventListener("change", (e) => {
    state.sort = e.target.value;
    runSearch();
  });

  $("#btn-clear-context").addEventListener("click", () => {
    hideContextBanner();
    runSearch();
  });

  // Transport controls
  $("#btn-play-pause").addEventListener("click", () => {
    if (!state.playingId) return;
    if (audio.paused) audio.play().catch(() => {});
    else audio.pause();
    updatePlayIcons();
    renderSamples();
  });

  $("#btn-loop-toggle").addEventListener("click", () => {
    state.loopEnabled = !state.loopEnabled;
    audio.loop = state.loopEnabled;
    $("#btn-loop-toggle").setAttribute("aria-pressed", String(state.loopEnabled));
  });

  const seekSlider = $("#seek-slider");
  syncSlider(seekSlider);
  seekSlider.addEventListener("input", () => {
    syncSlider(seekSlider);
    if (Number.isFinite(audio.duration) && audio.duration > 0) {
      audio.currentTime = (Number(seekSlider.value) / 1000) * audio.duration;
    }
  });

  const volSlider = $("#volume-slider");
  audio.volume = Number(volSlider.value);
  syncSlider(volSlider);
  volSlider.addEventListener("input", () => {
    syncSlider(volSlider);
    audio.volume = Number(volSlider.value);
  });

  audio.addEventListener("play", () => {
    updatePlayIcons();
    renderSamples();
  });
  audio.addEventListener("pause", () => {
    updatePlayIcons();
    renderSamples();
  });
  audio.addEventListener("ended", () => {
    updatePlayIcons();
    redrawAllWaveforms();
  });

  $("#btn-np-similar").addEventListener("click", () => {
    if (state.playingId) runFindSimilar(state.playingId);
  });

  $("#btn-np-finder").addEventListener("click", async () => {
    if (state.playingId) await api(`/api/samples/${state.playingId}/reveal`, { method: "POST" });
  });

  // Indexing & Folders modal
  $("#btn-open-folders").addEventListener("click", () => {
    loadCommonFolders();
    $("#folders-dialog").showModal();
  });
  $("#btn-empty-folders").addEventListener("click", () => {
    loadCommonFolders();
    $("#folders-dialog").showModal();
  });

  $("#btn-empty-demo").addEventListener("click", async () => {
    const common = await api("/api/folders/common");
    const demo = common.find((c) => c.label.includes("Demo"));
    if (demo) {
      await addAndIndexFolder(demo.path, demo.label);
    }
  });

  $("#add-folder-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const path = $("#folder-path-input").value.trim();
    if (path) await addAndIndexFolder(path);
  });

  $("#btn-pick-folder").addEventListener("click", async () => {
    try {
      const res = await api("/api/folders/pick", { method: "POST" });
      if (res.path) $("#folder-path-input").value = res.path;
    } catch (_) {}
  });

  $("#btn-reindex-all").addEventListener("click", async () => {
    await api("/api/index/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: false }),
    });
  });

  $("#btn-force-reindex").addEventListener("click", async () => {
    await api("/api/index/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
  });

  $("#btn-cancel-index").addEventListener("click", async () => {
    await api("/api/index/cancel", { method: "POST" });
  });

  // Auto-Organizer modal
  $("#btn-open-organize").addEventListener("click", () => {
    $("#organize-dialog").showModal();
  });
  $("#btn-pick-org-dir").addEventListener("click", async () => {
    try {
      const res = await api("/api/folders/pick", { method: "POST" });
      if (res.path) $("#org-target-dir").value = res.path;
    } catch (_) {}
  });
  $("#btn-org-preview").addEventListener("click", previewOrganize);
  $("#btn-org-apply").addEventListener("click", applyOrganize);
  $("#btn-org-undo").addEventListener("click", undoOrganize);

  // Sample Inspector modal
  $("#inspector-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.inspectingId) return;
    const bpmVal = $("#insp-bpm").value.trim();
    const customTags = $("#insp-custom-tags")
      .value.split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const payload = {
      sample_type: $("#insp-type").value,
      auto_category: $("#insp-category").value.trim() || "Uncategorized",
      bpm: bpmVal ? Number(bpmVal) : null,
      key_name: $("#insp-key").value.trim() || null,
      custom_tags: customTags,
    };
    const updated = await api(`/api/samples/${state.inspectingId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const idx = state.samples.findIndex((s) => s.id === state.inspectingId);
    if (idx !== -1) state.samples[idx] = { ...state.samples[idx], ...updated };
    $("#inspector-dialog").close();
    renderSamples();
    refreshStatus();
  });

  $("#btn-insp-finder").addEventListener("click", async () => {
    if (state.inspectingId) {
      await api(`/api/samples/${state.inspectingId}/reveal`, { method: "POST" });
    }
  });

  // Producer Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    const inInput = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    if (e.key === "/" && !inInput) {
      e.preventDefault();
      $("#search-input").focus();
      $("#search-input").select();
      return;
    }
    if (inInput || document.querySelector("dialog[open]")) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!state.samples.length) return;
      const next = Math.min(state.samples.length - 1, state.selectedIndex + 1);
      playSampleAtIndex(next, 0);
      document.querySelector(`.sample-row[data-index="${next}"]`)?.scrollIntoView({ block: "nearest" });
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (!state.samples.length) return;
      const prev = Math.max(0, state.selectedIndex - 1);
      playSampleAtIndex(prev, 0);
      document.querySelector(`.sample-row[data-index="${prev}"]`)?.scrollIntoView({ block: "nearest" });
    } else if (e.code === "Space") {
      e.preventDefault();
      if (state.selectedIndex >= 0) selectAndTogglePlay(state.selectedIndex);
    } else if (e.key.toLowerCase() === "l") {
      e.preventDefault();
      $("#btn-loop-toggle").click();
    } else if (e.key.toLowerCase() === "f") {
      e.preventDefault();
      const s = state.samples[state.selectedIndex];
      if (s) toggleFavorite(s);
    }
  });

  window.addEventListener("resize", () => redrawAllWaveforms());
}

// ── Bootstrap ───────────────────────────────────────────────────────────────

async function init() {
  initDialogs();
  initTheme();
  bindEvents();
  connectIndexEvents();
  await refreshStatus();
  await runSearch();
  requestAnimationFrame(animationLoop);
}

init();
