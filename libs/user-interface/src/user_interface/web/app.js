// pyurbanair browser interface: area -> case -> model & settings -> run -> results.
// Every request is relative (api/...), so the page works under its /<token>/
// prefix. The JSON API is described in docs/plans/user_interface.md.
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // Job states as mcp_server.jobs.registry defines them.
  const ACTIVE = new Set(["queued", "preparing", "running", "finalizing", "cancelling"]);
  const FAILED = new Set(["failed", "cancelled", "interrupted"]);
  const POLL_MS = 2000;
  const LARGE_GRID = 5e6;
  const STORE_KEY = "pyurbanair-ui:last-form";
  const NOMINATIM = "https://nominatim.openstreetmap.org/search";
  // Sequential ramp (YlGnBu) for building height, low = light, high = dark.
  const RAMP = ["#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"];
  const SOURCE_LABELS = { height: "OSM height tag", levels: "OSM building levels", default: "Default height" };
  const SOURCE_COLORS = { height: "#1d91c0", levels: "#7fcdbb", default: "#b8c2cc" };

  // All app state lives here.
  const state = {
    step: 1,
    maxStep: 1,
    view: "steps", // or "runs"
    options: null,
    drawing: false,
    bbox: null,
    caseRequest: null, // last POST api/cases body, for Retry
    caseId: null,
    caseData: null,
    caseStarted: 0,
    model: null,
    values: {}, // settings form values by field key
    runId: null,
    run: null,
    runStarted: null,
    runEnded: null,
    viz: null, // {runId, id, state, url, error, started}
    stage: "map",
    runs: [],
    timers: {},
  };

  // ---------------------------------------------------------------- helpers

  const nf = new Intl.NumberFormat();
  const fmt = (x, digits = 1) =>
    x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toLocaleString(undefined, { maximumFractionDigits: digits });

  function fmtCells(n) {
    if (n >= 1e9) return `${fmt(n / 1e9, 2)} B`;
    if (n >= 1e6) return `${fmt(n / 1e6, 2)} M`;
    if (n >= 1e4) return `${fmt(n / 1e3, 0)} k`;
    return nf.format(n);
  }

  function fmtDuration(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    const pad = (v) => String(v).padStart(2, "0");
    return h ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
  }

  // `created` may be an ISO string or epoch seconds/milliseconds.
  function parseTime(value) {
    if (value == null || value === "") return null;
    if (typeof value === "number" || /^\d+(\.\d+)?$/.test(String(value))) {
      const n = Number(value);
      return n < 1e12 ? n * 1000 : n;
    }
    const t = Date.parse(value);
    return Number.isNaN(t) ? null : t;
  }

  function fmtAgo(ms) {
    if (ms == null) return "";
    const s = (Date.now() - ms) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(ms).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  }

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of children.flat()) {
      if (child != null) node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function loadLast() {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY)) || {};
    } catch {
      return {};
    }
  }

  function saveLast(patch) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ ...loadLast(), ...patch }));
    } catch {
      /* storage unavailable: nothing to remember */
    }
  }

  function modelLabel(id) {
    const m = state.options?.models?.find((x) => x.id === id);
    return m ? m.label : id || "—";
  }

  function stateTone(s) {
    if (s === "succeeded") return "ok";
    if (s === "cancelling" || s === "cancelled") return "warn";
    if (FAILED.has(s)) return "bad";
    if (ACTIVE.has(s)) return "active";
    return "";
  }

  function setBadge(node, s) {
    node.textContent = s || "—";
    node.dataset.state = s || "";
    node.dataset.tone = stateTone(s);
  }

  function busy(button, on) {
    button.disabled = on;
    button.setAttribute("aria-busy", on ? "true" : "false");
  }

  // ---------------------------------------------------------------- toasts

  function toast(message, kind = "error") {
    const box = $("#toasts");
    const existing = $$(".toast", box).find((t) => t.dataset.msg === message);
    if (existing) existing.remove();
    const close = el("button", { type: "button", "aria-label": "Dismiss" }, "×");
    const node = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, el("p", { text: message }), close);
    node.dataset.msg = message;
    close.addEventListener("click", () => node.remove());
    box.append(node);
    while (box.children.length > 4) box.firstElementChild.remove();
    setTimeout(() => node.remove(), kind === "error" ? 10000 : 5000);
  }

  // ---------------------------------------------------------------- API

  async function api(path, body) {
    const init = body === undefined
      ? { headers: { Accept: "application/json" } }
      : { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(body) };
    let response;
    try {
      response = await fetch(path, init);
    } catch {
      throw new Error("Cannot reach the local server. Is it still running?");
    }
    let data = null;
    try {
      data = await response.json();
    } catch {
      /* non-JSON body */
    }
    if (!response.ok) {
      const error = new Error(data?.error || `${response.status} ${response.statusText}`);
      error.status = response.status;
      throw error;
    }
    return data || {};
  }

  // Timers by name; scheduling a name replaces its previous timer.
  function schedule(name, fn, ms) {
    clearTimeout(state.timers[name]);
    state.timers[name] = setTimeout(fn, ms);
  }
  function unschedule(name) {
    clearTimeout(state.timers[name]);
    delete state.timers[name];
  }

  // ---------------------------------------------------------------- map

  // worldCopyJump keeps the view on the [-180, 180] world the API accepts.
  const map = L.map("map", { zoomControl: true, worldCopyJump: true }).setView([52.0116, 4.3571], 15);
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  // Keyless CARTO basemaps are gone, so the neutral basemap is OSM's own
  // tiles toned down with a CSS filter (.basemap-neutral in app.css).
  const osmUrl = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
  const osmOptions = {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  };
  const neutral = L.tileLayer(osmUrl, { ...osmOptions, className: "basemap-neutral" }).addTo(map);
  const osm = L.tileLayer(osmUrl, osmOptions);
  L.control.layers({ Neutral: neutral, OpenStreetMap: osm }, null, { position: "topright" }).addTo(map);
  L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

  let areaRect = null; // the committed rectangle
  let draft = null; // {pointerId, start, x, y, rect} while dragging
  const caseLayer = L.layerGroup().addTo(map);
  let buildingsLayer = null;

  // Same spherical formula as mcp_server.area_selector.bbox_area_km2.
  function areaKm2(b) {
    const r = 6371.0088, rad = Math.PI / 180;
    return r * r * (b.east - b.west) * rad * (Math.sin(b.north * rad) - Math.sin(b.south * rad));
  }

  const boundsToBbox = (bounds) => ({
    west: bounds.getWest(), south: bounds.getSouth(), east: bounds.getEast(), north: bounds.getNorth(),
  });

  function rectStyle(over, editable) {
    const color = over ? cssVar("--danger") : cssVar("--accent");
    return editable
      ? { color, weight: 2, fillColor: color, fillOpacity: over ? 0.12 : 0.08, dashArray: null, interactive: false }
      : { color, weight: 1.5, fillOpacity: 0, dashArray: "6 5", interactive: false };
  }

  function newRect(bounds) {
    return L.rectangle(bounds, rectStyle(false, true))
      .bindTooltip("", { permanent: true, direction: "center", className: "area-tip" })
      .addTo(map);
  }

  // Updates the readout (and rect style) for `bounds`, or clears it.
  function showArea(bounds, rect) {
    const card = $("#area-card");
    const max = state.options?.max_area_km2;
    if (!bounds) {
      card.dataset.state = "empty";
      $("#area-value").textContent = "—";
      $("#area-meter").style.width = "0";
      $$("#bbox dd").forEach((dd) => (dd.textContent = "—"));
      $("#area-msg").innerHTML = "No area yet. Press <b>Draw area</b> and drag on the map.";
      updateBuildButton();
      return;
    }
    const b = boundsToBbox(bounds);
    const area = areaKm2(b);
    const over = max != null && area > max;
    card.dataset.state = over ? "over" : "ok";
    $("#area-value").textContent = area < 0.01 ? area.toFixed(4) : area.toFixed(area < 1 ? 3 : 2);
    $("#area-meter").style.width = max ? `${Math.min(100, (100 * area) / max)}%` : "0";
    $$("#bbox dd").forEach((dd) => (dd.textContent = b[dd.dataset.k].toFixed(5)));
    $("#area-msg").textContent = over
      ? `Too large: the maximum is ${max} km². Draw a smaller area.`
      : "Area within the limit.";
    if (rect) {
      rect.setStyle(rectStyle(over, state.step === 1));
      rect.setTooltipContent(`${area.toFixed(area < 1 ? 3 : 2)} km²`);
      if (state.step === 1) rect.openTooltip(bounds.getCenter());
      else rect.closeTooltip();
      const tip = rect.getTooltip()?.getElement();
      if (tip) tip.classList.toggle("over", over);
    }
    updateBuildButton();
  }

  function setDrawing(on) {
    state.drawing = on;
    const button = $("#draw-toggle");
    button.setAttribute("aria-pressed", String(on));
    $("#draw-label").textContent = on ? "Drawing… drag on the map" : areaRect ? "Redraw area" : "Draw area";
    map.getContainer().classList.toggle("drawing", on);
    for (const handler of [map.dragging, map.boxZoom, map.doubleClickZoom, map.touchZoom]) {
      if (on) handler.disable();
      else handler.enable();
    }
    $("#draw-hint").hidden = !on;
  }

  const mapEl = map.getContainer();
  mapEl.addEventListener(
    "pointerdown",
    (e) => {
      if (!state.drawing || draft || e.button > 0 || e.target.closest(".leaflet-control")) return;
      e.preventDefault();
      e.stopPropagation();
      try {
        mapEl.setPointerCapture(e.pointerId);
      } catch {
        /* capture is a nicety */
      }
      const start = map.mouseEventToLatLng(e);
      draft = { pointerId: e.pointerId, start, x: e.clientX, y: e.clientY, rect: newRect(L.latLngBounds(start, start)) };
    },
    true,
  );
  mapEl.addEventListener("pointermove", (e) => {
    if (!draft || e.pointerId !== draft.pointerId) return;
    const bounds = L.latLngBounds(draft.start, map.mouseEventToLatLng(e));
    draft.rect.setBounds(bounds); // grow the rectangle while dragging
    showArea(bounds, draft.rect);
  });
  // Released anywhere (also outside the map or window): finish the drag.
  function endDraft(e) {
    if (!draft || (e.pointerId !== undefined && e.pointerId !== draft.pointerId)) return;
    const d = draft;
    draft = null;
    const tiny = e.type === "pointercancel" || Math.hypot(e.clientX - d.x, e.clientY - d.y) < 5;
    if (tiny) {
      // A click, not a drag: drop it and keep the previous area.
      d.rect.remove();
      showArea(areaRect?.getBounds(), areaRect);
      return;
    }
    // Drawn on a wrapped world copy, longitudes can pass ±180: wrap them back.
    const bounds = map.wrapLatLngBounds(L.latLngBounds(d.start, map.mouseEventToLatLng(e)));
    if (areaRect) areaRect.remove();
    areaRect = d.rect;
    areaRect.setBounds(bounds);
    state.bbox = boundsToBbox(bounds);
    setDrawing(false);
    showArea(bounds, areaRect);
  }
  document.addEventListener("pointerup", endDraft);
  document.addEventListener("pointercancel", endDraft);
  window.addEventListener("blur", () => {
    if (draft) endDraft({ type: "pointercancel" });
  });

  $("#draw-toggle").addEventListener("click", () => setDrawing(!state.drawing));
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !state.drawing) return;
    if (draft) endDraft({ type: "pointercancel" });
    setDrawing(false);
  });

  // ---------------------------------------------------------------- place search (Nominatim)

  // Nominatim usage policy: no autocomplete, at most one request per second,
  // attribution. We search on submit only and space requests >= 1.1 s apart.
  let lastSearch = 0;
  $("#search-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#search-input").value.trim();
    if (!q) return;
    const button = $("#search-form button");
    busy(button, true);
    const wait = 1100 - (Date.now() - lastSearch);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    lastSearch = Date.now();
    const list = $("#search-results");
    try {
      const url = `${NOMINATIM}?format=jsonv2&limit=5&q=${encodeURIComponent(q)}`;
      const response = await fetch(url, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const hits = await response.json();
      list.replaceChildren();
      if (!hits.length) {
        list.append(el("li", { class: "empty", text: "No place found." }));
      }
      for (const hit of hits) {
        const go = () => {
          const [s, n, w, east] = hit.boundingbox.map(Number);
          map.fitBounds([[s, w], [n, east]], { maxZoom: 16 });
          list.hidden = true;
        };
        const type = [hit.type, hit.category].filter((x) => x && x !== "yes").join(" · ");
        list.append(el("li", {}, el("button", { type: "button", onclick: go }, hit.display_name, el("span", { class: "type", text: type }))));
      }
      list.append(el("li", { class: "empty" }, el("small", { text: "Search by Nominatim · © OpenStreetMap contributors" })));
      list.hidden = false;
      if (hits.length === 1) $("button", list).click();
    } catch (err) {
      toast(`Place search failed: ${err.message}`);
    } finally {
      busy(button, false);
    }
  });

  // ---------------------------------------------------------------- stepper & stage

  function renderStepper() {
    $$(".stepper li").forEach((li, i) => {
      const n = i + 1;
      const button = $("button", li);
      const current = state.view === "steps" && n === state.step;
      const done = n < state.maxStep || (n === state.maxStep && stepComplete(n) && n !== state.step);
      li.classList.toggle("current", current);
      li.classList.toggle("done", done && !current);
      li.classList.toggle("reachable", n <= state.maxStep);
      button.disabled = n > state.maxStep;
      if (current) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
      const status = current ? "current" : done ? "done" : n > state.maxStep ? "not available yet" : "";
      button.setAttribute("aria-label", `Step ${n}: ${$(".label", li).textContent}${status ? `, ${status}` : ""}`);
    });
    $("#runs-toggle").setAttribute("aria-pressed", String(state.view === "runs"));
  }

  function stepComplete(n) {
    if (n === 1) return !!state.caseId;
    if (n === 2) return state.caseData?.status === "ready";
    if (n === 3) return !!state.runId;
    if (n === 4) return state.run?.state === "succeeded";
    return false;
  }

  function unlock(n) {
    state.maxStep = Math.max(state.maxStep, n);
    renderStepper();
  }

  function go(n, { focus = true } = {}) {
    if (n > state.maxStep) return;
    state.step = n;
    state.view = "steps";
    if (state.drawing && n !== 1) setDrawing(false);
    $$(".panel-scroll > section").forEach((s) => (s.hidden = s.dataset.step !== String(n)));
    renderStepper();
    if (areaRect) showArea(areaRect.getBounds(), areaRect);
    if (n === 3) renderRunForm();
    if (n === 4) renderRun();
    if (n === 5) {
      renderResults();
      if (state.run?.state === "succeeded") startViz(state.runId);
      setStage(state.viz?.url ? "results" : "map");
    } else {
      setStage("map");
    }
    $(".panel-scroll").scrollTop = 0;
    if (focus) $(`section[data-step="${n}"] h2`).focus({ preventScroll: true });
  }

  function showRunsView(on) {
    state.view = on ? "runs" : "steps";
    $$(".panel-scroll > section").forEach((s) => {
      s.hidden = on ? s.dataset.view !== "runs" : s.dataset.step !== String(state.step);
    });
    renderStepper();
    if (on) {
      refreshRuns();
      $("#h-runs").focus({ preventScroll: true });
    }
  }

  function setStage(which) {
    const url = state.viz?.url;
    state.stage = which === "results" && url ? "results" : "map";
    $("#results-view").hidden = state.stage !== "results";
    $("#stage-switch").hidden = !url;
    $$("#stage-switch button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.stage === state.stage)));
    $("#viz-toggle").textContent = state.stage === "results" ? "Show map" : "Show results";
    $("#legend").hidden = state.stage !== "map" || state.caseData?.status !== "ready";
    if (state.stage === "map") setTimeout(() => map.invalidateSize(), 0);
  }

  document.addEventListener("click", (e) => {
    const target = e.target.closest("[data-goto]");
    if (target && !target.disabled) go(Number(target.dataset.goto));
  });
  $("#runs-toggle").addEventListener("click", () => showRunsView(state.view !== "runs"));
  $("#runs-close").addEventListener("click", () => showRunsView(false));
  $$("#stage-switch button").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.stage === "results" && state.step !== 5) go(5, { focus: false });
      setStage(b.dataset.stage);
    }),
  );
  $("#viz-toggle").addEventListener("click", () => setStage(state.stage === "results" ? "map" : "results"));

  // ---------------------------------------------------------------- 1 Area -> build case

  function updateBuildButton() {
    const max = state.options?.max_area_km2;
    const b = state.bbox;
    const area = b ? areaKm2(b) : 0;
    $("#build-case").disabled = !state.options || !b || area <= 0 || (max != null && area > max);
  }

  $("#case-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const res = $("#resolution"), height = $("#default-height");
    if (!e.target.reportValidity()) return;
    const body = { bbox: state.bbox, resolution: res.valueAsNumber, default_height: height.valueAsNumber };
    saveLast({ resolution: body.resolution, default_height: body.default_height });
    await startCase(body, $("#build-case"));
  });

  $("#case-retry").addEventListener("click", () => {
    if (state.caseRequest) startCase(state.caseRequest, $("#case-retry"));
  });

  async function startCase(body, button) {
    busy(button, true);
    try {
      const { case_id } = await api("api/cases", body);
      state.caseRequest = body;
      resetCase();
      state.caseId = case_id;
      state.caseData = { status: "building" };
      state.caseStarted = Date.now();
      state.maxStep = 2;
      go(2);
      pollCase(case_id);
    } catch (err) {
      toast(err.message);
    } finally {
      busy(button, false);
      updateBuildButton();
    }
  }

  // A new case invalidates everything after it.
  function resetCase() {
    unschedule("case");
    caseLayer.clearLayers();
    buildingsLayer = null;
    state.caseId = null;
    state.caseData = null;
    state.runId = null;
    state.run = null;
    state.viz = null;
    unschedule("run");
    unschedule("viz");
    $("#viewer-frame").removeAttribute("src");
    setStage("map");
  }

  // ---------------------------------------------------------------- 2 Case

  async function pollCase(caseId, { quiet = false } = {}) {
    if (state.caseId !== caseId) return;
    let data;
    try {
      data = await api(`api/cases/${encodeURIComponent(caseId)}`);
    } catch (err) {
      if (state.caseId !== caseId) return;
      if (quiet) return;
      toast(err.message);
      schedule("case", () => pollCase(caseId), POLL_MS * 2);
      return;
    }
    if (state.caseId !== caseId) return;
    state.caseData = data;
    if (data.status === "ready") {
      drawCase(data);
      unlock(3);
    } else if (data.status !== "failed") {
      schedule("case", () => pollCase(caseId), POLL_MS);
    }
    renderCase();
    renderRunForm();
    renderStepper();
  }

  function rampColor(t) {
    const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(x));
    const f = x - i;
    const a = RAMP[i].match(/\w\w/g).map((h) => parseInt(h, 16));
    const b = RAMP[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  }

  function maxHeight(data) {
    const fromSummary = Number(data.summary?.max_height);
    if (fromSummary > 0) return fromSummary;
    const hs = (data.buildings?.features || []).map((f) => Number(f.properties?.height) || 0);
    return Math.max(1, ...hs);
  }

  function buildingStyle(feature, hmax) {
    const h = Number(feature.properties?.height) || 0;
    const fallback = feature.properties?.height_source === "default";
    return {
      fillColor: rampColor(h / hmax),
      fillOpacity: 0.88,
      color: cssVar("--outline"),
      opacity: fallback ? 0.75 : 0.45,
      weight: fallback ? 1.2 : 0.8,
      dashArray: fallback ? "3 3" : null,
    };
  }

  function drawCase(data) {
    caseLayer.clearLayers();
    const hmax = maxHeight(data);
    buildingsLayer = L.geoJSON(data.buildings || null, {
      style: (f) => buildingStyle(f, hmax),
      onEachFeature: (f, layer) => {
        const p = f.properties || {};
        const source = SOURCE_LABELS[p.height_source] || p.height_source || "";
        layer.bindTooltip(`<b>${fmt(p.height)} m</b>${source ? ` · ${source}` : ""}`, { sticky: true, direction: "top" });
      },
    }).addTo(caseLayer);
    buildingsLayer.hmax = hmax;
    const sensors = data.sensors || [];
    sensors.forEach(([lon, lat], i) => {
      L.circleMarker([lat, lon], {
        radius: 5, color: "#fff", weight: 1.5, fillColor: cssVar("--sensor"), fillOpacity: 1, className: "sensor",
      }).bindTooltip(`Sensor ${i + 1}`, { direction: "top" }).addTo(caseLayer);
    });
    // Legend
    $("#legend-ramp").style.background = `linear-gradient(to right, ${RAMP.join(",")})`;
    $("#legend-mid").textContent = `${fmt(hmax / 2, 0)} m`;
    $("#legend-max").textContent = `${fmt(hmax, 0)} m`;
    $("#legend-sensors").hidden = !sensors.length;
    // Fit the domain (or the buildings if the area is unknown, e.g. a reopened run).
    const bounds = areaRect ? areaRect.getBounds() : buildingsLayer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] });
    setStage(state.stage);
  }

  function restyleMap() {
    if (buildingsLayer) buildingsLayer.setStyle((f) => buildingStyle(f, buildingsLayer.hmax));
    caseLayer.eachLayer((layer) => {
      if (layer instanceof L.CircleMarker) layer.setStyle({ fillColor: cssVar("--sensor") });
    });
    if (areaRect) showArea(areaRect.getBounds(), areaRect);
  }
  darkQuery.addEventListener("change", restyleMap);

  function renderCase() {
    const data = state.caseData || {};
    const building = data.status === "building" || (!data.status && !!state.caseId);
    $("#case-building").hidden = !building;
    $("#case-failed").hidden = data.status !== "failed";
    $("#case-ready").hidden = data.status !== "ready";
    $("#case-retry").hidden = !state.caseRequest;
    if (building) tickCase();
    if (data.status === "failed") $("#case-error").textContent = data.error || "Unknown error.";
    if (data.status !== "ready") return;

    const s = data.summary || {};
    $("#c-buildings").textContent = s.n_buildings != null ? nf.format(s.n_buildings) : "—";
    $("#c-maxh").textContent = fmt(s.max_height);
    const d = s.domain || {};
    const cells = (d.nx || 0) * (d.ny || 0) * (d.nz || 0);
    $("#c-grid").textContent = d.nx ? `${d.nx} × ${d.ny} × ${d.nz}` : "—";
    const L3 = Array.isArray(d.lengths_m) ? d.lengths_m : d.lengths_m ? [d.lengths_m.x, d.lengths_m.y, d.lengths_m.z] : [];
    $("#c-lengths").textContent = L3.length ? `${L3.map((v) => fmt(v, 0)).join(" × ")} m` : "";
    $("#c-cells").textContent = cells ? fmtCells(cells) : "—";
    $("#c-cells-sub").textContent = cells ? `${nf.format(cells)} cells` : "";
    const large = cells > LARGE_GRID;
    $("#c-cells-card").classList.toggle("warn", large);
    $("#c-cells-warning").hidden = !large;

    // Height sources: known keys first, then any others.
    const sources = s.height_sources || {};
    const keys = [...Object.keys(SOURCE_LABELS).filter((k) => k in sources), ...Object.keys(sources).filter((k) => !(k in SOURCE_LABELS))];
    const total = keys.reduce((acc, k) => acc + (Number(sources[k]) || 0), 0);
    const bar = $("#c-sources-bar"), legend = $("#c-sources");
    bar.replaceChildren();
    legend.replaceChildren();
    for (const k of keys) {
      const n = Number(sources[k]) || 0;
      const color = SOURCE_COLORS[k] || "#9a8fd1";
      if (n) bar.append(el("span", { style: `flex:${n};background:${color}`, title: `${SOURCE_LABELS[k] || k}: ${n}` }));
      legend.append(
        el("li", {},
          el("span", { class: "swatch", style: `background:${color}` }),
          SOURCE_LABELS[k] || k,
          el("span", { class: "n", text: nf.format(n) }),
          el("span", { class: "pct", text: total ? `${Math.round((100 * n) / total)}%` : "" })),
      );
    }
  }

  function tickCase() {
    $("#case-elapsed").textContent = fmtDuration(Date.now() - state.caseStarted);
  }

  // ---------------------------------------------------------------- 3 Model & settings

  function renderModels() {
    const box = $("#model-cards");
    box.replaceChildren();
    for (const m of state.options.models) {
      const input = el("input", { type: "radio", name: "model", value: m.id });
      input.checked = m.id === state.model;
      input.addEventListener("change", () => {
        state.model = m.id;
        saveLast({ model: m.id });
        renderFields();
      });
      box.append(
        el("label", { class: "model-card" },
          input,
          el("span", { class: "radio", "aria-hidden": "true" }),
          el("span", { class: "name" }, m.label, el("span", { class: "mono", text: m.id })),
          el("span", { class: "desc", text: m.description || "" })),
      );
    }
  }

  // A choice may be a plain value or {value, label}.
  const choiceValue = (c) => (c && typeof c === "object" && "value" in c ? c.value : c);
  const choiceLabel = (c) => (c && typeof c === "object" ? c.label ?? String(c.value) : String(c));

  const visibleFields = () =>
    (state.options?.fields || []).filter((f) => !f.models || !f.models.length || f.models.includes(state.model));

  function fieldControl(f, id) {
    const value = state.values[f.key];
    if (f.type === "bool") {
      const input = el("input", { type: "checkbox", id, role: "switch" });
      input.checked = !!value;
      input.addEventListener("change", () => (state.values[f.key] = input.checked));
      return input;
    }
    if (f.type === "choice") {
      const select = el("select", { id });
      (f.choices || []).forEach((c, i) => {
        const option = el("option", { value: String(i) }, choiceLabel(c));
        option.selected = choiceValue(c) === value;
        select.append(option);
      });
      select.addEventListener("change", () => (state.values[f.key] = choiceValue(f.choices[select.selectedIndex])));
      return select;
    }
    const int = f.type === "int";
    const input = el("input", {
      type: "number", id, required: true, inputmode: int ? "numeric" : "decimal",
      min: f.min, max: f.max, step: f.step ?? (int ? 1 : "any"),
    });
    if (value != null) input.value = String(value);
    input.addEventListener("input", () => {
      state.values[f.key] = input.value === "" ? null : int ? parseInt(input.value, 10) : input.valueAsNumber;
    });
    return input;
  }

  function renderFields() {
    const box = $("#fields");
    box.replaceChildren();
    const fields = visibleFields();
    if (!fields.length) box.append(el("p", { class: "empty", text: "This model has no settings to adjust." }));
    for (const f of fields) {
      const id = `f-${f.key.replace(/[^\w-]/g, "_")}`;
      const control = fieldControl(f, id);
      const hint = f.help ? el("p", { class: "hint", id: `${id}-hint`, text: f.help }) : null;
      if (hint) control.setAttribute("aria-describedby", hint.id);
      if (f.type === "bool") {
        box.append(
          el("div", { class: "field switch-field" },
            el("div", { class: "switch-text" }, el("label", { for: id, text: f.label }), hint),
            el("span", { class: "switch" }, control, el("span", { class: "track", "aria-hidden": "true" }))),
        );
        continue;
      }
      const range = f.type !== "choice" && (f.min != null || f.max != null)
        ? el("span", { class: "range", text: `${f.min ?? "…"} – ${f.max ?? "…"}` })
        : null;
      const wrapped = f.unit && f.type !== "choice"
        ? el("div", { class: "input-unit" }, control, el("span", { class: "unit", text: f.unit }))
        : control;
      box.append(
        el("div", { class: "field" },
          el("div", { class: "field-label-row" }, el("label", { for: id, text: f.label + (f.unit && f.type === "choice" ? ` (${f.unit})` : "") }), range),
          wrapped,
          hint),
      );
    }
  }

  function initValues() {
    const last = loadLast();
    for (const f of state.options.fields || []) {
      const saved = last.settings?.[f.key];
      let ok = saved !== undefined && saved !== null;
      if (ok && f.type === "bool") ok = typeof saved === "boolean";
      if (ok && (f.type === "number" || f.type === "int")) ok = typeof saved === "number";
      if (ok && f.type === "choice") ok = (f.choices || []).some((c) => choiceValue(c) === saved);
      state.values[f.key] = ok ? saved : f.default;
    }
    const ids = state.options.models.map((m) => m.id);
    state.model = ids.includes(last.model) ? last.model : ids[0] ?? null;
    if (typeof last.extra_overrides === "string") $("#extra-overrides").value = last.extra_overrides;
  }

  function renderRunForm() {
    const ready = state.caseData?.status === "ready";
    $("#run-blocked").hidden = ready;
    $("#run-start").disabled = !ready || !state.model;
  }

  $("#run-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!e.target.reportValidity()) return;
    const settings = {};
    for (const f of visibleFields()) settings[f.key] = state.values[f.key];
    const extraText = $("#extra-overrides").value;
    const extra = extraText.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
    saveLast({ model: state.model, settings: { ...loadLast().settings, ...settings }, extra_overrides: extraText });
    const body = { case_id: state.caseId, model: state.model, settings };
    if (extra.length) body.extra_overrides = extra;
    const button = $("#run-start");
    busy(button, true);
    try {
      const { run_id } = await api("api/runs", body);
      watchRun({ run_id, case_id: state.caseId, model: state.model, state: "queued" }, Date.now());
      refreshRuns();
    } catch (err) {
      toast(err.message);
    } finally {
      busy(button, false);
      renderRunForm();
    }
  });

  // ---------------------------------------------------------------- 4 Run

  function watchRun(run, started) {
    unschedule("run");
    state.runId = run.run_id;
    state.run = { log_tail: [], ...run };
    state.runStarted = started;
    state.runEnded = null;
    state.sawActive = ACTIVE.has(run.state);
    if (state.viz?.runId !== run.run_id) {
      unschedule("viz");
      state.viz = null;
      $("#viewer-frame").removeAttribute("src");
    }
    state.maxStep = Math.max(4, run.state === "succeeded" ? 5 : 4);
    if (run.state === "succeeded") go(5);
    else go(4);
    pollRun(run.run_id);
  }

  async function pollRun(runId) {
    if (state.runId !== runId) return;
    let data;
    try {
      data = await api(`api/runs/${encodeURIComponent(runId)}`);
    } catch (err) {
      if (state.runId !== runId) return;
      toast(err.message);
      schedule("run", () => pollRun(runId), POLL_MS * 2);
      return;
    }
    if (state.runId !== runId) return;
    const before = state.run?.state;
    state.run = { ...state.run, ...data };
    const s = data.state;
    if (ACTIVE.has(s) || !(FAILED.has(s) || s === "succeeded")) {
      state.sawActive = true;
      schedule("run", () => pollRun(runId), POLL_MS);
    } else {
      if (state.sawActive && state.runEnded == null) state.runEnded = Date.now();
      if (before !== s) refreshRuns();
      if (s === "succeeded") {
        unlock(5);
        if (state.step === 4 && state.view === "steps") setTimeout(() => state.runId === runId && state.step === 4 && go(5), 700);
      }
    }
    renderRun();
    if (state.step === 5) renderResults();
    renderStepper();
  }

  function renderRun() {
    const run = state.run;
    if (!run) return;
    const s = run.state;
    setBadge($("#run-state"), s);
    $("#run-id").textContent = run.run_id || "—";
    $("#run-phase").textContent = run.phase || "—";
    const failed = FAILED.has(s);
    $("#run-message").textContent = failed ? "" : run.message || "";
    const active = ACTIVE.has(s);
    $("#run-progress").hidden = !active;
    $("#run-cancel").hidden = !active;
    $("#run-cancel").disabled = s === "cancelling";
    $("#run-to-results").hidden = s !== "succeeded";
    $("#run-failed").hidden = !failed;
    $("#run-failed").classList.toggle("danger", s !== "cancelled");
    $("#run-failed").classList.toggle("warn", s === "cancelled");
    if (failed) {
      $("#run-failed-title").textContent =
        s === "cancelled" ? "The run was cancelled." : s === "interrupted" ? "The run was interrupted." : "The run failed.";
      $("#run-failed-text").textContent =
        run.error || run.message || (s === "cancelled" ? "Stopped on request." : "See the log below for details.");
    }
    tickRun();
    const log = $("#run-log");
    const lines = run.log_tail || [];
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
    const text = lines.length ? lines.join("\n") : active ? "Waiting for output…" : "No log output.";
    if (log.textContent !== text) {
      log.textContent = text;
      log.classList.toggle("empty", !lines.length);
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    $("#log-count").textContent = lines.length ? `last ${lines.length} lines` : "";
  }

  function tickRun() {
    const node = $("#run-elapsed");
    if (state.runStarted == null || (!ACTIVE.has(state.run?.state) && state.runEnded == null)) {
      node.textContent = "";
      return;
    }
    node.textContent = fmtDuration((state.runEnded ?? Date.now()) - state.runStarted);
  }

  $("#run-cancel").addEventListener("click", async () => {
    if (!state.runId || !window.confirm("Cancel this run? The solver is stopped and partial output is discarded.")) return;
    const button = $("#run-cancel");
    button.disabled = true;
    try {
      const data = await api(`api/runs/${encodeURIComponent(state.runId)}/cancel`, {});
      state.run = { ...state.run, state: data.state || state.run.state };
      renderRun();
      pollRun(state.runId);
    } catch (err) {
      toast(err.message);
      button.disabled = false;
    }
  });

  // ---------------------------------------------------------------- 5 Results

  async function startViz(runId) {
    if (!runId || (state.viz && state.viz.runId === runId && !state.viz.error)) return;
    state.viz = { runId, id: null, state: "starting", url: null, error: null, started: Date.now() };
    renderResults();
    try {
      const { visualization_id } = await api(`api/runs/${encodeURIComponent(runId)}/visualization`, {});
      if (state.viz?.runId !== runId) return;
      state.viz.id = visualization_id;
      pollViz(visualization_id);
    } catch (err) {
      if (state.viz?.runId !== runId) return;
      state.viz.error = err.message;
      toast(err.message);
      renderResults();
    }
  }

  async function pollViz(vizId) {
    if (state.viz?.id !== vizId) return;
    let data;
    try {
      data = await api(`api/visualizations/${encodeURIComponent(vizId)}`);
    } catch (err) {
      if (state.viz?.id !== vizId) return;
      if (err.status >= 400 && err.status < 500) {
        // The render cannot be shown (bad bundle, unknown ID): stop polling.
        state.viz.error = err.message;
        renderResults();
        return;
      }
      toast(err.message);
      schedule("viz", () => pollViz(vizId), POLL_MS * 2);
      return;
    }
    if (state.viz?.id !== vizId) return;
    state.viz.state = data.state;
    if (data.viewer_url) {
      state.viz.url = data.viewer_url;
      const frame = $("#viewer-frame");
      if (frame.getAttribute("src") !== data.viewer_url) frame.src = data.viewer_url;
      if (state.step === 5) setStage("results");
    } else if (data.state === "succeeded" || FAILED.has(data.state)) {
      // Succeeded without a URL: rendered, but the viewer cannot be served.
      state.viz.error = data.message || data.error || `Rendering ${data.state}.`;
    } else {
      schedule("viz", () => pollViz(vizId), POLL_MS);
    }
    renderResults();
  }

  $("#viz-retry").addEventListener("click", () => {
    state.viz = null;
    startViz(state.runId);
  });

  function renderResults() {
    const viz = state.viz;
    const succeeded = state.run?.state === "succeeded";
    $("#viz-pending").hidden = !succeeded || !viz || !!viz.url || !!viz.error;
    $("#viz-failed").hidden = !viz?.error;
    if (viz?.error) {
      $("#viz-failed-title").textContent = viz.state === "succeeded" ? "Viewer unavailable." : "Rendering failed.";
      $("#viz-error").textContent = viz.error;
    }
    $("#viz-ready").hidden = !viz?.url;
    if (viz?.url) $("#viz-open").href = viz.url;
    if (viz && !viz.url) $("#viz-elapsed").textContent = fmtDuration(Date.now() - viz.started);
    setStage(state.stage);

    const facts = $("#result-facts");
    facts.replaceChildren();
    const run = state.run;
    if (!run) return;
    const row = (label, value) => facts.append(el("div", {}, el("dt", { text: label }), el("dd", {}, value)));
    row("Model", modelLabel(run.model));
    row("Run", el("span", { class: "mono", text: run.run_id }));
    const r = run.results || {};
    if (Array.isArray(r.variables) && r.variables.length) {
      row("Variables", el("span", { class: "chips" }, r.variables.map((v) => el("span", { class: "chip", text: v }))));
    }
    const t = r.times;  // {size, first, last} of the time coordinate
    if (t && t.size) {
      // inspect_run_results reports first/last as strings.
      const [first, last] = [Number(t.first), Number(t.last)];
      const span = Number.isFinite(first) && Number.isFinite(last) ? ` (${fmt(first)} – ${fmt(last)} s)` : "";
      row("Snapshots", `${t.size}${span}`);
    }
  }

  // ---------------------------------------------------------------- runs list

  async function refreshRuns() {
    try {
      const data = await api("api/runs");
      state.runs = (data.runs || []).slice().sort((a, b) => (parseTime(b.created) ?? 0) - (parseTime(a.created) ?? 0));
    } catch (err) {
      toast(err.message);
      return;
    }
    renderRuns();
    if (state.runs.some((r) => ACTIVE.has(r.state))) schedule("runs", refreshRuns, POLL_MS * 3);
  }

  function renderRuns() {
    const count = $("#runs-count");
    count.hidden = !state.runs.length;
    count.textContent = String(state.runs.length);
    for (const list of $$("[data-runs-list]")) {
      list.replaceChildren();
      if (!state.runs.length) {
        list.append(el("li", { class: "empty", text: "No runs yet." }));
        continue;
      }
      for (const r of state.runs) {
        const badge = el("span", { class: "badge" });
        setBadge(badge, r.state);
        const created = parseTime(r.created);
        const meta = [r.run_id, fmtAgo(created)].filter(Boolean).join(" · ");
        list.append(
          el("li", {},
            el("button", {
              type: "button", class: "run-item", "aria-current": r.run_id === state.runId ? "true" : "false",
              title: created ? new Date(created).toLocaleString() : null,
              onclick: () => openRun(r),
            },
            el("span", { class: "title", text: modelLabel(r.model) }),
            badge,
            el("span", { class: "meta", text: meta }))),
        );
      }
    }
  }

  function openRun(r) {
    if (r.case_id && r.case_id !== state.caseId) {
      unschedule("case");
      caseLayer.clearLayers();
      buildingsLayer = null;
      if (areaRect) {
        areaRect.remove();
        areaRect = null;
        state.bbox = null;
        showArea(null);
      }
      state.caseId = r.case_id;
      state.caseData = null;
      state.caseRequest = null;
      pollCase(r.case_id, { quiet: true });
    }
    if (r.model) state.model = r.model;
    renderModels();
    renderFields();
    const ended = state.run?.run_id === r.run_id ? state.runEnded : null;
    watchRun(r, ACTIVE.has(r.state) ? parseTime(r.created) : null);
    if (ended != null) state.runEnded = ended;
    renderRuns();
  }

  // ---------------------------------------------------------------- clock

  setInterval(() => {
    if (state.step === 2 && state.caseData?.status === "building") tickCase();
    if (state.step === 4) tickRun();
    if (state.step === 5 && state.viz && !state.viz.url && !state.viz.error) {
      $("#viz-elapsed").textContent = fmtDuration(Date.now() - state.viz.started);
    }
  }, 1000);

  // ---------------------------------------------------------------- start

  async function init() {
    renderStepper();
    showArea(null);
    try {
      state.options = await api("api/options");
    } catch (err) {
      toast(`Could not load options: ${err.message}`);
      setTimeout(init, 4000);
      return;
    }
    const o = state.options;
    o.models = o.models || [];
    $("#area-max").textContent = o.max_area_km2 != null ? `max ${o.max_area_km2} km²` : "";
    const last = loadLast();
    const defaults = o.case_defaults || {};
    $("#resolution").value = String(typeof last.resolution === "number" ? last.resolution : defaults.resolution ?? "");
    $("#default-height").value = String(typeof last.default_height === "number" ? last.default_height : defaults.default_height ?? "");
    initValues();
    renderModels();
    renderFields();
    renderRunForm();
    updateBuildButton();
    refreshRuns();
  }

  init();
})();
