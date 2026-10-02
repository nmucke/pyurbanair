"use strict";
// Both directions select held, saved samples. No field interpolation is implied.
const ForwardTimeline = {
  physicalTime(frames, seconds) {
    if (!frames.length) return 0;
    let selected = frames[0].simulation_time;
    for (const frame of frames) { if (frame.video_time > seconds + 1e-6) break; selected = frame.simulation_time; }
    return selected;
  },
  videoTime(frames, simulationTime) {
    if (!frames.length) return 0;
    let selected = frames[0].video_time;
    for (const frame of frames) { if (frame.simulation_time > simulationTime) break; selected = frame.video_time; if (frame.simulation_time === simulationTime) break; }
    return selected;
  },
  clock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const whole = Math.floor(seconds), hours = Math.floor(whole / 3600);
    const fraction = seconds > 0 && seconds < 10 ? `.${Math.floor(seconds * 10) % 10}` : "";
    return `${hours ? `${hours}:` : ""}${hours ? String(Math.floor(whole / 60) % 60).padStart(2, "0") : Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}${fraction}`;
  }
};
if (typeof module !== "undefined") module.exports = ForwardTimeline;
if (typeof document !== "undefined") (async function () {
  const el = id => document.getElementById(id), cinema = document.querySelector(".cinema");
  let manifest, charts, panels = [], mode = "2d", primary = null;
  let physical = 0, presentation = 0, playing = false, lastTick = null, animation = null, generation = 0, minimumPhysical = -Infinity;
  const asset = path => {
    if (typeof path !== "string" || !path || path.startsWith("/") || path.includes("..") || path.includes(":") || path.includes("\\")) throw new Error("Invalid bundle asset path");
    return path;
  };
  const loadJSON = async path => {
    const response = await fetch(asset(path));
    if (!response.ok) throw new Error(`Missing bundle asset: ${path}`);
    return response.json();
  };
  const number = value => Number(value).toLocaleString(undefined, {maximumFractionDigits: 3});
  const visible = () => panels.filter(panel => panel.kind === mode);
  const canAnimate = panel => panel.frames.length > 1 && panel.frames.at(-1).simulation_time > panel.frames[0].simulation_time && (panel.snapshots.filter(snapshot => !panel.failedSnapshots.has(snapshot.path)).length > 1 || panel.movieReady);
  const duration = panel => {
    const last = panel.frames.at(-1).video_time;
    const delta = panel.frames.length > 1 ? Math.max(.001, last - panel.frames.at(-2).video_time) : 0;
    return Math.max(Number.isFinite(panel.view.duration) ? panel.view.duration : 0, last + delta);
  };
  function sample(frames, time) {
    let selected = frames[0];
    for (const frame of frames) { if (frame.simulation_time > time + 1e-6) break; selected = frame; }
    return selected;
  }
  function stopAnimation() {
    generation += 1;
    if (animation !== null) cancelAnimationFrame(animation);
    animation = null; lastTick = null; playing = false;
  }
  function updateStatus() {
    const active = visible();
    const pending = active.filter(panel => panel.view.media && !panel.movieReady && !panel.movieFailed && !panel.snapshots.length).length;
    const sequences = active.filter(panel => !panel.movieReady && panel.snapshots.length > 1).length;
    const stills = active.filter(panel => !canAnimate(panel)).length;
    el("status").textContent = pending ? `Loading ${pending} movie${pending === 1 ? "" : "s"} · previews remain available` : `Ready · ${active.length} synchronized ${mode === "2d" ? "slice" : "flow"} panel${active.length === 1 ? "" : "s"}${sequences ? ` · ${sequences} PNG sequence${sequences === 1 ? "" : "s"}` : ""}${stills ? ` · ${stills} still preview${stills === 1 ? "" : "s"}` : ""}`;
  }
  function updateControls() {
    const animated = primary && canAnimate(primary);
    el("play").disabled = !animated;
    el("restart").disabled = !animated;
    el("seek").disabled = !animated;
    el("play").textContent = playing ? "Pause" : "Play";
    el("play").setAttribute("aria-pressed", String(playing));
    el("playback").dataset.playing = String(playing);
    if (primary) {
      el("seek").min = String(primary.frames[0].simulation_time);
      el("seek").max = String(primary.frames.at(-1).simulation_time);
      el("seek").value = String(physical);
      el("clock").textContent = `${ForwardTimeline.clock(presentation)} / ${ForwardTimeline.clock(animated ? duration(primary) : 0)}`;
    }
    el("physical").textContent = `Simulation time: ${physical.toFixed(3)} s`;
    el("physical").dataset.time = String(physical);
    charts?.update(physical);
  }
  function choosePrimary() {
    const active = visible();
    const next = active.find(canAnimate) || active[0] || null;
    if (next !== primary) {
      primary = next;
      if (primary) {
        const first = primary.frames[0].simulation_time, last = primary.frames.at(-1).simulation_time;
        physical = Math.max(first, Math.min(last, physical));
        presentation = ForwardTimeline.videoTime(primary.frames, physical);
        minimumPhysical = physical;
      }
    }
    if (!primary || !canAnimate(primary)) stopAnimation();
    updateControls(); updateStatus();
  }
  function sourceLabel(panel, time, format) {
    const first = panel.frames[0].simulation_time, last = panel.frames.at(-1).simulation_time;
    const outside = physical < first ? "Before this view’s interval · " : physical > last ? "After this view’s interval · " : "";
    panel.status.textContent = `${outside}${format} · saved t = ${number(time)} s`;
    panel.element.dataset.sampleTime = String(time);
  }
  function showImage(panel) {
    panel.video.hidden = true; panel.image.hidden = false;
    panel.status.dataset.fallback = "true";
    const candidates = panel.snapshots.filter(snapshot => !panel.failedSnapshots.has(snapshot.path));
    const selected = candidates.length ? sample(candidates, physical) : {simulation_time: panel.frames[0].simulation_time, path: panel.view.poster};
    if (panel.failedSnapshots.has(selected.path)) {
      panel.image.hidden = true; panel.placeholder.hidden = false; panel.placeholder.textContent = "Preview unavailable";
      panel.status.textContent = "This preview file could not be loaded"; panel.element.dataset.ready = "error"; delete panel.element.dataset.sampleTime; return;
    }
    panel.pngLink.href = asset(selected.path); panel.pngLink.textContent = candidates.length ? "PNG" : "Poster PNG";
    const format = () => candidates.length > 1 ? (panel.movieFailed ? "Movie unavailable · PNG sequence" : "PNG sequence") : "Still preview";
    if (panel.imagePath === selected.path) { if (panel.element.dataset.ready === "true") sourceLabel(panel, selected.simulation_time, format()); return; }
    panel.imagePath = selected.path;
    panel.element.dataset.ready = "false"; panel.element.dataset.seeking = "true";
    panel.status.textContent = `Loading saved t = ${number(selected.simulation_time)} s…`;
    const token = ++panel.imageToken;
    panel.image.onload = () => {
      if (token !== panel.imageToken || panel.movieReady) return;
      panel.element.dataset.ready = "true"; panel.element.dataset.seeking = "false";
      sourceLabel(panel, selected.simulation_time, format());
      panel.placeholder.hidden = true;
    };
    panel.image.onerror = () => {
      if (token !== panel.imageToken || panel.movieReady) return;
      panel.failedSnapshots.add(selected.path);
      if (selected.path !== panel.view.poster && !panel.failedSnapshots.has(panel.view.poster)) { panel.imagePath = null; showImage(panel); choosePrimary(); return; }
      panel.image.hidden = true; panel.placeholder.hidden = false;
      panel.placeholder.textContent = "Preview unavailable";
      panel.status.textContent = "This preview file could not be loaded";
      panel.element.dataset.ready = "error"; delete panel.element.dataset.sampleTime; choosePrimary();
    };
    panel.image.src = asset(selected.path);
  }
  function displayPanel(panel) {
    if (!panel.movieReady) { showImage(panel); return; }
    panel.video.hidden = false; panel.image.hidden = true; panel.placeholder.hidden = true;
    panel.status.dataset.fallback = "false";
    const selected = sample(panel.frames, physical);
    const matchingSnapshot = panel.snapshots.find(snapshot => snapshot.simulation_time === selected.simulation_time && !panel.failedSnapshots.has(snapshot.path));
    panel.pngLink.href = asset(matchingSnapshot ? matchingSnapshot.path : panel.view.poster);
    panel.pngLink.textContent = matchingSnapshot ? "PNG" : "Poster PNG";
    const target = Math.max(0, Math.min(panel.video.duration - .000001, selected.video_time));
    panel.video.pause();
    if (Math.abs(panel.video.currentTime - target) > .001) {
      panel.status.textContent = `Seeking saved t = ${number(selected.simulation_time)} s…`;
      panel.element.dataset.seeking = "true";
      panel.video.currentTime = target;
    } else if (!panel.video.seeking) {
      panel.element.dataset.seeking = "false";
      sourceLabel(panel, selected.simulation_time, "Saved frame");
    }
  }
  function display() { visible().forEach(displayPanel); updateControls(); }
  function startAnimation() {
    if (!primary || !canAnimate(primary)) return;
    if (presentation >= duration(primary) - .001) { presentation = 0; physical = primary.frames[0].simulation_time; }
    playing = true; minimumPhysical = physical; lastTick = null;
    const token = ++generation;
    const tick = now => {
      if (token !== generation || !playing || !primary) return;
      if (lastTick !== null) presentation += Math.max(0, now - lastTick) / 1000 * Number(el("rate").value);
      lastTick = now;
      presentation = Math.min(duration(primary), presentation);
      physical = Math.max(minimumPhysical, ForwardTimeline.physicalTime(primary.frames, presentation));
      display();
      if (presentation >= duration(primary)) { stopAnimation(); updateControls(); return; }
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick); updateControls();
  }
  function seek(time) {
    if (!primary) return;
    const resume = playing; stopAnimation();
    const target = Math.max(primary.frames[0].simulation_time, Math.min(primary.frames.at(-1).simulation_time, time));
    presentation = ForwardTimeline.videoTime(primary.frames, target);
    physical = ForwardTimeline.physicalTime(primary.frames, presentation);
    minimumPhysical = physical; display();
    if (resume) startAnimation();
  }
  function selectMode(next) {
    if (!panels.some(panel => panel.kind === next)) return;
    const resume = playing; stopAnimation(); mode = next;
    el("views-2d").hidden = mode !== "2d"; el("views-3d").hidden = mode !== "3d";
    panels.forEach(panel => panel.video.pause());
    ["2d", "3d"].forEach(kind => { el(`mode-${kind}`).classList.toggle("active", kind === mode); el(`mode-${kind}`).setAttribute("aria-pressed", String(kind === mode)); });
    cinema.classList.toggle("slices-active", mode === "2d"); cinema.dataset.mode = mode;
    el("caption").textContent = mode === "2d" ? "Compare the horizontal maps and vertical exchange together. Colored markers match the virtual-probe traces at each height." : "Follow the instantaneous flow around and above the geometry. Streamlines show flow shape, not tracked particle trajectories.";
    primary = null; choosePrimary(); display();
    if (resume && primary && canAnimate(primary)) startAnimation();
  }
  function makePanel(view, index) {
    const kind = view.kind === "3d" ? "3d" : "2d";
    const frames = (view.frames || []).map(frame => ({video_time: Number(frame.video_time), simulation_time: Number(frame.simulation_time)}));
    if (!frames.length) frames.push({video_time: 0, simulation_time: manifest.time_range?.[0] || 0});
    if (frames.some((frame, i) => !Number.isFinite(frame.video_time) || !Number.isFinite(frame.simulation_time) || frame.video_time < 0 || (i && (frame.video_time < frames[i - 1].video_time || frame.simulation_time < frames[i - 1].simulation_time)))) throw new Error("Invalid frame timeline");
    const snapshots = (view.snapshots || []).filter(snapshot => Number.isFinite(snapshot.simulation_time)).map(snapshot => ({simulation_time: snapshot.simulation_time, path: asset(snapshot.path)})).sort((a, b) => a.simulation_time - b.simulation_time);
    const element = document.createElement("article"); element.className = "view-panel";
    element.dataset.viewId = String(view.id || index); element.dataset.kind = kind; element.dataset.axis = view.slice?.axis || ""; element.dataset.ready = "false";
    if (kind === "2d" && view.slice?.axis && view.slice.axis !== "z") element.classList.add("vertical-section");
    const heading = document.createElement("div"); heading.className = "panel-heading";
    const title = document.createElement("h2"); title.textContent = view.label || `${kind.toUpperCase()} view ${index + 1}`;
    const field = document.createElement("span"); field.className = "panel-field"; field.textContent = `${(view.field || "velocity").replaceAll("_", " ")} [${view.units || "m/s"}]`;
    heading.append(title, field);
    const visual = document.createElement("div"); visual.className = "panel-visual";
    const video = document.createElement("video"); video.className = "panel-video"; video.muted = true; video.playsInline = true; video.preload = "metadata"; video.controls = true; video.hidden = true; video.poster = asset(view.poster); video.setAttribute("aria-label", title.textContent);
    const image = document.createElement("img"); image.className = "panel-poster"; image.alt = `${title.textContent} · ${field.textContent}`;
    const placeholder = document.createElement("p"); placeholder.className = "panel-empty"; placeholder.hidden = true;
    visual.append(video, image, placeholder);
    const footer = document.createElement("div"); footer.className = "panel-footer";
    const status = document.createElement("span"); status.className = "panel-status panel-time";
    const downloads = document.createElement("div"); downloads.className = "panel-downloads";
    const pngLink = document.createElement("a"); pngLink.href = asset(view.poster); pngLink.download = ""; pngLink.textContent = "PNG"; pngLink.setAttribute("aria-label", `Download ${title.textContent} PNG`); downloads.append(pngLink);
    const panel = {view, kind, frames, snapshots, element, video, image, placeholder, status, pngLink, movieReady: false, movieFailed: false, imagePath: null, imageToken: 0, failedSnapshots: new Set()};
    if (view.media) {
      const mediaPath = asset(view.media), expected = new URL(mediaPath, document.baseURI).href;
      const movieLink = document.createElement("a"); movieLink.href = mediaPath; movieLink.download = ""; movieLink.textContent = "MP4"; movieLink.setAttribute("aria-label", `Download ${title.textContent} MP4`); downloads.append(movieLink);
      const movieLoaded = () => {
        if (video.currentSrc !== expected || panel.movieFailed) return;
        if (!Number.isFinite(video.duration) || video.duration <= 0) { panel.movieFailed = true; movieLink.hidden = true; showImage(panel); choosePrimary(); return; }
        if (video.readyState < 2) return;
        panel.movieReady = true; video.controls = false; element.dataset.ready = "true";
        choosePrimary(); if (kind === mode) displayPanel(panel);
      };
      video.addEventListener("loadedmetadata", movieLoaded);
      video.addEventListener("loadeddata", movieLoaded);
      video.addEventListener("seeked", () => { if (kind === mode && panel.movieReady) displayPanel(panel); });
      video.addEventListener("error", () => {
        if (video.currentSrc && video.currentSrc !== expected) return;
        panel.movieFailed = true; panel.movieReady = false; movieLink.hidden = true; video.pause();
        if (kind === mode) showImage(panel); choosePrimary();
      });
      video.src = mediaPath;
    }
    footer.append(status, downloads); element.append(heading, visual, footer);
    el(`views-${kind}`).append(element);
    return panel;
  }
  try {
    manifest = await loadJSON("viewer_manifest.json");
    if (manifest.version !== 1 || !manifest.views?.length) throw new Error("Unsupported or empty viewer bundle");
    el("title").textContent = manifest.case || manifest.run_id || "Forward simulation";
    const selection = manifest.selection?.reduction || (manifest.selection?.member !== null && manifest.selection?.member !== undefined ? `member ${manifest.selection.member}` : "single simulation");
    el("selection").textContent = `${manifest.backend || "Simulation"} · ${manifest.prediction_type || "simulation"} · ${selection}`;
    el("processing").textContent = `${manifest.processing?.collocation || "Physical grid coordinates"}. ${manifest.processing?.temporal || "Saved snapshots are held; no field interpolation"}. All panels share a physical-time cursor; each panel reports its own displayed saved sample.`;
    el("domain").textContent = ["x", "y", "z"].filter(axis => manifest.domain?.[axis]).map(axis => `${axis}: ${number(manifest.domain[axis].min)}–${number(manifest.domain[axis].max)} m`).join(" · ");
    (manifest.warnings || []).forEach(warning => { const item = document.createElement("li"); item.textContent = warning; el("warnings").append(item); });
    // Horizontal maps are paired above full-width vertical sections.
    const ordered = manifest.views.map((view, index) => ({view, index})).sort((a, b) => Number(a.view.kind !== "3d" && a.view.slice?.axis !== "z") - Number(b.view.kind !== "3d" && b.view.slice?.axis !== "z"));
    panels = ordered.map(({view, index}) => makePanel(view, index));
    ["2d", "3d"].forEach(kind => { const exists = panels.some(panel => panel.kind === kind); el(`mode-${kind}`).disabled = !exists; el(`mode-${kind}`).onclick = () => selectMode(kind); });
    el("mode-help").textContent = panels.some(panel => panel.kind === "3d") ? "Switch views while keeping the same simulation time." : "3D flow is unavailable in this bundle. Request a 3D render to enable it.";
    el("mode-3d").title = panels.some(panel => panel.kind === "3d") ? "Show three-dimensional flow" : "No 3D view was generated for this bundle";
    try {
      const data = await loadJSON(manifest.probes || "probes.json"); charts = new window.ForwardCharts(el("probes"), data);
      el("probe-description").textContent = `${(data.field || "velocity").replaceAll("_", " ")} [${data.units || "m/s"}] · grouped by actual sample height · simulation values, not observations`;
    } catch (error) { el("probe-description").textContent = `Probe charts unavailable: ${error.message}`; }
    physical = Number.isFinite(manifest.time_range?.[0]) ? manifest.time_range[0] : panels[0].frames[0].simulation_time;
    el("play").onclick = () => { if (playing) { stopAnimation(); updateControls(); } else startAnimation(); };
    el("restart").onclick = () => { if (primary) seek(primary.frames[0].simulation_time); };
    el("seek").oninput = () => seek(Number(el("seek").value));
    el("rate").onchange = () => { lastTick = null; };
    el("fullscreen").onclick = async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await cinema.requestFullscreen?.(); } catch (error) { el("status").textContent = error.message; } };
    document.addEventListener("visibilitychange", () => { if (document.hidden) { stopAnimation(); updateControls(); } });
    selectMode(panels.some(panel => panel.kind === "2d") ? "2d" : "3d");
    cinema.dataset.ready = "true";
  } catch (error) { stopAnimation(); el("status").textContent = `Cannot load visualization: ${error.message}. Serve this bundle through a local HTTP server.`; }
})();
