"use strict";
// Snapshots are held, so the cursor reports the actual source sample time.
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
    for (const frame of frames) { if (frame.simulation_time > simulationTime) break; selected = frame.video_time; }
    return selected;
  },
  clock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const whole = Math.floor(seconds), hours = Math.floor(whole / 3600);
    return `${hours ? `${hours}:` : ""}${hours ? String(Math.floor(whole / 60) % 60).padStart(2, "0") : Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }
};
if (typeof module !== "undefined") module.exports = ForwardTimeline;
if (typeof document !== "undefined") (async function () {
  const el = id => document.getElementById(id);
  const video = el("video"), status = el("status");
  let view, charts, token = 0, physical = 0;
  const asset = path => {
    if (typeof path !== "string" || path.startsWith("/") || path.includes("..") || path.includes(":") || path.includes("\\")) throw new Error("Invalid bundle asset path");
    return path;
  };
  const loadJSON = async path => {
    const response = await fetch(asset(path));
    if (!response.ok) throw new Error(`Missing bundle asset: ${path}`);
    return response.json();
  };
  function update(seconds = video.currentTime) {
    if (!view) return;
    physical = ForwardTimeline.physicalTime(view.frames, seconds);
    el("physical").textContent = `Simulation time: ${physical.toFixed(3)} s`;
    charts?.update(physical);
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    el("seek").max = String(duration);
    el("seek").value = String(Math.min(duration, seconds || 0));
    el("clock").textContent = `${ForwardTimeline.clock(seconds)} / ${ForwardTimeline.clock(duration)}`;
    el("play").textContent = video.paused ? "Play" : "Pause";
  }
  function still(message) {
    video.pause(); video.hidden = true; el("poster").hidden = false; el("playback").hidden = true;
    el("download").href = asset(view.poster); el("download").textContent = "Download PNG";
    status.textContent = message;
    update(0);
  }
  function selectView(next) {
    const previousPhysical = physical;
    const thisToken = ++token;
    video.pause(); video.removeAttribute("src"); video.load();
    view = next;
    el("poster").src = asset(view.poster);
    video.poster = asset(view.poster);
    el("caption").textContent = `${view.label} · ${view.field} [${view.units}]`;
    el("download").hidden = false;
    if (!view.media) { still("Still preview · no movie available for this view"); return; }
    video.hidden = false; el("poster").hidden = true; el("playback").hidden = true;
    status.textContent = "Loading movie…";
    el("download").href = asset(view.media); el("download").textContent = "Download MP4";
    const expected = new URL(asset(view.media), document.baseURI).href;
    video.onloadedmetadata = () => {
      if (token !== thisToken || video.currentSrc !== expected) return;
      if (!Number.isFinite(video.duration) || video.duration <= 0) { still("Movie duration unavailable; showing preview"); return; }
      video.currentTime = Math.min(video.duration, Math.max(0, ForwardTimeline.videoTime(view.frames, previousPhysical)));
      video.playbackRate = Number(el("rate").value);
      video.controls = false; el("playback").hidden = false; status.textContent = "Ready"; update();
    };
    video.onerror = () => { if (token === thisToken && video.currentSrc === expected) still("Movie could not be played; showing PNG preview"); };
    video.src = asset(view.media);
  }
  try {
    const manifest = await loadJSON("viewer_manifest.json");
    if (manifest.version !== 1 || !manifest.views?.length) throw new Error("Unsupported or empty viewer bundle");
    const probes = await loadJSON(manifest.probes);
    charts = new window.ForwardCharts(el("probes"), probes);
    el("title").textContent = `${manifest.case} · Urban wind`;
    const selection = manifest.selection.reduction || (manifest.selection.member !== null ? `member ${manifest.selection.member}` : "single simulation");
    el("selection").textContent = `${manifest.backend} · ${manifest.prediction_type} · ${selection}`;
    el("processing").textContent = `${manifest.processing.collocation}. ${manifest.processing.temporal}.`;
    manifest.warnings.forEach(warning => { const item = document.createElement("li"); item.textContent = warning; el("warnings").append(item); });
    manifest.views.forEach((item, index) => { const option = document.createElement("option"); option.value = String(index); option.textContent = item.label; el("view").append(option); });
    el("view").disabled = false;
    el("view").onchange = () => selectView(manifest.views[Number(el("view").value)]);
    el("play").onclick = async () => { if (video.paused) { try { await video.play(); } catch (error) { status.textContent = error.message; } } else video.pause(); };
    el("restart").onclick = () => { video.currentTime = 0; update(0); };
    el("seek").oninput = () => { if (Number.isFinite(video.duration)) { video.currentTime = Math.max(0, Math.min(video.duration, Number(el("seek").value))); update(); } };
    el("rate").onchange = () => { video.playbackRate = Number(el("rate").value); };
    el("fullscreen").onclick = () => document.querySelector(".cinema").requestFullscreen?.().catch(error => { status.textContent = error.message; });
    ["timeupdate", "seeked", "pause", "play", "ended"].forEach(event => video.addEventListener(event, () => update()));
    if (video.requestVideoFrameCallback) {
      const tick = (_, metadata) => { update(metadata.mediaTime); video.requestVideoFrameCallback(tick); };
      video.requestVideoFrameCallback(tick);
    }
    selectView(manifest.views[0]);
  } catch (error) { status.textContent = `Cannot load visualization: ${error.message}. Serve this bundle through a local HTTP server.`; }
})();
