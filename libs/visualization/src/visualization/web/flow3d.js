// Interactive WebGL view of a saved 3D velocity field (scene.json from the exporter).
// The shell owns loading, the frame cache, renderer, lights, buildings, camera
// controls and overlays; flow3d_slice.js and flow3d_volume.js draw the field.
import * as THREE from "./vendor/three.module.min.js";
import {createSliceMode} from "./flow3d_slice.js";
import {createVolumeMode} from "./flow3d_volume.js";

const CROSSING = 8;   // a fast tracer crosses the domain in no less than this many wall seconds
// Sampled bottom to top from the reference poster's colorbar (1920×1080 frame), smoothed, 17 even stops.
const STOPS = ["#100b11", "#170e2e", "#1e1149", "#261466", "#35198e", "#431fb8", "#5124e3", "#6533ef", "#7e48f0", "#965ef0", "#ab72e2", "#be85bc", "#d09798", "#e1ab75", "#e5c158", "#e8d634", "#e8eb11"].map((hex, i, all) => [i / (all.length - 1), hex]);
const STOP_RGB = STOPS.map(([t, hex]) => [t, [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)]);
// The shared speed palette: dark → indigo/violet → orange → yellow, sRGB components in [0, 1].
function color(t) {
  t = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  let i = 1;
  while (i < STOP_RGB.length - 1 && STOP_RGB[i][0] < t) i++;
  const [t0, c0] = STOP_RGB[i - 1], [t1, c1] = STOP_RGB[i], f = (t - t0) / (t1 - t0);
  return c0.map((c, j) => c + (c1[j] - c) * f);
}
function colormapTexture() {
  const data = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) { color(i / 255).forEach((c, j) => { data[i * 4 + j] = Math.round(c * 255); }); data[i * 4 + 3] = 255; }
  const texture = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace; texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping; texture.needsUpdate = true;
  return texture;
}
const relative = path => {
  if (typeof path !== "string" || !path || path.startsWith("/") || path.includes("..") || path.includes(":") || path.includes("\\")) throw new Error("Invalid scene asset path");
  return path;
};
async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Missing scene asset (${response.status})`);
  return response.arrayBuffer();
}
const node = (tag, className, text) => { const element = document.createElement(tag); if (className) element.className = className; if (text !== undefined) element.textContent = text; return element; };
const formatTime = seconds => {
  const minutes = Math.floor(Math.max(0, seconds) / 60), rest = Math.max(0, seconds) - minutes * 60;
  return `t = ${minutes} min ${rest.toFixed(1).padStart(4, "0")} s`;
};
// The exact ends plus round values in between that keep clear of them.
const colorTicks = (lo, hi) => {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / 5, step = [1, 2, 2.5, 5, 10].map(m => m * 10 ** Math.floor(Math.log10(raw))).find(s => s >= raw) || raw;
  const ticks = [lo];
  for (let v = Math.ceil(lo / step) * step; v < hi; v += step) if (Math.min(v - lo, hi - v) > .1 * (hi - lo)) ticks.push(+v.toFixed(6));
  return [...ticks, hi];
};
// Fractional index of p on increasing, possibly nonuniform cell centres c, clamped to [0, n - 1].
function locator(c) {
  const n = c.length, d = (c[n - 1] - c[0]) / Math.max(1, n - 1);
  if (n < 2) return () => 0;
  if (c.every((v, i) => Math.abs(v - c[0] - i * d) <= 1e-4 * d)) return p => Math.min(n - 1, Math.max(0, (p - c[0]) / d));
  return p => {
    if (p <= c[0]) return 0; if (p >= c[n - 1]) return n - 1;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c[m] <= p) lo = m; else hi = m; }
    return lo + (p - c[lo]) / (c[hi] - c[lo]);
  };
}
// Cell faces: midpoints between centres, the outer ones half a spacing beyond the end centres.
const faces = c => Float64Array.from({length: c.length + 1}, (_, i) => c.length < 2 ? c[0] + i - .5 : i === 0 ? c[0] - (c[1] - c[0]) / 2 : i === c.length ? c.at(-1) + (c.at(-1) - c.at(-2)) / 2 : (c[i - 1] + c[i]) / 2);

// Decoded saved states, a small LRU (each city state is ~3 MB as Float32). States that fail to load are not retried.
function frameStore(scene, base, onChange, capacity = 8) {
  const n = scene.x.length * scene.y.length * scene.z.length, times = scene.times.map(Number);
  const cache = new Map(), pending = new Map(), queue = [], failed = new Set();
  let lastIndex = 0, lost = null;
  const pump = () => {
    while (pending.size < 2 && queue.length) {
      const i = queue.shift();
      if (cache.has(i) || pending.has(i)) continue;
      pending.set(i, fetchBuffer(new URL(relative(scene.velocity[i]), base)).then(buffer => {
        const raw = new Int16Array(buffer), scale = Number(scene.velocity_scale[i]), values = new Float32Array(3 * n);
        if (raw.length !== 3 * n) throw new Error("Velocity file has the wrong size");
        for (let j = 0; j < values.length; j++) values[j] = raw[j] * scale;
        cache.set(i, {index: i, time: times[i], u: values.subarray(0, n), v: values.subarray(n, 2 * n), w: values.subarray(2 * n)});
        while (cache.size > capacity) {
          let evict = null, far = -1;
          for (const key of cache.keys()) { const d = Math.abs(key - lastIndex) + (key < lastIndex ? .5 : 0); if (d > far) { far = d; evict = key; } }
          cache.delete(evict);
        }
      }).catch(error => { failed.add(i); console.warn(`flow3d: saved state ${i} unavailable`, error); }).finally(() => { pending.delete(i); onChange(); pump(); }));
    }
  };
  const want = indices => { indices.filter(i => i >= 0 && i < times.length && !cache.has(i) && !pending.has(i) && !failed.has(i) && !queue.includes(i)).forEach(i => queue.push(i)); pump(); };
  const touch = i => { const value = cache.get(i); if (value) { cache.delete(i); cache.set(i, value); } return value; };
  return {
    // Saved time of a bracketing state that could not be loaded at the last frame() call, else null.
    get lost() { return lost; },
    frame(simTime) {
      if (!times.length) return null;
      let i = 0;
      while (i < times.length - 1 && times[i + 1] <= simTime) i++;
      const j = Math.min(times.length - 1, i + 1);
      const alpha = j === i || simTime <= times[i] ? 0 : Math.min(1, (simTime - times[i]) / (times[j] - times[i]));
      lastIndex = i; lost = failed.has(i) ? times[i] : alpha > 0 && failed.has(j) ? times[j] : null;
      queue.length = 0; want([i, j, j + 1, j + 2, j + 3]);
      const a = touch(i), b = touch(j);
      if (a && b) return {a, b, alpha};
      if (a || b) return {a: a || b, b: a || b, alpha: 0};
      // Hold the nearest decoded state while the bracketing ones load.
      let near = null;
      for (const value of cache.values()) if (!near || Math.abs(value.time - simTime) < Math.abs(near.time - simTime)) near = value;
      return near ? {a: near, b: near, alpha: 0} : null;
    },
  };
}

// Orbit around a target with z up; left drag rotates, right/shift drag or two fingers pan, wheel/pinch zoom.
function orbitControls(camera, element, onChange) {
  const target = new THREE.Vector3(), pointers = new Map();
  let radius = 1, azimuth = 0, polar = 1, home = null, scale = 1, floor = 0, gesture = null, moved = false;
  const apply = () => {
    const maxPolar = Math.min(Math.PI * .495, Math.acos(Math.max(-1, Math.min(1, (floor + .02 * scale - target.z) / radius))));
    polar = Math.max(.02, Math.min(maxPolar, polar));
    camera.position.set(target.x + radius * Math.sin(polar) * Math.cos(azimuth), target.y + radius * Math.sin(polar) * Math.sin(azimuth), target.z + radius * Math.cos(polar));
    camera.lookAt(target); onChange();
  };
  const set = (position, focus) => {
    target.copy(focus);
    const offset = position.clone().sub(focus);
    radius = Math.max(1e-3, offset.length()); azimuth = Math.atan2(offset.y, offset.x); polar = Math.acos(offset.z / radius);
    apply();
  };
  const pan = (dx, dy) => {
    const height = element.clientHeight || 1, perPixel = 2 * radius * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / height;
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
    target.addScaledVector(right, -dx * perPixel).addScaledVector(up, dy * perPixel); apply();
  };
  const zoom = factor => { radius = Math.max(.03 * scale, Math.min(6 * scale, radius * factor)); apply(); };
  const spread = () => { const [a, b] = [...pointers.values()]; return {d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2}; };
  element.addEventListener("pointerdown", event => {
    if (event.pointerType === "mouse" && event.button > 2) return;
    moved = true;
    element.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
    gesture = pointers.size > 1 ? {kind: "pinch", ...spread()} : {kind: event.button === 2 || event.button === 1 || event.shiftKey ? "pan" : "rotate"};
  });
  element.addEventListener("pointermove", event => {
    const last = pointers.get(event.pointerId);
    if (!last || !gesture) return;
    const dx = event.clientX - last.x, dy = event.clientY - last.y;
    last.x = event.clientX; last.y = event.clientY;
    if (gesture.kind === "pinch" && pointers.size > 1) {
      const now = spread();
      if (now.d > 0 && gesture.d > 0) zoom(gesture.d / now.d);
      pan(now.x - gesture.x, now.y - gesture.y);
      Object.assign(gesture, now);
    } else if (gesture.kind === "pan") pan(dx, dy);
    else if (gesture.kind === "rotate") {
      const k = 1.2 * Math.PI / Math.max(300, element.clientHeight || 300);
      azimuth -= dx * k; polar -= dy * k; apply();
    }
  });
  const release = event => {
    pointers.delete(event.pointerId);
    gesture = pointers.size === 1 ? {kind: "rotate"} : pointers.size ? gesture : null;
  };
  element.addEventListener("pointerup", release);
  element.addEventListener("pointercancel", release);
  element.addEventListener("wheel", event => { event.preventDefault(); moved = true; zoom(Math.exp(event.deltaY * (event.deltaMode ? .05 : .0015))); }, {passive: false});
  element.addEventListener("contextmenu", event => event.preventDefault());
  element.addEventListener("dblclick", () => { moved = false; reset(); });
  const reset = () => { if (home) { const view = home(); set(view.position, view.target); } };
  return {
    target,
    // home() -> {position, target}, evaluated on each reset so it fits the current aspect.
    setHome(view, size, ground) { scale = size; floor = ground; home = view; reset(); },
    reset,
    get moved() { return moved; },
  };
}

// host: an empty positioned element (the panel's visual area). Returns a handle the viewer drives.
export function createFlow3D(host, sceneUrl, {onReady = () => {}, onError = () => {}} = {}) {
  host.classList.add("flow3d");
  const message = node("p", "flow3d-message", "Loading the 3D flow…");
  host.append(message);
  let renderer;
  try { renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: false, powerPreference: "high-performance"}); }
  catch (error) {
    message.textContent = "The interactive 3D view needs WebGL, which this browser has disabled or does not support. The 2D slices remain available.";
    message.dataset.error = "true"; onError(error);
    return {setTime() {}, setActive() {}};
  }
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x0d0e12, 1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.shadowMap.autoUpdate = false;
  renderer.localClippingEnabled = true;
  const canvas = renderer.domElement; canvas.className = "flow3d-canvas";
  canvas.setAttribute("aria-label", "Interactive 3D flow: drag to rotate, scroll or pinch to zoom, right- or shift-drag to pan, double-click to reset");
  host.prepend(canvas);

  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(32, 1.6, .1, 1e5);
  camera.up.set(0, 0, 1);
  const light = new THREE.DirectionalLight(0xffffff, 2.4);
  scene.add(new THREE.HemisphereLight(0xc6cbe0, 0x26242a, 1.25), light, light.target);

  // Overlays in the reference style.
  const overlay = node("div", "flow3d-overlay");
  const clock = node("p", "flow3d-time", formatTime(0)), subtitle = node("p", "flow3d-subtitle", ""), notice = node("p", "flow3d-notice", "");
  const bar = node("div", "flow3d-colorbar"), barTitle = node("span", "flow3d-colorbar-title", "wind speed (m/s)");
  const ramp = node("canvas", "flow3d-ramp"), ticks = node("div", "flow3d-ticks");
  ramp.width = 1; ramp.height = 256;
  {
    const g = ramp.getContext("2d"), img = g.createImageData(1, 256);
    for (let i = 0; i < 256; i++) { color(1 - i / 255).forEach((c, j) => { img.data[i * 4 + j] = Math.round(c * 255); }); img.data[i * 4 + 3] = 255; }
    g.putImageData(img, 0, 0);
  }
  const rampBox = node("div", "flow3d-ramp-box"); rampBox.append(ramp, ticks); bar.append(barTitle, rampBox);
  const controls = node("div", "flow3d-controls");
  const toggle = node("div", "flow3d-toggle"); toggle.setAttribute("role", "group"); toggle.setAttribute("aria-label", "3D display");
  const sliceButton = node("button", "", "Height slice"), volumeButton = node("button", "", "Full 3D");
  [sliceButton, volumeButton].forEach(button => { button.type = "button"; toggle.append(button); });
  const levelLabel = node("label", "flow3d-level"), levelText = node("span", "", "z"), level = node("input");
  level.type = "range"; level.min = "0"; level.step = "1"; level.setAttribute("aria-label", "Slice height");
  levelLabel.append(level, levelText);
  const save = node("button", "flow3d-save", "Save image"); save.type = "button";
  const hint = node("p", "flow3d-hint", "drag to rotate · scroll to zoom · right-drag to pan · double-click to reset");
  controls.append(toggle, levelLabel, save);
  overlay.append(clock, notice, subtitle, bar, controls, hint);
  host.append(overlay);

  let ctx = null, modes = {}, modeName = "slice", active = false, dirty = true, raf = null, simTime = 0, lastSim = null, store = null, catcher = null;
  const requestRender = () => { dirty = true; schedule(); };
  const controller = orbitControls(camera, canvas, requestRender);
  function schedule() { if (active && raf === null) raf = requestAnimationFrame(loop); }
  function resize() {
    const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
    if (!controller.moved) controller.reset();
    requestRender();
  }
  function loop() {
    raf = null;
    if (!active || !dirty || !ctx) return;
    dirty = false;
    const dt = lastSim === null ? 0 : simTime - lastSim;
    lastSim = simTime;
    Object.values(modes).forEach(mode => mode.update(simTime, dt));
    notice.textContent = store.lost === null ? "" : `Could not load saved state t = ${+store.lost.toFixed(2)} s`;
    // A mode may settle its colour range or label only once its data is decoded.
    const mode = modes[modeName];
    if (String(mode.range || ctx.speedLimits) !== bar.dataset.range || (mode.label && !subtitle.textContent.startsWith(mode.label))) refreshLabels(false);
    renderer.render(scene, camera);
  }
  function setRange() {
    const [lo, hi] = modes[modeName].range || ctx.speedLimits;
    ticks.replaceChildren(...colorTicks(lo, hi).map(value => { const tick = node("span", "", String(+value.toPrecision(3))); tick.style.bottom = `${(value - lo) / (hi - lo || 1) * 100}%`; return tick; }));
    bar.dataset.range = `${lo},${hi}`;
  }
  function refreshLabels(render = true) {
    const z = ctx.grid.z[Number(level.value)];
    levelText.textContent = `z = ${z.toFixed(1)} m`;
    subtitle.textContent = `${modes[modeName].label} · flow left to right`;
    catcher.visible = modeName === "slice"; catcher.position.z = z + .02 * (ctx.grid.bounds[5] - ctx.grid.bounds[4]) / ctx.grid.nz;
    setRange(); if (render) requestRender();
  }
  function selectMode(name) {
    modeName = name;
    Object.entries(modes).forEach(([key, mode]) => mode.setVisible(key === name));
    sliceButton.setAttribute("aria-pressed", String(name === "slice")); volumeButton.setAttribute("aria-pressed", String(name === "volume"));
    sliceButton.classList.toggle("active", name === "slice"); volumeButton.classList.toggle("active", name === "volume");
    levelLabel.hidden = name !== "slice";
    host.dataset.mode = name; lastSim = null; refreshLabels();
  }
  sliceButton.onclick = () => selectMode("slice");
  volumeButton.onclick = () => selectMode("volume");
  level.oninput = () => { modes.slice.setLevel(Number(level.value)); refreshLabels(); };
  save.onclick = () => {
    renderer.render(scene, camera);
    const out = document.createElement("canvas"), g = out.getContext("2d"), ratio = canvas.width / Math.max(1, host.clientWidth);
    out.width = canvas.width; out.height = canvas.height; g.drawImage(canvas, 0, 0);
    g.scale(ratio, ratio);
    g.fillStyle = "#f2f2f2"; g.font = "13px ui-monospace, Menlo, monospace"; g.fillText(clock.textContent, 16, 26);
    g.fillStyle = "#c9c9cf"; g.font = "12px Arial, sans-serif"; g.textAlign = "right"; g.fillText(subtitle.textContent, host.clientWidth - 16, 26);
    const box = rampBox.getBoundingClientRect(), origin = host.getBoundingClientRect(), x = box.left - origin.left, y = box.top - origin.top;
    g.drawImage(ramp, x, y, 8, box.height);
    g.textAlign = "left"; g.font = "11px Arial, sans-serif"; g.fillStyle = "#c9c9cf";
    [...ticks.children].forEach(tick => { const t = parseFloat(tick.style.bottom) / 100; g.fillText(tick.textContent, x + 13, y + box.height * (1 - t) + 4); });
    g.textAlign = "right"; g.fillText(barTitle.textContent, x + 24, y - 8);
    out.toBlob(blob => {
      if (!blob) return;
      const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
      link.download = `flow3d_${modeName}_t${simTime.toFixed(1)}s.png`; link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, "image/png");
  };

  new ResizeObserver(resize).observe(host);
  let onScreen = false, wanted = false;
  const update = () => { active = wanted && onScreen && !document.hidden; if (active) { dirty = true; schedule(); } };
  new IntersectionObserver(entries => { onScreen = entries.at(-1).isIntersecting; update(); }).observe(host);
  document.addEventListener("visibilitychange", update);

  (async () => {
    const base = new URL(sceneUrl, document.baseURI), meta = await (await fetch(base)).json();
    if (meta.version !== 1 || !meta.x?.length || !meta.times?.length || meta.velocity?.length !== meta.times.length) throw new Error("Unsupported 3D scene");
    const [fluidBuffer, buildingBuffer] = await Promise.all([fetchBuffer(new URL(relative(meta.fluid), base)), meta.buildings ? fetchBuffer(new URL(relative(meta.buildings), base)) : null]);
    const x = Float32Array.from(meta.x), y = Float32Array.from(meta.y), z = Float32Array.from(meta.z);
    const xf = faces(x), yf = faces(y), zf = faces(z), bounds = [xf[0], xf.at(-1), yf[0], yf.at(-1), zf[0], zf.at(-1)];
    const [x0, x1, y0, y1, z0, z1] = bounds, size = Math.max(x1 - x0, y1 - y0);
    const fluid = new Uint8Array(fluidBuffer);
    if (fluid.length !== x.length * y.length * z.length) throw new Error("Fluid mask has the wrong size");

    // Buildings, clipped to the domain so neighbouring geometry does not stick out.
    let top = z0;
    const clip = [new THREE.Plane(new THREE.Vector3(1, 0, 0), -x0), new THREE.Plane(new THREE.Vector3(-1, 0, 0), x1), new THREE.Plane(new THREE.Vector3(0, 1, 0), -y0), new THREE.Plane(new THREE.Vector3(0, -1, 0), y1)];
    if (buildingBuffer && buildingBuffer.byteLength >= 36) {
      // Drop ground-plate triangles (all three corners on the floor); the shell draws its own ground.
      const all = new Float32Array(buildingBuffer), keep = [], floorZ = z0 + 1e-3 * size;
      for (let t = 0; t + 9 <= all.length; t += 9) if (!(all[t + 2] <= floorZ && all[t + 5] <= floorZ && all[t + 8] <= floorZ)) keep.push(t);
      const soup = new Float32Array(keep.length * 9), geometry = new THREE.BufferGeometry();
      keep.forEach((t, i) => soup.set(all.subarray(t, t + 9), i * 9));
      geometry.setAttribute("position", new THREE.BufferAttribute(soup, 3)); geometry.computeVertexNormals();
      for (let i = 2; i < soup.length; i += 3) { const px = soup[i - 2], py = soup[i - 1]; if (px >= x0 && px <= x1 && py >= y0 && py <= y1) top = Math.max(top, soup[i]); }
      const buildings = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: 0x8d8e93, roughness: .88, metalness: 0, flatShading: true, clippingPlanes: clip}));
      buildings.castShadow = buildings.receiveShadow = true;
      // Back faces only show where the domain clips a building: draw them as a flat cap.
      const caps = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0x6d6e73, side: THREE.BackSide, clippingPlanes: clip}));
      scene.add(buildings, caps);
    }
    // An unlit near-black floor (like the reference's) with the building shadows laid over it.
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), new THREE.MeshBasicMaterial({color: 0x08090c}));
    const groundShadow = new THREE.Mesh(ground.geometry, new THREE.ShadowMaterial({opacity: .6, depthWrite: false}));
    ground.position.set((x0 + x1) / 2, (y0 + y1) / 2, z0 - .001 * size); groundShadow.position.copy(ground.position).setZ(z0);
    groundShadow.receiveShadow = true; groundShadow.renderOrder = .5; scene.add(ground, groundShadow);
    catcher = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), new THREE.ShadowMaterial({opacity: .6, depthWrite: false}));
    catcher.position.set((x0 + x1) / 2, (y0 + y1) / 2, z0); catcher.receiveShadow = true; catcher.renderOrder = 1.5; scene.add(catcher);

    // Key light from upstream-left and above; the shadow camera covers the whole domain.
    const centre = new THREE.Vector3((x0 + x1) / 2, (y0 + y1) / 2, z0);
    const radius = .5 * Math.hypot(x1 - x0, y1 - y0, Math.max(top, z0 + 1) - z0) * 1.05;
    light.target.position.copy(centre);
    light.position.copy(centre).add(new THREE.Vector3(-1, -.45, .9).normalize().multiplyScalar(3 * radius));
    Object.assign(light.shadow.camera, {left: -radius, right: radius, top: radius, bottom: -radius, near: radius, far: 5 * radius});
    light.shadow.camera.updateProjectionMatrix();
    const mapSize = Math.min(4096, renderer.capabilities.maxTextureSize);
    light.shadow.mapSize.set(mapSize, mapSize); light.shadow.radius = 3; light.shadow.bias = -4e-4; light.shadow.normalBias = 3 * radius / mapSize;
    light.castShadow = true; renderer.shadowMap.needsUpdate = true;
    camera.near = size / 2000; camera.far = size * 20; camera.updateProjectionMatrix();

    // Default camera: scene.camera, else upstream-left and above with +x running left to right.
    const span = Math.min(z1 - z0, Math.max(1.4 * (top - z0), .2 * size));
    const focus = new THREE.Vector3(centre.x, centre.y, z0 + .15 * span);
    let home = null;
    if (Array.isArray(meta.camera) && meta.camera.length >= 2) {
      const [position, focal] = meta.camera.map(p => new THREE.Vector3(...p.map(Number)));
      if ([position, focal].every(p => Number.isFinite(p.x + p.y + p.z)) && position.distanceTo(focal) > 0) home = () => ({position, target: focal});
    }
    if (!home) {
      // Move along the view direction until the footprint box (up to the building tops) fills the view.
      const corners = [x0, x1].flatMap(cx => [y0, y1].flatMap(cy => [z0, z0 + span].map(cz => new THREE.Vector3(cx, cy, cz))));
      const direction = new THREE.Vector3(-.28, -1, .62).normalize(), probe = new THREE.PerspectiveCamera(camera.fov, 1, camera.near, camera.far);
      probe.up.set(0, 0, 1);
      home = () => {
        let distance = 1.5 * size;
        probe.aspect = camera.aspect; probe.updateProjectionMatrix();
        for (let i = 0; i < 12; i++) {
          probe.position.copy(focus).addScaledVector(direction, distance); probe.lookAt(focus); probe.updateMatrixWorld();
          const extent = Math.max(...corners.map(c => { const p = c.clone().project(probe); return Math.max(Math.abs(p.x), Math.abs(p.y)); }));
          distance *= Math.pow(extent / .96, .8);
        }
        return {position: focus.clone().addScaledVector(direction, distance), target: focus.clone()};
      };
    }
    resize(); controller.setHome(home, size, z0);

    store = frameStore(meta, base, requestRender);
    // pace(dt, speed): a tracer step of dt simulated seconds, capped so that air at speed crosses the domain in no less than CROSSING wall seconds.
    const pacer = () => {
      let last = null;
      return (dt, speed) => { const now = performance.now(), wall = last === null ? 0 : Math.min(.1, (now - last) / 1000); last = now; return Math.min(dt, wall * size / speed / CROSSING); };
    };
    ctx = {
      THREE, scene, camera, renderer, locator, pacer,
      grid: {x, y, z, nx: x.length, ny: y.length, nz: z.length, bounds, xf, yf, zf, fx: locator(x), fy: locator(y), fz: locator(z)},
      fluid, speedLimits: (meta.speed_limits || [0, 1]).map(Number), colormap: colormapTexture(),
      frame: t => store.frame(t), requestRender, buildingTop: top,
      levelSpeedLimits: Array.isArray(meta.level_speed_limits) ? meta.level_speed_limits.map(v => Array.isArray(v) ? v.map(Number) : null) : undefined,
    };
    level.max = String(z.length - 1);
    modes = {slice: createSliceMode(ctx), volume: createVolumeMode(ctx)};
    level.value = String(modes.slice.level);
    store.frame(simTime);
    message.remove();
    selectMode("slice");
    host.dataset.ready = "true"; onReady();
  })().catch(error => { message.textContent = `3D flow unavailable: ${error.message}`; message.dataset.error = "true"; host.dataset.ready = "error"; onError(error); });

  return {
    setTime(t) { if (Number.isFinite(t) && t !== simTime) { simTime = t; clock.textContent = formatTime(t); requestRender(); } },
    setActive(on) { wanted = on; update(); },
  };
}
