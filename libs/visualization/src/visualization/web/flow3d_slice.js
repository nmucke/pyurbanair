// Height slice: a glowing horizontal wind-speed sheet at one z level with tracer streaks drifting over it,
// optionally crossed by a vertical x–z sheet at one y row.
const TRACERS = 2200, CROSS_TRACERS = 700, TRAIL = 6, TRAIL_FRACTION = 0.02, BRIGHTNESS = 0.55, PERCENTILE = 0.99, LIFE_FRACTION = 0.25, DEFAULT_HEIGHT = 9;
const STREAK = [1, 0.9, 0.55], FULL_AREA = 1600 * 900;

const random = seed => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const niceCeil = x => { const e = 10 ** Math.floor(Math.log10(x)), step = e * (x / e < 2 ? 0.1 : 0.5); return +(Math.ceil(x / step - 1e-9) * step).toFixed(6); };

// A speed sheet with tracer streaks on the grid plane of axes a, b (the group's local x, y), held at index m
// of the third axis c, which set(m) moves the group along. ground: b is height, so its lower edge does not fade.
function createSheet(ctx, a, b, tracers, seed, ground = false) {
  const {THREE, grid, fluid} = ctx, c = "xyz".replace(a, "").replace(b, ""), stride = {x: 1, y: grid.nx, z: grid.nx * grid.ny};
  const na = grid[`n${a}`], nb = grid[`n${b}`], plane = na * nb, fa = grid[`f${a}`], fb = grid[`f${b}`], af = grid[`${a}f`], bf = grid[`${b}f`];
  const [x0, x1, y0, y1] = grid.bounds, a0 = af[0], a1 = af.at(-1), b0 = bf[0], b1 = bf.at(-1);
  const L = Math.max(x1 - x0, y1 - y0), pace = ctx.pacer();
  const fade = 0.06 * L, e0 = ground ? b0 - fade : b0;
  const minCell = Math.min(...[af, bf].flatMap(f => Array.from(f.subarray(1), (v, i) => v - f[i])));
  const [ua, ub] = [a, b].map(axis => ({x: "u", y: "v", z: "w"})[axis]);
  const group = new THREE.Group();
  group.visible = false;
  ctx.scene.add(group);

  // Sheet: vertices at the cell centres plus the outer faces, so nonuniform grids interpolate piecewise linearly.
  const texture = (data, type) => {
    const t = new THREE.DataTexture(data, na, nb, THREE.RedFormat, type);
    t.magFilter = t.minFilter = THREE.LinearFilter;
    t.unpackAlignment = 1;
    t.needsUpdate = true;
    return t;
  };
  const speedA = new Uint16Array(plane), speedB = new Uint16Array(plane), mask = new Uint8Array(plane), flat = new Int32Array(plane);
  const texA = texture(speedA, THREE.HalfFloatType), texB = texture(speedB, THREE.HalfFloatType), texM = texture(mask, THREE.UnsignedByteType);
  const vx = [a0, ...grid[a], a1], vy = [b0, ...grid[b], b1], cols = vx.length, rows = vy.length;
  const position = new Float32Array(cols * rows * 3), uv = new Float32Array(cols * rows * 2), index = [];
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const p = j * cols + i;
    position.set([vx[i], vy[j], 0], p * 3);
    uv.set([(Math.min(Math.max(i - 1, 0), na - 1) + 0.5) / na, (Math.min(Math.max(j - 1, 0), nb - 1) + 0.5) / nb], p * 2);
    if (i && j) index.push(p - cols - 1, p - cols, p, p - cols - 1, p, p - 1);
  }
  const sheetGeometry = new THREE.BufferGeometry();
  sheetGeometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  sheetGeometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  sheetGeometry.setIndex(index);
  const uniforms = {
    speedA: {value: texA}, speedB: {value: texB}, mask: {value: texM}, colormap: {value: ctx.colormap},
    alpha: {value: 0}, range: {value: new THREE.Vector2(...ctx.speedLimits)},
    rect: {value: new THREE.Vector4(a0, a1, e0, b1)}, fade: {value: fade},
  };
  const vertexShader = `#include <clipping_planes_pars_vertex>
    varying vec2 vUv; varying vec2 vXY;
    void main() {
      vUv = uv; vXY = position.xy; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mvPosition;
      #include <clipping_planes_vertex>
    }`;
  const fragmentShader = `#include <clipping_planes_pars_fragment>
    uniform sampler2D speedA, speedB, mask, colormap; uniform float alpha, fade; uniform vec2 range; uniform vec4 rect;
    varying vec2 vUv; varying vec2 vXY;
    void main() {
      #include <clipping_planes_fragment>
      float s = mix(texture2D(speedA, vUv).r, texture2D(speedB, vUv).r, alpha);
      float t = clamp((s - range.x) / (range.y - range.x), 0.0, 1.0);
      vec3 c = texture2D(colormap, vec2(t * 255.0 / 256.0 + 0.5 / 256.0, 0.5)).rgb * (1.0 + 0.35 * t * t);
      vec2 d = min(vXY - rect.xz, rect.yw - vXY);
      float a = smoothstep(0.02, 0.38, t) * smoothstep(0.35, 0.75, texture2D(mask, vUv).r) * smoothstep(0.0, fade, min(d.x, d.y));
      gl_FragColor = vec4(c, 0.96 * a);
      #include <colorspace_fragment>
    }`;

  // Tracers: TRAIL + 1 segments each, from the head back to an interpolated tail TRAIL record intervals ago.
  const R = TRAIL + 1, SEG = TRAIL + 1;
  const px = new Float32Array(tracers), py = new Float32Array(tracers), age = new Float32Array(tracers), life = new Float32Array(tracers);
  const hist = new Float32Array(tracers * R * 2), count = new Uint8Array(tracers), pts = new Float32Array((R + 1) * 2);
  const linePos = new Float32Array(tracers * SEG * 6), lineCol = new Float32Array(tracers * SEG * 6);
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage));
  lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineCol, 3).setUsage(THREE.DynamicDrawUsage));
  const streak = new THREE.Color().setRGB(...STREAK, THREE.SRGBColorSpace);
  // A sheet and its streaks, optionally clipped (world-space planes), sharing the data above: -> [sheet, lines].
  const piece = (clippingPlanes = []) => {
    const sheet = new THREE.Mesh(sheetGeometry, new THREE.ShaderMaterial({
      uniforms, vertexShader, fragmentShader, clipping: true, clippingPlanes, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    }));
    const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, clippingPlanes,
    }));
    sheet.frustumCulled = lines.frustumCulled = false;
    group.add(sheet, lines);
    return [sheet, lines];
  };

  const scratch = new THREE.Vector2(), U = new Float32Array(plane), V = new Float32Array(plane);
  let m = 0, cells = new Int32Array(0), areas = new Float64Array(0), range = [...ctx.speedLimits];
  let rand = random(seed), head = 0, acc = 0, trail = 1, record = 1, lifetime = 1, maxSpeed = 0;
  let lastTime = null, lastA = null, lastB = null, stale = true;

  let su = 0, sv = 0;
  const sample = (x, y) => {
    const gx = fa(x), gy = fb(y), i = Math.min(Math.max(Math.floor(gx), 0), na - 2), j = Math.min(Math.max(Math.floor(gy), 0), nb - 2);
    const tx = Math.min(Math.max(gx - i, 0), 1), ty = Math.min(Math.max(gy - j, 0), 1), p = j * na + i;
    const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    su = U[p] * w00 + U[p + 1] * w10 + U[p + na] * w01 + U[p + na + 1] * w11;
    sv = V[p] * w00 + V[p + 1] * w10 + V[p + na] * w01 + V[p + na + 1] * w11;
  };
  const solid = (x, y) => {
    if (x < a0 || x > a1 || y < b0 || y > b1) return true;
    const i = Math.min(Math.max(Math.round(fa(x)), 0), na - 1), j = Math.min(Math.max(Math.round(fb(y)), 0), nb - 1);
    return !fluid[flat[j * na + i]];
  };
  const spawn = (n, staggered) => {
    if (!cells.length) { px[n] = py[n] = NaN; return; }
    // Pick a fluid cell with probability proportional to its area, then a uniform point inside it.
    const target = rand() * areas[areas.length - 1];
    let lo = 0, hi = areas.length - 1;
    while (lo < hi) { const k = (lo + hi) >> 1; if (areas[k] <= target) lo = k + 1; else hi = k; }
    const p = cells[lo], i = p % na, j = (p - i) / na;
    px[n] = af[i] + rand() * (af[i + 1] - af[i]);
    py[n] = bf[j] + rand() * (bf[j + 1] - bf[j]);
    life[n] = lifetime * (0.5 + 0.5 * rand());
    age[n] = staggered ? rand() * life[n] : 0;
    count[n] = 0;
  };
  // RK2 (midpoint), sub-stepped so no step moves more than half a cell.
  const advect = dt => {
    const steps = Math.min(32, Math.max(1, Math.ceil(maxSpeed * dt / (0.5 * minCell)))), h = dt / steps;
    for (let n = 0; n < tracers; n++) {
      let x = px[n], y = py[n];
      if (x !== x) continue;
      age[n] += dt;
      if (age[n] > life[n]) { spawn(n, false); continue; }
      for (let s = 0; s < steps; s++) {
        sample(x, y);
        sample(x + 0.5 * h * su, y + 0.5 * h * sv);
        x += h * su; y += h * sv;
        if (solid(x, y)) break;
      }
      if (solid(x, y)) spawn(n, false); else { px[n] = x; py[n] = y; }
    }
  };
  const remember = () => {
    head = (head + 1) % R;
    for (let n = 0; n < tracers; n++) {
      hist[(n * R + head) * 2] = px[n]; hist[(n * R + head) * 2 + 1] = py[n];
      if (count[n] < R) count[n]++;
    }
  };
  const step = dt => {
    advect(dt);
    for (acc += dt; acc >= record; acc -= record) remember();
  };
  // In-plane velocity of the blended state.
  const blend = f => {
    const {a: s, b: e, alpha} = f, sa = s[ua], ea = e[ua], sb = s[ub], eb = e[ub];
    maxSpeed = 0;
    for (let p = 0; p < plane; p++) {
      const q = flat[p], u = sa[q] + alpha * (ea[q] - sa[q]), v = sb[q] + alpha * (eb[q] - sb[q]);
      U[p] = u; V[p] = v;
      const s2 = u * u + v * v;
      if (s2 > maxSpeed) maxSpeed = s2;
    }
    maxSpeed = Math.sqrt(maxSpeed);
  };
  const speeds = (state, out) => {
    const {u, v, w} = state;
    for (let p = 0; p < plane; p++) { const q = flat[p]; out[p] = THREE.DataUtils.toHalfFloat(Math.hypot(u[q], v[q], w[q])); }
  };
  const reset = () => {
    rand = random(seed + m);
    trail = TRAIL_FRACTION * L / range[1];
    record = trail / TRAIL;
    lifetime = LIFE_FRACTION * L / range[1];
    for (let n = 0; n < tracers; n++) spawn(n, true);
    for (let r = 0; r < R; r++) { advect(record); remember(); }
    acc = 0;
  };
  const write = () => {
    const col = streak, cf = grid[`${c}f`], z = 0.1 * (cf[m + 1] - cf[m]);
    const f = 1 - acc / record;
    let o = 0;
    for (let n = 0; n < tracers; n++) {
      const x = px[n];
      const fadeIn = Math.min(1, age[n] / (0.5 * trail)), fadeOut = Math.min(1, (life[n] - age[n]) / (0.25 * life[n]));
      const edge = Math.min(x - a0, a1 - x, py[n] - e0, b1 - py[n]) / fade;
      const g = x === x ? BRIGHTNESS * Math.max(0, Math.min(fadeIn, fadeOut, edge * edge * (3 - 2 * Math.min(1, edge)))) : 0;
      pts[0] = x; pts[1] = py[n];
      for (let j = 1; j <= R; j++) {
        const k = Math.min(j, count[n]), q = k ? (n * R + (head - k + 1 + R) % R) * 2 : -1;
        pts[2 * j] = k ? hist[q] : x; pts[2 * j + 1] = k ? hist[q + 1] : py[n];
      }
      for (let j = 0; j < SEG; j++) {
        const last = j === SEG - 1;
        const ax = pts[2 * j], ay = pts[2 * j + 1];
        const bx = last ? pts[2 * j] + f * (pts[2 * j + 2] - pts[2 * j]) : pts[2 * j + 2];
        const by = last ? pts[2 * j + 1] + f * (pts[2 * j + 3] - pts[2 * j + 1]) : pts[2 * j + 3];
        const sa = j ? (acc + (j - 1) * record) / trail : 0, sb = last ? 1 : (acc + j * record) / trail;
        const ba = g * (1 - sa) ** 1.5, bb = g * (1 - Math.min(1, sb)) ** 1.5;
        linePos[o] = ax; linePos[o + 1] = ay; linePos[o + 2] = z; linePos[o + 3] = bx; linePos[o + 4] = by; linePos[o + 5] = z;
        lineCol[o] = col.r * ba; lineCol[o + 1] = col.g * ba; lineCol[o + 2] = col.b * ba;
        lineCol[o + 3] = col.r * bb; lineCol[o + 4] = col.g * bb; lineCol[o + 5] = col.b * bb;
        o += 6;
      }
    }
    lineGeometry.attributes.position.needsUpdate = lineGeometry.attributes.color.needsUpdate = true;
  };

  return {
    group, piece, flat,
    get cells() { return cells; },
    get visible() { return group.visible; },
    set visible(on) { group.visible = on; if (!on) stale = true; },
    set(k) {
      m = k;
      const list = [];
      for (let j = 0, p = 0; j < nb; j++) for (let i = 0; i < na; i++, p++) {
        const q = flat[p] = m * stride[c] + i * stride[a] + j * stride[b];
        mask[p] = fluid[q] ? 255 : 0;
        if (fluid[q]) list.push(p);
      }
      cells = Int32Array.from(list);
      areas = new Float64Array(cells.length);
      cells.forEach((p, n) => { const i = p % na, j = (p - i) / na; areas[n] = (n ? areas[n - 1] : 0) + (af[i + 1] - af[i]) * (bf[j + 1] - bf[j]); });
      texM.needsUpdate = true;
      group.position[c] = grid[c][m];
      lastA = lastB = null;
      stale = true;
    },
    setRange(r) { range = [...r]; uniforms.range.value.set(...range); stale = true; },
    update(f, simTime, dtSim) {
      if (f.a !== lastA || f.b !== lastB) {
        speeds(f.a, speedA); speeds(f.b, speedB);
        texA.needsUpdate = texB.needsUpdate = true;
        lastA = f.a; lastB = f.b;
      }
      uniforms.alpha.value = f.alpha;
      blend(f);
      const jump = lastTime === null || dtSim < 0 || Math.abs(simTime - lastTime - dtSim) > 1e-4 * Math.max(1, Math.abs(simTime)) || dtSim * range[1] > 0.5 * L;
      const dt = pace(dtSim, range[1]);
      lastTime = simTime;
      if (stale || jump) { reset(); stale = false; write(); }
      else if (dtSim > 0) { step(dt); write(); }
      // Keep the on-screen streak density near the reference's: fewer streaks in a small canvas.
      const size = ctx.renderer.getSize(scratch), shown = Math.round(tracers * Math.min(1, Math.max(0.35, size.x * size.y / FULL_AREA)));
      lineGeometry.setDrawRange(0, shown * SEG * 2);
    },
  };
}

export function createSliceMode(ctx) {
  const {THREE, grid, fluid} = ctx, {nx, ny, nz, bounds: [, , y0, y1, z0]} = grid, plane = nx * ny;
  const flatSheet = createSheet(ctx, "x", "y", TRACERS, 1), crossSheet = createSheet(ctx, "x", "z", CROSS_TRACERS, 7919, true);
  const [sheet, lines] = flatSheet.piece();
  sheet.renderOrder = 1;
  lines.renderOrder = 1.6;
  // The vertical sheet in two halves cut at the slice height: drawn beyond and in front of the
  // horizontal sheet (and the shell's shadow catcher at 1.5) in painter's order, so the translucent
  // sheets blend correctly from any side.
  crossSheet.group.rotation.x = Math.PI / 2;
  const below = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0), above = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const halves = [crossSheet.piece([below]), crossSheet.piece([above])];
  // With the vertical sheet both share one colour range: the full span of the exported per-level ranges.
  const limits = (ctx.levelSpeedLimits || []).filter(Boolean);
  const shared = limits.length ? [Math.min(...limits.map(l => l[0])), Math.max(...limits.map(l => l[1]))] : [...ctx.speedLimits];
  let level = 0, row = 0, vertical = false, shown = false, levelRange = [...ctx.speedLimits], ranged = false;
  const round = v => v >= 10 ? Math.round(v) : Number(v.toFixed(1));
  const applyRange = () => { flatSheet.setRange(mode.range); crossSheet.setRange(mode.range); ctx.requestRender?.(); };

  const mode = {
    get level() { return level; },
    get row() { return row; },
    get vertical() { return vertical; },
    get label() { return `wind speed ${round(grid.z[level] - z0)} m above the ground${vertical ? ` and at y = ${round(grid.y[row])} m` : ""}`; },
    get range() { return vertical ? shared : levelRange; },
    setLevel(k) {
      level = Math.min(Math.max(Math.round(k), 0), nz - 1);
      flatSheet.set(level);
      below.constant = grid.z[level]; above.constant = -grid.z[level];
      // The exported per-level limits keep the colorbar fixed per level; else the first decoded state sets it.
      const limits = ctx.levelSpeedLimits?.[level];
      ranged = Array.isArray(limits) && limits[1] > limits[0];
      levelRange = ranged ? [...limits] : [...ctx.speedLimits];
      applyRange();
    },
    setRow(j) { row = Math.min(Math.max(Math.round(j), 0), ny - 1); crossSheet.set(row); ctx.requestRender?.(); },
    setVertical(on) { vertical = on; crossSheet.visible = shown && on; applyRange(); },
    update(simTime, dtSim) {
      if (!shown) return;
      const f = ctx.frame(simTime);
      if (!f) return;
      if (!ranged) {
        const s = [];
        for (const p of flatSheet.cells) { const q = flatSheet.flat[p]; s.push(Math.hypot(f.a.u[q], f.a.v[q], f.a.w[q])); }
        s.sort((a, b) => a - b);
        const hi = s.length ? s[Math.min(s.length - 1, Math.floor(PERCENTILE * s.length))] : 0;
        levelRange = hi > 0 ? [0, niceCeil(hi)] : [...ctx.speedLimits];
        ranged = true;
        applyRange();
      }
      flatSheet.update(f, simTime, dtSim);
      if (vertical) {
        const order = ctx.camera.position.z > grid.z[level] ? [0, 1] : [1, 0];
        halves.forEach(([s, l], i) => { s.renderOrder = order[i] ? 1.7 : 0.7; l.renderOrder = s.renderOrder + 0.1; });
        crossSheet.update(f, simTime, dtSim);
      }
    },
    setVisible(on) { shown = on; flatSheet.visible = on; crossSheet.visible = on && vertical; },
  };
  let k = 0, best = Infinity;
  grid.z.forEach((z, i) => { const d = Math.abs(z - z0 - DEFAULT_HEIGHT); if (d < best) { best = d; k = i; } });
  for (let i = 0; i < nz; i++) {
    let n = 0;
    for (let p = 0; p < plane; p++) n += fluid[i * plane + p];
    if (n >= 0.6 * plane) { k = Math.max(k, i); break; }
  }
  mode.setLevel(k);
  best = Infinity;
  grid.y.forEach((y, j) => { const d = Math.abs(y - (y0 + y1) / 2); if (d < best) { best = d; row = j; } });
  mode.setRow(row);
  return mode;
}
