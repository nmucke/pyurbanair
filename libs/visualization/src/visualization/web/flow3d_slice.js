// Height slice: a glowing horizontal wind-speed sheet at one z level with tracer streaks drifting over it.
const TRACERS = 2200, TRAIL = 6, TRAIL_FRACTION = 0.02, BRIGHTNESS = 0.55, PERCENTILE = 0.99, LIFE_FRACTION = 0.25, DEFAULT_HEIGHT = 9;
const STREAK = [1, 0.9, 0.55], FULL_AREA = 1600 * 900;

const random = seed => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const niceCeil = x => { const e = 10 ** Math.floor(Math.log10(x)), step = e * (x / e < 2 ? 0.1 : 0.5); return +(Math.ceil(x / step - 1e-9) * step).toFixed(6); };

export function createSliceMode(ctx) {
  const {THREE, grid, fluid} = ctx, {nx, ny, nz, bounds: [x0, x1, y0, y1, z0], fx, fy, xf, yf, zf} = grid, plane = nx * ny;
  const L = Math.max(x1 - x0, y1 - y0), pace = ctx.pacer();
  const fade = 0.06 * L;
  const minCell = Math.min(...[xf, yf].flatMap(f => Array.from(f.subarray(1), (v, i) => v - f[i])));
  const group = new THREE.Group();
  group.visible = false;
  ctx.scene.add(group);

  // Sheet: vertices at the cell centres plus the outer faces, so nonuniform grids interpolate piecewise linearly.
  const texture = (data, type) => {
    const t = new THREE.DataTexture(data, nx, ny, THREE.RedFormat, type);
    t.magFilter = t.minFilter = THREE.LinearFilter;
    t.unpackAlignment = 1;
    t.needsUpdate = true;
    return t;
  };
  const speedA = new Uint16Array(plane), speedB = new Uint16Array(plane), mask = new Uint8Array(plane);
  const texA = texture(speedA, THREE.HalfFloatType), texB = texture(speedB, THREE.HalfFloatType), texM = texture(mask, THREE.UnsignedByteType);
  const vx = [x0, ...grid.x, x1], vy = [y0, ...grid.y, y1], cols = vx.length, rows = vy.length;
  const position = new Float32Array(cols * rows * 3), uv = new Float32Array(cols * rows * 2), index = [];
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const p = j * cols + i;
    position.set([vx[i], vy[j], 0], p * 3);
    uv.set([(Math.min(Math.max(i - 1, 0), nx - 1) + 0.5) / nx, (Math.min(Math.max(j - 1, 0), ny - 1) + 0.5) / ny], p * 2);
    if (i && j) index.push(p - cols - 1, p - cols, p, p - cols - 1, p, p - 1);
  }
  const sheetGeometry = new THREE.BufferGeometry();
  sheetGeometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  sheetGeometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  sheetGeometry.setIndex(index);
  const uniforms = {
    speedA: {value: texA}, speedB: {value: texB}, mask: {value: texM}, colormap: {value: ctx.colormap},
    alpha: {value: 0}, range: {value: new THREE.Vector2(...ctx.speedLimits)},
    rect: {value: new THREE.Vector4(x0, x1, y0, y1)}, fade: {value: fade},
  };
  const sheetMaterial = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    vertexShader: `varying vec2 vUv; varying vec2 vXY;
      void main() { vUv = uv; vXY = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform sampler2D speedA, speedB, mask, colormap; uniform float alpha, fade; uniform vec2 range; uniform vec4 rect;
      varying vec2 vUv; varying vec2 vXY;
      void main() {
        float s = mix(texture2D(speedA, vUv).r, texture2D(speedB, vUv).r, alpha);
        float t = clamp((s - range.x) / (range.y - range.x), 0.0, 1.0);
        vec3 c = texture2D(colormap, vec2(t * 255.0 / 256.0 + 0.5 / 256.0, 0.5)).rgb * (1.0 + 0.35 * t * t);
        vec2 d = min(vXY - rect.xz, rect.yw - vXY);
        float a = smoothstep(0.02, 0.38, t) * smoothstep(0.35, 0.75, texture2D(mask, vUv).r) * smoothstep(0.0, fade, min(d.x, d.y));
        gl_FragColor = vec4(c, 0.96 * a);
        #include <colorspace_fragment>
      }`,
  });
  const sheet = new THREE.Mesh(sheetGeometry, sheetMaterial);
  sheet.renderOrder = 1;
  sheet.frustumCulled = false;

  // Tracers: TRAIL + 1 segments each, from the head back to an interpolated tail TRAIL record intervals ago.
  const R = TRAIL + 1, SEG = TRAIL + 1;
  const px = new Float32Array(TRACERS), py = new Float32Array(TRACERS), age = new Float32Array(TRACERS), life = new Float32Array(TRACERS);
  const hist = new Float32Array(TRACERS * R * 2), count = new Uint8Array(TRACERS), pts = new Float32Array((R + 1) * 2);
  const linePos = new Float32Array(TRACERS * SEG * 6), lineCol = new Float32Array(TRACERS * SEG * 6);
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage));
  lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineCol, 3).setUsage(THREE.DynamicDrawUsage));
  const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
  }));
  lines.renderOrder = 2;
  lines.frustumCulled = false;
  const streak = new THREE.Color().setRGB(...STREAK, THREE.SRGBColorSpace);
  group.add(sheet, lines);

  const scratch = new THREE.Vector2(), U = new Float32Array(plane), V = new Float32Array(plane);
  let level = 0, cells = new Int32Array(0), areas = new Float64Array(0), range = [...ctx.speedLimits], ranged = false, label = "";
  let rand = random(1), head = 0, acc = 0, trail = 1, record = 1, lifetime = 1, maxSpeed = 0;
  let lastTime = null, lastA = null, lastB = null, stale = true;

  let su = 0, sv = 0;
  const sample = (x, y) => {
    const gx = fx(x), gy = fy(y), i = Math.min(Math.max(Math.floor(gx), 0), nx - 2), j = Math.min(Math.max(Math.floor(gy), 0), ny - 2);
    const tx = Math.min(Math.max(gx - i, 0), 1), ty = Math.min(Math.max(gy - j, 0), 1), p = j * nx + i;
    const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    su = U[p] * w00 + U[p + 1] * w10 + U[p + nx] * w01 + U[p + nx + 1] * w11;
    sv = V[p] * w00 + V[p + 1] * w10 + V[p + nx] * w01 + V[p + nx + 1] * w11;
  };
  const solid = (x, y) => {
    if (x < x0 || x > x1 || y < y0 || y > y1) return true;
    const i = Math.min(Math.max(Math.round(fx(x)), 0), nx - 1), j = Math.min(Math.max(Math.round(fy(y)), 0), ny - 1);
    return !fluid[level * plane + j * nx + i];
  };
  const spawn = (n, staggered) => {
    if (!cells.length) { px[n] = py[n] = NaN; return; }
    // Pick a fluid cell with probability proportional to its area, then a uniform point inside it.
    const target = rand() * areas[areas.length - 1];
    let lo = 0, hi = areas.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (areas[m] <= target) lo = m + 1; else hi = m; }
    const c = cells[lo], i = c % nx, j = (c - i) / nx;
    px[n] = xf[i] + rand() * (xf[i + 1] - xf[i]);
    py[n] = yf[j] + rand() * (yf[j + 1] - yf[j]);
    life[n] = lifetime * (0.5 + 0.5 * rand());
    age[n] = staggered ? rand() * life[n] : 0;
    count[n] = 0;
  };
  // RK2 (midpoint), sub-stepped so no step moves more than half a cell.
  const advect = dt => {
    const steps = Math.min(32, Math.max(1, Math.ceil(maxSpeed * dt / (0.5 * minCell)))), h = dt / steps;
    for (let n = 0; n < TRACERS; n++) {
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
    for (let n = 0; n < TRACERS; n++) {
      hist[(n * R + head) * 2] = px[n]; hist[(n * R + head) * 2 + 1] = py[n];
      if (count[n] < R) count[n]++;
    }
  };
  const step = dt => {
    advect(dt);
    for (acc += dt; acc >= record; acc -= record) remember();
  };
  const blend = f => {
    const off = level * plane, {a, b, alpha} = f;
    maxSpeed = 0;
    for (let p = 0; p < plane; p++) {
      const u = a.u[off + p] + alpha * (b.u[off + p] - a.u[off + p]), v = a.v[off + p] + alpha * (b.v[off + p] - a.v[off + p]);
      U[p] = u; V[p] = v;
      const s = u * u + v * v;
      if (s > maxSpeed) maxSpeed = s;
    }
    maxSpeed = Math.sqrt(maxSpeed);
  };
  const speeds = (state, out) => {
    const off = level * plane, {u, v, w} = state;
    for (let p = 0; p < plane; p++) out[p] = THREE.DataUtils.toHalfFloat(Math.hypot(u[off + p], v[off + p], w[off + p]));
  };
  const reset = () => {
    rand = random(1 + level);
    trail = TRAIL_FRACTION * L / range[1];
    record = trail / TRAIL;
    lifetime = LIFE_FRACTION * L / range[1];
    for (let n = 0; n < TRACERS; n++) spawn(n, true);
    for (let r = 0; r < R; r++) { advect(record); remember(); }
    acc = 0;
  };
  const write = () => {
    const c = streak, z = 0.1 * (zf[level + 1] - zf[level]);
    const f = 1 - acc / record;
    let o = 0;
    for (let n = 0; n < TRACERS; n++) {
      const x = px[n];
      const fadeIn = Math.min(1, age[n] / (0.5 * trail)), fadeOut = Math.min(1, (life[n] - age[n]) / (0.25 * life[n]));
      const edge = Math.min(x - x0, x1 - x, py[n] - y0, y1 - py[n]) / fade;
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
        lineCol[o] = c.r * ba; lineCol[o + 1] = c.g * ba; lineCol[o + 2] = c.b * ba;
        lineCol[o + 3] = c.r * bb; lineCol[o + 4] = c.g * bb; lineCol[o + 5] = c.b * bb;
        o += 6;
      }
    }
    lineGeometry.attributes.position.needsUpdate = lineGeometry.attributes.color.needsUpdate = true;
  };

  const mode = {
    get level() { return level; },
    get label() { return label; },
    get range() { return range; },
    setLevel(k) {
      level = Math.min(Math.max(Math.round(k), 0), nz - 1);
      const off = level * plane, list = [];
      for (let p = 0; p < plane; p++) { mask[p] = fluid[off + p] ? 255 : 0; if (fluid[off + p]) list.push(p); }
      cells = Int32Array.from(list);
      areas = new Float64Array(cells.length);
      cells.forEach((p, n) => { const i = p % nx, j = (p - i) / nx; areas[n] = (n ? areas[n - 1] : 0) + (xf[i + 1] - xf[i]) * (yf[j + 1] - yf[j]); });
      texM.needsUpdate = true;
      sheet.position.z = grid.z[level];
      lines.position.z = grid.z[level];
      const h = grid.z[level] - z0;
      label = `wind speed ${h >= 10 ? Math.round(h) : Number(h.toFixed(1))} m above the ground`;
      // The exported per-level limits keep the colorbar fixed per level; else the first decoded state sets it.
      const limits = ctx.levelSpeedLimits?.[level];
      ranged = Array.isArray(limits) && limits[1] > limits[0];
      range = ranged ? [...limits] : [...ctx.speedLimits];
      uniforms.range.value.set(...range);
      lastA = lastB = null;
      stale = true;
      ctx.requestRender?.();
    },
    update(simTime, dtSim) {
      if (!group.visible) { stale = true; return; }
      const f = ctx.frame(simTime);
      if (!f) return;
      if (!ranged) {
        const off = level * plane, s = [];
        for (const p of cells) s.push(Math.hypot(f.a.u[off + p], f.a.v[off + p], f.a.w[off + p]));
        s.sort((a, b) => a - b);
        const hi = s.length ? s[Math.min(s.length - 1, Math.floor(PERCENTILE * s.length))] : 0;
        range = hi > 0 ? [0, niceCeil(hi)] : [...ctx.speedLimits];
        uniforms.range.value.set(...range);
        ranged = stale = true;
        ctx.requestRender?.();
      }
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
      const size = ctx.renderer.getSize(scratch), shown = Math.round(TRACERS * Math.min(1, Math.max(0.35, size.x * size.y / FULL_AREA)));
      lineGeometry.setDrawRange(0, shown * SEG * 2);
    },
    setVisible(on) { group.visible = on; if (!on) stale = true; },
  };
  let k = 0, best = Infinity;
  grid.z.forEach((z, i) => { const d = Math.abs(z - z0 - DEFAULT_HEIGHT); if (d < best) { best = d; k = i; } });
  for (let i = 0; i < nz; i++) {
    let n = 0;
    for (let p = 0; p < plane; p++) n += fluid[i * plane + p];
    if (n >= 0.6 * plane) { k = Math.max(k, i); break; }
  }
  mode.setLevel(k);
  return mode;
}
