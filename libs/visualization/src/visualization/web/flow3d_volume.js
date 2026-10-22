// Full 3D mode of the 3D flow view. The vortices the buildings shed (Q criterion, Q > 0) are
// raymarched as a soft glow coloured by wind speed, and sparse tracer streaks, seeded mostly in
// those vortices and in the canopy layer, show where and how fast the air moves. The colour range
// is fitted to the wind speeds of the canopy layer (up to ~2× the building height), so the street
// channels and wakes use the whole palette; the faster air above saturates to yellow.
const MAX_CELLS = 1.5e6;     // volume texture budget: larger grids are averaged over c×c columns
const CACHE_BYTES = 96e6;    // derived volumes kept per saved state
const LUT = 1024;            // world -> texture coordinate lookup per axis (nonuniform grids)
const RAYS = 6e5;            // volume rays per frame

export function createVolumeMode(ctx) {
  const {THREE, grid: {x, y, z, nx, ny, nz, bounds, zf, fx, fy, fz}, fluid} = ctx, [x0, x1, y0, y1, z0, z1] = bounds;
  const nxy = nx * ny, L = Math.max(x1 - x0, y1 - y0);
  const cellAt = (px, py, pz) => Math.round(fz(pz)) * nxy + Math.round(fy(py)) * nx + Math.round(fx(px));

  // Canopy height: 90th percentile of the building-column tops.
  const tops = [];
  for (let j = 0; j < nxy; j++) { let top = -1; for (let k = 0; k < nz; k++) if (!fluid[k * nxy + j]) top = k; if (top >= 0) tops.push(top); }
  tops.sort((a, b) => a - b);
  const dz = (z1 - z0) / nz, topK = tops[Math.floor(0.9 * (tops.length - 1))];
  const hc = Math.max(3 * dz, tops.length ? z[topK] + (z[Math.min(nz - 1, topK + 1)] - z[topK]) / 2 - z0 : 0.15 * (z1 - z0));

  // Coarse grid for the volume: columns averaged c×c so the 3D textures stay within MAX_CELLS.
  let c = 1; while (nxy * nz / (c * c) > MAX_CELLS) c++;
  const mx = Math.ceil(nx / c), my = Math.ceil(ny / c), mxy = mx * my, m = mxy * nz;
  const block = (a, mn) => Float32Array.from({length: mn}, (_, i) => { const s = a.slice(i * c, i * c + c); return s.reduce((p, q) => p + q, 0) / s.length; });
  const cx = c === 1 ? x : block(x, mx), cy = c === 1 ? y : block(y, my);
  const column = Int32Array.from({length: nxy}, (_, f) => Math.floor(f / nx / c) * mx + Math.floor(f % nx / c));
  const cfluid = new Uint8Array(m), cweight = new Float32Array(m);
  for (let f = 0; f < nxy * nz; f++) { const o = Math.floor(f / nxy) * mxy + column[f % nxy]; cweight[o] += fluid[f]; cfluid[o] += 1; }
  for (let o = 0; o < m; o++) { cfluid[o] = cweight[o] * 2 >= cfluid[o] && cweight[o] > 0 ? 1 : 0; cweight[o] = cweight[o] ? 1 / cweight[o] : 0; }
  const coarseAt = (px, py, pz) => Math.round(fz(pz)) * mxy + Math.floor(Math.round(fy(py)) / c) * mx + Math.floor(Math.round(fx(px)) / c);
  // Glow damping next to buildings, where Q hugs every facade and roof as a skin: 0.05, 0.25, 0.6 one to three cells away.
  const damp = new Float32Array(m).fill(1);
  const neighbours = o => {
    const i = o % mx, j = Math.floor(o / mx) % my, k = Math.floor(o / mxy), out = [];
    if (i > 0) out.push(o - 1); if (i < mx - 1) out.push(o + 1); if (j > 0) out.push(o - mx); if (j < my - 1) out.push(o + mx);
    if (k > 0) out.push(o - mxy); if (k < nz - 1) out.push(o + mxy);
    return out;
  };
  for (let o = 0; o < m; o++) if (cfluid[o] && neighbours(o).some(n => !cfluid[n])) damp[o] = 0.05;
  for (const [from, to] of [[0.05, 0.25], [0.25, 0.6]]) for (let o = 0; o < m; o++) if (damp[o] === 1 && neighbours(o).some(n => damp[n] === Math.fround(from))) damp[o] = to;
  const coarsen = s => {
    if (c === 1) return s;
    const out = {u: new Float32Array(m), v: new Float32Array(m), w: new Float32Array(m)};
    const {u, v, w} = out, su = s.u, sv = s.v, sw = s.w;
    for (let k = 0; k < nz; k++) for (let j = 0, f = k * nxy; j < nxy; j++, f++) {
      if (!fluid[f]) continue;
      const o = k * mxy + column[j], wt = cweight[o];
      u[o] += su[f] * wt; v[o] += sv[f] * wt; w[o] += sw[f] * wt;
    }
    return out;
  };

  // Per saved state: RG bytes = (speed in the colour range, sqrt(Q+) / its 99.5th percentile),
  // cached by state index so the shell's frame cache can drop the decoded state.
  const mode = {range: null, label: "wind speed in 3D · glow marks vortices (Q criterion)"};
  let qNorm = 0, zTop = z0 + 2 * hc;
  const cache = new Map();
  const derive = s => {
    let out = cache.get(s.index);
    if (out) { cache.delete(s.index); cache.set(s.index, out); return out; }
    const {u, v, w} = coarsen(s), q = new Float32Array(m);
    // velocity gradient: central differences, one-sided next to solids; Q = -1/2 Σ ∂i uj ∂j ui
    const ends = new Int32Array(6), inv = new Float32Array(3);
    const pick = (a, o, stride, co, i, len) => {
      const lo = i > 0 && cfluid[o - stride] === 1, hi = i < len - 1 && cfluid[o + stride] === 1;
      ends[2 * a] = hi ? o + stride : o; ends[2 * a + 1] = lo ? o - stride : o;
      inv[a] = lo || hi ? 1 / (co[hi ? i + 1 : i] - co[lo ? i - 1 : i]) : 0;
    };
    for (let k = 0; k < nz; k++) for (let j = 1; j < my - 1; j++) for (let i = 1; i < mx - 1; i++) {   // open boundaries stay 0
      const o = k * mxy + j * mx + i; if (!cfluid[o]) continue;
      pick(0, o, 1, cx, i, mx); pick(1, o, mx, cy, j, my); pick(2, o, mxy, z, k, nz);
      const xp = ends[0], xm = ends[1], yp = ends[2], ym = ends[3], zp = ends[4], zm = ends[5], ix = inv[0], iy = inv[1], iz = inv[2];
      const ux = (u[xp] - u[xm]) * ix, uy = (u[yp] - u[ym]) * iy, uz = (u[zp] - u[zm]) * iz;
      const vx = (v[xp] - v[xm]) * ix, vy = (v[yp] - v[ym]) * iy, vz = (v[zp] - v[zm]) * iz;
      const wx = (w[xp] - w[xm]) * ix, wy = (w[yp] - w[ym]) * iy, wz = (w[zp] - w[zm]) * iz;
      const Q = -0.5 * (ux * ux + vy * vy + wz * wz) - (uy * vx + uz * wx + vz * wy);
      q[o] = Q > 0 ? Math.sqrt(Q) * damp[o] : 0;
    }
    if (!qNorm) {
      const sample = []; for (let o = 0; o < m; o += 3) if (cfluid[o]) sample.push(q[o]);
      sample.sort((a, b) => a - b); qNorm = sample[Math.floor(0.995 * (sample.length - 1))] || 1;
    }
    const [lo, top] = mode.range, scale = 255 / (top - lo);
    out = new Uint8Array(2 * m);
    for (let o = 0; o < m; o++) {
      out[2 * o] = Math.min(255, Math.max(0, (Math.sqrt(u[o] * u[o] + v[o] * v[o] + w[o] * w[o]) - lo) * scale));
      const b = Math.min(255, 255 * q[o] / qNorm); out[2 * o + 1] = b;
      if (b > 40) zTop = Math.max(zTop, z[Math.floor(o / mxy)] + 2 * dz);
    }
    cache.set(s.index, out);
    while (cache.size > 2 && cache.size * 2 * m > CACHE_BYTES) cache.delete(cache.keys().next().value);
    return out;
  };
  // Colour range: the exported per-level limits (scene.json, over all frames) of the levels below
  // 2× the building top, [min of lowers, max of uppers]; without them, [0, canopy p99 of the
  // first decoded state rounded up to 0.5].
  const fitRange = s => {
    const below = zz => zz - z0 < 2 * Math.max(hc, (ctx.buildingTop ?? z0) - z0);
    const levels = ctx.levelSpeedLimits?.filter((l, k) => l && below(z[k])) || [];
    const lo = Math.min(...levels.map(l => l[0])), top = Math.max(...levels.map(l => l[1]));
    if (top > lo) { mode.range = [lo, top]; return; }
    const v = [];
    for (let k = 0; k < nz && below(z[k]); k++) for (let o = k * nxy; o < (k + 1) * nxy; o += 2) if (fluid[o]) v.push(Math.hypot(s.u[o], s.v[o], s.w[o]));
    v.sort((a, b) => a - b);
    const hi = Math.ceil(2 * (v[Math.floor(0.99 * (v.length - 1))] || 0)) / 2;
    mode.range = hi > 0 ? [0, hi] : ctx.speedLimits.slice();
  };

  // Volume: a box raymarched front to back, emission coloured by speed, opacity from Q, the field
  // blended between the bracketing states. Rays stop at the scene depth (buildings, ground) from
  // a depth-only pre-pass of the scene, so buildings in front of a vortex stay clean.
  const texture3d = data => {
    const t = new THREE.Data3DTexture(data, mx, my, nz);
    t.format = THREE.RGFormat; t.type = THREE.UnsignedByteType; t.minFilter = t.magFilter = THREE.LinearFilter; t.unpackAlignment = 1; t.needsUpdate = true;
    return t;
  };
  const texA = texture3d(new Uint8Array(2 * m)), texB = texture3d(new Uint8Array(2 * m));
  const lut = new Float32Array(3 * LUT);
  [[cx, c === 1 ? fx : ctx.locator(cx), x0, x1], [cy, c === 1 ? fy : ctx.locator(cy), y0, y1], [z, fz, z0, z1]].forEach(([co, f, a, b], r) => {
    for (let j = 0; j < LUT; j++) lut[r * LUT + j] = (f(a + (b - a) * j / (LUT - 1)) + 0.5) / co.length;
  });
  const lutTex = new THREE.DataTexture(lut, LUT, 3, THREE.RedFormat, THREE.FloatType);
  lutTex.minFilter = lutTex.magFilter = THREE.NearestFilter; lutTex.needsUpdate = true;
  const depthTarget = new THREE.WebGLRenderTarget(1, 1, {depthTexture: new THREE.DepthTexture(1, 1)});
  const target = new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter});
  const view = {depth: {value: depthTarget.depthTexture}, near: {value: 1}, far: {value: 2}};
  const linearDepth = `#include <packing>
      uniform sampler2D depth; uniform float near, far;
      float sceneZ(ivec2 p) { return -perspectiveDepthToViewZ(texelFetch(depth, p, 0).r, near, far); }`;
  const cell = Math.min((x1 - x0) / mx, (y1 - y0) / my, dz);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      ...view, A: {value: texA}, B: {value: texB}, lut: {value: lutTex}, cmap: {value: ctx.colormap}, alpha: {value: 0}, forward: {value: new THREE.Vector3()},
      scale: {value: 1}, b0: {value: new THREE.Vector3(x0, y0, z0)}, b1: {value: new THREE.Vector3(x1, y1, z1)}, top: {value: z1}, stepLen: {value: (m < 5e5 ? 0.5 : 0.75) * cell},
    },
    vertexShader: `varying vec3 vW; void main() { vec4 w = modelMatrix * vec4(position, 1.); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `precision highp float; precision highp sampler3D;
      ${linearDepth}
      uniform sampler3D A, B; uniform sampler2D lut, cmap; uniform float alpha, stepLen, top, scale; uniform vec3 b0, b1, forward; varying vec3 vW;
      float axis(float u, int row) {
        float f = clamp(u, 0., 1.) * ${LUT - 1}.; int i = int(min(f, ${LUT - 2}.));
        return mix(texelFetch(lut, ivec2(i, row), 0).r, texelFetch(lut, ivec2(i + 1, row), 0).r, f - float(i));
      }
      void main() {
        vec3 hi = vec3(b1.xy, top), ro = cameraPosition, rd = normalize(vW - ro), inv = 1. / rd;
        vec3 ta = (b0 - ro) * inv, tb = (hi - ro) * inv, tn3 = min(ta, tb), tf3 = max(ta, tb);
        float t = max(max(max(tn3.x, tn3.y), tn3.z), 0.), tf = min(min(tf3.x, tf3.y), tf3.z);
        tf = min(tf, sceneZ(ivec2((gl_FragCoord.xy - .5) * scale + .5)) / dot(rd, forward));
        if (tf <= t) discard;
        t += stepLen * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        vec3 C = vec3(0.); float T = 1.;
        for (int i = 0; i < 1536; i++) {
          if (t > tf || T < .03) break;
          vec3 u = (ro + rd * t - b0) / (b1 - b0), p = vec3(axis(u.x, 0), axis(u.y, 1), axis(u.z, 2));
          vec2 s = mix(texture(A, p).rg, texture(B, p).rg, alpha);
          float q = smoothstep(.3, 1., s.g);
          if (q > 0.) {
            float a = 1. - exp(-.03 * q * stepLen / ${cell.toFixed(4)});
            C += T * a * 2.6 * texture(cmap, vec2(clamp(s.r, .002, .998), .5)).rgb;
            T *= 1. - a;
          }
          t += stepLen;
        }
        if (T > .995) discard;
        gl_FragColor = vec4(C, 1. - T);   // linear, premultiplied
      }`,
    side: THREE.BackSide, depthWrite: false, depthTest: false, blending: THREE.NoBlending,
  });
  const volume = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material), volumeScene = new THREE.Scene();
  volume.frustumCulled = false; volumeScene.add(volume);
  // The march runs at reduced resolution (≤ RAYS rays); a full-screen quad upsamples it with
  // depth-aware weights (no glow bleeds onto a nearer building edge) and composites it.
  const composite = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    uniforms: {...view, map: {value: target.texture}, scale: {value: 1}},
    vertexShader: `void main() { gl_Position = vec4(position.xy, 0., 1.); }`,
    fragmentShader: `${linearDepth}
      uniform sampler2D map; uniform float scale;
      void main() {
        ivec2 full = textureSize(depth, 0) - 1, low = textureSize(map, 0) - 1;
        float zf = sceneZ(ivec2(gl_FragCoord.xy)), best = 1e30;
        vec2 pl = gl_FragCoord.xy / scale - .5, f = fract(pl);
        ivec2 i0 = ivec2(floor(pl));
        vec4 sum = vec4(0.), nearest = vec4(0.); float wsum = 0.;
        for (int n = 0; n < 4; n++) {
          ivec2 o = ivec2(n & 1, n >> 1), q = clamp(i0 + o, ivec2(0), low);
          float zn = sceneZ(clamp(ivec2((vec2(q) + .5) * scale), ivec2(0), full)), dz = abs(zn - zf) / zf;
          vec4 c = texelFetch(map, q, 0);
          float w = (o.x == 1 ? f.x : 1. - f.x) * (o.y == 1 ? f.y : 1. - f.y) * exp(-dz * dz * 2500.);
          sum += w * c; wsum += w;
          if (dz < best) { best = dz; nearest = c; }
        }
        gl_FragColor = wsum > 1e-4 ? sum / wsum : nearest;
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
  }));
  composite.renderOrder = 1; composite.frustumCulled = false;
  const size = new THREE.Vector2(), clear = new THREE.Color(), depthOnly = new THREE.MeshBasicMaterial({colorWrite: false});
  composite.onBeforeRender = (renderer, scene, camera) => {
    const previous = renderer.getRenderTarget();
    if (previous) size.set(previous.width, previous.height); else renderer.getDrawingBufferSize(size);
    const k = Math.min(1, Math.sqrt(RAYS / (size.x * size.y))), w = Math.max(1, Math.round(size.x * k)), h = Math.max(1, Math.round(size.y * k));
    if (target.width !== w || target.height !== h) target.setSize(w, h);
    if (depthTarget.width !== size.x || depthTarget.height !== size.y) depthTarget.setSize(size.x, size.y);
    material.uniforms.scale.value = size.x / w; composite.material.uniforms.scale.value = size.x / w;
    view.near.value = camera.near; view.far.value = camera.far; camera.getWorldDirection(material.uniforms.forward.value);
    const alpha = renderer.getClearAlpha(), shadows = renderer.shadowMap.autoUpdate, override = scene.overrideMaterial, background = scene.background;
    renderer.getClearColor(clear); renderer.shadowMap.autoUpdate = false;
    group.visible = false; scene.overrideMaterial = depthOnly; scene.background = null;
    renderer.setRenderTarget(depthTarget); renderer.clear(); renderer.render(scene, camera);
    group.visible = true; scene.overrideMaterial = override; scene.background = background;
    renderer.setRenderTarget(target); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.render(volumeScene, camera);
    renderer.setRenderTarget(previous); renderer.setClearColor(clear, alpha); renderer.shadowMap.autoUpdate = shadows;
  };
  const fitBox = () => {
    const top = Math.min(z1, zTop); material.uniforms.top.value = top;
    volume.scale.set(x1 - x0, y1 - y0, top - z0); volume.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + top) / 2);
  };

  // Streaks: tracers advected (RK2) through the trilinear, time-blended velocity, drawn as
  // fading ribbons of constant screen width with additive glow.
  const N = Math.round(Math.min(1200, Math.max(400, nxy / 12))), K = 14, segs = N * K;
  const levelWeight = Float64Array.from(z, (zz, k) => {
    let f = 0; for (let j = k * nxy; j < (k + 1) * nxy; j++) f += fluid[j];
    const h = zz - z0; return f * (h < 2 * hc ? 1 : Math.exp(-(h - 2 * hc) / (0.6 * hc))) + 1e-9;
  });
  const cdf = Float64Array.from(levelWeight); for (let k = 1; k < nz; k++) cdf[k] += cdf[k - 1];
  const P = new Float32Array(3 * N), age = new Float32Array(N), life = new Float32Array(N);
  const hist = new Float32Array(4 * segs), count = new Uint32Array(N);   // count: points recorded since spawn, ring of K
  let lo = 0, hi = 1, trailDt = 1, lifeT = 1, acc = 0, measure = null;
  const spawn = (i, scatter) => {
    for (let t = 0; t < 40; t++) {
      const r = Math.random() * cdf[nz - 1]; let k = 0; while (cdf[k] < r) k++;
      const px = x0 + Math.random() * (x1 - x0), py = y0 + Math.random() * (y1 - y0), pz = zf[k] + Math.random() * (zf[k + 1] - zf[k]);
      if (!fluid[cellAt(px, py, pz)]) continue;
      if (t < 39 && measure && Math.random() > 0.03 + measure[2 * coarseAt(px, py, pz) + 1] / 255) continue;
      P[3 * i] = px; P[3 * i + 1] = py; P[3 * i + 2] = pz; break;
    }
    life[i] = lifeT * (0.6 + 0.8 * Math.random()); age[i] = scatter ? Math.random() * life[i] : 0; count[i] = 0;
  };
  const vel = [0, 0, 0];
  const sample = (px, py, pz, a, b, al) => {    // -> false outside fluid
    const gx = fx(px), gy = fy(py), gz = fz(pz);
    const i0 = Math.max(0, Math.min(nx - 2, gx | 0)), j0 = Math.max(0, Math.min(ny - 2, gy | 0)), k0 = Math.max(0, Math.min(nz - 2, gz | 0));
    const tx = gx - i0, ty = gy - j0, tz = gz - k0;
    let u = 0, v = 0, w = 0;
    for (let n = 0; n < 8; n++) {
      const di = n & 1, dj = (n >> 1) & 1, dk = n >> 2, o = (k0 + dk) * nxy + (j0 + dj) * nx + i0 + di;
      const wt = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty) * (dk ? tz : 1 - tz);
      u += wt * (a.u[o] + al * (b.u[o] - a.u[o])); v += wt * (a.v[o] + al * (b.v[o] - a.v[o])); w += wt * (a.w[o] + al * (b.w[o] - a.w[o]));
    }
    vel[0] = u; vel[1] = v; vel[2] = w;
    return fluid[Math.round(gz) * nxy + Math.round(gy) * nx + Math.round(gx)] === 1;
  };
  const dmin = Math.min((x1 - x0) / nx, (y1 - y0) / ny, dz);
  const advect = (dt, f) => {
    const steps = Math.min(8, Math.max(1, Math.ceil(hi * dt / (0.7 * dmin)))), h = dt / steps;
    for (let i = 0; i < N; i++) {
      let px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2], ok = true;
      for (let s = 0; s < steps && ok; s++) {
        sample(px, py, pz, f.a, f.b, f.alpha);
        ok = sample(px + 0.5 * h * vel[0], py + 0.5 * h * vel[1], pz + 0.5 * h * vel[2], f.a, f.b, f.alpha);
        px += h * vel[0]; py += h * vel[1]; pz += h * vel[2];
        ok = ok && px > x0 && px < x1 && py > y0 && py < y1 && pz > z0 && pz < z1;
      }
      age[i] += dt;
      if (!ok || age[i] > life[i]) spawn(i, false); else { P[3 * i] = px; P[3 * i + 1] = py; P[3 * i + 2] = pz; }
    }
  };
  const pushPoint = f => {
    for (let i = 0; i < N; i++) {
      const o = 4 * (i * K + count[i]++ % K);
      sample(P[3 * i], P[3 * i + 1], P[3 * i + 2], f.a, f.b, f.alpha);
      hist[o] = P[3 * i]; hist[o + 1] = P[3 * i + 1]; hist[o + 2] = P[3 * i + 2]; hist[o + 3] = Math.max(0, Math.hypot(vel[0], vel[1], vel[2]) - lo) / (hi - lo);
    }
  };
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute("corner", new THREE.BufferAttribute(new Float32Array([0, -1, 1, -1, 1, 1, 0, 1]), 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const p0 = new Float32Array(3 * segs), p1 = new Float32Array(3 * segs), sa = new Float32Array(4 * segs);
  for (const [name, data, size] of [["p0", p0, 3], ["p1", p1, 3], ["sa", sa, 4]]) geometry.setAttribute(name, new THREE.InstancedBufferAttribute(data, size).setUsage(THREE.DynamicDrawUsage));
  geometry.instanceCount = 0;
  const ribbons = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
    uniforms: {cmap: {value: ctx.colormap}, res: {value: new THREE.Vector2(1, 1)}, width: {value: 1.4}},
    vertexShader: `attribute vec2 corner; attribute vec3 p0, p1; attribute vec4 sa; uniform vec2 res; uniform float width; varying float vS, vA, vSide;
      void main() {
        vec4 c0 = projectionMatrix * modelViewMatrix * vec4(p0, 1.), c1 = projectionMatrix * modelViewMatrix * vec4(p1, 1.);
        if (c0.w <= 0. || c1.w <= 0.) { gl_Position = vec4(2., 2., 2., 1.); return; }
        vec2 d = c1.xy / c1.w * res - c0.xy / c0.w * res; d = length(d) > 1e-4 ? normalize(d) : vec2(1., 0.);
        vec4 c = mix(c0, c1, corner.x); c.xy += vec2(-d.y, d.x) * corner.y * width / res * c.w;
        gl_Position = c; vS = mix(sa.x, sa.y, corner.x); vA = mix(sa.z, sa.w, corner.x); vSide = corner.y;
      }`,
    fragmentShader: `uniform sampler2D cmap; varying float vS, vA, vSide;
      void main() {
        float a = vA * (1. - vSide * vSide);
        gl_FragColor = vec4(texture2D(cmap, vec2(clamp(vS, .002, .998), .5)).rgb * a, 1.);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  ribbons.renderOrder = 2; ribbons.frustumCulled = false;
  ribbons.onBeforeRender = renderer => {
    renderer.getDrawingBufferSize(size);
    ribbons.material.uniforms.res.value.set(size.x / 2, size.y / 2); ribbons.material.uniforms.width.value = 0.7 * renderer.getPixelRatio() + 0.6;
  };
  const build = () => {
    let s = 0;
    for (let i = 0; i < N; i++) {
      const n = Math.min(K, count[i]), newest = (count[i] - 1) % K; if (!n) continue;
      const fade = Math.min(1, age[i] / (0.15 * life[i]), (life[i] - age[i]) / (0.15 * life[i]));
      // dimmer away from vortices and high up, faded in over the first 10% of the domain (no inlet "rain")
      const h = P[3 * i + 2] - z0, inlet = Math.min(1, (P[3 * i] - x0) / (0.1 * (x1 - x0)));
      const qh = measure ? measure[2 * coarseAt(P[3 * i], P[3 * i + 1], P[3 * i + 2]) + 1] / 255 : 1;
      const gain = 0.45 * fade * inlet * inlet * (0.25 + 0.75 * Math.min(1, 2 * qh)) * (h < 2 * hc ? 1 : Math.max(0.3, Math.exp(-(h - 2 * hc) / (2 * hc))));
      if (gain <= 0.01) continue;
      let px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2], sp = hist[4 * (i * K + newest) + 3];
      for (let j = 0; j < n; j++) {
        const o = 4 * (i * K + (newest - j + K) % K), qs = hist[o + 3];
        p0[3 * s] = px; p0[3 * s + 1] = py; p0[3 * s + 2] = pz;
        px = hist[o]; py = hist[o + 1]; pz = hist[o + 2];
        p1[3 * s] = px; p1[3 * s + 1] = py; p1[3 * s + 2] = pz;
        sa[4 * s] = sp; sa[4 * s + 1] = qs;
        sa[4 * s + 2] = gain * (0.3 + 0.7 * Math.min(1, sp + 0.15)) * (1 - j / n); sa[4 * s + 3] = gain * (0.3 + 0.7 * Math.min(1, qs + 0.15)) * (1 - (j + 1) / n);
        sp = qs; s++;
      }
    }
    geometry.instanceCount = s;
    for (const name of ["p0", "p1", "sa"]) geometry.attributes[name].needsUpdate = true;
  };

  const group = new THREE.Group(); group.add(composite, ribbons); group.visible = false;
  ctx.scene.add(group);
  let shownA = null, shownB = null, seeded = false, spacing = 0, pace = ctx.pacer();
  return Object.assign(mode, {
    update(simTime, dt) {
      if (!group.visible) return;
      const f = ctx.frame(simTime); if (!f) return;
      if (!mode.range) { fitRange(f.a); [lo, hi] = mode.range; trailDt = 0.1 * L / hi / K; lifeT = 0.3 * L / hi; ctx.requestRender(); }
      if (f.a !== shownA) { texA.image.data = derive(f.a); texA.needsUpdate = true; shownA = f.a; }
      if (f.b !== shownB) { texB.image.data = derive(f.b); texB.needsUpdate = true; shownB = f.b; }
      material.uniforms.alpha.value = f.alpha; measure = texA.image.data; fitBox();
      if (Number.isFinite(f.b?.time - f.a?.time) && f.b.time > f.a.time) spacing = f.b.time - f.a.time;
      if (!seeded || dt < 0 || dt > (spacing ? 2 * spacing : 0.1 * L / hi)) {   // start, restart or seek: reseed
        for (let i = 0; i < N; i++) spawn(i, true);
        pushPoint(f);
        for (let j = 1; j < K; j++) { advect(trailDt, f); pushPoint(f); }   // grow full trails in the current field, so a paused view shows them
        seeded = true; acc = 0; build(); return;
      }
      dt = pace(dt, hi); if (dt <= 0) return;
      advect(dt, f); acc += dt;
      if (acc >= trailDt) { pushPoint(f); acc %= trailDt; }
      build();
    },
    setVisible(on) { if (on && !group.visible) { seeded = false; pace = ctx.pacer(); } group.visible = on; },
  });
}
