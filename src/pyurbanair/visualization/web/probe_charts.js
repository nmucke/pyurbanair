"use strict";
window.ForwardCharts = class {
  constructor(container, data) {
    this.data = data;
    this.time = data.times?.[0] ?? 0;
    const groups = new Map();
    const palette = ["#73dbd4", "#ffbe73", "#b4a7ff", "#f17c9b", "#acd77e", "#7cbbf1"];
    (data.probes || []).forEach((probe, index) => {
      const height = probe.actual?.z ?? probe.requested?.z;
      const key = Number.isFinite(height) ? String(height) : "unknown";
      if (!groups.has(key)) groups.set(key, {height, probes: []});
      const color = typeof probe.color === "string" && /^#[\da-f]{3,8}$/i.test(probe.color) ? probe.color : palette[index % palette.length];
      groups.get(key).probes.push({...probe, color, values: probe.values || []});
    });
    this.low = Infinity;
    this.high = -Infinity;
    groups.forEach(group => group.probes.forEach(probe => probe.values.forEach(value => {
      if (value !== null && Number.isFinite(value)) { this.low = Math.min(this.low, value); this.high = Math.max(this.high, value); }
    })));
    if (!Number.isFinite(this.low)) { this.low = 0; this.high = 1; }
    if (["u", "v", "w"].includes(data.field)) { this.high = Math.max(Math.abs(this.low), Math.abs(this.high), 1e-9); this.low = -this.high; }
    else this.low = Math.min(0, this.low);
    if (this.low === this.high) this.high = this.low + 1;
    this.panels = [...groups.values()].sort((a, b) => (a.height ?? Infinity) - (b.height ?? Infinity)).map(group => {
      const panel = document.createElement("section");
      panel.className = "probe-chart sensor-chart";
      panel.dataset.height = Number.isFinite(group.height) ? String(group.height) : "unknown";
      panel.dataset.traceCount = String(group.probes.length);
      const heading = document.createElement("div"); heading.className = "probe-heading";
      const title = document.createElement("h3");
      title.textContent = Number.isFinite(group.height) ? `Height · z = ${group.height.toLocaleString(undefined, {maximumFractionDigits: 3})} m` : "Virtual probe samples";
      const time = document.createElement("span"); time.className = "sensor-time";
      heading.append(title, time);
      const canvas = document.createElement("canvas"); canvas.className = "series-chart";
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", `${data.field || "Velocity"} [${data.units || "m/s"}] at ${group.probes.map(probe => probe.label || probe.id).join(", ")}, ${title.textContent}`);
      const legend = document.createElement("div"); legend.className = "sensor-legend";
      group.probes.forEach(probe => {
        const item = document.createElement("span"); item.dataset.probeId = String(probe.id);
        const swatch = document.createElement("canvas"); swatch.className = "trace-color"; swatch.width = swatch.height = 14; swatch.setAttribute("aria-hidden", "true");
        const context = swatch.getContext("2d"); context.fillStyle = probe.color; context.beginPath(); context.arc(7, 7, 5, 0, 2 * Math.PI); context.fill();
        const label = document.createElement("span"); label.textContent = String(probe.label || probe.id);
        if (probe.actual) item.title = `x = ${probe.actual.x} m, y = ${probe.actual.y} m, z = ${probe.actual.z} m`;
        item.append(swatch, label); legend.append(item);
      });
      panel.append(heading, canvas, legend); container.append(panel);
      return {...group, canvas, time};
    });
    if (!this.panels.length) {
      const empty = document.createElement("p"); empty.className = "probe-empty";
      empty.textContent = "No virtual probes were requested for this bundle. Probe locations can be added in a new render without rerunning the simulation.";
      container.append(empty);
    }
    this.observer = new ResizeObserver(() => this.draw());
    this.panels.forEach(panel => this.observer.observe(panel.canvas));
    this.draw();
  }
  update(time) { if (this.time === time) return; this.time = time; this.draw(); }
  draw() {
    const times = this.data.times || [];
    if (!times.length) return;
    this.panels.forEach(({probes, canvas, time}) => {
      time.textContent = `t = ${this.time.toFixed(3)} s`;
      const width = canvas.clientWidth, height = canvas.clientHeight;
      if (!width || !height) return;
      const density = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * density); canvas.height = Math.round(height * density);
      const ctx = canvas.getContext("2d"); ctx.scale(density, density);
      const start = times[0], end = times[times.length - 1];
      const left = 40, right = width - 9, top = 14, bottom = height - 25;
      const x = t => left + (t - start) / (end - start || 1) * Math.max(1, right - left);
      const y = value => top + (this.high - value) / (this.high - this.low) * (bottom - top);
      ctx.font = "10px Arial"; ctx.lineWidth = 1;
      [0, .5, 1].forEach(fraction => {
        const value = this.low + fraction * (this.high - this.low);
        ctx.strokeStyle = "#254552"; ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
        ctx.fillStyle = "#adc6cd"; ctx.fillText(value.toLocaleString(undefined, {maximumFractionDigits: 2}), 1, y(value) + 3);
      });
      ctx.fillStyle = "#adc6cd";
      ctx.fillText(`${start.toLocaleString()} s`, left, height - 7);
      const endText = `${end.toLocaleString()} s`; ctx.fillText(endText, Math.max(left, right - ctx.measureText(endText).width), height - 7);
      probes.forEach(probe => {
        ctx.strokeStyle = probe.color; ctx.lineWidth = 1.7; ctx.beginPath();
        let gap = true;
        probe.values.forEach((value, index) => {
          if (index >= times.length || value === null || !Number.isFinite(value)) { gap = true; return; }
          if (gap) ctx.moveTo(x(times[index]), y(value)); else ctx.lineTo(x(times[index]), y(value));
          gap = false;
        });
        ctx.stroke();
        let selected = -1;
        for (let index = 0; index < times.length && times[index] <= this.time; index++) selected = index;
        const value = probe.values[selected];
        if (selected >= 0 && value !== null && Number.isFinite(value)) { ctx.fillStyle = probe.color; ctx.beginPath(); ctx.arc(x(times[selected]), y(value), 2.5, 0, 2 * Math.PI); ctx.fill(); }
      });
      ctx.strokeStyle = "#ffd788"; ctx.lineWidth = 1;
      const cursor = x(Math.max(start, Math.min(end, this.time)));
      ctx.beginPath(); ctx.moveTo(cursor, top - 4); ctx.lineTo(cursor, bottom + 1); ctx.stroke();
    });
  }
};
