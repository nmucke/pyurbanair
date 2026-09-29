"use strict";
window.ForwardCharts = class {
  constructor(container, data) {
    this.data = data;
    this.time = data.times[0] || 0;
    this.panels = data.probes.map(probe => {
      const panel = document.createElement("section");
      panel.className = "probe-chart";
      const title = document.createElement("h3");
      title.textContent = `${probe.id} · ${data.field} [${data.units}]`;
      const canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", `${probe.id} virtual probe series`);
      panel.append(title, canvas);
      container.append(panel);
      return {probe, canvas};
    });
    this.observer = new ResizeObserver(() => this.draw());
    this.panels.forEach(panel => this.observer.observe(panel.canvas));
    this.draw();
  }
  update(time) { this.time = time; this.draw(); }
  draw() {
    const times = this.data.times;
    if (!times.length) return;
    this.panels.forEach(({probe, canvas}) => {
      const width = canvas.clientWidth, height = canvas.clientHeight;
      const density = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(width * density));
      canvas.height = Math.max(1, Math.round(height * density));
      const ctx = canvas.getContext("2d");
      ctx.scale(density, density);
      const valid = probe.values.filter(value => value !== null && Number.isFinite(value));
      let low = valid.length ? Math.min(...valid) : 0;
      let high = valid.length ? Math.max(...valid) : 1;
      if (low === high) { low -= 0.5; high += 0.5; }
      const start = times[0], end = times[times.length - 1];
      const x = t => 38 + (t - start) / (end - start || 1) * Math.max(1, width - 50);
      const y = v => 12 + (high - v) / (high - low) * (height - 34);
      ctx.strokeStyle = "#31505d";
      ctx.beginPath(); ctx.moveTo(38, 10); ctx.lineTo(38, height - 22); ctx.lineTo(width - 8, height - 22); ctx.stroke();
      ctx.fillStyle = "#adc6cd"; ctx.font = "11px Arial";
      ctx.fillText(high.toPrecision(3), 0, 16); ctx.fillText(low.toPrecision(3), 0, height - 24);
      ctx.fillText(`${start.toFixed(2)} s`, 38, height - 5);
      ctx.fillText(`${end.toFixed(2)} s`, Math.max(40, width - 70), height - 5);
      ctx.strokeStyle = probe.color; ctx.lineWidth = 2; ctx.beginPath();
      let gap = true;
      probe.values.forEach((value, i) => {
        if (value === null || !Number.isFinite(value)) { gap = true; return; }
        if (gap) ctx.moveTo(x(times[i]), y(value)); else ctx.lineTo(x(times[i]), y(value));
        gap = false;
      });
      ctx.stroke();
      if (times.length === 1 && valid.length) { ctx.beginPath(); ctx.arc(x(start), y(valid[0]), 3, 0, 2 * Math.PI); ctx.fillStyle = probe.color; ctx.fill(); }
      ctx.strokeStyle = "#ffd788"; ctx.lineWidth = 1;
      const cursor = x(Math.max(start, Math.min(end, this.time)));
      ctx.beginPath(); ctx.moveTo(cursor, 8); ctx.lineTo(cursor, height - 20); ctx.stroke();
    });
  }
};
