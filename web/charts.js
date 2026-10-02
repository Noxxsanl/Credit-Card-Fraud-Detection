/*
 * Biểu đồ Chart.js của Fraud Console.
 *
 * Chỉ những biểu đồ cần trục số thực sự mới dùng Chart.js: đường cong chi phí (UI-03, trục
 * hoành thang log) và các đường PR/ROC (UI-04). Thác nước SHAP, thanh sai số bootstrap và ma
 * trận nhầm lẫn dựng bằng HTML/CSS trong index.html — chúng vốn là bảng có kèm thanh, nên
 * đọc được bằng trình đọc màn hình mà không cần bảng thay thế.
 *
 * Màu và chữ lấy từ biến CSS ở styles.css (:root), không gõ mã màu ở đây.
 * Phiên bản thư viện: web/vendor/README.md.
 */
(function () {
  "use strict";

  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  let themed = false;

  function theme() {
    if (themed) return;
    themed = true;
    const d = Chart.defaults;
    d.font.family = token("--font-sans");
    d.font.size = 12;
    d.color = token("--ink-2");
    d.borderColor = token("--grid");
    d.animation = false;                 // thanh trượt ngưỡng vẽ lại liên tục — không hoạt hình
    d.maintainAspectRatio = false;
    d.responsive = true;
    d.elements.line.borderWidth = 2;
    d.elements.line.borderCapStyle = "round";
    d.elements.line.borderJoinStyle = "round";
    d.elements.point.radius = 0;
    d.elements.point.hoverRadius = 5;
    d.elements.point.hoverBorderWidth = 2;
    const tip = d.plugins.tooltip;
    tip.backgroundColor = token("--surface");
    tip.titleColor = token("--ink");
    tip.bodyColor = token("--ink");
    tip.footerColor = token("--ink-2");
    tip.borderColor = token("--border-strong");
    tip.borderWidth = 1;
    tip.padding = 10;
    tip.cornerRadius = 6;
    tip.boxPadding = 4;
    tip.usePointStyle = true;
    tip.titleFont = { weight: "600" };
    const legend = d.plugins.legend.labels;
    legend.usePointStyle = true;
    legend.pointStyle = "line";
    legend.boxWidth = 18;
    legend.color = token("--ink");
  }

  /*
   * Vạch dọc đánh dấu ngưỡng. Danh sách vạch gắn thẳng lên đối tượng biểu đồ (chart.$markers)
   * thay vì đi qua options: bộ giải tùy chọn của Chart.js coi mảng là tùy chọn "theo chỉ số"
   * và không trả nguyên mảng về cho plugin.
   */
  const markersPlugin = {
    id: "fraudMarkers",
    afterDatasetsDraw(chart) {
      const lines = chart.$markers;
      if (!lines || !lines.length) return;
      const { ctx, chartArea: area, scales } = chart;
      const x = scales.x;
      ctx.save();
      ctx.font = `600 11px ${token("--font-sans")}`;
      ctx.textBaseline = "middle";
      lines.forEach((m, i) => {
        if (m.value == null || !Number.isFinite(m.value) || m.value <= 0) return;
        const px = x.getPixelForValue(m.value);
        if (px < area.left - 0.5 || px > area.right + 0.5) return;
        ctx.strokeStyle = m.color;
        ctx.lineWidth = m.width || 2;
        ctx.beginPath();
        ctx.moveTo(px, area.top);
        ctx.lineTo(px, area.bottom);
        ctx.stroke();

        const text = m.label;
        const w = ctx.measureText(text).width;
        const right = px + 6 + w <= area.right;
        const tx = right ? px + 6 : px - 6 - w;
        const ty = area.top + 10 + i * 18;
        ctx.fillStyle = token("--surface");
        ctx.fillRect(tx - 3, ty - 8, w + 6, 16);
        ctx.fillStyle = token("--ink");
        ctx.fillText(text, tx, ty);
      });
      ctx.restore();
    },
  };

  /* Đường gióng dọc theo con trỏ — người đọc nhắm vào một ngưỡng, không nhắm vào đường 2px. */
  const crosshairPlugin = {
    id: "fraudCrosshair",
    afterDatasetsDraw(chart) {
      if (!chart.$crosshair) return;
      const active = chart.tooltip && chart.tooltip.getActiveElements();
      if (!active || !active.length) return;
      const px = active[0].element.x;
      const { ctx, chartArea: area } = chart;
      ctx.save();
      ctx.strokeStyle = token("--axis");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, area.top);
      ctx.lineTo(px, area.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  Chart.register(markersPlugin, crosshairPlugin);

  const isPowerOfTen = (v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9;

  function axisTitle(text) {
    return { display: true, text, color: token("--ink-2"), font: { weight: "600" } };
  }

  /**
   * UI-03 — tổng chi phí ước tính mỗi ngày theo ngưỡng (trục hoành thang log).
   * Điểm dữ liệu: {x: τ, y: EUR/ngày, apd: cảnh báo/ngày, recall}.
   */
  function costCurve(canvas, fmt) {
    theme();
    const color = token("--series-1");
    const chart = new Chart(canvas, {
      type: "line",
      data: {
        datasets: [{
          label: "Chi phí ước tính mỗi ngày",
          data: [],
          parsing: false,
          normalized: true,
          borderColor: color,
          backgroundColor: color,
          pointHoverBackgroundColor: color,
          pointHoverBorderColor: token("--surface"),
        }],
      },
      options: {
        interaction: { mode: "index", axis: "x", intersect: false },
        layout: { padding: { top: 4, right: 8 } },
        scales: {
          x: {
            type: "logarithmic",
            min: 1e-4,
            max: 1,
            title: axisTitle("Ngưỡng τ (thang log)"),
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              callback: (v) => (isPowerOfTen(v) ? fmt.tauTick(v) : null),
            },
          },
          y: {
            beginAtZero: true,
            title: axisTitle("EUR mỗi ngày"),
            ticks: { callback: (v) => fmt.int(v) },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `τ = ${fmt.tau(items[0].raw.x)}`,
              label: (item) => `Chi phí ≈ ${fmt.int(item.raw.y)} EUR/ngày`,
              afterLabel: (item) => [
                `Cảnh báo ≈ ${fmt.int(item.raw.apd)}/ngày`,
                `Recall ${fmt.pct(item.raw.recall)}`,
              ],
            },
          },
        },
      },
    });
    chart.$crosshair = true;
    chart.$markers = [];
    return chart;
  }

  /**
   * Đường cong trên mặt phẳng [0, 1] × [0, 1]: PR hoặc ROC, một hoặc nhiều đường.
   *
   * series: [{label, x: [], y: [], color, width}]
   * reference: {label, points: [{x, y}, …]} — đường cơ sở (PR) hoặc đường chéo (ROC).
   */
  function unitCurves(canvas, { series, reference, xTitle, yTitle, fmt, legend = true }) {
    theme();
    const datasets = series.map((s) => ({
      label: s.label,
      data: s.x.map((x, i) => ({ x, y: s.y[i] })),
      parsing: false,
      normalized: false,
      borderColor: s.color,
      backgroundColor: s.color,
      borderWidth: s.width || 2,
      pointHoverBackgroundColor: s.color,
      pointHoverBorderColor: token("--surface"),
    }));
    if (reference) {
      datasets.push({
        label: reference.label,
        data: reference.points,
        parsing: false,
        borderColor: token("--ink-3"),
        backgroundColor: token("--ink-3"),
        borderWidth: 1,
        pointHoverRadius: 0,
      });
    }
    return new Chart(canvas, {
      type: "line",
      data: { datasets },
      options: {
        interaction: { mode: "nearest", axis: "xy", intersect: false },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: axisTitle(xTitle), ticks: { callback: (v) => fmt.num(v, 1) } },
          y: { type: "linear", min: 0, max: 1, title: axisTitle(yTitle), ticks: { callback: (v) => fmt.num(v, 1) } },
        },
        plugins: {
          legend: { display: legend, position: "bottom", align: "start" },
          tooltip: {
            callbacks: {
              title: (items) => items[0].dataset.label,
              label: (item) => `${xTitle} ${fmt.num(item.raw.x, 3)} · ${yTitle} ${fmt.num(item.raw.y, 3)}`,
            },
          },
        },
      },
    });
  }

  window.FraudCharts = { costCurve, unitCurves, token };
})();
