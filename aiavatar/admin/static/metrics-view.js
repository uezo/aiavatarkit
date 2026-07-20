const phases = [
  ["avg_silence_detection_phase", "Silence detection", "#94a3b8"],
  ["avg_streaming_stt_finalization_phase", "Streaming STT finalization", "#60a5fa"],
  ["avg_turn_end_gate_phase", "Turn-end gate", "#22d3ee"],
  ["avg_stt_phase", "STT", "#2dd4bf"],
  ["avg_stop_response_phase", "Stop current response", "#4ade80"],
  ["avg_before_llm_phase", "Before-LLM handlers", "#fbbf24"],
  ["avg_llm_phase", "LLM", "#fb923c"],
  ["avg_processing_phase", "Processing", "#f472b6"],
  ["avg_tts_phase", "TTS", "#f87171"],
];

function seconds(value) {
  return value == null ? "—" : `${value.toFixed(3)} s`;
}

function chartTheme() {
  const dark = document.documentElement.dataset.theme === "dark";
  return dark
    ? { grid: "#3f3f46", text: "#a1a1aa", title: "#d4d4d8", tooltip: "#27272a", tooltipBorder: "#52525b" }
    : { grid: "#e4e4e7", text: "#71717a", title: "#52525b", tooltip: "#18181b", tooltipBorder: "#18181b" };
}

export function renderMetrics(root, { api, setStatus }) {
  root.innerHTML = `
    <section class="page-heading"><h2>Metrics</h2><p>Latency measured from the end of the user's speech.</p></section>
    <form class="toolbar">
      <div class="field"><label for="metrics-period">Period</label><select id="metrics-period" name="period"><option>1h</option><option>6h</option><option selected>24h</option><option>7d</option><option>30d</option></select></div>
      <div class="field"><label for="metrics-interval">Interval</label><select id="metrics-interval" name="interval"><option>1m</option><option>5m</option><option>15m</option><option selected>1h</option><option>1d</option></select></div>
      <button class="primary" type="submit">Refresh</button>
    </form>
    <div class="cards" data-summary></div>
    <section class="panel"><h3>First response breakdown</h3><p class="hint" data-coverage></p><div class="empty" data-empty hidden>No detailed timing data in this period.</div><div class="chart-wrap" data-chart><canvas></canvas></div></section>`;
  const form = root.querySelector("form");
  const cards = root.querySelector("[data-summary]");
  const coverage = root.querySelector("[data-coverage]");
  const empty = root.querySelector("[data-empty]");
  const chartWrap = root.querySelector("[data-chart]");
  let chart = null;
  let stopped = false;

  function syncChartTheme() {
    if (!chart) return;
    const theme = chartTheme();
    chart.options.scales.x.grid.color = theme.grid;
    chart.options.scales.x.ticks.color = theme.text;
    chart.options.scales.y.grid.color = theme.grid;
    chart.options.scales.y.ticks.color = theme.text;
    chart.options.scales.y.title.color = theme.title;
    chart.options.plugins.legend.labels.color = theme.title;
    chart.options.plugins.tooltip.backgroundColor = theme.tooltip;
    chart.options.plugins.tooltip.borderColor = theme.tooltipBorder;
    chart.update("none");
  }

  window.addEventListener("admin-theme-change", syncChartTheme);

  function card(label, value, sub = "") {
    const element = document.createElement("div");
    element.className = "card";
    element.innerHTML = `<div class="label"></div><div class="value"></div><div class="sub"></div>`;
    element.querySelector(".label").textContent = label;
    element.querySelector(".value").textContent = value;
    element.querySelector(".sub").textContent = sub;
    return element;
  }

  async function load(event) {
    event?.preventDefault();
    const period = form.period.value;
    const interval = form.interval.value;
    setStatus("Loading metrics…");
    try {
      const [summary, timeline] = await Promise.all([
        api.get(`metrics/summary?period=${encodeURIComponent(period)}`),
        api.get(`metrics/timeline?period=${encodeURIComponent(period)}&interval=${encodeURIComponent(interval)}`),
      ]);
      if (stopped) return;
      cards.replaceChildren(
        card("Requests", String(summary.total_requests), `${summary.error_count} errors`),
        card("Average first response", seconds(summary.avg_first_response_time)),
        card("Median first response", seconds(summary.p50_first_response_time)),
        card("P95 first response", seconds(summary.p95_first_response_time)),
      );
      const ratio = summary.success_count ? Math.round(summary.measured_count / summary.success_count * 100) : 0;
      coverage.textContent = `${summary.measured_count} measured / ${summary.success_count} successful (${ratio}% coverage). Older, text-only, and unsupported VAD records are excluded.`;
      empty.hidden = summary.measured_count !== 0;
      chartWrap.hidden = summary.measured_count === 0;
      if (summary.measured_count === 0) {
        if (chart) chart.destroy();
        chart = null;
        setStatus();
        return;
      }
      if (!window.Chart) throw new Error("Chart library is unavailable");
      if (chart) chart.destroy();
      const theme = chartTheme();
      chart = new Chart(root.querySelector("canvas"), {
        type: "bar",
        data: {
          labels: timeline.buckets.map(bucket => new Date(`${bucket.timestamp}Z`).toLocaleString()),
          datasets: phases.map(([key, label, color]) => ({
            label,
            data: timeline.buckets.map(bucket => bucket[key] || 0),
            backgroundColor: color,
            stack: "latency",
          })),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          scales: {
            x: {
              stacked: true,
              grid: { color: theme.grid },
              ticks: { color: theme.text, maxRotation: 45 },
            },
            y: {
              stacked: true,
              beginAtZero: true,
              grid: { color: theme.grid },
              ticks: { color: theme.text },
              title: { display: true, text: "Seconds", color: theme.title },
            },
          },
          plugins: {
            legend: {
              position: "bottom",
              labels: { color: theme.title, boxWidth: 12, boxHeight: 12, padding: 16 },
            },
            tooltip: {
              backgroundColor: theme.tooltip,
              borderColor: theme.tooltipBorder,
              borderWidth: 1,
              titleColor: "#fafafa",
              bodyColor: "#e4e4e7",
              padding: 10,
            },
          },
        },
      });
      setStatus();
    } catch (error) {
      setStatus(error.message, true);
    }
  }
  form.addEventListener("submit", load);
  load();
  return () => {
    stopped = true;
    window.removeEventListener("admin-theme-change", syncChartTheme);
    if (chart) chart.destroy();
  };
}
