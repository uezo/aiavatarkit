const speechPhases = [
  ["avg_silence_detection_phase", "Silence detection", "#94a3b8"],
  ["avg_streaming_stt_finalization_phase", "Streaming STT finalization", "#60a5fa"],
  ["avg_turn_end_gate_phase", "Turn-end gate", "#22d3ee"],
  ["avg_stt_phase", "Input / STT", "#2dd4bf"],
  ["avg_stop_response_phase", "Stop current response", "#4ade80"],
  ["avg_before_llm_phase", "Before-LLM handlers", "#fbbf24"],
  ["avg_llm_phase", "LLM", "#fb923c"],
  ["avg_processing_phase", "Processing", "#f472b6"],
  ["avg_tts_phase", "TTS", "#f87171"],
];

const pipelinePhases = speechPhases.slice(3);

function seconds(value) {
  return value == null ? "—" : `${value.toFixed(3)} s`;
}

function coverage(summary) {
  const ratio = summary.success_count
    ? Math.round(summary.measured_count / summary.success_count * 100)
    : 0;
  return `${summary.measured_count} measured / ${summary.success_count} successful (${ratio}% coverage).`;
}

function chartTheme() {
  const dark = document.documentElement.dataset.theme === "dark";
  return dark
    ? { grid: "#3f3f46", text: "#a1a1aa", title: "#d4d4d8", tooltip: "#27272a", tooltipBorder: "#52525b" }
    : { grid: "#e4e4e7", text: "#71717a", title: "#52525b", tooltip: "#18181b", tooltipBorder: "#18181b" };
}

function card(label, value, sub = "") {
  const element = document.createElement("div");
  element.className = "card";
  element.innerHTML = `<div class="label"></div><div class="value"></div><div class="sub"></div>`;
  element.querySelector(".label").textContent = label;
  element.querySelector(".value").textContent = value;
  element.querySelector(".sub").textContent = sub;
  return element;
}

function createChart(canvas, buckets, phases) {
  const theme = chartTheme();
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: buckets.map(bucket => new Date(`${bucket.timestamp}Z`).toLocaleString()),
      datasets: phases.map(([key, label, color]) => ({
        label,
        data: buckets.map(bucket => bucket[key] ?? 0),
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
}

export function renderMetrics(root, { api, setStatus }) {
  root.innerHTML = `
    <section class="page-heading"><h2>Metrics</h2></section>
    <form class="toolbar">
      <div class="field"><label for="metrics-period">Period</label><select id="metrics-period" name="period"><option>1h</option><option>6h</option><option selected>24h</option><option>7d</option><option>30d</option></select></div>
      <div class="field"><label for="metrics-interval">Interval</label><select id="metrics-interval" name="interval"><option>1m</option><option>5m</option><option>15m</option><option selected>1h</option><option>1d</option></select></div>
      <button class="primary" type="submit">Refresh</button>
    </form>
    <div class="cards" data-summary></div>
    <div class="empty" data-empty hidden>No metrics in this period.</div>
    <div data-channels></div>`;
  const form = root.querySelector("form");
  const cards = root.querySelector("[data-summary]");
  const empty = root.querySelector("[data-empty]");
  const channelContainer = root.querySelector("[data-channels]");
  let charts = [];
  let stopped = false;

  function clearCharts() {
    charts.forEach(chart => chart.destroy());
    charts = [];
  }

  function syncChartTheme() {
    const theme = chartTheme();
    charts.forEach(chart => {
      chart.options.scales.x.grid.color = theme.grid;
      chart.options.scales.x.ticks.color = theme.text;
      chart.options.scales.y.grid.color = theme.grid;
      chart.options.scales.y.ticks.color = theme.text;
      chart.options.scales.y.title.color = theme.title;
      chart.options.plugins.legend.labels.color = theme.title;
      chart.options.plugins.tooltip.backgroundColor = theme.tooltip;
      chart.options.plugins.tooltip.borderColor = theme.tooltipBorder;
      chart.update("none");
    });
  }

  window.addEventListener("admin-theme-change", syncChartTheme);

  function renderChannel(channelMetrics) {
    const panel = document.createElement("section");
    panel.className = "panel";
    panel.innerHTML = `
      <h3 data-channel></h3>
      <div class="cards" data-channel-summary></div>
      <h3 data-latency-title></h3>
      <p class="hint" data-coverage></p>
      <div class="empty" data-latency-empty hidden>No timing data in this period.</div>
      <div class="chart-wrap" data-latency-chart><canvas></canvas></div>`;

    const channelName = channelMetrics.channel || "Unclassified";
    panel.querySelector("[data-channel]").textContent = `Channel: ${channelName}`;
    const pipelineSummary = channelMetrics.pipeline_summary;
    const useSpeech = channelMetrics.speech_summary.measured_count !== 0;
    const summary = useSpeech ? channelMetrics.speech_summary : pipelineSummary;
    const buckets = useSpeech ? channelMetrics.speech_buckets : channelMetrics.pipeline_buckets;
    const phases = useSpeech ? speechPhases : pipelinePhases;
    const latencyTitle = useSpeech
      ? "Speech end to first output"
      : "Pipeline first output";
    const coverageDetail = useSpeech
      ? "Includes speech detection, turn-end, and Pipeline phases."
      : "Pipeline invocation to first content output.";

    panel.querySelector("[data-channel-summary]").replaceChildren(
      card("Requests", String(pipelineSummary.total_requests), `${pipelineSummary.error_count} errors`),
      card("Average first output", seconds(summary.avg_first_response_time)),
      card("Median first output", seconds(summary.p50_first_response_time)),
      card("P95 first output", seconds(summary.p95_first_response_time)),
    );
    panel.querySelector("[data-latency-title]").textContent = latencyTitle;
    panel.querySelector("[data-coverage]").textContent = `${coverage(summary)} ${coverageDetail}`;

    const latencyEmpty = panel.querySelector("[data-latency-empty]");
    const latencyChart = panel.querySelector("[data-latency-chart]");
    latencyEmpty.hidden = summary.measured_count !== 0;
    latencyChart.hidden = summary.measured_count === 0;

    channelContainer.append(panel);

    if (summary.measured_count !== 0) {
      charts.push(createChart(
        panel.querySelector("[data-latency-chart] canvas"),
        buckets,
        phases,
      ));
    }
  }

  async function load(event) {
    event?.preventDefault();
    const period = form.period.value;
    const interval = form.interval.value;
    setStatus("Loading metrics…");
    try {
      const metrics = await api.get(
        `metrics/by-channel?period=${encodeURIComponent(period)}&interval=${encodeURIComponent(interval)}`,
      );
      if (stopped) return;
      clearCharts();
      cards.replaceChildren(
        card("Requests", String(metrics.total_requests)),
        card("Successful", String(metrics.success_count)),
        card("Errors", String(metrics.error_count)),
        card("Channels", String(metrics.channels.length)),
      );
      channelContainer.replaceChildren();
      empty.hidden = metrics.channels.length !== 0;
      const hasMeasurements = metrics.channels.some(item =>
        item.pipeline_summary.measured_count || item.speech_summary.measured_count
      );
      if (hasMeasurements && !window.Chart) {
        throw new Error("Chart library is unavailable");
      }
      metrics.channels.forEach(renderChannel);
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
    clearCharts();
  };
}
