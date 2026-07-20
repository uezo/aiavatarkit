import { createConfigPanel } from "./config-panel.js";

const components = [
  ["Pipeline", "config/pipeline"],
  ["VAD", "config/vad"],
  ["STT", "config/stt"],
  ["LLM", "config/llm"],
  ["TTS", "config/tts"],
];

export function renderConfig(root, { api, setStatus }) {
  root.innerHTML = `
    <section class="page-heading"><h2>Config</h2><p>Runtime settings for the active pipeline and adapters.</p></section>
    <div class="config-grid" data-grid></div>`;
  const grid = root.querySelector("[data-grid]");
  let stopped = false;

  async function load() {
    setStatus("Loading configuration…");
    try {
      const [componentResults, adapters] = await Promise.all([
        Promise.all(components.map(async ([title, endpoint]) => [title, endpoint, await api.get(endpoint)])),
        api.get("config/adapters"),
      ]);
      if (stopped) return;
      grid.replaceChildren();
      for (const [title, endpoint, response] of componentResults) {
        grid.append(createConfigPanel({
          title: response.type ? `${title} · ${response.type}` : title,
          endpoint,
          config: response.config,
          api,
          setStatus,
        }));
      }
      for (const adapter of adapters) {
        grid.append(createConfigPanel({
          title: `Adapter · ${adapter.name} (${adapter.type})`,
          endpoint: `config/adapter/${encodeURIComponent(adapter.name)}`,
          config: adapter.config,
          api,
          setStatus,
        }));
      }
      setStatus();
    } catch (error) {
      setStatus(error.message, true);
      grid.innerHTML = `<div class="panel empty">Configuration could not be loaded.</div>`;
    }
  }
  load();
  return () => { stopped = true; };
}
