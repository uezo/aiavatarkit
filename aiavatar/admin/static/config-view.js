import { createRuntimeConfigPanel } from "./config-panel.js";

export function renderConfig(root, { api, setStatus }) {
  root.innerHTML = `
    <section class="page-heading">
      <h2>Config</h2>
      <p>Changes apply to the running process only and are not persisted.</p>
    </section>
    <div class="config-grid" data-grid></div>`;
  const grid = root.querySelector("[data-grid]");
  let stopped = false;

  async function load(appliedStatus = null) {
    if (appliedStatus === null) setStatus("Loading configuration…");
    try {
      const schema = await api.get("config/runtime");
      if (stopped) return;
      grid.replaceChildren();
      for (const section of schema.sections) {
        grid.append(createRuntimeConfigPanel({
          section,
          api,
          setStatus,
          onApplied: status => load(status),
        }));
      }
      setStatus(appliedStatus ?? "");
    } catch (error) {
      setStatus(error.message, true);
      grid.innerHTML = `<div class="panel empty">Configuration could not be loaded.</div>`;
    }
  }

  load();
  return () => { stopped = true; };
}
