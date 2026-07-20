export function renderEvaluation(root, { api, setStatus }) {
  root.innerHTML = `
    <section class="page-heading"><h2>Evaluation</h2><p>Run a scenario set and retrieve its result.</p></section>
    <section class="panel">
      <form><div class="field"><label for="evaluation-scenarios">Scenarios (JSON array)</label><textarea id="evaluation-scenarios" name="scenarios">[]</textarea></div>
      <div class="inline-error" data-error></div><button class="primary" type="submit">Start evaluation</button></form>
    </section>
    <section class="panel"><h3>Result</h3><pre data-result>No evaluation started.</pre></section>`;
  const form = root.querySelector("form");
  const error = root.querySelector("[data-error]");
  const result = root.querySelector("[data-result]");
  let timer = null;
  let stopped = false;

  async function poll(id) {
    if (stopped) return;
    try {
      const data = await api.get(`evaluate/${encodeURIComponent(id)}`);
      result.textContent = JSON.stringify(data.scenarios, null, 2);
      setStatus(`Evaluation ${id} finished`);
    } catch (reason) {
      if (reason.message.startsWith("404")) {
        timer = setTimeout(() => poll(id), 2000);
      } else {
        error.textContent = reason.message;
        setStatus(reason.message, true);
      }
    }
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    error.textContent = "";
    try {
      const scenarios = JSON.parse(form.scenarios.value);
      if (!Array.isArray(scenarios)) throw new Error("Scenarios must be a JSON array");
      const data = await api.post("evaluate", { scenarios });
      result.textContent = `Running ${data.evaluation_id}…`;
      setStatus(`Evaluation ${data.evaluation_id} started`);
      poll(data.evaluation_id);
    } catch (reason) {
      error.textContent = reason.message;
      setStatus(reason.message, true);
    }
  });
  return () => { stopped = true; if (timer) clearTimeout(timer); };
}
