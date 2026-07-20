let fieldSequence = 0;

function fieldFor(key, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.textContent = key.replaceAll("_", " ");
  const fieldId = `config-field-${fieldSequence++}`;
  let input;
  if (typeof value === "boolean") {
    input = document.createElement("select");
    input.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
    input.value = String(value);
  } else if (value !== null && typeof value === "object") {
    input = document.createElement("textarea");
    input.value = JSON.stringify(value, null, 2);
    input.rows = 3;
  } else {
    input = document.createElement("input");
    input.type = typeof value === "number" ? "number" : "text";
    if (typeof value === "number") input.step = "any";
    input.value = value ?? "";
  }
  input.name = key;
  input.id = fieldId;
  label.htmlFor = fieldId;
  input.dataset.kind = value === null ? "null" : typeof value;
  wrapper.append(label, input);
  return wrapper;
}

function readValue(input) {
  if (input.dataset.kind === "boolean") return input.value === "true";
  if (input.dataset.kind === "number") return input.value === "" ? null : Number(input.value);
  if (input.dataset.kind === "object") return JSON.parse(input.value);
  if (input.dataset.kind === "null") {
    if (input.value === "") return null;
    try { return JSON.parse(input.value); }
    catch (_) { return input.value; }
  }
  return input.value;
}

export function createConfigPanel({ title, endpoint, config, api, setStatus }) {
  const card = document.createElement("form");
  card.className = "config-card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const fields = document.createElement("div");
  fields.className = "config-fields";
  Object.entries(config).forEach(([key, value]) => fields.append(fieldFor(key, value)));
  const error = document.createElement("div");
  error.className = "inline-error";
  const actions = document.createElement("div");
  actions.className = "config-actions";
  const save = document.createElement("button");
  save.className = "primary";
  save.type = "submit";
  save.textContent = "Save";
  actions.append(save);
  card.append(heading, fields, error, actions);
  card.addEventListener("submit", async event => {
    event.preventDefault();
    save.disabled = true;
    error.textContent = "";
    try {
      const next = {};
      card.querySelectorAll("[name]").forEach(input => { next[input.name] = readValue(input); });
      await api.post(endpoint, { config: next });
      setStatus(`${title} saved`);
    } catch (reason) {
      error.textContent = reason.message;
      setStatus(reason.message, true);
    } finally {
      save.disabled = false;
    }
  });
  return card;
}
