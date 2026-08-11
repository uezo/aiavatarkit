let fieldSequence = 0;

function readValue(input) {
  if (input.dataset.nullable === "true" && input.value === "") return null;
  if (input.dataset.kind === "boolean") return input.value === "true";
  if (input.dataset.kind === "number") return input.value === "" ? null : Number(input.value);
  if (input.dataset.kind === "json") return input.value === "" ? null : JSON.parse(input.value);
  return input.value;
}

function fieldForSpec(field) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.textContent = field.label;
  const fieldId = `config-field-${fieldSequence++}`;
  let input;
  if (field.kind === "boolean") {
    input = document.createElement("select");
    input.innerHTML = field.nullable
      ? `<option value=""></option><option value="true">true</option><option value="false">false</option>`
      : `<option value="true">true</option><option value="false">false</option>`;
    input.value = field.value == null ? "" : String(field.value);
  } else if (field.kind === "json") {
    input = document.createElement("textarea");
    input.value = field.value == null ? "" : JSON.stringify(field.value, null, 2);
    input.rows = 3;
  } else {
    input = document.createElement("input");
    input.type = field.secret ? "password" : field.kind === "number" ? "number" : "text";
    if (field.kind === "number") input.step = "any";
    input.value = field.value ?? "";
    if (field.secret && field.configured) input.placeholder = "Configured (leave blank to keep)";
  }
  input.name = field.name;
  input.id = fieldId;
  input.dataset.kind = field.kind;
  input.dataset.secret = String(Boolean(field.secret));
  input.dataset.nullable = String(Boolean(field.nullable));
  label.htmlFor = fieldId;
  wrapper.append(label, input);
  return wrapper;
}

export function createRuntimeConfigPanel({ section, api, setStatus, onApplied }) {
  const card = document.createElement("form");
  card.className = "config-card";
  const heading = document.createElement("h3");
  heading.textContent = section.component
    ? `${section.title}: ${section.component}`
    : section.title;
  const fields = document.createElement("div");
  fields.className = "config-fields";
  const error = document.createElement("div");
  error.className = "inline-error";
  const actions = document.createElement("div");
  actions.className = "config-actions";
  const apply = document.createElement("button");
  apply.className = "primary";
  apply.type = "submit";
  apply.textContent = "Apply";
  actions.append(apply);

  card.append(heading, fields, error, actions);
  fields.replaceChildren(...section.fields.map(fieldForSpec));

  card.addEventListener("submit", async event => {
    event.preventDefault();
    apply.disabled = true;
    error.textContent = "";
    try {
      const next = {};
      card.querySelectorAll("[name]").forEach(input => {
        if (input.dataset.secret === "true" && input.value === "") return;
        next[input.name] = readValue(input);
      });
      await api.post(`config/runtime/${encodeURIComponent(section.name)}`, {config: next});
      await onApplied(`${section.title} applied`);
    } catch (reason) {
      error.textContent = reason.message;
      setStatus(reason.message, true);
    } finally {
      apply.disabled = false;
    }
  });
  return card;
}
