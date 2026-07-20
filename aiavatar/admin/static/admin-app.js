import { api } from "./admin-api.js";
import { renderMetrics } from "./metrics-view.js";
import { renderLogs } from "./logs-view.js";
import { renderConfig } from "./config-view.js";
import { renderEvaluation } from "./evaluation-view.js";

const navigation = document.querySelector("#navigation");
const content = document.querySelector("#content");
const status = document.querySelector("#global-status");
let cleanup = null;

function setStatus(message = "", isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

async function start() {
  try {
    const capabilities = await api.get("capabilities");
    const sections = [
      ["metrics", "Metrics", renderMetrics],
      ["logs", "Logs", renderLogs],
      ["config", "Config", renderConfig],
    ];
    if (capabilities.evaluation) sections.push(["evaluation", "Evaluation", renderEvaluation]);

    function activate(name) {
      const section = sections.find(item => item[0] === name) || sections[0];
      if (cleanup) cleanup();
      content.replaceChildren();
      navigation.querySelectorAll("button").forEach(button => {
        button.classList.toggle("active", button.dataset.section === section[0]);
      });
      window.location.hash = section[0];
      cleanup = section[2](content, { api, setStatus }) || null;
      content.focus({ preventScroll: true });
    }

    for (const [name, label] of sections) {
      const button = document.createElement("button");
      button.className = "nav-button";
      button.dataset.section = name;
      button.textContent = label;
      button.addEventListener("click", () => activate(name));
      navigation.append(button);
    }
    window.addEventListener("hashchange", () => activate(window.location.hash.slice(1)));
    activate(window.location.hash.slice(1));
  } catch (error) {
    setStatus(error.message, true);
    content.innerHTML = `<div class="empty">Admin could not be initialized.</div>`;
  }
}

start();
