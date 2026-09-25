import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (await readFile(new URL("../../aiavatar/admin/static/admin-app.js", import.meta.url), "utf8"))
  .replace(/^import .*;\n/gm, "")
  .replace(/\nstart\(\);\s*$/, "\nreturn start();");

async function harness(capabilities, hash = "") {
  const requests = [];
  const renders = [];
  const listeners = {};
  const buttons = [];
  const element = () => ({
    dataset: {}, classList: { toggle() {} },
    replaceChildren() {}, focus() {}, addEventListener() {},
  });
  const navigation = {
    append: button => buttons.push(button),
    querySelectorAll: () => buttons,
  };
  const elements = { "#navigation": navigation, "#content": element(), "#global-status": element() };
  const document = { querySelector: selector => elements[selector], createElement: element };
  const window = { location: { hash }, addEventListener: (name, handler) => { listeners[name] = handler; } };
  const api = {
    async get(path) {
      requests.push(path);
      return path === "capabilities" ? capabilities : {};
    },
  };
  const render = name => (_root, { api }) => {
    renders.push(name);
    if (name === "config") void api.get("config/runtime");
  };
  await new Function("document", "window", "api", "renderMetrics", "renderLogs", "renderConfig", "renderEvaluation", source)(
    document, window, api, ...["metrics", "logs", "config", "evaluation"].map(render),
  );
  return {
    requests, renders, window,
    sections: buttons.map(button => button.dataset.section),
    navigate(name) {
      window.location.hash = `#${name}`;
      listeners.hashchange();
    },
  };
}

test("Metrics/Logs-only capabilities reject stale Config navigation without a config request", async () => {
  const app = await harness({ config: false, evaluation: false }, "#config");
  assert.deepEqual(app.sections, ["metrics", "logs"]);
  assert.deepEqual(app.renders, ["metrics"]);
  assert.equal(app.window.location.hash, "metrics");
  app.navigate("logs");
  app.navigate("config");
  app.navigate("evaluation");
  assert.deepEqual(app.renders, ["metrics", "logs", "metrics", "metrics"]);
  assert.deepEqual(app.requests, ["capabilities"]);
});

for (const capabilities of [{ config: true, evaluation: false }, { evaluation: false }]) {
  test(`Config remains available when its capability is ${capabilities.config === true ? "true" : "omitted"}`, async () => {
    const app = await harness(capabilities, "#config");
    assert.deepEqual(app.sections, ["metrics", "logs", "config"]);
    assert.deepEqual(app.renders, ["config"]);
    assert.deepEqual(app.requests, ["capabilities", "config/runtime"]);
  });
}

test("Evaluation remains independently gated when Config is disabled", async () => {
  const app = await harness({ config: false, evaluation: true }, "#evaluation");
  assert.deepEqual(app.sections, ["metrics", "logs", "evaluation"]);
  assert.deepEqual(app.renders, ["evaluation"]);
  assert.deepEqual(app.requests, ["capabilities"]);
});
