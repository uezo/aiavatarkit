function textBlock(className, label, value) {
  const block = document.createElement("div");
  block.className = className;
  const heading = document.createElement("div");
  heading.className = "label";
  heading.textContent = label;
  const content = document.createElement("div");
  content.textContent = value || "—";
  block.append(heading, content);
  return block;
}

const timingPhases = [
  ["silence_detection", "Silence detection", "#94a3b8"],
  ["streaming_stt_finalization", "Streaming STT finalization", "#60a5fa"],
  ["turn_end_gate", "Turn-end gate", "#22d3ee"],
  ["stt", "STT", "#2dd4bf"],
  ["stop_response", "Stop current response", "#4ade80"],
  ["before_llm", "Before-LLM handlers", "#fbbf24"],
  ["llm", "LLM to first chunk", "#fb923c"],
  ["processing", "Response processing", "#f472b6"],
  ["tts", "TTS to first audio", "#f87171"],
];

function renderTiming(timing) {
  const section = document.createElement("section");
  section.className = "turn-timing";
  const header = document.createElement("div");
  header.className = "turn-timing-header";
  header.innerHTML = `<span>First response breakdown</span><strong>${timing.total_first_response.toFixed(3)} s</strong>`;
  const bar = document.createElement("div");
  bar.className = "turn-timing-bar";
  bar.setAttribute("aria-label", `First response ${timing.total_first_response.toFixed(3)} seconds`);
  const legend = document.createElement("div");
  legend.className = "turn-timing-legend";
  for (const [key, label, color] of timingPhases) {
    const value = Math.max(Number(timing[key]) || 0, 0);
    if (value > 0) {
      const segment = document.createElement("span");
      segment.style.flexGrow = String(value);
      segment.style.backgroundColor = color;
      segment.title = `${label}: ${value.toFixed(3)} s`;
      bar.append(segment);
    }
    const item = document.createElement("div");
    item.innerHTML = `<i style="background:${color}"></i><span></span><strong>${value.toFixed(3)} s</strong>`;
    item.querySelector("span").textContent = label;
    legend.append(item);
  }
  section.append(header, bar, legend);
  return section;
}

function voiceButton(label, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "voice-button";
  button.dataset.label = label;
  button.textContent = `▶ ${label}`;
  button.addEventListener("click", () => onClick(button));
  return button;
}

function renderLog(log, { voiceRecorderEnabled, playVoice }) {
  const entry = document.createElement("article");
  entry.className = "log-entry";
  const meta = document.createElement("div");
  meta.className = "meta";
  const values = [
    `time: ${log.created_at || "—"}`,
    `session: ${log.session_id || "—"}`,
    `user: ${log.user_id || "—"}`,
    `context: ${log.context_id || "—"}`,
    `transaction: ${log.transaction_id || "—"}`,
  ];
  for (const value of values) {
    const span = document.createElement("span");
    span.textContent = value;
    meta.append(span);
  }
  const voiceActions = document.createElement("div");
  voiceActions.className = "voice-actions";
  if (voiceRecorderEnabled && log.transaction_id) {
    voiceActions.append(
      voiceButton("Request audio", button => playVoice(log, "request", button)),
      voiceButton("Response audio", button => playVoice(log, "response", button)),
    );
  }
  const conversation = document.createElement("div");
  conversation.className = "conversation";
  conversation.append(
    textBlock("bubble", "Request", log.request_text),
    textBlock("bubble response", log.quick_response_text ? "Response (quick response used)" : "Response", log.response_text),
  );
  if (log.error_info) conversation.append(textBlock("bubble error", "Error", log.error_info));
  entry.append(meta);
  if (voiceActions.childElementCount) entry.append(voiceActions);
  entry.append(conversation);
  if (log.timing_breakdown) entry.append(renderTiming(log.timing_breakdown));
  if (log.tool_calls) {
    const pre = document.createElement("pre");
    try { pre.textContent = JSON.stringify(JSON.parse(log.tool_calls), null, 2); }
    catch (_) { pre.textContent = log.tool_calls; }
    entry.append(pre);
  }
  return entry;
}

function formatTimestamp(value) {
  if (!value) return "—";
  return value
    .replace("T", " ")
    .replace(/\.\d+(?=Z|[+-]\d{2}:\d{2}$|$)/, "")
    .replace(/Z$|[+-]\d{2}:\d{2}$/, "");
}

function messageCount(count) {
  return `${count} message${count === 1 ? "" : "s"}`;
}

function renderGroupRow(group, openDrawer) {
  const row = document.createElement("tr");
  row.className = `log-row${group.has_error ? " has-error" : ""}`;
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.setAttribute("aria-haspopup", "dialog");

  const firstTime = formatTimestamp(group.logs[0]?.created_at);
  const lastTime = formatTimestamp(group.logs[group.logs.length - 1]?.created_at);
  const values = [
    ["log-context", group.context_id || "No context"],
    ["log-message-count", messageCount(group.logs.length)],
    ["log-period", firstTime === lastTime ? firstTime : `${firstTime} ~ ${lastTime}`],
    ["log-user", group.logs.find(log => log.user_id)?.user_id || "—"],
  ];
  for (const [className, value] of values) {
    const cell = document.createElement("td");
    cell.className = className;
    cell.textContent = value;
    row.append(cell);
  }
  const activate = () => openDrawer(group, row);
  row.addEventListener("click", activate);
  row.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activate();
    }
  });
  return row;
}

export function renderLogs(root, { api, setStatus }) {
  root.innerHTML = `
    <section class="page-heading"><h2>Logs</h2><p>Search conversation messages. All conditions are combined.</p></section>
    <form class="filter-grid">
      <div class="field grow"><label for="logs-user">User ID</label><input id="logs-user" name="user_id" autocomplete="off"></div>
      <div class="field grow"><label for="logs-session">Session ID</label><input id="logs-session" name="session_id" autocomplete="off"></div>
      <div class="field grow"><label for="logs-context">Context ID</label><input id="logs-context" name="context_id" autocomplete="off"></div>
      <div class="field grow"><label for="logs-keyword">Keyword</label><input id="logs-keyword" name="keyword" autocomplete="off" placeholder="request, response, error, tool call"></div>
      <div class="field"><label for="logs-error">Error</label><select id="logs-error" name="has_error"><option value="">All</option><option value="true">With error</option><option value="false">Without error</option></select></div>
      <div class="field"><label for="logs-limit">Limit</label><input id="logs-limit" name="limit" type="number" min="1" max="10000" value="200"></div>
      <button class="primary" type="submit">Search</button><button class="secondary" type="reset">Reset</button>
    </form>
    <div class="log-table-wrap">
      <table class="log-table">
        <thead><tr><th>Context ID</th><th>Messages</th><th>Period</th><th>User ID</th></tr></thead>
        <tbody data-results></tbody>
      </table>
    </div>
    <div class="log-drawer-backdrop" data-log-drawer hidden>
      <aside class="log-drawer" role="dialog" aria-modal="true" aria-labelledby="log-drawer-title">
        <header class="log-drawer-header">
          <div><div class="eyebrow">Conversation log</div><h3 id="log-drawer-title"></h3></div>
          <button class="secondary" type="button" data-log-drawer-close aria-label="Close log details">Close</button>
        </header>
        <div class="log-drawer-content" data-log-drawer-content></div>
      </aside>
    </div>`;
  const form = root.querySelector("form");
  const results = root.querySelector("[data-results]");
  const drawerBackdrop = root.querySelector("[data-log-drawer]");
  const drawerTitle = root.querySelector("#log-drawer-title");
  const drawerContent = root.querySelector("[data-log-drawer-content]");
  const drawerClose = root.querySelector("[data-log-drawer-close]");
  let stopped = false;
  let trigger = null;
  let voiceRecorderEnabled = false;
  let playbackToken = 0;
  let currentAudio = null;
  let currentAudioUrl = null;
  let currentAudioDone = null;
  let currentVoiceButton = null;

  function releaseCurrentAudio() {
    currentAudio?.pause();
    currentAudio = null;
    if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
    const done = currentAudioDone;
    currentAudioDone = null;
    done?.();
  }

  function stopVoice() {
    playbackToken += 1;
    releaseCurrentAudio();
    if (currentVoiceButton) {
      currentVoiceButton.textContent = `▶ ${currentVoiceButton.dataset.label}`;
      currentVoiceButton = null;
    }
  }

  async function playBlob(blob, token) {
    if (token !== playbackToken) return;
    await new Promise((resolve, reject) => {
      currentAudioUrl = URL.createObjectURL(blob);
      currentAudio = new Audio(currentAudioUrl);
      currentAudioDone = resolve;
      currentAudio.onended = releaseCurrentAudio;
      currentAudio.onerror = () => {
        currentAudioDone = null;
        releaseCurrentAudio();
        reject(new Error("Audio playback failed"));
      };
      currentAudio.play().catch(error => {
        currentAudioDone = null;
        releaseCurrentAudio();
        reject(error);
      });
    });
  }

  async function playVoice(log, voiceType, button) {
    if (currentVoiceButton === button) {
      stopVoice();
      return;
    }
    stopVoice();
    const token = playbackToken;
    currentVoiceButton = button;
    button.textContent = `■ ${button.dataset.label}`;
    try {
      if (voiceType === "request") {
        const response = await api.get(`logs/voice/${log.transaction_id}/request`);
        await playBlob(await response.blob(), token);
      } else {
        let played = false;
        if (log.quick_response_text) {
          try {
            const quick = await api.get(`logs/voice/${log.transaction_id}/quick_response`);
            await playBlob(await quick.blob(), token);
            played = true;
          } catch (_) {}
        }
        const { count } = await api.get(`logs/voice/${log.transaction_id}/response`);
        for (let index = 0; index < count && token === playbackToken; index += 1) {
          const response = await api.get(`logs/voice/${log.transaction_id}/response_${index}`);
          await playBlob(await response.blob(), token);
          played = true;
        }
        if (!played) throw new Error("Voice file not found");
      }
    } catch (error) {
      if (token === playbackToken) setStatus(error.message, true);
    } finally {
      if (token === playbackToken) stopVoice();
    }
  }

  function closeDrawer() {
    if (drawerBackdrop.hidden) return;
    stopVoice();
    drawerBackdrop.hidden = true;
    document.body.classList.remove("drawer-open");
    trigger?.focus();
    trigger = null;
  }

  function openDrawer(group, button) {
    trigger = button;
    drawerTitle.textContent = `${group.context_id || "No context"} · ${messageCount(group.logs.length)}`;
    drawerContent.replaceChildren(...group.logs.map(log => renderLog(log, { voiceRecorderEnabled, playVoice })));
    drawerBackdrop.hidden = false;
    document.body.classList.add("drawer-open");
    drawerClose.focus();
  }

  function handleKeydown(event) {
    if (event.key === "Escape") closeDrawer();
  }

  drawerClose.addEventListener("click", closeDrawer);
  drawerBackdrop.addEventListener("click", event => {
    if (event.target === drawerBackdrop) closeDrawer();
  });
  document.addEventListener("keydown", handleKeydown);

  async function load(event) {
    event?.preventDefault();
    const params = new URLSearchParams();
    for (const name of ["user_id", "session_id", "context_id", "keyword", "has_error", "limit"]) {
      const value = form.elements[name].value.trim();
      if (value) params.set(name, value);
    }
    setStatus("Loading logs…");
    try {
      const data = await api.get(`logs?${params}`);
      if (stopped) return;
      closeDrawer();
      voiceRecorderEnabled = data.voice_recorder_enabled;
      results.replaceChildren();
      if (!data.groups.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.className = "empty";
        cell.textContent = "No matching logs.";
        row.append(cell);
        results.append(row);
      }
      for (const group of data.groups) {
        results.append(renderGroupRow(group, openDrawer));
      }
      setStatus();
    } catch (error) {
      setStatus(error.message, true);
    }
  }
  form.addEventListener("submit", load);
  form.addEventListener("reset", () => setTimeout(load, 0));
  load();
  return () => {
    stopped = true;
    closeDrawer();
    document.removeEventListener("keydown", handleKeydown);
  };
}
