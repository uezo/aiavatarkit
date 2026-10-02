import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const [uiSource, messageSource] = await Promise.all([
    readFile(new URL("ui.js", htmlDirectory), "utf8"),
    readFile(new URL("avatar3d/common/message-controller.js", htmlDirectory), "utf8"),
]);

function eventTarget(properties = {}) {
    const listeners = new Map();
    return {
        ...properties,
        addEventListener(name, callback) {
            if (!listeners.has(name)) listeners.set(name, new Set());
            listeners.get(name).add(callback);
        },
        removeEventListener(name, callback) {
            listeners.get(name)?.delete(callback);
        },
        dispatch(name, event = {}) {
            for (const callback of [...(listeners.get(name) || [])]) callback(event);
        },
        listenerCount(name) { return listeners.get(name)?.size || 0; },
    };
}

function element() {
    const classes = new Set();
    const node = eventTarget({
        hidden: false,
        textContent: "",
        className: "",
        scrollTop: 0,
        scrollHeight: 100,
        value: "100",
        contains: () => false,
        classList: {
            add: (...names) => names.forEach(name => classes.add(name)),
            remove: (...names) => names.forEach(name => classes.delete(name)),
            contains: name => classes.has(name),
            toggle(name) {
                if (classes.has(name)) classes.delete(name);
                else classes.add(name);
            },
        },
    });
    Object.defineProperty(node, "innerHTML", {
        set() { assert.fail("Transcript markup must be rendered using textContent"); },
    });
    return node;
}

function fakeClock() {
    let now = 0;
    let nextId = 1;
    const tasks = new Map();
    const schedule = (callback, delay = 0, interval = 0) => {
        const id = nextId++;
        tasks.set(id, { callback, due: now + delay, interval });
        return id;
    };
    return {
        now: () => now,
        setTimeout: (callback, delay) => schedule(callback, delay),
        clearTimeout: id => tasks.delete(id),
        setInterval: (callback, delay) => schedule(callback, delay, delay),
        clearInterval: id => tasks.delete(id),
        pendingCount: () => tasks.size,
        advance(milliseconds) {
            const target = now + milliseconds;
            let count = 0;
            while (true) {
                const next = [...tasks.entries()].sort((a, b) => a[1].due - b[1].due)[0];
                if (!next || next[1].due > target) break;
                assert.ok(count++ < 10000, "Timer loop must terminate");
                const [id, task] = next;
                now = task.due;
                if (task.interval) task.due += task.interval;
                else tasks.delete(id);
                task.callback();
            }
            now = target;
        },
    };
}

function harness({ preview = true, with3d = false, separatePartialTranscript } = {}) {
    const elements = Object.fromEntries([
        "avatarFrame", "inputLevel", "chatBtn", "interruptToggle", "cameraToggle",
        "volumeBtn", "volumePopup", "volumeSlider", "volumeValue", "volumeControl",
        "microphoneVolumeSlider", "microphoneVolumeValue", "messageBox", "messageSpeaker",
        "messageText", "toolStatus",
        ...(preview ? ["partialTranscript", "partialTranscriptLabel", "partialTranscriptText"] : []),
    ].map(id => [id, element()]));
    if (preview) elements.partialTranscript.hidden = true;
    const document = eventTarget({ getElementById: id => elements[id] || null });
    const clock = fakeClock();
    const storage = new Map();
    const calls = [];
    const aiavatar = {
        ws: eventTarget(),
        startListening: (...args) => calls.push(["start", ...args]),
        stopListening: (...args) => calls.push(["stop", ...args]),
        sendConfig() {}, setVolume() {}, setMicrophoneVolume() {},
        isAudioPlaying: false,
    };
    let nextUuid = 1;
    const AvatarUI = new Function(
        "document", "crypto", "localStorage", "setTimeout", "clearTimeout", "Date",
        `${uiSource}; return AvatarUI;`,
    )(
        document, { randomUUID: () => `id-${nextUuid++}` },
        { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
        clock.setTimeout, clock.clearTimeout, { now: clock.now },
    );
    const ui = new AvatarUI({ aiavatar, userId: "user", camera: {}, onStop: () => calls.push(["onStop"]), separatePartialTranscript });
    const state = { showUserText: true, showAIText: true, messageSpeed: 70, autoHide: false };
    let controller = null;
    if (with3d) {
        const installMessageController = new Function(
            "document", "setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date",
            `${messageSource.replace(/^export /gm, "")}; return installMessageController;`,
        )(
            document, clock.setTimeout, clock.clearTimeout, clock.setInterval, clock.clearInterval,
            { now: clock.now },
        );
        controller = installMessageController({ aiavatar, ui, state });
    }
    return {
        ui, state, elements, clock, aiavatar, calls, controller,
        receive: response => ui.handleResponse({ text: null, metadata: {}, ...response }),
        partial: text => ui.handleResponse({ type: "info", text: null, metadata: { partial_request_text: text } }),
        dispose() {
            controller?.dispose();
            ui.dispose();
            assert.equal(clock.pendingCount(), 0, "Owned timers must be cleared on disposal");
        },
    };
}

test("partial text has its own safe preview without changing the main speaker or message", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        assert.equal(h.ui.partialTranscript, h.elements.partialTranscript);
        assert.equal(h.ui.partialTranscriptLabel, h.elements.partialTranscriptLabel);
        assert.equal(h.ui.partialTranscriptText, h.elements.partialTranscriptText);
        assert.equal(h.ui._partialTranscriptTimer, null);
        h.ui.showMessage("ai", "説明を続けています。");
        h.ui.speakerLabelUser = "利用者";
        h.partial("<img src=x onerror=alert(1)>");
        assert.equal(h.ui.partialTranscript.hidden, false);
        assert.equal(h.ui.partialTranscriptLabel.textContent, "利用者 · Transcribing");
        assert.equal(h.ui.partialTranscriptText.textContent, "<img src=x onerror=alert(1)>");
        h.partial("うん、なるほど");
        assert.equal(h.ui.partialTranscriptText.textContent, "うん、なるほど");
        assert.equal(h.ui.messageText.textContent, "説明を続けています。");
        assert.equal(h.ui.messageSpeaker.textContent, "AI");
        assert.equal(h.ui.currentUserText, "");
    } finally { h.dispose(); }
});

test("accepted, old response chunks and final retain preview until start promotes recognized text", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        h.partial("質問の途中");
        h.receive({ type: "accepted" });
        assert.equal(h.ui.partialTranscript.hidden, false);
        h.receive({ type: "chunk", voice_text: "前の回答の続き。" });
        h.receive({ type: "final", voice_text: "前の回答が完了。" });
        assert.equal(h.ui.partialTranscript.hidden, false);
        assert.equal(h.ui.partialTranscriptText.textContent, "質問の途中");
        assert.equal(h.ui.messageText.textContent, "前の回答が完了。");
        h.receive({ type: "start", metadata: { recognized_text: "確定した質問です。" } });
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.ui._partialTranscriptTimer, null);
        assert.equal(h.ui.messageText.textContent, "確定した質問です。");
        assert.equal(h.ui.messageSpeaker.textContent, "User");
    } finally { h.dispose(); }
});

test("empty recognition and command inputs cannot leave a preview or overwrite the main message", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        h.ui.showMessage("ai", "現在の回答");
        for (const text of ["", " ", null, "$internal-command"]) {
            h.partial("前の認識");
            h.partial(text);
            assert.equal(h.ui.partialTranscript.hidden, true);
            assert.equal(h.ui.partialTranscriptText.textContent, "");
            assert.equal(h.ui._partialTranscriptTimer, null);
            assert.equal(h.ui.messageText.textContent, "現在の回答");
        }
    } finally { h.dispose(); }
});

test("connection, start without recognized text, cancellation and errors clear the preview", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        for (const type of ["connected", "start", "canceled", "error"]) {
            h.partial("認識中");
            h.receive({ type, user_id: "user", voice_text: "" });
            assert.equal(h.ui.partialTranscript.hidden, true, type);
            assert.equal(h.ui._partialTranscriptTimer, null, type);
        }
    } finally { h.dispose(); }
});

test("preview expiry resets for new partials and disposal clears its timer", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        h.partial("古い認識");
        h.clock.advance(2000);
        h.partial("新しい認識");
        h.clock.advance(1000);
        assert.equal(h.ui.partialTranscript.hidden, false, "The canceled old expiry must not hide newer text");
        assert.equal(h.ui.partialTranscriptText.textContent, "新しい認識");
        h.clock.advance(1999);
        assert.equal(h.ui.partialTranscript.hidden, false);
        h.clock.advance(1);
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.ui._partialTranscriptTimer, null);
        h.partial("破棄予定の認識");
        h.ui.dispose();
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.clock.pendingCount(), 0);
    } finally { h.dispose(); }
});

test("Stop clears the preview immediately along with the listening session", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        h.ui.isChatActive = true;
        h.partial("認識中");
        h.elements.chatBtn.dispatch("click");
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.ui._partialTranscriptTimer, null);
        assert.equal(h.ui.isChatActive, false);
        assert.deepEqual(h.calls, [["stop", h.ui.sessionId], ["onStop"]]);
    } finally { h.dispose(); }
});

test("socket close clears preview and reconnect removes obsolete close listeners", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        const oldSocket = h.aiavatar.ws;
        h.receive({ type: "connected", user_id: "user" });
        assert.equal(oldSocket.listenerCount("close"), 1);
        h.partial("切断前");
        oldSocket.dispatch("close");
        assert.equal(h.ui.partialTranscript.hidden, true);

        const currentSocket = eventTarget();
        h.aiavatar.ws = currentSocket;
        h.receive({ type: "connected", user_id: "user" });
        assert.equal(oldSocket.listenerCount("close"), 0);
        assert.equal(currentSocket.listenerCount("close"), 1);
        h.partial("再接続後");
        oldSocket.dispatch("close");
        assert.equal(h.ui.partialTranscript.hidden, false);
        currentSocket.dispatch("close");
        assert.equal(h.ui.partialTranscript.hidden, true);
        h.ui.dispose();
        assert.equal(currentSocket.listenerCount("close"), 0);
    } finally { h.dispose(); }
});

test("pages without preview DOM retain the legacy partial and final user message behavior", () => {
    const h = harness({ preview: false, separatePartialTranscript: true });
    try {
        assert.equal(h.ui.partialTranscript, null);
        h.partial("認識途中");
        assert.equal(h.ui.messageText.textContent, "認識途中");
        assert.equal(h.ui.messageSpeaker.textContent, "User");
        assert.equal(h.ui.currentUserText, "認識途中");
        h.receive({ type: "start", metadata: { recognized_text: "認識確定" } });
        assert.equal(h.ui.messageText.textContent, "認識確定");
        assert.equal(h.ui.currentUserText, "");
    } finally { h.dispose(); }
});

test("partial text uses the main message box by default even when preview DOM exists", () => {
    for (const config of [{}, { separatePartialTranscript: false }]) {
        const h = harness(config);
        try {
            assert.equal(h.ui.separatePartialTranscript, false);
            h.ui.showMessage("ai", "Current answer");
            h.partial("First partial");
            assert.equal(h.ui.messageText.textContent, "First partial");
            assert.equal(h.ui.messageSpeaker.textContent, "User");
            assert.equal(h.ui.currentUserText, "First partial");
            assert.equal(h.ui.partialTranscript.hidden, true);
            assert.equal(h.ui._partialTranscriptTimer, null);
            h.ui.separatePartialTranscript = undefined;
            h.partial("Updated partial");
            assert.equal(h.ui.messageText.textContent, "Updated partial");
            h.receive({ type: "start", metadata: { recognized_text: "Final recognition" } });
            assert.equal(h.ui.messageText.textContent, "Final recognition");
            assert.equal(h.ui.currentUserText, "");
        } finally { h.dispose(); }
    }
});

test("turning separate preview off clears it before the next partial uses the main box", () => {
    const h = harness({ separatePartialTranscript: true });
    try {
        h.partial("Preview text");
        assert.equal(h.ui.partialTranscript.hidden, false);
        h.ui.separatePartialTranscript = false;
        h.partial("Main text");
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.ui.partialTranscriptText.textContent, "");
        assert.equal(h.ui._partialTranscriptTimer, null);
        assert.equal(h.ui.messageText.textContent, "Main text");
    } finally { h.dispose(); }
});

test("3D main-box partials retain the default user display and stop the AI typewriter", () => {
    const h = harness({ with3d: true });
    try {
        h.receive({ type: "chunk", voice_text: "Answer in progress" });
        h.clock.advance(31);
        assert.equal(h.ui.messageText.textContent, "A");
        h.partial("A question");
        assert.equal(h.ui.messageText.textContent, "A question");
        assert.equal(h.ui.messageSpeaker.textContent, "User");
        assert.equal(h.ui.partialTranscript.hidden, true);
        h.clock.advance(310);
        assert.equal(h.ui.messageText.textContent, "A question");
        h.state.showUserText = false;
        h.partial("Hidden partial");
        assert.equal(h.ui.messageText.textContent, "A question");
    } finally { h.dispose(); }
});

test("3D partial previews preserve the running AI typewriter through accepted and old final events", () => {
    const h = harness({ with3d: true, separatePartialTranscript: true });
    try {
        h.receive({ type: "chunk", voice_text: "説明を続けます。" });
        h.clock.advance(31);
        assert.equal(h.ui.messageText.textContent, "説");
        h.partial("うん");
        assert.equal(h.ui.messageText.textContent, "説");
        assert.equal(h.ui.messageSpeaker.textContent, "AI");
        assert.equal(h.ui.partialTranscriptText.textContent, "うん");
        h.receive({ type: "accepted" });
        h.receive({ type: "final", voice_text: "説明を続けます。" });
        h.clock.advance(31);
        assert.equal(h.ui.messageText.textContent, "説明");
        assert.equal(h.ui.partialTranscript.hidden, false);
        h.clock.advance(31 * 20);
        assert.equal(h.ui.messageText.textContent, "説明を続けます。");
        assert.equal(h.ui.currentAIText, "説明を続けます。");
        h.receive({ type: "start", metadata: { recognized_text: "実際の質問" } });
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.ui.messageText.textContent, "実際の質問");
        assert.equal(h.ui.messageSpeaker.textContent, "User");
    } finally { h.dispose(); }
});

test("3D user-text visibility applies to previews and controller disposal restores the UI method", () => {
    const h = harness({ with3d: true, separatePartialTranscript: true });
    try {
        h.partial("表示する認識");
        assert.equal(h.ui.partialTranscript.hidden, false);
        h.state.showUserText = false;
        h.partial("表示しない認識");
        assert.equal(h.ui.partialTranscript.hidden, true);
        assert.equal(h.ui._partialTranscriptTimer, null);
        h.receive({ type: "start", metadata: { recognized_text: "表示しない確定文" } });
        assert.equal(h.ui.messageText.textContent, "");
        const wrappedMethod = h.ui.showPartialTranscript;
        h.controller.dispose();
        assert.notEqual(h.ui.showPartialTranscript, wrappedMethod);
        h.ui.showPartialTranscript("復元したメソッド");
        assert.equal(h.ui.partialTranscript.hidden, false);
        assert.equal(h.ui.partialTranscriptText.textContent, "復元したメソッド");
    } finally { h.dispose(); }
});
