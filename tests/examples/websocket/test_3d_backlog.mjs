import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const controllerSource = await readFile(
    new URL("avatar3d/common/backlog-controller.js", htmlDirectory),
    "utf8",
);
const threeDSource = await readFile(new URL("3d.html", htmlDirectory), "utf8");
const { BacklogController } = await import(
    `data:text/javascript;base64,${Buffer.from(controllerSource).toString("base64")}`
);

class FakeView {
    constructor() {
        this.renders = [];
        this.playing = null;
        this.handlers = null;
    }

    bind(handlers) {
        this.handlers = handlers;
    }

    render(entries) {
        this.renders.push([...entries]);
    }

    setPlaying(id) {
        this.playing = id;
    }

    open() {}
    close() {}
    dispose() {}
}

class FakeStore {
    constructor({ contextId = null, entries = [], maxEntries = 100 } = {}) {
        this.contextId = contextId;
        this.entries = [...entries];
        this.maxEntries = maxEntries;
        this.appends = [];
    }

    async load() {
        return { contextId: this.contextId, entries: [...this.entries] };
    }

    async appendTurn(contextId, entries) {
        if (this.contextId && this.contextId !== contextId) this.entries = [];
        this.contextId = contextId;
        this.entries = [...this.entries, ...entries].slice(-this.maxEntries);
        this.appends.push({ contextId, entries });
    }

    async removeOldest(count) {
        this.entries.splice(0, count);
    }
}

function createController(store = new FakeStore()) {
    const aiavatar = {
        chatContextId: null,
        isAudioPlaying: false,
        isBacklogAudioPlaying: false,
        volume: 1,
    };
    const ui = {
        speakerLabelUser: "User",
        speakerLabelAI: "AI",
        isServerProcessing: false,
    };
    const view = new FakeView();
    const controller = new BacklogController({ aiavatar, ui, store, view, maxEntries: 100 });
    return { controller, aiavatar, ui, view, store };
}

test("3D viewer provides a themed backlog capped at 100 messages", () => {
    assert.match(threeDSource, /id="backlogBtn">LOG</);
    assert.match(threeDSource, /id="backlogOverlay"[\s\S]*role="dialog"[\s\S]*hidden/);
    assert.match(threeDSource, /class="backlog-header message-inner"/);
    assert.match(controllerSource, /className = "backlog-entry-body message-inner"/);
    assert.match(threeDSource, /backlog:\s*{[\s\S]*enabled: true,[\s\S]*maxEntries: 100/);
});

test("backlog saves the user and AI messages together only after final", async () => {
    const { controller, store } = createController();
    await controller.ready;

    controller.stageUser({
        text: "What is this?",
        imageDataUrl: "data:image/jpeg;base64,AQID",
    });
    controller.handleResponse({
        type: "start",
        context_id: "context-1",
        metadata: { request_text: "What is this?" },
    });
    controller.handleResponse({
        type: "chunk",
        context_id: "context-1",
        voice_text: "It is a device.",
        audio_data: Buffer.from("RIFF0000WAVEdata").toString("base64"),
        metadata: {},
    });

    assert.equal(store.appends.length, 0);
    await controller.handleResponse({
        type: "final",
        context_id: "context-1",
        voice_text: "It is a small device.",
    });

    assert.equal(store.appends.length, 1);
    assert.equal(store.appends[0].contextId, "context-1");
    assert.deepEqual(store.appends[0].entries.map((entry) => entry.role), ["user", "ai"]);
    assert.equal(store.appends[0].entries[0].text, "What is this?");
    assert.ok(store.appends[0].entries[0].image instanceof Blob);
    assert.equal(store.appends[0].entries[1].text, "It is a small device.");
    assert.equal(store.appends[0].entries[1].audioChunks.length, 1);
    assert.ok(store.appends[0].entries[1].audioChunks[0] instanceof Blob);
});

test("backlog discards unfinished turns on errors", async () => {
    const { controller, store } = createController();
    await controller.ready;
    controller.stageUser({ text: "Do not retain this" });
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    controller.handleResponse({ type: "chunk", context_id: "context-1", voice_text: "Partial" });
    controller.handleResponse({ type: "error", context_id: "context-1" });

    await controller.handleResponse({ type: "final", context_id: "context-1", voice_text: "Late final" });
    assert.equal(store.appends.length, 0);
    assert.equal(controller.entries.length, 0);
});

test("backlog replaces persisted history when the completed context changes", async () => {
    const oldEntries = [{
        id: "old-ai",
        contextId: "old-context",
        role: "ai",
        speaker: "AI",
        text: "Old answer",
        image: null,
        audioChunks: [],
        createdAt: 1,
    }];
    const store = new FakeStore({ contextId: "old-context", entries: oldEntries });
    const { controller } = createController(store);
    await controller.ready;

    controller.stageUser({ text: "New request" });
    controller.handleResponse({ type: "start", context_id: "new-context", metadata: {} });
    await controller.handleResponse({
        type: "final",
        context_id: "new-context",
        voice_text: "New answer",
    });

    assert.equal(controller.contextId, "new-context");
    assert.deepEqual(controller.entries.map((entry) => entry.text), ["New request", "New answer"]);
    assert.deepEqual(store.entries.map((entry) => entry.text), ["New request", "New answer"]);
});

test("an interrupted response with final retains its accumulated text without a flag", async () => {
    const { controller, store } = createController();
    await controller.ready;
    controller.stageUser({ text: "Tell me something" });
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    controller.handleResponse({ type: "chunk", context_id: "context-1", voice_text: "Part one. " });
    controller.handleResponse({ type: "chunk", context_id: "context-1", voice_text: "Part two." });
    await controller.handleResponse({
        type: "final",
        context_id: "context-1",
        voice_text: "",
        metadata: { interrupted: true },
    });

    const aiEntry = store.entries.at(-1);
    assert.equal(aiEntry.text, "Part one. Part two.");
    assert.equal(Object.hasOwn(aiEntry, "interrupted"), false);
});

test("backlog keeps at most 100 messages in memory and storage", async () => {
    const oldEntries = Array.from({ length: 100 }, (_, index) => ({
        id: `old-${index}`,
        contextId: "context-1",
        role: index % 2 ? "ai" : "user",
        speaker: index % 2 ? "AI" : "User",
        text: `Old ${index}`,
        image: null,
        audioChunks: [],
        createdAt: index,
    }));
    const store = new FakeStore({ contextId: "context-1", entries: oldEntries });
    const { controller } = createController(store);
    await controller.ready;
    controller.stageUser({ text: "Newest request" });
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    await controller.handleResponse({ type: "final", context_id: "context-1", voice_text: "Newest answer" });

    assert.equal(controller.entries.length, 100);
    assert.equal(store.entries.length, 100);
    assert.deepEqual(controller.entries.slice(-2).map((entry) => entry.text), [
        "Newest request",
        "Newest answer",
    ]);
});
