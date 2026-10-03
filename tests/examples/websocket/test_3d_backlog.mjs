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

const pcmFormat = { sample_rate: 24000, channels: 1, sample_width: 2 };

function pcmDescriptor(audioId, text, frameCount, metadata = {}) {
    return {
        type: "chunk",
        context_id: "context-1",
        voice_text: text,
        audio_data: null,
        metadata: {
            ...metadata,
            audio_id: audioId,
            audio_frame_count: frameCount,
            pcm_format: pcmFormat,
        },
    };
}

function pcmChunk(audioId, bytes) {
    return {
        type: "chunk",
        context_id: "context-1",
        audio_data: Buffer.from(bytes).toString("base64"),
        metadata: { audio_id: audioId, pcm_format: pcmFormat },
    };
}

test("continuous PCM never accumulates in the finite-turn backlog", async () => {
    const { controller, store } = createController();
    await controller.ready;
    const metadata = { audio_id: "live", pcm_format: pcmFormat, continuous_audio: true };
    controller.handleResponse({ type: "chunk", metadata });
    for (let i = 0; i < 100; i++) {
        controller.handleResponse({ type: "chunk", metadata, audio_data: "AAA=" });
    }
    assert.equal(controller.pendingTurn, null);
    assert.equal(store.appends.length, 0);
    controller.dispose();
});

async function assertPcmWav(blob, expectedPcm) {
    assert.equal(blob.type, "audio/wav");
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const view = new DataView(bytes.buffer);
    assert.equal(Buffer.from(bytes.subarray(0, 4)).toString(), "RIFF");
    assert.equal(Buffer.from(bytes.subarray(8, 16)).toString(), "WAVEfmt ");
    assert.equal(view.getUint32(4, true), 36 + expectedPcm.length);
    assert.equal(view.getUint16(22, true), 1);
    assert.equal(view.getUint32(24, true), 24000);
    assert.equal(view.getUint16(34, true), 16);
    assert.equal(view.getUint32(40, true), expectedPcm.length);
    assert.deepEqual([...bytes.subarray(44)], expectedPcm);
}

test("backlog groups PCM by original audio, including Nod, alongside WAV responses", async () => {
    const { controller, store } = createController();
    await controller.ready;
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    controller.handleResponse(pcmDescriptor("answer", "Answer.", 3));
    controller.handleResponse(pcmChunk("answer", [1, 0]));
    controller.handleResponse(pcmDescriptor("nod", "Yes.", 1, { nod: true }));
    controller.handleResponse(pcmChunk("nod", [2, 0]));
    controller.handleResponse(pcmChunk("answer", [3, 0, 4, 0]));
    const wav = Buffer.from("RIFF0000WAVEdata");
    controller.handleResponse({
        type: "chunk",
        context_id: "context-1",
        audio_data: wav.toString("base64"),
    });

    assert.equal(store.appends.length, 0);
    await controller.handleResponse({ type: "final", context_id: "context-1" });

    const entry = store.entries.at(-1);
    assert.equal(entry.text, "Answer.Yes.");
    assert.equal(entry.audioChunks.length, 3);
    await assertPcmWav(entry.audioChunks[0], [1, 0, 3, 0, 4, 0]);
    await assertPcmWav(entry.audioChunks[1], [2, 0]);
    assert.deepEqual(Buffer.from(await entry.audioChunks[2].arrayBuffer()), wav);
});

test("backlog retains the received part of PCM on interrupted final without empty audio files", async () => {
    const { controller, store } = createController();
    await controller.ready;
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    controller.handleResponse(pcmDescriptor("partial", "Interrupted.", 100));
    controller.handleResponse(pcmChunk("partial", [1, 0, 2, 0]));
    controller.handleResponse(pcmDescriptor("empty", "", 100));
    await controller.handleResponse({
        type: "final",
        context_id: "context-1",
        metadata: { interrupted: true },
    });

    const entry = store.entries.at(-1);
    assert.equal(entry.text, "Interrupted.");
    assert.equal(entry.audioChunks.length, 1);
    await assertPcmWav(entry.audioChunks[0], [1, 0, 2, 0]);
});

test("backlog requires a PCM descriptor and releases unfinished PCM on error", async () => {
    const { controller, store } = createController();
    await controller.ready;
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    controller.handleResponse(pcmDescriptor("failed", "Failed.", 1));
    controller.handleResponse(pcmChunk("failed", [1, 0]));
    controller.handleResponse({ type: "error", context_id: "context-1" });
    assert.equal(controller.pendingTurn, null);
    controller.handleResponse({ type: "start", context_id: "context-1", metadata: {} });
    controller.handleResponse(pcmChunk("failed", [2, 0]));
    controller.handleResponse(pcmDescriptor("next", "Next.", 1));
    controller.handleResponse(pcmChunk("next", [3, 0]));
    await controller.handleResponse({ type: "final", context_id: "context-1" });

    const entry = store.entries.at(-1);
    assert.equal(entry.text, "Next.");
    assert.equal(entry.audioChunks.length, 1);
    await assertPcmWav(entry.audioChunks[0], [3, 0]);
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
