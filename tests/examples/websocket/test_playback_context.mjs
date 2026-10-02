import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
const helperSource = await readFile(new URL("playback-context.js", htmlDirectory), "utf8");
const installPlaybackContext = new Function(`
    ${helperSource.replace("export function", "function")}
    return installPlaybackContext;
`)();

async function harness({ autoDecode = true, install = true } = {}) {
    const sources = [];
    const animationFrames = [];
    const order = [];
    const errors = [];
    class AudioContext {
        constructor() {
            this.currentTime = 10;
            this.state = "running";
            this.destination = {};
            this.decodes = [];
        }
        async resume() { this.state = "running"; }
        async close() { this.state = "closed"; }
        createGain() { return { gain: {}, connect() {} }; }
        createMediaStreamSource() { return { connect() {} }; }
        createScriptProcessor() { return { connect() {}, disconnect() {} }; }
        decodeAudioData(bytes, success, failure) {
            const complete = () => success({
                duration: 4,
                sampleRate: 16000,
                getChannelData: () => new Float32Array(64000),
            });
            this.decodes.push({ complete, failure });
            if (autoDecode) complete();
        }
        createBufferSource() {
            const context = this;
            const source = {
                connect() {},
                start() {
                    order.push("audio:start");
                    if (context.startError) throw context.startError;
                    this.started = true;
                },
                stop() { this.stopped = true; this.onended?.(); },
                end() { this.onended?.(); },
            };
            sources.push(source);
            return source;
        }
    }
    class WebSocket {
        static OPEN = 1;
        static CONNECTING = 0;
        constructor() { this.readyState = WebSocket.OPEN; this.sent = []; }
        send(data) {
            if (this.sendError) throw this.sendError;
            const message = JSON.parse(data);
            this.sent.push(message);
            order.push(message.type === "playback" ? `control:${message.metadata.event}` : message.type);
        }
        close() { this.readyState = 3; this.onclose?.(); }
        receive(message) { this.onmessage({ data: JSON.stringify(message) }); }
    }
    const testConsole = { log() {}, error: (...args) => errors.push(args) };
    const Client = new Function("WebSocket", "window", "navigator", "requestAnimationFrame", "console", `
        ${clientSource}
        return AIAvatarClient;
    `)(WebSocket, { AudioContext }, {
        mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) },
    }, callback => animationFrames.push(callback), testConsole);
    const installContextHelper = new Function("console", `
        ${helperSource.replace("export function", "function")}
        return installPlaybackContext;
    `)(testConsole);
    const installContext = client => {
        const helper = installContextHelper(client);
        const previousOnResponse = client.onResponseReceived;
        client.onResponseReceived = function (response) {
            helper.handleResponse(response);
            previousOnResponse.call(client, response);
        };
        return helper;
    };
    const client = new Client({ webSocketUrl: "ws://example.test" });
    client.isMicrophoneMuted = () => false;
    const playbackContext = install ? installContext(client) : null;
    await client.startListening("session", "user");
    const sendInput = () => {
        client.scriptNode.onaudioprocess({
            inputBuffer: { getChannelData: () => Float32Array.from([0.5, -0.5]) },
        });
        return client.ws.sent.at(-1);
    };
    return {
        client, context: client.audioContext, sources, animationFrames,
        sendInput, order, errors, playbackContext, installContext,
    };
}

const audio = (text = "A😀うえ", overrides = {}) => ({
    type: "chunk",
    session_id: "session",
    voice_text: text,
    audio_data: Buffer.from("wav").toString("base64"),
    ...overrides,
});
const controls = socket => socket.sent.filter(message => message.type === "playback");
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
const finalResponse = (overrides = {}) => ({
    type: "final", session_id: "session", transaction_id: "transaction", ...overrides,
});

test("start control is sent once after decode and before playback, with no microphone progress metadata", async () => {
    const { client, context, sources, animationFrames, sendInput, order, installContext } = await harness({
        autoDecode: false, install: false,
    });
    const lifecycle = [];
    let frames = 0;
    let received = 0;
    const onPlaybackAudio = () => frames++;
    client.onPlaybackAudio = onPlaybackAudio;
    client.onResponseReceived = () => received++;
    client.onPlaybackStart = function (state) {
        assert.equal(this, client);
        assert.equal(sources[0].started, undefined);
        assert.deepEqual(Object.keys(state), ["message", "playbackId", "durationSeconds"]);
        assert.equal(state.durationSeconds, 4);
        lifecycle.push(state);
        order.push("callback:start");
    };
    client.onPlaybackEnd = state => lifecycle.push(state);
    installContext(client);
    try {
        client.ws.receive(audio());
        assert.equal(controls(client.ws).length, 0);
        context.decodes[0].complete();
        const start = controls(client.ws)[0];
        assert.deepEqual(order, ["control:start", "callback:start", "audio:start"]);
        assert.equal(start.session_id, "session");
        assert.deepEqual(start.metadata, {
            event: "start", playback_id: lifecycle[0].playbackId,
            text: "A😀うえ", duration_seconds: 4,
        });
        assert.equal(typeof start.metadata.playback_id, "string");
        assert.ok(start.metadata.playback_id.length > 0);
        assert.equal(client.onPlaybackAudio, onPlaybackAudio);
        assert.equal(frames, 0);
        animationFrames.shift()(0);
        assert.equal(frames, 1);
        for (let index = 0; index < 50; index++) {
            const request = sendInput();
            assert.deepEqual(Object.keys(request), ["type", "session_id", "audio_data"]);
            assert.equal(Buffer.from(request.audio_data, "base64").length, 4);
        }
        assert.equal(controls(client.ws).length, 1);
        client.ws.receive({ type: "final" });
        assert.equal(controls(client.ws).length, 1);
        sources[0].end();
        await flush();
        assert.deepEqual(controls(client.ws)[1].metadata, {
            event: "end", playback_id: start.metadata.playback_id, completed: true,
        });
        assert.deepEqual(lifecycle[1], { ...lifecycle[0], completed: true });
        assert.equal(received, 2);
    } finally { await client.stopListening("session"); }
});

test("queued chunks notify only when each is ready to play and receive unique playback IDs", async () => {
    const { client, context, sources } = await harness({ autoDecode: false });
    try {
        client.ws.receive(audio("first"));
        client.ws.receive(audio("second"));
        assert.equal(controls(client.ws).length, 0);
        context.decodes[0].complete();
        assert.equal(controls(client.ws).length, 1);
        sources[0].end();
        await flush();
        assert.equal(controls(client.ws).length, 2);
        context.decodes[1].complete();
        sources[1].end();
        await flush();
        const messages = controls(client.ws).map(message => message.metadata);
        assert.deepEqual(messages.map(message => message.event), ["start", "end", "start", "end"]);
        assert.deepEqual(messages.filter(message => message.event === "start").map(message => message.text), ["first", "second"]);
        assert.notEqual(messages[0].playback_id, messages[2].playback_id);
        assert.equal(messages[0].playback_id, messages[1].playback_id);
        assert.equal(messages[2].playback_id, messages[3].playback_id);
    } finally { await client.stopListening("session"); }
});

test("playback starts echo each response chunk's top-level transaction ID unchanged", async () => {
    const { client, sources } = await harness();
    const transactionIds = ["transaction-one", "transaction-one", "transaction-two"];
    try {
        for (const [index, transactionId] of transactionIds.entries()) {
            client.ws.receive(audio(`chunk ${index}`, {
                transaction_id: transactionId,
                metadata: { transaction_id: "unrelated-metadata-value" },
            }));
            sources[index].end();
            await flush();
        }
        const messages = controls(client.ws).map(message => message.metadata);
        const starts = messages.filter(message => message.event === "start");
        assert.deepEqual(starts.map(message => message.transaction_id), transactionIds);
        assert.equal(new Set(starts.map(message => message.playback_id)).size, 3);
        for (const [index, start] of starts.entries()) {
            assert.notEqual(start.playback_id, start.transaction_id);
            assert.deepEqual(messages[index * 2 + 1], {
                event: "end", playback_id: start.playback_id, completed: true,
            });
        }
    } finally { await client.stopListening("session"); }
});

test("missing and null transaction IDs still announce playback without inventing an ID", async () => {
    const { client, sources } = await harness();
    try {
        for (const [index, overrides] of [{}, { transaction_id: null }].entries()) {
            client.ws.receive(audio("legacy response", overrides));
            sources[index].end();
            await flush();
        }
        const messages = controls(client.ws);
        assert.equal(messages.length, 4);
        assert.equal(messages.filter(message => message.metadata.event === "start").length, 2);
        assert.ok(messages.every(message => !Object.hasOwn(message.metadata, "transaction_id")));
    } finally { await client.stopListening("session"); }
});

test("final received with queued audio announces only the last chunk, immediately after its start", async () => {
    const { client, context, sources, order } = await harness({ autoDecode: false });
    try {
        client.ws.receive(audio("first", { transaction_id: "transaction" }));
        client.ws.receive(audio("last", { transaction_id: "transaction" }));
        client.ws.receive(finalResponse());
        context.decodes[0].complete();
        assert.deepEqual(controls(client.ws).map(message => message.metadata.event), ["start"]);
        sources[0].end();
        await flush();
        context.decodes[1].complete();
        const messages = controls(client.ws).map(message => message.metadata);
        assert.deepEqual(messages.map(message => message.event), ["start", "end", "start", "final"]);
        assert.deepEqual(messages[3], {
            event: "final", playback_id: messages[2].playback_id, transaction_id: "transaction",
        });
        assert.deepEqual(order.slice(-3), ["control:start", "control:final", "audio:start"]);
        client.ws.receive(finalResponse());
        assert.equal(controls(client.ws).length, 4);
    } finally { await client.stopListening("session"); }
});

test("final received while the last chunk is decoding waits for its playback start", async () => {
    const { client, context, order } = await harness({ autoDecode: false });
    try {
        client.ws.receive(audio("last", { transaction_id: "transaction" }));
        client.ws.receive(finalResponse());
        assert.equal(controls(client.ws).length, 0);
        context.decodes[0].complete();
        assert.deepEqual(order, ["control:start", "control:final", "audio:start"]);
    } finally { await client.stopListening("session"); }
});

test("final received during playback is sent once and does not restart the audio", async () => {
    const { client, context, sources, order } = await harness();
    try {
        client.ws.receive(audio("last", { transaction_id: "transaction" }));
        context.currentTime = 13.8;
        client.ws.receive(finalResponse());
        client.ws.receive(finalResponse());
        const messages = controls(client.ws).map(message => message.metadata);
        assert.deepEqual(messages.map(message => message.event), ["start", "final"]);
        assert.deepEqual(messages[1], {
            event: "final", playback_id: messages[0].playback_id, transaction_id: "transaction",
        });
        assert.equal(sources.length, 1);
        assert.equal(client.currentAudioSource, sources[0]);
        assert.equal(order.filter(event => event === "audio:start").length, 1);
    } finally { await client.stopListening("session"); }
});

test("a late final can identify the most recently completed playback but not an earlier response", async () => {
    const { client, sources } = await harness();
    try {
        client.ws.receive(audio("finished", { transaction_id: "transaction" }));
        const firstId = controls(client.ws)[0].metadata.playback_id;
        sources[0].end();
        await flush();
        client.ws.receive(finalResponse());
        assert.deepEqual(controls(client.ws).at(-1).metadata, {
            event: "final", playback_id: firstId, transaction_id: "transaction",
        });
        client.ws.receive(audio("newer", { transaction_id: "new-transaction" }));
        sources[1].end();
        await flush();
        const count = controls(client.ws).length;
        client.ws.receive(finalResponse());
        assert.equal(controls(client.ws).length, count);
        client.ws.receive(finalResponse({ transaction_id: "new-transaction" }));
        assert.equal(controls(client.ws).at(-1).metadata.transaction_id, "new-transaction");
    } finally { await client.stopListening("session"); }
});

test("later non-text audio prevents earlier spoken audio from being called final", async () => {
    const { client, context, sources } = await harness({ autoDecode: false });
    try {
        client.ws.receive(audio("spoken", { transaction_id: "transaction" }));
        context.decodes[0].complete();
        client.ws.receive(audio("", { transaction_id: "transaction" }));
        client.ws.receive(finalResponse());
        assert.equal(controls(client.ws).length, 1);
        sources[0].end();
        await flush();
        context.decodes[1].complete();
        sources[1].end();
        await flush();
        client.ws.receive(finalResponse());
        assert.deepEqual(controls(client.ws).map(message => message.metadata.event), ["start", "end"]);
    } finally { await client.stopListening("session"); }
});

test("unmatched, interrupted, error, Nod, and legacy finals do not announce response completion", async () => {
    const { client } = await harness();
    try {
        client.ws.receive(audio("last", { transaction_id: "transaction" }));
        for (const overrides of [
            { session_id: "another-session" }, { transaction_id: "another-transaction" },
            { transaction_id: null }, { transaction_id: undefined },
            { metadata: { interrupted: true } }, { metadata: { error: "failed" } },
            { metadata: { nod: true } },
        ]) client.ws.receive(finalResponse(overrides));
        assert.equal(controls(client.ws).length, 1);
    } finally { await client.stopListening("session"); }
});

test("stop and reconnect discard pending and completed final candidates", async () => {
    const { client, context, sources } = await harness({ autoDecode: false });
    const oldSocket = client.ws;
    try {
        oldSocket.receive(audio("canceled", { transaction_id: "transaction" }));
        oldSocket.receive(finalResponse());
        oldSocket.receive({ type: "stop" });
        context.decodes[0].complete();
        await flush();
        assert.equal(controls(oldSocket).length, 0);

        oldSocket.receive(audio("completed", { transaction_id: "transaction" }));
        context.decodes[1].complete();
        sources[0].end();
        await flush();
        oldSocket.receive({ type: "stop" });
        oldSocket.receive(finalResponse());
        assert.deepEqual(controls(oldSocket).map(message => message.metadata.event), ["start", "end"]);

        await client.stopListening("session");
        await client.startListening("session", "user");
        client.ws.receive(finalResponse());
        assert.equal(controls(client.ws).length, 0);
    } finally { await client.stopListening("session"); }
});

test("new Nod or backlog playback cannot reuse a previously completed final candidate", async () => {
    const { client, sources } = await harness();
    try {
        for (const [index, backlog] of [false, true].entries()) {
            client.ws.receive(audio("spoken", { transaction_id: "transaction" }));
            sources[index * 2].end();
            await flush();
            client.isBacklogAudioPlaying = backlog;
            const playback = client.playAudioSync("AA==", audio("auxiliary", {
                transaction_id: "transaction", metadata: { nod: !backlog },
            }));
            client.ws.receive(finalResponse());
            sources[index * 2 + 1].end();
            await playback;
            client.isBacklogAudioPlaying = false;
            client.ws.receive(finalResponse());
        }
        assert.deepEqual(controls(client.ws).map(message => message.metadata.event), ["start", "end", "start", "end"]);
    } finally { await client.stopListening("session"); }
});

test("a canceled decode never announces playback even if its callback returns later", async () => {
    const { client, context, sources } = await harness({ autoDecode: false });
    try {
        client.ws.receive(audio());
        client.ws.receive({ type: "stop" });
        await flush();
        context.decodes[0].complete();
        assert.equal(sources.length, 0);
        assert.equal(controls(client.ws).length, 0);
        assert.equal(client.currentAudioMessage, null);
        assert.equal(client.currentAudioSource, null);
    } finally { await client.stopListening("session"); }
});

test("barge-in and explicit Stop send one interrupted end before disconnect", async () => {
    const { client, context } = await harness();
    const socket = client.ws;
    try {
        socket.receive(audio());
        context.currentTime = 11;
        socket.receive({ type: "stop" });
        socket.receive({ type: "stop" });
        await flush();
        assert.equal(controls(socket).length, 2);
        assert.equal(controls(socket)[1].metadata.completed, false);
        socket.receive(audio("another chunk"));
        await client.stopListening("session");
        assert.deepEqual(controls(socket).map(message => message.metadata.event), ["start", "end", "start", "end"]);
        assert.equal(controls(socket)[3].metadata.completed, false);
        assert.equal(socket.sent.at(-1).type, "stop");
    } finally { await client.stopListening("session"); }
});

test("start and end callback failures are isolated while controls and audio continue", async () => {
    const { client, sources, errors, installContext } = await harness({ install: false });
    client.onPlaybackStart = () => { throw new Error("start callback failed"); };
    client.onPlaybackEnd = () => { throw new Error("end callback failed"); };
    installContext(client);
    try {
        client.ws.receive(audio());
        assert.equal(sources[0].started, true);
        sources[0].end();
        await flush();
        assert.equal(controls(client.ws).length, 2);
        assert.equal(client.processingQueue, false);
        assert.equal(errors.length, 2);
    } finally { await client.stopListening("session"); }
});

test("a failed source.start after asynchronous decode sends matching interrupted end", async () => {
    const { client, context, sources } = await harness({ autoDecode: false });
    try {
        context.startError = new Error("cannot start audio");
        const playback = client.playAudioSync("AA==", audio());
        const rejected = assert.rejects(playback, /cannot start audio/);
        context.decodes[0].complete();
        await rejected;
        const messages = controls(client.ws);
        assert.equal(messages.length, 2);
        assert.equal(messages[1].metadata.completed, false);
        assert.equal(messages[0].metadata.playback_id, messages[1].metadata.playback_id);
        assert.equal(sources[0].started, undefined);
        assert.equal(client.currentAudioMessage, null);
        assert.equal(client.currentAudioSource, null);
    } finally { await client.stopListening("session"); }
});

test("canceling from a start callback never starts an orphaned audio source", async () => {
    const { client, sources, installContext } = await harness({ install: false });
    client.onPlaybackStart = () => client.stopAudio();
    installContext(client);
    try {
        client.ws.receive(audio());
        await flush();
        assert.equal(sources[0].started, undefined);
        assert.equal(controls(client.ws).length, 2);
        assert.equal(controls(client.ws)[1].metadata.completed, false);
    } finally { await client.stopListening("session"); }
});

test("Nod, backlog, non-text, and unaddressed audio do not send controls", async () => {
    const { client } = await harness();
    try {
        for (const [message, backlog] of [
            [audio("nod", { metadata: { nod: true } }), false],
            [audio("backlog"), true],
            [audio("   "), false],
            [audio(null), false],
            [audio("no session", { session_id: null }), false],
        ]) {
            client.isBacklogAudioPlaying = backlog;
            const playback = client.playAudioSync("AA==", message);
            client.stopAudio();
            await playback;
        }
        assert.equal(controls(client.ws).length, 0);
    } finally { await client.stopListening("session"); }
});

test("disconnected sockets and send failures do not prevent local audio playback", async () => {
    const { client, sources, errors } = await harness();
    const socket = client.ws;
    try {
        socket.sendError = new Error("send failed");
        const first = client.playAudioSync("AA==", audio());
        assert.equal(sources[0].started, true);
        sources[0].end();
        await first;
        assert.equal(errors.length, 1);
        assert.equal(controls(socket).length, 0);
        socket.sendError = null;
        socket.readyState = 3;
        const second = client.playAudioSync("AA==", audio());
        sources[1].end();
        await second;
        assert.equal(sources[1].started, true);
        assert.equal(controls(socket).length, 0);
    } finally { await client.stopListening("session"); }
});

test("late ends from replaced playback cannot clear a newer connection's context", async () => {
    const { client, sources, installContext } = await harness({ install: false });
    const started = [];
    client.onPlaybackStart = state => started.push(state);
    installContext(client);
    const oldSocket = client.ws;
    try {
        const first = client.playAudioSync("AA==", audio());
        const oldState = started[0];
        client.ws = new oldSocket.constructor();
        const second = client.playAudioSync("AA==", audio("replacement"));
        const newSocket = client.ws;
        const newId = controls(newSocket)[0].metadata.playback_id;
        client.onPlaybackEnd({ ...oldState, completed: true });
        sources[0].end();
        await first;
        assert.equal(controls(oldSocket).length, 1);
        assert.equal(controls(newSocket).length, 1);
        sources[1].end();
        await second;
        assert.equal(controls(newSocket).length, 2);
        assert.equal(controls(newSocket)[1].metadata.playback_id, newId);
        assert.notEqual(newId, oldState.playbackId);
    } finally { await client.stopListening("session"); }
});

test("reconnection uses the new message's session ID and does not revive old end events", async () => {
    const { client, sources } = await harness();
    const oldSocket = client.ws;
    oldSocket.receive(audio());
    await client.stopListening("session");
    await client.startListening("next-session", "user");
    try {
        client.ws.receive(audio("next", { session_id: "next-session" }));
        sources[0].end();
        assert.equal(controls(client.ws).length, 1);
        assert.equal(controls(client.ws)[0].session_id, "next-session");
        sources[1].end();
        await flush();
        assert.equal(controls(client.ws).length, 2);
        assert.notEqual(controls(oldSocket)[0].metadata.playback_id, controls(client.ws)[0].metadata.playback_id);
    } finally { await client.stopListening("next-session"); }
});

test("dispose closes the active report and restores existing callbacks without stopping audio", async () => {
    const { client, sources, installContext } = await harness({ install: false });
    const previousStart = () => {};
    let ends = 0;
    const previousEnd = () => ends++;
    client.onPlaybackStart = previousStart;
    client.onPlaybackEnd = previousEnd;
    const helper = installContext(client);
    try {
        client.ws.receive(audio());
        helper.dispose();
        helper.dispose();
        assert.equal(client.onPlaybackStart, previousStart);
        assert.equal(client.onPlaybackEnd, previousEnd);
        assert.equal(sources[0].stopped, undefined);
        assert.equal(controls(client.ws).length, 2);
        assert.equal(controls(client.ws)[1].metadata.completed, false);
        sources[0].end();
        await flush();
        assert.equal(ends, 1);
        assert.equal(controls(client.ws).length, 2);
    } finally { await client.stopListening("session"); }
});

test("dispose preserves callbacks replaced by another component", () => {
    const client = {};
    const helper = installPlaybackContext(client);
    const replacement = () => {};
    client.onPlaybackStart = replacement;
    client.onPlaybackEnd = replacement;
    helper.dispose();
    assert.equal(client.onPlaybackStart, replacement);
    assert.equal(client.onPlaybackEnd, replacement);
});

test("muted microphone input keeps the existing silent data message shape", async () => {
    const { client, sendInput } = await harness();
    try {
        client.mute();
        const request = sendInput();
        assert.deepEqual(Object.keys(request), ["type", "session_id", "audio_data"]);
        assert.deepEqual([...Buffer.from(request.audio_data, "base64")], [0, 0, 0, 0]);
        assert.equal(controls(client.ws).length, 0);
    } finally { await client.stopListening("session"); }
});

test("maintained pages install playback context after avatar callbacks are bound", async () => {
    const index = await readFile(new URL("index.html", htmlDirectory), "utf8");
    const app = await readFile(new URL("avatar3d/common/app.js", htmlDirectory), "utf8");
    assert.ok(index.indexOf("installPlaybackContext(aiavatar)") > index.indexOf("await avatar.bind(aiavatar)"));
    assert.ok(app.indexOf("installPlaybackContext(aiavatar)") > app.indexOf("await modelAdapter.initialize({ aiavatar, ui })"));
    for (const source of [index, app]) {
        assert.match(source, /onResponseReceived = \(response\) => \{\s*playbackContext\.handleResponse\(response\);/);
    }
});
