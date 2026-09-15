import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
const appSource = await readFile(new URL("avatar3d/common/app.js", htmlDirectory), "utf8");
const uiSource = await readFile(new URL("ui.js", htmlDirectory), "utf8");

async function harness({ autoDecode = true, socketError = null, getUserMedia = null, resume = null, close = null, start = true } = {}) {
    const sources = [];
    const microphoneTrack = { stop() { this.stopped = true; } };
    class AudioContext {
        constructor() {
            this.currentTime = 10;
            this.state = "running";
            this.destination = {};
            this.autoDecode = autoDecode;
            this.decodes = [];
        }
        async resume() { await resume?.(); }
        async close() { await close?.(); this.state = "closed"; }
        createGain() { return { gain: {}, connect() {} }; }
        createMediaStreamSource() { return { connect() {} }; }
        createScriptProcessor() { return { connect() {}, disconnect() { this.disconnected = true; } }; }
        decodeAudioData(bytes, onSuccess) {
            const decoded = {
                marker: Buffer.from(bytes).toString(),
                sampleRate: 24000,
                getChannelData: () => new Float32Array(2400),
            };
            const complete = () => onSuccess(decoded);
            this.decodes.push(complete);
            if (this.autoDecode) complete();
        }
        createBufferSource() {
            const source = {
                connect() {},
                start() { this.started = true; },
                stop() { this.stopped = true; },
                end() { this.onended(); },
            };
            sources.push(source);
            return source;
        }
    }
    class WebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        constructor() {
            if (socketError) throw socketError;
            this.readyState = WebSocket.OPEN;
            this.sent = [];
        }
        send(data) { this.sent.push(JSON.parse(data)); }
        close() { this.readyState = 3; this.onclose?.(); }
        receive(message) { this.onmessage({ data: JSON.stringify(message) }); }
    }
    const Client = new Function("WebSocket", "window", "navigator", `
        ${clientSource}
        return AIAvatarClient;
    `)(WebSocket, { AudioContext }, {
        mediaDevices: { getUserMedia: getUserMedia || (async () => ({ getTracks: () => [microphoneTrack] })) },
    });
    const client = new Client({ webSocketUrl: "ws://example.test" });
    if (start) await client.startListening("session", "user");
    return { client, socket: client.ws, context: client.audioContext, sources, microphoneTrack };
}

const audio = marker => ({ type: "chunk", audio_data: Buffer.from(marker).toString("base64") });
const nod = id => ({ ...audio(id), metadata: { nod: true, nod_id: id } });
const stopResponse = { type: "stop" };
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
};

test("disconnect without nods stops main audio and microphone without generating a response", async () => {
    const { client, socket, context, sources, microphoneTrack } = await harness();
    const received = [];
    client.onResponseReceived = message => received.push(message);
    socket.receive(audio("main"));
    socket.receive(audio("queued-main"));
    const processor = client.scriptNode;
    socket.close();
    // Even a callback already scheduled before disconnection must not send.
    processor.onaudioprocess({ inputBuffer: { getChannelData: () => new Float32Array([0.5]) } });
    await flush();
    assert.equal(socket.sent.length, 0);
    assert.equal(sources[0].stopped, true);
    assert.equal(client.messageQueue.length, 0);
    assert.equal(processor.disconnected, true);
    assert.equal(microphoneTrack.stopped, true);
    assert.equal(context.state, "closed");
    assert.equal(client.ws, null);
    assert.deepEqual(received, [audio("main"), audio("queued-main")]);
});

test("disconnect during AudioContext resume never starts microphone capture", async () => {
    const resumed = deferred();
    let microphoneRequests = 0;
    const { client } = await harness({ start: false, resume: () => resumed.promise,
        getUserMedia: async () => { microphoneRequests++; },
    });
    const starting = client.startListening("session", "user");
    const context = client.audioContext;
    client.ws.close();
    resumed.resolve();
    await starting;
    assert.equal(microphoneRequests, 0);
    assert.equal(context.state, "closed");
    assert.equal(client.micStream, null);
});

test("a microphone acquired after disconnect is released without replacing a new session", async () => {
    const permission = deferred();
    const requested = deferred();
    const oldTrack = { stop() { this.stopped = true; } };
    const newTrack = { stop() { this.stopped = true; } };
    let requests = 0;
    const { client } = await harness({ start: false, getUserMedia: () => {
        if (++requests === 1) { requested.resolve(); return permission.promise; }
        return Promise.resolve({ getTracks: () => [newTrack] });
    } });
    const starting = client.startListening("old", "user");
    const oldSocket = client.ws;
    const onOldClose = oldSocket.onclose;
    await requested.promise;
    oldSocket.close();
    await client.startListening("new", "user");
    const newSocket = client.ws;
    const context = client.audioContext;
    try {
        onOldClose(); // A delayed event from an old socket cannot close the new one.
        permission.resolve({ getTracks: () => [oldTrack] });
        await starting;
        assert.equal(oldTrack.stopped, true);
        assert.equal(newTrack.stopped, undefined);
        assert.equal(client.ws, newSocket);
        assert.equal(client.audioContext, context);
        assert.equal(context.state, "running");
        assert.deepEqual(client.micStream.getTracks(), [newTrack]);
    } finally { await client.stopListening("new"); }
});

test("old audio shutdown cannot release a new session's resources", async () => {
    const closing = deferred();
    const { client, microphoneTrack } = await harness({ close: () => closing.promise });
    const stopped = client.stopListening("old");
    assert.equal(microphoneTrack.stopped, true);
    await client.startListening("new", "user");
    const context = client.audioContext;
    const gainNode = client.gainNode;
    const micStream = client.micStream;
    closing.resolve();
    await stopped;
    assert.equal(client.audioContext, context);
    assert.equal(client.gainNode, gainNode);
    assert.equal(client.micStream, micStream);
    assert.equal(context.state, "running");
    await client.stopListening("new");
});

test("a page may stop again after disconnect without duplicate audio shutdown", async () => {
    let closes = 0;
    const { client, socket, microphoneTrack } = await harness({ close: () => { closes++; } });
    socket.close();
    await client.stopListening("session");
    assert.equal(closes, 1);
    assert.equal(microphoneTrack.stopped, true);
});

test("nod metadata without a playback ID preserves resume and main-response ordering", async () => {
    const { client, socket, context, sources } = await harness({ autoDecode: false });
    try {
        socket.receive({ ...audio("bc"), metadata: { nod: true } });
        socket.receive(stopResponse);
        context.autoDecode = true;
        context.decodes[0]();
        assert.equal(sources[0].buffer.marker, "bc");
        socket.receive({ type: "accepted" });
        socket.receive(audio("main"));
        assert.equal(sources[0].stopped, undefined);
        sources[0].end();
        await flush();
        assert.deepEqual(sources.map(source => source.buffer.marker), ["bc", "main"]);
    } finally { await client.stopListening("session"); }
});

test("an audible nod finishes across user resumption and main response transition", async () => {
    const { client, socket, sources } = await harness();
    try {
        socket.receive(nod("bc-1"));
        socket.receive(stopResponse);
        socket.receive({ type: "accepted" });
        socket.receive(audio("main"));
        assert.equal(sources.length, 1);
        assert.equal(sources[0].stopped, undefined);
        assert.equal(client.currentAudioMessage.metadata.nod_id, "bc-1");
        sources[0].end();
        await flush();
        assert.equal(sources.length, 2);
        assert.equal(sources[1].buffer.marker, "main");
        assert.equal(sources[0].stopped, undefined);
    } finally { await client.stopListening("session"); }
});

test("barge-in discards queued main audio and preserves queued and audible nods", async () => {
    const { client, socket, sources } = await harness();
    try {
        socket.receive(nod("bc-1"));
        socket.receive(audio("canceled-main"));
        socket.receive(nod("pending-bc"));
        socket.receive(stopResponse);
        assert.deepEqual(client.messageQueue.map(message => message.metadata.nod_id), ["pending-bc"]);
        assert.equal(sources[0].stopped, undefined);
        sources[0].end();
        await flush();
        assert.deepEqual(sources.map(source => source.buffer.marker), ["bc-1", "pending-bc"]);
        sources[1].end();
        await flush();
        assert.equal(client.isAudioPlaying, false);
    } finally { await client.stopListening("session"); }
});

test("barge-in preserves a decoding nod without restarting its playback", async () => {
    const { client, socket, context, sources } = await harness({ autoDecode: false });
    try {
        socket.receive(nod("decoding-bc"));
        socket.receive(nod("queued-bc"));
        socket.receive(stopResponse);
        socket.receive(stopResponse);
        assert.equal(client.currentAudioMessage.metadata.nod_id, "decoding-bc");
        assert.deepEqual(client.messageQueue.map(message => message.metadata.nod_id), ["queued-bc"]);
        context.autoDecode = true;
        context.decodes[0]();
        assert.deepEqual(sources.map(source => source.buffer.marker), ["decoding-bc"]);
        sources[0].end();
        await flush();
        assert.deepEqual(sources.map(source => source.buffer.marker), ["decoding-bc", "queued-bc"]);
        sources[1].end();
        await flush();
        assert.equal(client.processingQueue, false);
    } finally { await client.stopListening("session"); }
});

for (const autoDecode of [true, false]) {
    test(`barge-in stops ${autoDecode ? "playing" : "decoding"} main audio then plays the retained nod`, async () => {
        const { client, socket, context, sources } = await harness({ autoDecode });
        try {
            socket.receive(audio("main"));
            socket.receive(audio("queued-main"));
            socket.receive(nod("queued-bc"));
            socket.receive(stopResponse);
            assert.deepEqual(client.messageQueue.map(message => message.metadata.nod_id), ["queued-bc"]);
            context.autoDecode = true;
            await flush();
            if (autoDecode) assert.equal(sources[0].stopped, true);
            else context.decodes[0](); // A canceled main decode must not start later.
            assert.deepEqual(sources.map(source => source.buffer.marker), autoDecode ? ["main", "queued-bc"] : ["queued-bc"]);
            assert.equal(client.currentAudioMessage.metadata.nod_id, "queued-bc");
            sources.at(-1).end();
            await flush();
            assert.equal(client.processingQueue, false);
        } finally { await client.stopListening("session"); }
    });
}

for (const transition of ["accepted", "start", "chunk"]) {
    test(`${transition} drops a decoding and a queued nod before main playback`, async () => {
        const { client, socket, context, sources } = await harness({ autoDecode: false });
        try {
            socket.receive(nod("decoding-bc"));
            socket.receive(nod("queued-bc"));
            context.autoDecode = true;
            socket.receive({ type: transition });
            socket.receive(audio("main"));
            await flush();
            assert.deepEqual(sources.map(source => source.buffer.marker), ["main"]);
            context.decodes[0]();
            assert.equal(sources.length, 1);
        } finally { await client.stopListening("session"); }
    });
}

for (const autoDecode of [true, false]) {
    test(`ordinary barge-in cancels ${autoDecode ? "playing" : "decoding"} main audio and queued chunks`, async () => {
        const { client, socket, context, sources } = await harness({ autoDecode });
        try {
            socket.receive(audio("main"));
            socket.receive(audio("queued-main"));
            socket.receive(stopResponse);
            await flush();
            assert.equal(client.messageQueue.length, 0);
            assert.equal(client.isAudioPlaying, false);
            if (autoDecode) assert.equal(sources[0].stopped, true);
            else {
                context.decodes[0]();
                assert.equal(sources.length, 0);
            }
        } finally { await client.stopListening("session"); }
    });
}

for (const autoDecode of [true, false]) {
    for (const disconnect of [true, false]) {
        test(`${disconnect ? "disconnect" : "explicit Stop"} terminates ${autoDecode ? "playing" : "decoding"} nods and releases resources`, async () => {
            const { client, socket, context, sources, microphoneTrack } = await harness({ autoDecode });
            socket.receive(nod("bc-1"));
            socket.receive(audio("queued-main"));
            if (disconnect) socket.close();
            else await client.stopListening("session");
            await flush();
            assert.equal(client.messageQueue.length, 0);
            assert.equal(context.state, "closed");
            assert.equal(microphoneTrack.stopped, true);
            if (autoDecode) assert.equal(sources[0].stopped, true);
            else {
                context.decodes[0]();
                assert.equal(sources.length, 0);
            }
        });
    }
}

test("startup preserves an error raised before the WebSocket is created", async () => {
    const socketError = new Error("Invalid WebSocket URL");
    await assert.rejects(harness({ socketError }), error => error === socketError);
});

test("nod wire messages preserve playing audio and clear pending audio", async () => {
    const { client, socket, sources } = await harness();
    try {
        const nodAudio = id => ({ ...audio(id), metadata: { nod: true, nod_id: id } });
        socket.receive(nodAudio("nod-1"));
        socket.receive(audio("old-main"));
        socket.receive(nodAudio("nod-2"));
        socket.receive(stopResponse);
        assert.equal(sources[0].stopped, undefined);
        assert.equal(client.messageQueue.length, 1);
        socket.receive({ type: "accepted" });
        socket.receive(audio("new-main"));
        assert.equal(sources[0].stopped, undefined);
        sources[0].end();
        await flush();
        assert.deepEqual(sources.map(source => source.buffer.marker), ["nod-1", "new-main"]);
        socket.receive(stopResponse);
        assert.equal(sources[1].stopped, true);
    } finally { await client.stopListening("session"); }
});

test("plain stop preserves a playing nod without a playback ID", async () => {
    const { client, socket, sources } = await harness();
    try {
        socket.receive({ ...audio("nod"), metadata: { nod: true } });
        socket.receive({ type: "stop" });
        assert.equal(sources[0].stopped, undefined);
        assert.equal(client.messageQueue.length, 0);
    } finally { await client.stopListening("session"); }
});

test("empty-text nods leave the 3D message window unchanged while audio and normal responses continue", async () => {
    const { client, socket, sources } = await harness();
    const AvatarUI = new Function(`${uiSource}; return AvatarUI;`)();
    const ui = Object.create(AvatarUI.prototype);
    const messageClasses = new Set(["hidden"]);
    Object.assign(ui, {
        aiavatar: client, currentUserText: "", currentAIText: "",
        speakerLabelUser: "User", speakerLabelAI: "AI",
        messageSpeaker: {}, messageText: {}, toolStatus: {},
        messageBox: { classList: {
            add: name => messageClasses.add(name),
            remove: name => messageClasses.delete(name),
        } },
    });
    const received = [];
    const consumer = name => ({ handleResponse: response => received.push([name, response]) });
    const handler = appSource.match(/    aiavatar\.onResponseReceived = \(response\) => \{[\s\S]*?\n    \};/)[0];
    new Function("aiavatar", "ui", "backlog", "artifacts", "modelAdapter", "vision", "display", handler)(
        client, ui, consumer("backlog"), consumer("artifacts"), consumer("model"), consumer("vision"), {},
    );
    const nodAudio = id => ({
        ...audio(id), text: "", voice_text: "",
        metadata: { nod: true, nod_id: id, recording_id: "record-a" },
    });
    try {
        socket.receive(nodAudio("nod-1"));
        assert.deepEqual([...messageClasses], ["hidden"]);
        assert.equal(ui.currentAIText, "");
        assert.ok(received.some(([name, response]) => name === "backlog" && response.metadata?.nod === true));
        assert.equal(sources[0].buffer.marker, "nod-1");

        socket.receive({ type: "info", text: null, metadata: { partial_request_text: "話の続き" } });
        socket.receive(nodAudio("nod-2"));
        assert.equal(ui.messageText.textContent, "話の続き");
        assert.equal(ui.messageSpeaker.textContent, "User");

        socket.receive({ ...stopResponse, text: null });
        socket.receive({ type: "accepted", text: null });
        socket.receive({ ...audio("main"), text: "回答です。", voice_text: "回答です。" });
        socket.receive({ type: "final", text: "回答です。", voice_text: "回答です。" });
        assert.equal(ui.messageText.textContent, "回答です。");
        assert.equal(ui.messageSpeaker.textContent, "AI");
        assert.equal(ui.isServerProcessing, false);
        assert.ok(received.some(([name, response]) => name === "backlog" && response.type === "final"));
        assert.ok(received.filter(([, response]) => response.metadata?.nod === true)
            .every(([, response]) => response.text === "" && response.voice_text === ""));
        assert.equal(sources[0].stopped, undefined);
        sources[0].end();
        await flush();
        assert.deepEqual(sources.map(source => source.buffer.marker), ["nod-1", "main"]);
    } finally { await client.stopListening("session"); }
});
