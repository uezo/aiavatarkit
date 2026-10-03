import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const html = new URL("../../../examples/websocket/html/", import.meta.url);
const source = await readFile(new URL("aiavatar.js", html), "utf8");
const contextSource = await readFile(new URL("playback-context.js", html), "utf8");
const installPlaybackContext = new Function(`${contextSource.replace("export function", "function")}; return installPlaybackContext;`)();
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);

async function harness(t) {
    const sources = [], timers = new Map(), frames = new Map(), errors = [];
    let identifier = 0;
    class AudioContext {
        constructor() { this.currentTime = 10; this.state = "running"; this.destination = {}; this.decodes = 0; }
        async resume() {}
        async close() { this.state = "closed"; }
        createGain() { return { gain: {}, connect() {} }; }
        createMediaStreamSource() { return { connect() {} }; }
        createScriptProcessor() { return { connect() {}, disconnect() {} }; }
        createBuffer(channels, length, sampleRate) {
            const data = Array.from({ length: channels }, () => new Float32Array(length));
            return { length, sampleRate, duration: length / sampleRate, getChannelData: channel => data[channel] };
        }
        decodeAudioData(bytes, success) { this.decodes++; success(this.createBuffer(1, 1600, 16000)); }
        createBufferSource() {
            const node = {
                connect(destination) { this.destination = destination; },
                start(time) { this.startedAt = time; },
                stop() { this.stopped = true; },
                disconnect() { this.disconnected = true; },
                end() { this.onended?.(); },
            };
            sources.push(node);
            return node;
        }
    }
    class WebSocket {
        static OPEN = 1;
        constructor() { this.readyState = 1; this.sent = []; }
        send(data) { this.sent.push(JSON.parse(data)); }
        close() { this.readyState = 3; }
        receive(message) { this.onmessage({ data: JSON.stringify(message) }); }
    }
    let client;
    const Client = new Function("window", "navigator", "WebSocket", "setTimeout", "clearTimeout",
        "requestAnimationFrame", "cancelAnimationFrame", "console", `${source}; return AIAvatarClient;`)(
        { AudioContext }, { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } }, WebSocket,
        (callback, ms) => { timers.set(++identifier, { callback, at: client.audioContext.currentTime + ms / 1000 }); return identifier; },
        id => timers.delete(id), callback => { frames.set(++identifier, callback); return identifier; },
        id => frames.delete(id), { log() {}, error: (...args) => errors.push(args) },
    );
    client = new Client({ webSocketUrl: "ws://example.test" });
    const context = installPlaybackContext(client);
    client.onResponseReceived = message => context.handleResponse(message);
    await client.startListening("session", "user");
    t.after(async () => {
        await client.stopListening("session");
        context.dispose();
        assert.deepEqual(errors, []);
        assert.equal(timers.size, 0);
        assert.equal(frames.size, 0);
    });
    const advance = time => {
        client.audioContext.currentTime = time;
        for (const [id, timer] of [...timers]) {
            if (timer.at <= time + 1e-9) { timers.delete(id); timer.callback(); }
        }
        for (const [id, callback] of [...frames]) { frames.delete(id); callback(time * 1000); }
    };
    return { client, sources, advance, frames,
        receive: message => client.ws.receive(message),
        controls: () => client.ws.sent.filter(message => message.type === "playback").map(message => message.metadata),
    };
}

function descriptor(id, count, { rate = 16000, channels = 1, nod = false } = {}) {
    return { type: "chunk", session_id: "session", transaction_id: nod ? null : "transaction",
        voice_text: nod ? "" : id, audio_data: null,
        metadata: { audio_id: id, audio_frame_count: count, pcm_format: { sample_rate: rate, channels, sample_width: 2 },
            ...(nod ? { nod: true } : {}) },
    };
}
function pcm(header, values) {
    const bytes = Buffer.alloc(values.length * 2);
    values.forEach((value, i) => bytes.writeInt16LE(value, i * 2));
    return { type: "chunk", session_id: header.session_id, transaction_id: header.transaction_id,
        audio_data: bytes.toString("base64"),
        metadata: { audio_id: header.metadata.audio_id, pcm_format: header.metadata.pcm_format },
    };
}
const final = { type: "final", session_id: "session", transaction_id: "transaction" };

test("PCM starts with the first fragment and reports the original audio only once", async t => {
    const { client, receive, sources, advance, controls } = await harness(t);
    const header = descriptor("speech", 4800, { rate: 24000 });
    receive(header);
    receive(pcm(header, Array(2400).fill(16384)));
    assert.equal(sources.length, 1, "start scheduling before the full audio arrives");
    assert.equal(client.audioContext.decodes, 0);
    near(sources[0].startedAt, 10.05);
    assert.equal(controls().length, 0, "scheduled audio has not sounded yet");
    advance(10.06);
    assert.deepEqual(controls()[0], { event: "start", playback_id: "speech", text: "speech",
        duration_seconds: 0.2, transaction_id: "transaction" });
    receive(pcm(header, Array(2400).fill(-16384)));
    near(sources[1].startedAt, 10.15);
    receive(final);
    assert.equal(controls().at(-1).event, "final");
    sources[0].end();
    assert.equal(client.isAudioPlaying, true);
    sources[1].end();
    await flush();
    assert.deepEqual(controls().map(event => event.event), ["start", "final", "end"]);
    assert.equal(controls().at(-1).completed, true);
    assert.equal(client.isAudioPlaying, false);
    assert.equal(client.pcmAudio.size, 0);
});

test("stereo PCM deinterleaves and follows the output sample rate", async t => {
    const { receive, sources } = await harness(t);
    const header = descriptor("stereo", 2, { rate: 48000, channels: 2 });
    receive(header);
    receive(pcm(header, [-32768, 16384, 32767, -16384]));
    assert.deepEqual([...sources[0].buffer.getChannelData(0)], [-1, 32767 / 32768]);
    assert.deepEqual([...sources[0].buffer.getChannelData(1)], [0.5, -0.5]);
    assert.equal(sources[0].buffer.sampleRate, 48000);
});

test("buffer starvation does not finish the audio or repeat its start notification", async t => {
    const { client, receive, sources, advance, controls } = await harness(t);
    const header = descriptor("gap", 3200);
    receive(header);
    receive(pcm(header, Array(1600).fill(1)));
    advance(10.15);
    sources[0].end();
    assert.equal(client.isAudioPlaying, true);
    assert.deepEqual(controls().map(event => event.event), ["start"]);
    advance(10.3);
    receive(pcm(header, Array(1600).fill(2)));
    near(sources[1].startedAt, 10.35);
    sources[1].end();
    await flush();
    assert.deepEqual(controls().map(event => event.event), ["start", "end"]);
});

test("PCM lip sync retains history across short fragments at the scheduled position", async t => {
    const { client, receive, sources, advance } = await harness(t);
    const analyzed = [];
    client.onPlaybackAudio = data => analyzed.push(data);
    const header = descriptor("short", 1600);
    receive(header);
    for (let i = 0; i < 10; i++) receive(pcm(header, Array(160).fill(16384)));
    advance(10.01);
    assert.equal(analyzed.length, 0);
    advance(10.125);
    assert.equal(analyzed.length, 1);
    assert.ok(analyzed[0].samplePosition >= 1024);
    assert.ok(analyzed[0].pcm.every(value => value === 0.5));
    receive({ type: "stop" });
    assert.ok(sources.every(node => node.stopped && node.disconnected));
});

test("queued original audio keeps its text and final notification, while WAV still works", async t => {
    const { receive, sources, advance, controls } = await harness(t);
    const first = descriptor("first", 1600), last = descriptor("last", 1600);
    receive(first);
    receive(pcm(first, Array(1600).fill(1)));
    receive(last);
    receive(pcm(last, Array(1600).fill(2)));
    receive(final);
    assert.equal(sources.length, 1);
    advance(10.15);
    sources[0].end();
    await flush();
    assert.equal(sources.length, 2);
    near(sources[1].startedAt, 10.15);
    advance(10.15);
    assert.deepEqual(controls().map(event => event.event), ["start", "end", "start", "final"]);
    assert.equal(controls().at(-1).playback_id, "last");
    receive({ type: "chunk", audio_data: "AAAA" });
    sources[1].end();
    await flush();
    assert.equal(sources.length, 3);
    sources[2].end();
});

test("stop cancels active and queued PCM and late fragments cannot restart them", async t => {
    const { client, receive, sources, advance, controls } = await harness(t);
    const first = descriptor("first", 1600), queued = descriptor("queued", 1600);
    receive(first);
    receive(pcm(first, Array(1600).fill(1)));
    receive(queued);
    receive(pcm(queued, Array(1600).fill(2)));
    advance(10.06);
    receive({ type: "stop" });
    receive(pcm(first, [3]));
    receive(pcm(queued, [4]));
    await flush();
    assert.equal(sources.length, 1);
    assert.equal(sources[0].stopped, true);
    assert.equal(client.pcmAudio.size, 0);
    assert.equal(controls().at(-1).completed, false);
});

test("an audible PCM Nod drains on stop and main response waits behind it", async t => {
    const { client, receive, sources, advance, controls } = await harness(t);
    const nod = descriptor("nod", 3200, { nod: true }), main = descriptor("main", 1600);
    receive(nod);
    receive(pcm(nod, Array(1600).fill(1))); // Sender canceled before its remaining frames.
    advance(10.06);
    receive({ type: "stop" });
    receive({ type: "accepted" });
    receive(main);
    receive(pcm(main, Array(1600).fill(2)));
    assert.equal(sources.length, 1);
    assert.equal(sources[0].stopped, undefined);
    assert.equal(controls().length, 0);
    advance(10.15);
    sources[0].end();
    await flush();
    assert.equal(sources.length, 2);
    advance(10.15);
    assert.equal(controls()[0].text, "main");
    assert.equal(client.currentAudioMessage.metadata.audio_id, "main");
});

test("main acceptance cancels a PCM Nod before audible start and drops queued Nods", async t => {
    const { client, receive, sources, controls } = await harness(t);
    const nod = descriptor("nod", 1600, { nod: true }), next = descriptor("next", 1600, { nod: true });
    receive(nod);
    receive(pcm(nod, Array(1600).fill(1)));
    receive(next);
    receive(pcm(next, Array(1600).fill(1)));
    receive({ type: "accepted" });
    await flush();
    assert.equal(sources[0].stopped, true);
    assert.equal(sources.length, 1);
    assert.equal(client.pcmAudio.size, 0);
    assert.equal(controls().length, 0);
});

for (const receivedFrames of [0, 1600]) {
    test(`stop drains a queued Nod with ${receivedFrames} received frames without blocking the queue`, async t => {
        const { client, receive, sources, advance } = await harness(t);
        const main = descriptor("main", 3200), nod = descriptor("nod", 3200, { nod: true });
        receive(main);
        receive(pcm(main, Array(1600).fill(1)));
        receive(nod);
        if (receivedFrames) receive(pcm(nod, Array(receivedFrames).fill(2)));
        receive({ type: "stop" });
        await flush();
        if (receivedFrames) {
            advance(sources[1].startedAt + sources[1].buffer.duration);
            sources[1].end();
            await flush();
        }
        assert.equal(client.isAudioPlaying, false);
        assert.equal(client.processingQueue, false);
        assert.equal(client.pcmAudio.size, 0);
    });
}

test("an interrupted final closes partial PCM so the next response can play", async t => {
    const { client, receive, sources, advance, controls } = await harness(t);
    const first = descriptor("old", 3200), next = descriptor("new", 1600);
    next.transaction_id = "next";
    receive(first);
    receive(pcm(first, Array(1600).fill(1)));
    advance(10.06);
    receive({ ...final, metadata: { interrupted: true } });
    receive(next);
    receive(pcm(next, Array(1600).fill(2)));
    sources[0].end();
    await flush();
    assert.equal(controls().at(-1).completed, false);
    assert.equal(sources.length, 2);
    assert.equal(client.currentAudioMessage.metadata.audio_id, "new");
});

test("disconnect clears PCM sources, descriptors and callbacks before reconnect", async t => {
    const { client, receive, sources, advance } = await harness(t);
    const header = descriptor("before", 3200);
    client.onPlaybackAudio = () => {};
    receive(header);
    receive(pcm(header, Array(1600).fill(1)));
    advance(10.06);
    await client.stopListening("session");
    assert.equal(client.pcmAudio.size, 0);
    assert.equal(sources[0].stopped, true);
    await client.startListening("session", "user");
    receive(pcm(header, Array(1600).fill(2)));
    const next = descriptor("after", 1600);
    receive(next);
    receive(pcm(next, Array(1600).fill(3)));
    assert.equal(sources.length, 2);
});
