import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../../../../examples/websocket/html/aiavatar.js", import.meta.url), "utf8");
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };

async function harness(t, options = {}) {
    const sources = [], timers = new Map(), frames = new Map(), errors = [], events = [];
    let nextId = 0;
    class AudioContext {
        constructor() { this.currentTime = 10; this.state = "running"; this.destination = {}; }
        async resume() {}
        async close() { this.state = "closed"; }
        createGain() { return { gain: {}, connect() {} }; }
        createMediaStreamSource() { return { connect() {} }; }
        createScriptProcessor() { return { connect() {}, disconnect() {} }; }
        createBuffer(channels, length, sampleRate) {
            const samples = Array.from({ length: channels }, () => new Float32Array(length));
            return { length, sampleRate, duration: length / sampleRate, getChannelData: channel => samples[channel] };
        }
        createBufferSource() {
            const node = {
                connect() {}, start(time) { this.startedAt = time; },
                stop() { this.stopped = true; }, disconnect() { this.disconnected = true; },
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
        close() { this.readyState = 3; this.onclose?.(); }
        receive(message) { this.onmessage({ data: JSON.stringify(message) }); }
    }
    let client;
    const Client = new Function("window", "navigator", "WebSocket", "setTimeout", "clearTimeout",
        "requestAnimationFrame", "cancelAnimationFrame", "console", `${source}; return AIAvatarClient;`)(
        { AudioContext }, { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } }, WebSocket,
        (callback, ms) => { timers.set(++nextId, { callback, at: client.audioContext.currentTime + ms / 1000 }); return nextId; },
        id => timers.delete(id), callback => { frames.set(++nextId, callback); return nextId; },
        id => frames.delete(id), { log() {}, error: (...args) => errors.push(args) },
    );
    client = new Client({ webSocketUrl: "ws://example.test", ...options });
    client.onPlaybackStart = state => events.push({ type: "start", ...state });
    client.onPlaybackEnd = state => events.push({ type: "end", ...state });
    client.onPlaybackAudio = state => events.push({ type: "audio", ...state });
    await client.startListening("session", "user");
    t.after(async () => {
        await client.stopListening("session");
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
    return { client, sources, events, frames, advance,
        receive: message => client.ws.receive(message),
        microphone: () => client.scriptNode.onaudioprocess({ inputBuffer: { getChannelData: () => new Float32Array([0.5]) } }),
    };
}

function descriptor(id = "live") {
    return { type: "chunk", session_id: "session", transaction_id: null, audio_data: null,
        metadata: { audio_id: id, continuous_audio: true, pcm_format: { sample_rate: 16000, channels: 1, sample_width: 2 } },
    };
}
function pcm(header, frames = 1600) {
    const bytes = Buffer.alloc(frames * 2);
    for (let i = 0; i < frames; i++) bytes.writeInt16LE(16384, i * 2);
    return { ...header, audio_data: bytes.toString("base64") };
}

test("continuous PCM survives gaps, resets visemes and resumes without another descriptor", async t => {
    const { client, receive, sources, advance, events, frames } = await harness(t);
    const header = descriptor();
    receive(header);
    assert.equal(client.isAudioPlaying, true);
    receive(pcm(header));
    advance(10.06);
    assert.equal(client.isAudioPlaying, true);
    assert.ok(Number.isNaN(events[0].durationSeconds));
    assert.equal(events.find(event => event.type === "audio").pcm[0], 0.5);
    advance(10.15);
    sources[0].end();
    assert.equal(client.isAudioPlaying, false);
    assert.equal(events.at(-1).type, "end");
    assert.equal(frames.size, 0);
    assert.equal(client.pcmAudio.size, 1, "keep the descriptor while no output is arriving");
    assert.equal(client.processingQueue, true);
    receive({ type: "final", session_id: "session" });
    advance(11);
    receive(pcm(header));
    assert.equal(client.isAudioPlaying, true, "resume playback state when the next audio is scheduled");
    advance(11.06);
    assert.equal(client.isAudioPlaying, true);
    assert.equal(events.filter(event => event.type === "start").length, 2);
    assert.equal(sources.length, 2);
});

test("stop and disconnect reject late PCM and reconnect uses only its new stream", async t => {
    const { client, receive, sources, advance } = await harness(t);
    const header = descriptor();
    receive(header);
    receive(pcm(header));
    advance(10.06);
    receive({ type: "stop" });
    receive(pcm(header));
    await flush();
    assert.equal(sources.length, 1);
    assert.equal(sources[0].stopped, true);
    assert.equal(client.processingQueue, false);
    assert.equal(client.pcmAudio.size, 0);
    const second = descriptor("second");
    receive(second);
    receive(pcm(second));
    const oldReceive = client.ws.onmessage;
    client.ws.close();
    await flush();
    assert.equal(sources[1].stopped, true);
    assert.equal(client.pcmAudio.size, 0);
    await client.startListening("session", "user");
    oldReceive({ data: JSON.stringify(pcm(second)) });
    receive(pcm(second));
    const next = descriptor("reconnected");
    receive(next);
    receive(pcm(next));
    assert.equal(sources.length, 3);
    assert.equal(client.currentAudioMessage.metadata.audio_id, "reconnected");
});

test("realtime microphone input follows the existing mute controls", async t => {
    const { client, receive, microphone } = await harness(t);
    client.isAudioPlaying = true;
    microphone();
    assert.equal(client.ws.sent.at(-1).audio_data, "AAA=");
    receive({ type: "connected", metadata: { realtime: true } });
    microphone();
    assert.equal(client.ws.sent.at(-1).audio_data, "AAA=", "connection metadata does not override mute");
    client.isMicrophoneMuted = () => false;
    microphone();
    assert.notEqual(client.ws.sent.at(-1).audio_data, "AAA=");
    client.toggleMute();
    microphone();
    assert.equal(client.ws.sent.at(-1).audio_data, "AAA=");
});
