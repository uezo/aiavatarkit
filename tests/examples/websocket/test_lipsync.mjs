import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const lipSyncSource = await readFile(new URL("lipsync.js", htmlDirectory), "utf8");
const mfccLipSyncSource = await readFile(new URL("mfcc-lipsync.js", htmlDirectory), "utf8");
const vrmIdleSource = await readFile(new URL("vrm-idle.js", htmlDirectory), "utf8");
const imageAvatarSource = await readFile(new URL("image-avatar.js", htmlDirectory), "utf8");
const vrmAdapterSource = await readFile(
    new URL("avatar3d/models/vrm/vrm-adapter.js", htmlDirectory),
    "utf8",
);
const mmdAdapterSource = await readFile(
    new URL("avatar3d/models/mmd/mmd-adapter.js", htmlDirectory),
    "utf8",
);
const mptAvatarSource = await readFile(new URL("mpt-avatar.js", htmlDirectory), "utf8");
const motionPngLipSyncSource = await readFile(
    new URL("motionpngtuber/lipsync.js", htmlDirectory),
    "utf8",
);
const indexSource = await readFile(new URL("index.html", htmlDirectory), "utf8");
const threeDSource = await readFile(new URL("3d.html", htmlDirectory), "utf8");
const defaultFemaleProfile = JSON.parse(await readFile(
    new URL("profiles/default-female.json", htmlDirectory),
    "utf8",
));
const {
    LipSyncEngine,
    MFCCLipSyncEngine,
} = new Function(`
    ${lipSyncSource}
    ${mfccLipSyncSource}
    return { LipSyncEngine, MFCCLipSyncEngine };
`)();
const ImageAvatar = new Function(`${imageAvatarSource}; return ImageAvatar;`)();
const MmdAdapter = new Function(`
    ${mmdAdapterSource
        .replace(/^import .*$/gm, "")
        .replace("export class MmdAdapter", "class MmdAdapter")}
    return MmdAdapter;
`)();

function sine(frequency, { sampleRate = 16000, sampleCount = 1024, amplitude = 0.2 } = {}) {
    return Float32Array.from(
        { length: sampleCount },
        (_, index) => amplitude * Math.sin(2 * Math.PI * frequency * index / sampleRate),
    );
}

function playbackInput(pcm, sampleRate = 16000, overrides = {}) {
    return {
        pcm,
        sampleRate,
        samplePosition: pcm.length,
        tSec: pcm.length / sampleRate,
        ...overrides,
    };
}

function profileWith(calibrations, overrides = {}) {
    return {
        mfccNum: 12,
        mfccDataCount: 16,
        melFilterBankChannels: 30,
        targetSampleRate: 16000,
        sampleCount: 1024,
        useStandardization: false,
        compareMethod: 1,
        mfccs: Object.entries(calibrations).map(([name, vectors]) => ({
            name,
            mfccCalibrationDataList: vectors.map((array) => ({ array: Array.from(array) })),
        })),
        ...overrides,
    };
}

test("legacy engine owns its PCM RMS and spectral centroid analysis", () => {
    const low = LipSyncEngine.analyze(playbackInput(sine(300)));
    const high = LipSyncEngine.analyze(playbackInput(sine(3000)));

    assert.ok(Math.abs(low.rms - Math.SQRT1_2 * 0.2) < 0.002);
    assert.ok(high.centroid01 > low.centroid01 * 4);
});

test("legacy engine defaults to 30Hz and keeps threshold updates periodic after history fills", () => {
    const engine = new LipSyncEngine({ historySeconds: 4 });
    let updateCount = 0;
    engine.autoUpdateThresholds = () => { updateCount++; };

    for (let frame = 0; frame < 180; frame++) {
        engine._update({ rms: 0.1, centroid01: 0.2, tSec: frame / 30 });
    }

    assert.equal(engine.cfg.audioHz, 30);
    assert.equal(engine.histories.env.length, 120);
    assert.equal(updateCount, 3);
});

test("uLipSync v3 JSON profile classifies MFCCs and emits five vowel weights", () => {
    const seedProfile = profileWith({
        A: [new Float64Array(12)],
        I: [new Float64Array(12)],
    });
    const calibrator = new MFCCLipSyncEngine({ profile: seedProfile });
    const aAudio = playbackInput(sine(450));
    const iAudio = playbackInput(sine(2400));
    const aMfcc = calibrator.extractMfcc(aAudio).mfcc;
    const iMfcc = calibrator.extractMfcc(iAudio).mfcc;

    const engine = new MFCCLipSyncEngine({
        profile: profileWith({ A: [aMfcc], I: [iMfcc] }),
    });
    const result = engine.processAudioData(aAudio);

    assert.ok(result.visemes.A > result.visemes.I);
    assert.equal(result.mainViseme, "A");
    assert.ok(result.mainVisemeWeight > 0);
    assert.deepEqual(Object.keys(result.visemes), ["A", "I", "U", "E", "O"]);
});

test("bundled default female MFCC profile is ready for the 3D viewer", async () => {
    const engine = new MFCCLipSyncEngine({ profile: defaultFemaleProfile });
    await engine.initialize();
    const result = engine.processAudioData(playbackInput(sine(700)));

    assert.deepEqual(Object.keys(result.visemes), ["A", "I", "U", "E", "O"]);
    assert.ok(Object.values(result.visemes).every(Number.isFinite));
    assert.ok(Math.max(...Object.values(result.visemes)) > 0);
    assert.deepEqual(
        defaultFemaleProfile.mfccs.map(({ name }) => name),
        ["A", "I", "U", "E", "O"],
    );
    assert.equal(
        threeDSource.match(/profileUrl: "profiles\/default-female\.json"/g)?.length,
        2,
    );
    assert.doesNotMatch(threeDSource, /my-tts3|ulipsync-sample/);
    assert.match(threeDSource, /maxVolume: -0\.8/);
    assert.equal(threeDSource.match(/usePhonemeBlend: false/g)?.length, 2);
    assert.equal(threeDSource.match(/maxVisemeWeight:/g)?.length, 2);
    assert.doesNotMatch(mfccLipSyncSource, /usePhonemeBlend/);
    assert.doesNotMatch(threeDSource, /playbackAudioHz:/);
    assert.doesNotMatch(threeDSource, /topK:/);
    assert.doesNotMatch(mfccLipSyncSource, /topK/);
    assert.doesNotMatch(threeDSource, /phonemeScoreMultipliers/);
    assert.doesNotMatch(threeDSource, /lipSyncLogBtn|downloadDiagnostics|debug:\s*true/);
    assert.doesNotMatch(
        mfccLipSyncSource,
        /diagnostic|localStorage|downloadDiagnostics|exportDiagnostics|debug\s*=/i,
    );
});

test("MMD adapter scales blended and winner-only visemes by its configured maximum", () => {
    const adapter = Object.create(MmdAdapter.prototype);
    adapter.config = {
        lipsync: { usePhonemeBlend: true, maxVisemeWeight: 0.5 },
    };
    const result = {
        visemes: { A: 0.7, I: 0.2, U: 0, E: 0, O: 0 },
        mainViseme: "A",
        mainVisemeWeight: 0.7,
    };

    assert.deepEqual(adapter.lipSyncWeights(result), {
        A: 0.35,
        I: 0.1,
        U: 0,
        E: 0,
        O: 0,
    });
    adapter.config.lipsync.usePhonemeBlend = false;
    assert.deepEqual(adapter.lipSyncWeights(result), {
        A: 0.35,
        I: 0,
        U: 0,
        E: 0,
        O: 0,
    });
});

test("MFCC cosine scores use uLipSync Float32 underflow before classification", () => {
    const mfcc = new Float64Array(12);
    mfcc[0] = 1;
    const unitVectorWithCosine = (similarity) => {
        const vector = new Float64Array(12);
        vector[0] = similarity;
        vector[1] = Math.sqrt(1 - similarity * similarity);
        return vector;
    };
    const aMfcc = unitVectorWithCosine(0.2);
    const iMfcc = unitVectorWithCosine(0.3);
    const engine = new MFCCLipSyncEngine({
        profile: profileWith({ A: [aMfcc], I: [iMfcc] }, { compareMethod: 2 }),
    });
    const audio = playbackInput(sine(700));
    engine.extractMfcc = (input) => ({
        mfcc,
        rawVolume: 0.1,
        audio: MFCCLipSyncEngine.input(input),
    });

    const iScore = engine._scoreDetail(mfcc, engine.entries[1].average);
    assert.equal(iScore.metric, 0.3);
    assert.ok(Math.pow(iScore.metric, 100) > 0);
    assert.equal(iScore.score, 0);

    const result = engine.processAudioData(audio);
    assert.deepEqual(result.visemes, { A: 0, I: 0, U: 0, E: 0, O: 0 });
    assert.equal(result.mainViseme, "A");
    assert.ok(result.mainVisemeWeight > 0);
    assert.equal(result.visemes.I, 0);
});

test("MFCC result supports both winner-only and ratio-blended application", () => {
    const audio = playbackInput(sine(700));
    const calibrator = new MFCCLipSyncEngine({
        profile: profileWith({ A: [new Float64Array(12)] }),
    });
    const mfcc = calibrator.extractMfcc(audio).mfcc;
    const profile = profileWith({ "-": [mfcc], A: [mfcc] });

    const result = new MFCCLipSyncEngine({ profile }).processAudioData(audio);

    assert.equal(result.mainViseme, null);
    assert.equal(result.mainVisemeWeight, 0);
    assert.ok(result.visemes.A > 0);
    assert.ok(result.visemes.A < 1);
});

test("uLipSync profile handling averages calibration data and aggregates duplicate phonemes", () => {
    const audio = playbackInput(sine(700));
    const vector = (first) => {
        const values = new Float64Array(12);
        values[0] = first;
        return values;
    };
    const engine = new MFCCLipSyncEngine({
        profile: {
            ...profileWith({ A: [vector(0)] }),
            mfccs: [
                {
                    name: "A",
                    mfccCalibrationDataList: [vector(0), vector(2)].map((array) => ({ array })),
                },
                { name: "A", mfccCalibrationDataList: [{ array: vector(2) }] },
                { name: "I", mfccCalibrationDataList: [{ array: vector(3) }] },
            ],
        },
    });
    engine.extractMfcc = (input) => ({
        mfcc: new Float64Array(12),
        rawVolume: 0.1,
        audio: MFCCLipSyncEngine.input(input),
    });
    engine._score = (_mfcc, average) => ({ 1: 0.3, 2: 0.3, 3: 0.4 })[average[0]];

    const result = engine.processAudioData(audio);

    assert.equal(engine.entries[0].average[0], 1);
    assert.ok(Math.abs(result.visemes.A - 0.6) < 1e-12);
    assert.ok(Math.abs(result.visemes.I - 0.4) < 1e-12);
    assert.equal(result.mainViseme, "A");
    assert.equal(result.mainVisemeWeight, 1);
    assert.deepEqual(
        Object.keys(result),
        ["visemes", "mainViseme", "mainVisemeWeight"],
    );
});

test("MFCC phoneme score multipliers bias normalized scores case-insensitively", () => {
    const vector = (first) => {
        const values = new Float64Array(12);
        values[0] = first;
        return values;
    };
    const engine = new MFCCLipSyncEngine({
        profile: profileWith({ A: [vector(1)], U: [vector(2)] }),
        phonemeScoreMultipliers: { u: 1.4 },
    });
    engine.extractMfcc = (input) => ({
        mfcc: new Float64Array(12),
        rawVolume: 0.1,
        audio: MFCCLipSyncEngine.input(input),
    });
    engine._score = (_mfcc, average) => (average[0] === 1 ? 0.6 : 0.5);

    const result = engine.processAudioData(playbackInput(sine(700)));

    assert.equal(result.mainViseme, "U");
    assert.ok(Math.abs(result.visemes.A - 0.6 / 1.3) < 1e-12);
    assert.ok(Math.abs(result.visemes.U - 0.7 / 1.3) < 1e-12);
    assert.equal(engine._applyPhonemeScoreMultiplier(0.8, "U"), 1);
});

test("MFCC phoneme score multipliers reject invalid values", () => {
    const profile = profileWith({ A: [new Float64Array(12)] });
    for (const phonemeScoreMultipliers of [[], "invalid"]) {
        assert.throws(
            () => new MFCCLipSyncEngine({ profile, phonemeScoreMultipliers }),
            /phonemeScoreMultipliers must be an object/,
        );
    }
    for (const multiplier of [-1, Number.NaN, Number.POSITIVE_INFINITY, "1.4"]) {
        assert.throws(
            () => new MFCCLipSyncEngine({
                profile,
                phonemeScoreMultipliers: { U: multiplier },
            }),
            /must be a finite non-negative number/,
        );
    }
});

test("legacy and MFCC implementations are separate injectable engine objects", async () => {
    const profile = profileWith({ A: [new Float64Array(12)] });
    const legacy = new LipSyncEngine({
        levels: [{ thresh: 0, shape: "closed" }],
        vowelBands: [{ upper: 1, shape: "closed" }],
    });
    const mfcc = new MFCCLipSyncEngine({ profile });

    assert.ok(legacy instanceof LipSyncEngine);
    assert.ok(mfcc instanceof MFCCLipSyncEngine);
    assert.equal(typeof legacy.processAudioData, "function");
    assert.equal(typeof mfcc.processAudioData, "function");
    const legacyResult = legacy.processAudioData(playbackInput(sine(300)));
    const mfccResult = mfcc.processAudioData(playbackInput(sine(300)));
    assert.deepEqual(Object.keys(legacyResult.visemes), ["A", "I", "U", "E", "O"]);
    assert.deepEqual(Object.keys(mfccResult.visemes), ["A", "I", "U", "E", "O"]);
    assert.deepEqual(
        Object.keys(legacyResult),
        ["visemes", "mainViseme", "mainVisemeWeight"],
    );
    assert.deepEqual(Object.keys(mfccResult), Object.keys(legacyResult));
    assert.doesNotMatch(lipSyncSource, /LipSyncAudio|MFCCLipSyncEngine|createLipSyncEngine/);
    assert.doesNotMatch(mfccLipSyncSource, /LipSyncAudio|class LipSyncEngine|ULipSyncEngine|createLipSyncEngine/);
    assert.doesNotMatch(lipSyncSource, /^\s+(?:update|apply)\(/m);
    assert.doesNotMatch(mfccLipSyncSource, /^\s+(?:update|apply)\(/m);
    assert.doesNotMatch(`${lipSyncSource}\n${mfccLipSyncSource}`, /outputType/);
    assert.doesNotMatch(mfccLipSyncSource, /console\.|localStorage|CustomEvent/);
    assert.doesNotMatch(lipSyncSource, /applyTarget|mouthPathTemplate|mouthCache|fetchMouth/);
    assert.match(imageAvatarSource, /applyLipSyncResult\(result\)/);
    for (const adapterSource of [vrmAdapterSource, mmdAdapterSource]) {
        assert.match(adapterSource, /usePhonemeBlend/);
        assert.match(adapterSource, /result\.mainVisemeWeight/);
        assert.match(adapterSource, /result\.visemes/);
    }
    assert.doesNotMatch(mptAvatarSource, /audioAnalyzer|mouthRenderer/);
    assert.match(mptAvatarSource, /new LipsyncEngine/);
    assert.match(motionPngLipSyncSource, /class LipsyncEngine/);
    assert.match(motionPngLipSyncSource, /processAudioData\(data\)/);
    assert.equal(await mfcc.initialize(), mfcc);
    await assert.rejects(
        () => new MFCCLipSyncEngine().initialize(),
        /requires profile or profileUrl/,
    );

    const previousFetch = globalThis.fetch;
    globalThis.fetch = async (url) => ({
        ok: url === "profiles/test.json",
        status: url === "profiles/test.json" ? 200 : 404,
        async json() { return profile; },
    });
    try {
        const loaded = new MFCCLipSyncEngine({
            profileUrl: "profiles/test.json",
        });
        await loaded.initialize();
        assert.ok(loaded instanceof MFCCLipSyncEngine);
        assert.equal(loaded.entries[0].name, "A");
    } finally {
        if (previousFetch === undefined) delete globalThis.fetch;
        else globalThis.fetch = previousFetch;
    }
});

test("image avatar selects a stateless discrete mouth from volume and visemes", () => {
    const avatar = Object.create(ImageAvatar.prototype);
    avatar.mouthImage = { src: "", style: { display: "none" } };
    avatar.mouthOpenThreshold = 0.52;
    avatar.visemeConfidenceThreshold = 0.55;
    avatar.mouthPreloaded = true;
    avatar.mouthCache = new Map([
        ["half", "mouth-half"],
        ["open", "mouth-open"],
        ["u", "mouth-u"],
        ["e", "mouth-e"],
    ]);

    const apply = (mainViseme, mainVisemeWeight, visemes = {}) => {
        avatar.applyLipSyncResult({ visemes, mainViseme, mainVisemeWeight });
        return [avatar.mouthImage.src, avatar.mouthImage.style.display];
    };

    assert.deepEqual(apply(null, 0), ["", "none"]);
    assert.deepEqual(apply("A", 0.4, { A: 0.4 }), ["mouth-half", "block"]);
    assert.deepEqual(apply("O", 0.4, { O: 0.4 }), ["mouth-half", "block"]);
    assert.deepEqual(apply("A", 1, { A: 1 }), ["mouth-open", "block"]);
    assert.deepEqual(apply("O", 0.8, { O: 0.8 }), ["mouth-open", "block"]);
    assert.deepEqual(apply("U", 0.2, { U: 0.2 }), ["mouth-u", "block"]);
    assert.deepEqual(apply("I", 0.2, { I: 0.2 }), ["mouth-e", "block"]);
    assert.deepEqual(apply("E", 0.2, { E: 0.2 }), ["mouth-e", "block"]);
    assert.deepEqual(
        apply("A", 0.8, { A: 0.4, I: 0.4 }),
        ["mouth-half", "block"],
    );
    assert.deepEqual(apply("A", 0.8), ["mouth-half", "block"]);
});

test("image avatar initializes and uses an injected lip sync engine", async () => {
    const events = [];
    const result = {
        visemes: { A: 1, I: 0, U: 0, E: 0, O: 0 },
        mainViseme: "A",
        mainVisemeWeight: 1,
    };
    const engine = {
        async initialize() {
            events.push("initialize");
        },
        processAudioData(audio) {
            events.push(["process", audio]);
            return result;
        },
    };
    const avatar = new ImageAvatar({
        faceImage: { hidden: true },
        mouthImage: { hidden: true, src: "", style: { display: "none" } },
        faceImagePaths: {},
        lipsyncEngine: engine,
        rmsScale: 0.5,
        blinkEnabled: false,
    });
    avatar.preloadMouths = () => events.push("preload");
    avatar.applyLipSyncResult = (value) => events.push(["apply", value]);
    const aiavatar = {};

    await avatar.bind(aiavatar);
    const audio = { pcm: new Float32Array([0.1]), sampleRate: 16000 };
    aiavatar.onPlaybackAudio(audio);

    assert.equal(avatar.lipsyncEngine, engine);
    assert.deepEqual(events, [
        "preload",
        "initialize",
        ["process", { ...audio, gain: 0.5 }],
        ["apply", result],
    ]);
    assert.match(indexSource, /<script src="mfcc-lipsync\.js"><\/script>/);
    assert.match(indexSource, /lipsyncEngine: new MFCCLipSyncEngine\(\{/);
    assert.match(indexSource, /profileUrl: "profiles\/default-female\.json"/);
    assert.match(indexSource, /await avatar\.bind\(aiavatar\)/);
    assert.match(indexSource, /new MPTAvatar\(\{/);
});

test("VRM mouth weights use three-vrm expression preset names", () => {
    const VRMIdle = new Function(`${vrmIdleSource}; return VRMIdle;`)();
    const calls = [];
    const idle = Object.create(VRMIdle.prototype);
    idle._visemeTarget = {};
    idle._vrm = {
        meta: { metaVersion: "0" },
        expressionManager: {
            setValue(name, value) {
                calls.push({ name, value });
            },
        },
    };
    idle.applyVisemeWeights({
        A: 0.1,
        I: 0.2,
        U: 0.3,
        E: 0.4,
        O: 0.5,
    });
    for (const viseme of VRMIdle.VISEMES) {
        idle._setExpr(viseme, idle._visemeTarget[viseme]);
    }

    assert.deepEqual(calls.map(({ name }) => name), ["aa", "ih", "ou", "ee", "oh"]);
    assert.deepEqual(calls.map(({ value }) => value), [0.1, 0.2, 0.3, 0.4, 0.5]);
    assert.doesNotMatch(
        vrmIdleSource,
        /getExpression|VISEME_ALIASES|mixed\/fallback|_logVisemeMapping|console\./,
    );
});

test("AIAvatarClient sends decoded playback PCM with the current sample position", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    assert.doesNotMatch(clientSource, /onPlaybackAnalyze|createAnalyser|legacyAnalyser/);
    assert.doesNotMatch(clientSource, /LipSyncEngine|MFCCLipSyncEngine|outputType|processAudioData/);
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const pcm = sine(440);
    const decoded = {
        sampleRate: 16000,
        duration: pcm.length / 16000,
        getChannelData(channel) {
            assert.equal(channel, 0);
            return pcm;
        },
    };
    let source;
    const audioContext = {
        currentTime: 5,
        destination: {},
        decodeAudioData(_buffer, onSuccess) {
            onSuccess(decoded);
        },
        createBufferSource() {
            source = {
                connect(target) { this.target = target; },
                start() { this.started = true; },
            };
            return source;
        },
    };
    const callbacks = [];
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback) => {
        callbacks.push(callback);
        return callbacks.length;
    };

    try {
        const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
        client.audioContext = audioContext;
        const frames = [];
        client.onPlaybackAudio = (frame) => frames.push(frame);

        const playback = client.playAudioSync("AA==");
        assert.equal(source.started, true);
        audioContext.currentTime = 5.025;
        callbacks.shift()(25);

        assert.equal(frames.length, 1);
        assert.equal(frames[0].pcm, pcm);
        assert.equal(frames[0].sampleRate, 16000);
        assert.equal(frames[0].samplePosition, 400);
        assert.equal(frames[0].tSec, 5.025);
        assert.deepEqual(Object.keys(frames[0]), ["pcm", "sampleRate", "samplePosition", "tSec"]);

        audioContext.currentTime = 5 + decoded.duration;
        source.onended();
        await playback;
        assert.equal(frames.length, 1);
    } finally {
        if (previousRequestAnimationFrame === undefined) delete globalThis.requestAnimationFrame;
        else globalThis.requestAnimationFrame = previousRequestAnimationFrame;
    }
});

test("AIAvatarClient preserves 60Hz and 30Hz playback cadence with rounded frame times", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const pcm = sine(440, { sampleCount: 32000 });
    const decoded = {
        sampleRate: 16000,
        duration: pcm.length / 16000,
        getChannelData() { return pcm; },
    };
    const animationCallbacks = [];
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback) => {
        animationCallbacks.push(callback);
        return animationCallbacks.length;
    };

    const runCadence = async (playbackAudioHz, timestamps) => {
        animationCallbacks.length = 0;
        let source;
        const audioContext = {
            currentTime: 10,
            destination: {},
            decodeAudioData(_buffer, onSuccess) { onSuccess(decoded); },
            createBufferSource() {
                source = {
                    connect() {},
                    start() {},
                };
                return source;
            },
        };
        const client = new AIAvatarClient({
            webSocketUrl: "ws://example.test",
            playbackAudioHz,
        });
        client.audioContext = audioContext;
        const frames = [];
        client.onPlaybackAudio = (frame) => frames.push(frame);

        const playback = client.playAudioSync("AA==");
        for (const timestamp of timestamps) {
            assert.equal(animationCallbacks.length, 1);
            audioContext.currentTime = 10 + timestamp / 1000;
            animationCallbacks.shift()(timestamp);
        }
        source.onended();
        await playback;
        return frames.map(({ samplePosition, sampleRate }) => (
            Math.round(samplePosition / sampleRate * 1000)
        ));
    };

    try {
        const rounded60HzTimestamps = Array.from(
            { length: 61 },
            (_, index) => Math.round(index * 1000 / 60),
        );
        assert.deepEqual(
            await runCadence(60, rounded60HzTimestamps),
            rounded60HzTimestamps,
        );
        assert.deepEqual(
            await runCadence(30, rounded60HzTimestamps),
            rounded60HzTimestamps.filter((_, index) => index % 2 === 0),
        );

        const delayedTimestamps = [0, 17, 100, 117, 133, 150, 167, 183, 200];
        assert.deepEqual(
            await runCadence(30, delayedTimestamps),
            [0, 100, 133, 167, 200],
        );
    } finally {
        if (previousRequestAnimationFrame === undefined) delete globalThis.requestAnimationFrame;
        else globalThis.requestAnimationFrame = previousRequestAnimationFrame;
    }
});

test("AIAvatarClient keeps a replaced source from ending the current mouth", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const pcm = sine(440);
    const decoded = {
        sampleRate: 16000,
        duration: pcm.length / 16000,
        getChannelData() { return pcm; },
    };
    const sources = [];
    const audioContext = {
        currentTime: 0,
        destination: {},
        decodeAudioData(_buffer, onSuccess) { onSuccess(decoded); },
        createBufferSource() {
            const source = {
                connect() {},
                start() {},
                stop() { this.stopped = true; },
            };
            sources.push(source);
            return source;
        },
    };
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = () => 1;

    try {
        const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
        client.audioContext = audioContext;
        let playbackEndCount = 0;
        client.onPlaybackAudio = () => {};
        client.onPlaybackEnd = () => { playbackEndCount += 1; };

        const firstPlayback = client.playAudioSync("AA==");
        const secondPlayback = client.playAudioSync("AA==");
        assert.equal(sources.length, 2);
        assert.equal(sources[0].stopped, true);
        assert.equal(client.currentAudioSource, sources[1]);

        sources[0].onended();
        await firstPlayback;
        assert.equal(client.currentAudioSource, sources[1]);
        assert.equal(playbackEndCount, 1);

        sources[1].onended();
        await secondPlayback;
        assert.equal(client.currentAudioSource, null);
        assert.equal(playbackEndCount, 2);
    } finally {
        if (previousRequestAnimationFrame === undefined) delete globalThis.requestAnimationFrame;
        else globalThis.requestAnimationFrame = previousRequestAnimationFrame;
    }
});

test("AIAvatarClient stopListening discards queued audio chunks", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const pcm = sine(440);
    const decoded = {
        sampleRate: 16000,
        duration: pcm.length / 16000,
        getChannelData() { return pcm; },
    };
    const sources = [];
    const audioContext = {
        state: "running",
        currentTime: 0,
        destination: {},
        decodeAudioData(_buffer, onSuccess) { onSuccess(decoded); },
        createBufferSource() {
            const source = {
                connect() {},
                start() {},
                stop() { this.stopped = true; },
            };
            sources.push(source);
            return source;
        },
        async close() { this.state = "closed"; },
    };
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = () => 1;

    try {
        const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
        client.audioContext = audioContext;
        client.onPlaybackAudio = () => {};
        client.messageQueue.push({ audio_data: "AA==" }, { audio_data: "AA==" });

        const processing = client.processQueue();
        assert.equal(sources.length, 1);
        await client.stopListening("session");
        await processing;

        assert.equal(sources.length, 1);
        assert.equal(sources[0].stopped, true);
        assert.equal(client.messageQueue.length, 0);
        assert.equal(client.processingQueue, false);
    } finally {
        if (previousRequestAnimationFrame === undefined) delete globalThis.requestAnimationFrame;
        else globalThis.requestAnimationFrame = previousRequestAnimationFrame;
    }
});

test("AIAvatarClient old queue cannot clear a newer playback state", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
    const playbackResolvers = [];
    client.playAudioSync = () => new Promise((resolve) => playbackResolvers.push(resolve));

    client.messageQueue.push({ audio_data: "old" });
    const oldQueue = client.processQueue(0);
    assert.equal(client.isAudioPlaying, true);

    client.queueGeneration = 1;
    client.processingQueue = false;
    client.messageQueue.push({ audio_data: "new" });
    const newQueue = client.processQueue(1);
    assert.equal(client.isAudioPlaying, true);

    playbackResolvers[0]();
    await oldQueue;
    assert.equal(client.isAudioPlaying, true);

    playbackResolvers[1]();
    await newQueue;
    assert.equal(client.isAudioPlaying, false);
});

test("AIAvatarClient ignores a decode callback after stopListening", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const pcm = sine(440);
    const decoded = {
        sampleRate: 16000,
        duration: pcm.length / 16000,
        getChannelData() { return pcm; },
    };
    let decodeSuccess;
    let sourceCount = 0;
    const audioContext = {
        state: "running",
        currentTime: 0,
        destination: {},
        decodeAudioData(_buffer, onSuccess) { decodeSuccess = onSuccess; },
        createBufferSource() {
            sourceCount++;
            return { connect() {}, start() {}, stop() {} };
        },
        async close() { this.state = "closed"; },
    };

    const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
    client.audioContext = audioContext;
    const playback = client.playAudioSync("AA==");
    await client.stopListening("session");
    decodeSuccess(decoded);
    await playback;

    assert.equal(sourceCount, 0);
    assert.equal(client.currentAudioSource, null);
});

test("AIAvatarClient stopListening closes a connecting WebSocket", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const previousWebSocket = globalThis.WebSocket;
    globalThis.WebSocket = { CONNECTING: 0, OPEN: 1 };

    try {
        const socket = {
            readyState: globalThis.WebSocket.CONNECTING,
            onopen() {},
            onmessage() {},
            onerror() {},
            close() { this.closed = true; },
        };
        const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
        client.ws = socket;

        await client.stopListening("session");

        assert.equal(socket.closed, true);
        assert.equal(socket.onopen, null);
        assert.equal(socket.onmessage, null);
        assert.equal(socket.onerror, null);
        assert.equal(client.ws, null);
    } finally {
        if (previousWebSocket === undefined) delete globalThis.WebSocket;
        else globalThis.WebSocket = previousWebSocket;
    }
});

test("AIAvatarClient stopListening finalizes playback before closing audio", async () => {
    const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
    const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
    const pcm = sine(440);
    const decoded = {
        sampleRate: 16000,
        duration: pcm.length / 16000,
        getChannelData() { return pcm; },
    };
    let source;
    let contextClosed = false;
    const audioContext = {
        currentTime: 0,
        destination: {},
        decodeAudioData(_buffer, onSuccess) { onSuccess(decoded); },
        createBufferSource() {
            source = {
                connect() {},
                start() {},
                stop() { this.stopped = true; },
            };
            return source;
        },
        async close() { contextClosed = true; },
    };
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = () => 1;

    try {
        const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
        client.audioContext = audioContext;
        let playbackEndCount = 0;
        client.onPlaybackAudio = () => {};
        client.onPlaybackEnd = () => { playbackEndCount += 1; };

        const playback = client.playAudioSync("AA==");
        await client.stopListening("session");
        await playback;

        assert.equal(source.stopped, true);
        assert.equal(contextClosed, true);
        assert.equal(playbackEndCount, 1);
        assert.equal(client.currentAudioSource, null);
    } finally {
        if (previousRequestAnimationFrame === undefined) delete globalThis.requestAnimationFrame;
        else globalThis.requestAnimationFrame = previousRequestAnimationFrame;
    }
});
