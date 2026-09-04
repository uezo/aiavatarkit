import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const clientSource = await readFile(new URL("aiavatar.js", htmlDirectory), "utf8");
const uiSource = await readFile(new URL("ui.js", htmlDirectory), "utf8");
const threeDSource = await readFile(new URL("3d.html", htmlDirectory), "utf8");
const AIAvatarClient = new Function(`${clientSource}; return AIAvatarClient;`)();
const AvatarUI = new Function(`${uiSource}; return AvatarUI;`)();

function eventTarget(properties = {}) {
    const listeners = new Map();
    return {
        ...properties,
        addEventListener(name, listener) {
            listeners.set(name, listener);
        },
        dispatch(name, event = {}) {
            listeners.get(name)?.(event);
        },
    };
}

test("3D volume popup provides separate speaker and microphone sliders", () => {
    assert.match(threeDSource, /id="volumeSlider"[^>]*max="100"/s);
    assert.match(threeDSource, /id="microphoneVolumeSlider"[^>]*max="200"/s);
    assert.match(threeDSource, />🔊<.*>🎙️</s);
    assert.match(threeDSource, /aria-label="Speaker volume"/);
    assert.match(threeDSource, /aria-label="Microphone volume"/);
    assert.match(threeDSource, /initialMicrophoneVolume: 1\.0/);
});

test("volume controls update speaker and microphone volumes independently", () => {
    const previousDocument = globalThis.document;
    const speakerSlider = eventTarget({ value: "100" });
    const microphoneSlider = eventTarget({ value: "100" });
    const speakerValue = { textContent: "100" };
    const microphoneValue = { textContent: "100" };
    const volumeButton = eventTarget({ textContent: "VOL" });
    const volumePopup = { classList: { toggle() {}, remove() {} } };
    const volumeControl = { contains: () => false };
    const calls = [];
    globalThis.document = { addEventListener() {} };

    try {
        const ui = Object.create(AvatarUI.prototype);
        Object.assign(ui, {
            aiavatar: {
                setVolume: (value) => calls.push(["speaker", value]),
                setMicrophoneVolume: (value) => calls.push(["microphone", value]),
            },
            volumeBtn: volumeButton,
            volumePopup,
            volumeSlider: speakerSlider,
            volumeValue: speakerValue,
            microphoneVolumeSlider: microphoneSlider,
            microphoneVolumeValue: microphoneValue,
            volumeControl,
        });

        ui._setupVolumeControl();
        speakerSlider.value = "35";
        speakerSlider.dispatch("input");
        microphoneSlider.value = "150";
        microphoneSlider.dispatch("input");

        assert.deepEqual(calls, [["speaker", 0.35], ["microphone", 1.5]]);
        assert.equal(speakerValue.textContent, "35");
        assert.equal(microphoneValue.textContent, "150");
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test("microphone volume scales and clips outgoing PCM", () => {
    const client = new AIAvatarClient({ webSocketUrl: "ws://example.test" });
    client.setMicrophoneVolume(1.5);

    assert.equal(client.microphoneVolume, 1.5);
    const pcm = client.float32To16BitPCMBuffer(
        Float32Array.from([0.5, -0.5, 0.9, -0.9]),
        client.microphoneVolume,
    );
    const view = new DataView(pcm);
    assert.equal(view.getInt16(0, true), 24575);
    assert.equal(view.getInt16(2, true), -24576);
    assert.equal(view.getInt16(4, true), 32767);
    assert.equal(view.getInt16(6, true), -32768);

    client.setMicrophoneVolume(3);
    assert.equal(client.microphoneVolume, 2);
    client.setMicrophoneVolume(-1);
    assert.equal(client.microphoneVolume, 0);
});

test("shared UI remains compatible with pages without a microphone slider", () => {
    const previousDocument = globalThis.document;
    globalThis.document = { addEventListener() {} };
    try {
        const ui = Object.create(AvatarUI.prototype);
        Object.assign(ui, {
            aiavatar: { setVolume() {} },
            volumeBtn: eventTarget(),
            volumePopup: { classList: { toggle() {}, remove() {} } },
            volumeSlider: eventTarget({ value: "100" }),
            volumeValue: { textContent: "100" },
            microphoneVolumeSlider: null,
            microphoneVolumeValue: null,
            volumeControl: { contains: () => false },
        });
        assert.doesNotThrow(() => ui._setupVolumeControl());
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test("backlog audio mutes the microphone even when barge-in is enabled", () => {
    const ui = Object.create(AvatarUI.prototype);
    Object.assign(ui, {
        aiavatar: {
            isAudioPlaying: false,
            isBacklogAudioPlaying: true,
        },
        interruptEnabled: true,
        isServerProcessing: false,
        isBargeInBlocked: false,
    });
    ui._setupMicrophoneMute();
    assert.equal(ui.aiavatar.isMicrophoneMuted(), true);
});
