import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const vrmDirectory = new URL(
    "../../../examples/websocket/html/avatar3d/models/vrm/",
    import.meta.url,
);

function sourceDataUrl(source) {
    return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}

const threeDataUrl = sourceDataUrl(`
    export class Vector3 {
        constructor(x = 0, y = 0, z = 0) {
            this.set(x, y, z);
        }
        set(x, y, z) {
            this.x = x;
            this.y = y;
            this.z = z;
            return this;
        }
        clone() {
            return new Vector3(this.x, this.y, this.z);
        }
        add(other) {
            this.x += other.x;
            this.y += other.y;
            this.z += other.z;
            return this;
        }
        multiplyScalar(value) {
            this.x *= value;
            this.y *= value;
            this.z *= value;
            return this;
        }
        length() {
            return Math.hypot(this.x, this.y, this.z);
        }
        project(camera) {
            if (!camera.matrixWorldIsCurrent) throw new Error("projected with a stale camera matrix");
            this.x -= camera.projectOffsetX || 0;
            return this;
        }
    }

    export class Box3 {
        constructor() {
            throw new Error("the control surface must not use cached model bounds");
        }
    }
`);
const loaderDataUrl = sourceDataUrl("export class GLTFLoader {}");
const controlsDataUrl = sourceDataUrl("export class OrbitControls {}");
const vrmDataUrl = sourceDataUrl("export class VRMLoaderPlugin {}; export const VRMUtils = {};");
const animationDataUrl = sourceDataUrl(`
    export class VRMAnimationLoaderPlugin {}
    export function createVRMAnimationClip() {}
`);
const settingsDataUrl = sourceDataUrl("export function installVrmSettings() {}");
const actualSettingsDataUrl = sourceDataUrl(
    await readFile(new URL("vrm-settings.js", vrmDirectory), "utf8"),
);

const adapterDataUrl = sourceDataUrl(
    (await readFile(new URL("vrm-adapter.js", vrmDirectory), "utf8"))
        .replace('"three"', `"${threeDataUrl}"`)
        .replace('"three/addons/loaders/GLTFLoader.js"', `"${loaderDataUrl}"`)
        .replace('"three/addons/controls/OrbitControls.js"', `"${controlsDataUrl}"`)
        .replace('"@pixiv/three-vrm"', `"${vrmDataUrl}"`)
        .replace('"@pixiv/three-vrm-animation"', `"${animationDataUrl}"`)
        .replace('"./vrm-settings.js"', `"${settingsDataUrl}"`),
);
const { VrmAdapter } = await import(adapterDataUrl);
const { installVrmSettings } = await import(actualSettingsDataUrl);

function bone(position) {
    return {
        getWorldPosition(point) {
            return point.set(...position);
        },
    };
}

function createControlSurfaceAdapter({
    bones = {
        hips: bone([0.1, 0, 0]),
        head: bone([0, 0.5, 0]),
        leftFoot: bone([-0.1, -0.5, 0]),
        rightFoot: bone([0.1, -0.5, 0]),
    },
} = {}) {
    let modelMatrixUpdates = 0;
    let cameraMatrixUpdates = 0;
    const rawBoneRequests = [];
    const adapter = Object.create(VrmAdapter.prototype);
    adapter.currentModel = {
        scene: {
            updateMatrixWorld(force) {
                assert.equal(force, true);
                modelMatrixUpdates += 1;
            },
        },
        humanoid: {
            getRawBoneNode(name) {
                rawBoneRequests.push(name);
                return bones[name] || null;
            },
            getNormalizedBoneNode() {
                throw new Error("the rendered hit area must not use normalized proxy bones");
            },
        },
    };
    adapter.viewCamera = {
        matrixWorldIsCurrent: false,
        projectOffsetX: 0,
        position: {
            set(x) {
                adapter.viewCamera.projectOffsetX = x * 0.2;
                adapter.viewCamera.matrixWorldIsCurrent = false;
            },
        },
        updateMatrixWorld(force) {
            assert.equal(force, true);
            this.matrixWorldIsCurrent = true;
            cameraMatrixUpdates += 1;
        },
    };
    adapter.controls = {
        maxDistance: 5,
        target: { copy() {} },
        update() {},
    };
    adapter.controlSurface = { hidden: true, style: {} };
    return {
        adapter,
        rawBoneRequests,
        modelMatrixUpdates: () => modelMatrixUpdates,
        cameraMatrixUpdates: () => cameraMatrixUpdates,
    };
}

function withViewport(callback) {
    const previousWindow = globalThis.window;
    globalThis.window = { innerWidth: 1000, innerHeight: 800 };
    try {
        callback();
    } finally {
        if (previousWindow === undefined) delete globalThis.window;
        else globalThis.window = previousWindow;
    }
}

test("control surface projects raw rendered bones with the current camera matrix", () => {
    withViewport(() => {
        const {
            adapter,
            rawBoneRequests,
            modelMatrixUpdates,
            cameraMatrixUpdates,
        } = createControlSurfaceAdapter();

        adapter.updateControlSurface();

        assert.equal(modelMatrixUpdates(), 0);
        assert.equal(cameraMatrixUpdates(), 1);
        assert.deepEqual(rawBoneRequests, ["hips", "head", "leftFoot", "rightFoot"]);
        assert.equal(adapter.controlSurface.hidden, false);
        assert.equal(
            adapter.controlSurface.style.clipPath,
            "inset(128px 380px 184px 480px)",
        );
    });
});

test("render loop updates the surface after the VRM pose update", () => {
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    const events = [];
    globalThis.requestAnimationFrame = () => 17;

    try {
        const adapter = Object.create(VrmAdapter.prototype);
        adapter.renderRequest = null;
        adapter.controls = { update: () => events.push("controls") };
        adapter.clock = { getDelta: () => 0.016 };
        adapter.idle = { update: () => events.push("idle") };
        adapter.updateControlSurface = () => events.push("surface");
        adapter.renderer = { render: () => events.push("render") };
        adapter.scene = {};
        adapter.viewCamera = {};

        adapter.start();

        assert.equal(adapter.renderRequest, 17);
        assert.deepEqual(events, ["controls", "idle", "surface", "render"]);
    } finally {
        if (previousRequestAnimationFrame === undefined) {
            delete globalThis.requestAnimationFrame;
        } else {
            globalThis.requestAnimationFrame = previousRequestAnimationFrame;
        }
    }
});

test("render loop measures the raw pose produced by the same idle tick", () => {
    const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
    const previousWindow = globalThis.window;
    const headPosition = [0, 0.1, 0];
    const footPosition = [0, -0.1, 0];
    globalThis.requestAnimationFrame = () => 19;
    globalThis.window = { innerWidth: 1000, innerHeight: 800 };

    try {
        const { adapter } = createControlSurfaceAdapter({
            bones: {
                hips: bone([0.1, 0, 0]),
                head: bone(headPosition),
                leftFoot: bone(footPosition),
            },
        });
        adapter.renderRequest = null;
        adapter.clock = { getDelta: () => 0.016 };
        adapter.idle = {
            update() {
                headPosition[1] = 0.5;
                footPosition[1] = -0.5;
            },
        };
        adapter.renderer = { render() {} };
        adapter.scene = {};

        adapter.start();

        assert.equal(adapter.renderRequest, 19);
        assert.equal(
            adapter.controlSurface.style.clipPath,
            "inset(128px 380px 184px 480px)",
        );
    } finally {
        if (previousRequestAnimationFrame === undefined) {
            delete globalThis.requestAnimationFrame;
        } else {
            globalThis.requestAnimationFrame = previousRequestAnimationFrame;
        }
        if (previousWindow === undefined) delete globalThis.window;
        else globalThis.window = previousWindow;
    }
});

test("camera state changes are projected by the next rendered frame", () => {
    withViewport(() => {
        const { adapter, cameraMatrixUpdates } = createControlSurfaceAdapter();

        const restored = adapter.applyCameraState({
            px: 1,
            py: 2,
            pz: 3,
            tx: 0,
            ty: 1,
            tz: 0,
        });

        assert.equal(restored, true);
        assert.equal(cameraMatrixUpdates(), 0);
        assert.equal(adapter.controlSurface.hidden, true);

        adapter.updateControlSurface();

        assert.equal(cameraMatrixUpdates(), 1);
        assert.equal(adapter.controlSurface.hidden, false);
        assert.equal(
            adapter.controlSurface.style.clipPath,
            "inset(128px 480px 184px 380px)",
        );
    });
});

test("one rendered foot is sufficient and missing vertical anchors hide the surface", () => {
    withViewport(() => {
        const bones = {
            hips: bone([0, 0, 0]),
            head: bone([0, 0.5, 0]),
            leftFoot: bone([0, -0.5, 0]),
        };
        const { adapter } = createControlSurfaceAdapter({ bones });

        adapter.updateControlSurface();
        assert.equal(adapter.controlSurface.hidden, false);

        delete bones.leftFoot;
        adapter.updateControlSurface();
        assert.equal(adapter.controlSurface.hidden, true);
    });
});

test("default artifact camera uses the cached raw skeleton frame instead of model bounds", () => {
    const { adapter, modelMatrixUpdates } = createControlSurfaceAdapter();
    let applied = null;
    adapter.modelSkeletonFrame = adapter.captureRawSkeletonFrame();
    adapter.applyCameraState = (state, options) => {
        applied = { state, options };
        return true;
    };

    const result = adapter.applyDefaultArtifactCamera(5);

    assert.equal(result, true);
    assert.equal(modelMatrixUpdates(), 1);
    assert.ok(Object.values(applied.state).every(Number.isFinite));
    assert.ok(applied.options.maxDistance >= 5);
});

test("reset view restores the default camera for the current mode without unloading the model", () => {
    const model = {};
    const adapter = Object.create(VrmAdapter.prototype);
    adapter.currentModel = model;
    adapter.config = {
        camera: {
            maxDistance: 5,
        },
    };
    adapter.persistence = { cameraKey: "camera" };
    adapter.cameraSaveTimer = null;
    adapter.controlSurface = { hidden: false };
    adapter.viewCamera = {
        zoom: 4,
        projectionUpdates: 0,
        updateProjectionMatrix() {
            this.projectionUpdates += 1;
        },
    };
    const calls = [];
    adapter.applyDefaultModelCamera = (options) => {
        calls.push(["normal", options]);
        return true;
    };
    adapter.applyDefaultArtifactCamera = (maxDistance) => {
        calls.push(["artifact", maxDistance]);
        return true;
    };
    adapter.saveCameraState = (key) => calls.push(["save", key]);

    adapter.artifactMode = false;
    assert.equal(adapter.resetView(), true);
    assert.equal(adapter.currentModel, model);
    assert.equal(adapter.viewCamera.zoom, 1);
    assert.equal(adapter.viewCamera.projectionUpdates, 1);
    assert.deepEqual(calls, [
        ["normal", { resetControls: true }],
        ["save", "camera"],
    ]);
    assert.equal(adapter.controlSurface.hidden, true);

    calls.length = 0;
    adapter.artifactMode = true;
    adapter.normalMaxDistance = 8;
    assert.equal(adapter.resetView(), true);
    assert.equal(adapter.currentModel, model);
    assert.deepEqual(calls, [
        ["artifact", 15],
        ["save", "camera_artifact"],
    ]);
});

test("VRM adapter uses the injected lip sync engine object", async () => {
    const events = [];
    const engine = {
        async initialize() {
            events.push("initialize");
        },
        processAudioData(audio) {
            events.push(["processAudioData", audio]);
            return {
                visemes: { A: 0.2, I: 0.1, U: 0, E: 0, O: 0 },
                mainViseme: "A",
                mainVisemeWeight: 0.75,
            };
        },
    };
    const adapter = Object.create(VrmAdapter.prototype);
    adapter.config = {
        expression: { neutralName: "neutral", defaultDurationSeconds: 2 },
        lipsync: { engine, usePhonemeBlend: false },
    };
    adapter.idle = {
        applyExpression() {},
        applyVisemeWeights(weights) {
            events.push(["weights", weights]);
        },
        clearVisemes() {},
    };
    const aiavatar = {};

    await adapter.bind(aiavatar);
    const audio = { pcm: new Float32Array([0.1]), sampleRate: 16000 };
    aiavatar.onPlaybackAudio(audio);

    assert.equal(adapter.lipsyncEngine, engine);
    assert.deepEqual(events, [
        "initialize",
        ["processAudioData", audio],
        ["weights", { A: 0.75, I: 0, U: 0, E: 0, O: 0 }],
    ]);

    adapter.config.lipsync.usePhonemeBlend = true;
    assert.deepEqual(adapter.lipSyncWeights(engine.processAudioData(audio)), {
        A: 0.2,
        I: 0.1,
        U: 0,
        E: 0,
        O: 0,
    });

    adapter.config.lipsync.maxVisemeWeight = 0.5;
    assert.deepEqual(adapter.lipSyncWeights(engine.processAudioData(audio)), {
        A: 0.1,
        I: 0.05,
        U: 0,
        E: 0,
        O: 0,
    });
    adapter.config.lipsync.usePhonemeBlend = false;
    assert.deepEqual(adapter.lipSyncWeights(engine.processAudioData(audio)), {
        A: 0.375,
        I: 0,
        U: 0,
        E: 0,
        O: 0,
    });

    adapter.config.lipsync.maxVisemeWeight = 2;
    assert.deepEqual(adapter.lipSyncWeights(engine.processAudioData(audio)), {
        A: 0.75,
        I: 0,
        U: 0,
        E: 0,
        O: 0,
    });
});

test("VRM adapter previews mouth shapes using the configured lip sync maximum", () => {
    const calls = [];
    const adapter = Object.create(VrmAdapter.prototype);
    adapter.config = {
        lipsync: { usePhonemeBlend: false, maxVisemeWeight: 0.8 },
    };
    adapter.idle = {
        applyVisemeWeights(weights) {
            calls.push(weights);
        },
    };

    adapter.previewViseme("U");
    adapter.previewViseme();

    assert.deepEqual(calls, [
        { A: 0, I: 0, U: 0.8, E: 0, O: 0 },
        { A: 0, I: 0, U: 0, E: 0, O: 0 },
    ]);
});

test("VRM adapter persists and restores the lip sync maximum", () => {
    const previousLocalStorage = globalThis.localStorage;
    const values = new Map();
    globalThis.localStorage = {
        getItem(key) {
            return values.get(key) ?? null;
        },
        setItem(key, value) {
            values.set(key, value);
        },
        removeItem(key) {
            values.delete(key);
        },
    };

    try {
        const adapter = Object.create(VrmAdapter.prototype);
        adapter.config = { lipsync: { maxVisemeWeight: 1 } };
        adapter.persistence = {
            enabled: true,
            restoreUserSettings: true,
            lipSyncKey: "vrm_lipsync",
        };
        adapter.defaultMaxVisemeWeight = 1;
        adapter.idle = { clearVisemes() {} };

        assert.equal(adapter.setMaxVisemeWeight(0.6), 0.6);
        assert.deepEqual(JSON.parse(values.get("vrm_lipsync")), { maxVisemeWeight: 0.6 });

        adapter.config.lipsync.maxVisemeWeight = 0;
        adapter.loadLipSyncSettings();
        assert.equal(adapter.getMaxVisemeWeight(), 0.6);

        assert.equal(adapter.setMaxVisemeWeight(3), 1);
        adapter.resetLipSyncSettings();
        assert.equal(adapter.getMaxVisemeWeight(), 1);
        assert.equal(values.has("vrm_lipsync"), false);
    } finally {
        if (previousLocalStorage === undefined) delete globalThis.localStorage;
        else globalThis.localStorage = previousLocalStorage;
    }
});

test("VRM adapter preserves an explicit zero face duration", async () => {
    const calls = [];
    const adapter = Object.create(VrmAdapter.prototype);
    adapter.config = {
        expression: { neutralName: "neutral", defaultDurationSeconds: 2 },
        lipsync: {
            usePhonemeBlend: false,
            engine: { async initialize() {} },
        },
    };
    adapter.idle = {
        applyExpression(name, duration) {
            calls.push([name, duration]);
        },
        clearVisemes() {},
    };
    const aiavatar = {};

    await adapter.bind(aiavatar);
    aiavatar.updateFace("neutral", 0);

    assert.deepEqual(calls, [["neutral", 0]]);
});

test("Load settings expose view, mouth preview, and lip sync maximum controls", () => {
    const previousDocument = globalThis.document;

    class FakeElement {
        constructor(tagName) {
            this.tagName = tagName;
            this.children = [];
            this.listeners = {};
            this.style = {};
            this.textContent = "";
        }
        append(...children) {
            this.children.push(...children);
        }
        appendChild(child) {
            this.children.push(child);
            return child;
        }
        replaceChildren(...children) {
            this.children = children;
        }
        addEventListener(type, listener) {
            this.listeners[type] = listener;
        }
        click() {
            this.listeners.click?.();
        }
    }

    globalThis.document = {
        createElement: (tagName) => new FakeElement(tagName),
    };

    try {
        const panels = {};
        let resetCount = 0;
        const previewedVisemes = [];
        const maximums = [];
        const adapter = {
            settingsHost: {
                addTab(name, render) {
                    const panel = new FakeElement("div");
                    panels[name] = panel;
                    render(panel);
                },
                onTabReset() {},
            },
            animationNames: [],
            lightDefinitions: [],
            lighting: {},
            resetView() {
                resetCount += 1;
            },
            getMaxVisemeWeight() {
                return 1;
            },
            setMaxVisemeWeight(value) {
                maximums.push(value);
                return value;
            },
            previewViseme(viseme) {
                previewedVisemes.push(viseme ?? null);
            },
        };

        installVrmSettings(adapter);
        const resetButton = panels.Load.children.find(
            (child) => child.textContent === "Reset view",
        );
        assert.ok(resetButton);
        resetButton.click();
        assert.equal(resetCount, 1);

        const descendants = (element) => [
            element,
            ...element.children.flatMap(descendants),
        ];
        const controls = descendants(panels.Load);
        const maxWeightSlider = controls.find(
            (element) => element.type === "range" && element.max === 1,
        );
        assert.ok(maxWeightSlider);
        assert.equal(maxWeightSlider.min, 0);
        assert.equal(maxWeightSlider.step, 0.1);
        maxWeightSlider.value = "0.7";
        maxWeightSlider.listeners.input();
        assert.deepEqual(maximums, [0.7]);

        controls.find((element) => element.textContent === "U").click();
        controls.find((element) => element.textContent === "Close").click();
        assert.deepEqual(previewedVisemes, ["U", null]);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});
