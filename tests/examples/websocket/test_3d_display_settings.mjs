import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const [displaySource, threeDSource] = await Promise.all([
    readFile(new URL("avatar3d/common/display-controller.js", htmlDirectory), "utf8"),
    readFile(new URL("3d.html", htmlDirectory), "utf8"),
]);

function element(tagName = "div") {
    const listeners = new Map();
    const classes = new Set();
    return {
        tagName, children: [], style: {}, textContent: "", checked: false,
        append(...nodes) { this.children.push(...nodes); },
        appendChild(node) { this.children.push(node); return node; },
        replaceChildren(...nodes) { this.children = nodes; },
        addEventListener(name, callback) { listeners.set(name, callback); },
        dispatch(name) { return listeners.get(name)?.(); },
        querySelector() { return null; },
        classList: {
            toggle(name, value) { if (value) classes.add(name); else classes.delete(name); },
            remove(name) { classes.delete(name); },
        },
    };
}

function descendants(node) {
    return [node, ...node.children.flatMap(descendants)];
}

function harness({ stored = new Map(), config = {} } = {}) {
    const elements = Object.fromEntries(
        ["messageBox", "requestTextForm", "micGlow", "bgLayer"].map(id => [id, element()]),
    );
    const document = {
        getElementById: id => elements[id] || null,
        querySelector: () => null,
        createElement: element,
    };
    const localStorage = {
        getItem: key => stored.get(key) ?? null,
        setItem: (key, value) => stored.set(key, value),
        removeItem: key => stored.delete(key),
    };
    const DisplayController = new Function("document", "localStorage", "fetch",
        `${displaySource.replace(/^export /gm, "")}; return DisplayController;`,
    )(document, localStorage, () => assert.fail("Display settings tests must not use the network"));
    const ui = {
        separatePartialTranscript: undefined,
        previewVisible: true,
        clearCount: 0,
        messageText: { textContent: "AI answer" },
        clearPartialTranscript() { this.previewVisible = false; this.clearCount++; },
    };
    const tabs = new Map();
    const resets = new Map();
    const actions = [];
    const controller = new DisplayController({
        aiavatar: { unmute: () => actions.push("unmute") }, ui,
        settingsHost: {
            addTab: (name, build) => tabs.set(name, build),
            onTabReset: (name, reset) => resets.set(name, reset),
        },
        blobStore: { get: async () => null, delete: async key => actions.push(["delete", key]) },
        config: {
            messageBoxOpacity: 80, characterName: "", userName: "", showUserText: true,
            showAIText: true, showMicGlow: true, showMenu: true, autoHide: false,
            messageSpeed: 70, ...config,
        },
        persistence: {
            enabled: true, restoreUserSettings: true, displayKey: "display", backgroundKey: "background",
        },
    });
    const panel = element();
    tabs.get("UI")(panel);
    function checkbox(label) {
        const rows = descendants(panel).filter(node =>
            node.tagName === "label" && node.children.some(child => child.textContent === label));
        assert.equal(rows.length, 1, `One settings toggle must be labeled ${label}`);
        return rows[0].children.find(node => node.tagName === "input");
    }
    return {
        controller, ui, stored, actions, checkbox,
        toggle(label, checked) {
            const input = checkbox(label);
            input.checked = checked;
            input.dispatch("change");
        },
        reset: () => resets.get("UI")(),
        dispose: () => controller.dispose(),
    };
}

test("3D separate live transcript defaults off for new and older saved settings", () => {
    assert.match(threeDSource, /separatePartialTranscript:\s*false/);
    for (const saved of [null, { showUserText: true, msgSpeed: 50 }]) {
        const stored = new Map(saved ? [["display", JSON.stringify(saved)]] : []);
        const h = harness({ stored });
        try {
            assert.equal(h.controller.defaults.separatePartialTranscript, false);
            assert.equal(h.controller.state.separatePartialTranscript, false);
            assert.equal(h.ui.separatePartialTranscript, false);
            assert.equal(h.checkbox("Live transcript below").checked, false);
            assert.equal(h.ui.previewVisible, false);
        } finally { h.dispose(); }
    }
});

test("UI toggles update the flag and immediately clear preview when it or user speech is hidden", () => {
    const h = harness();
    try {
        const initialClearCount = h.ui.clearCount;
        h.toggle("Live transcript below", true);
        assert.equal(h.controller.state.separatePartialTranscript, true);
        assert.equal(h.ui.separatePartialTranscript, true);
        assert.equal(h.ui.clearCount, initialClearCount);
        h.ui.previewVisible = true;
        h.toggle("Live transcript below", false);
        assert.equal(h.ui.separatePartialTranscript, false);
        assert.equal(h.ui.previewVisible, false);
        assert.equal(h.ui.messageText.textContent, "AI answer");

        h.toggle("Live transcript below", true);
        h.ui.previewVisible = true;
        h.toggle("Show user speech", false);
        assert.equal(h.controller.state.showUserText, false);
        assert.equal(h.ui.previewVisible, false);
        assert.equal(h.ui.separatePartialTranscript, true);
        assert.equal(h.ui.messageText.textContent, "AI answer");
        const saved = JSON.parse(h.stored.get("display"));
        assert.equal(saved.separatePartialTranscript, true);
        assert.equal(saved.showUserText, false);
    } finally { h.dispose(); }
});

test("separate transcript preference is saved and restored with the settings toggle", () => {
    const h = harness();
    let restored;
    try {
        h.toggle("Live transcript below", true);
        assert.equal(JSON.parse(h.stored.get("display")).separatePartialTranscript, true);
        restored = harness({ stored: h.stored });
        assert.equal(restored.controller.state.separatePartialTranscript, true);
        assert.equal(restored.ui.separatePartialTranscript, true);
        assert.equal(restored.checkbox("Live transcript below").checked, true);
    } finally {
        restored?.dispose();
        h.dispose();
    }
});

test("reset removes saved opt-in and restores the default display with no lingering preview", async () => {
    const h = harness({ stored: new Map([["display", JSON.stringify({ separatePartialTranscript: true })]]) });
    try {
        assert.equal(h.ui.separatePartialTranscript, true);
        h.ui.previewVisible = true;
        await h.reset();
        assert.equal(h.stored.has("display"), false);
        assert.equal(h.controller.state.separatePartialTranscript, false);
        assert.equal(h.ui.separatePartialTranscript, false);
        assert.equal(h.ui.previewVisible, false);
        assert.equal(h.ui.messageText.textContent, "AI answer");
    } finally { h.dispose(); }
});
