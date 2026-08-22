import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactDirectory = new URL("../../../examples/websocket/html/artifact/", import.meta.url);

function sourceDataUrl(source) {
    return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}

const parserDataUrl = sourceDataUrl(
    await readFile(new URL("artifact-parser.js", artifactDirectory), "utf8"),
);
const urlDataUrl = sourceDataUrl(
    await readFile(new URL("artifact-url.js", artifactDirectory), "utf8"),
);
const pluginDataUrl = sourceDataUrl(
    (await readFile(new URL("webapp-artifact-plugin.js", artifactDirectory), "utf8"))
        .replace('"./artifact-parser.js"', `"${parserDataUrl}"`)
        .replace('"./artifact-url.js"', `"${urlDataUrl}"`),
);
const { WebAppArtifactPlugin } = await import(pluginDataUrl);

class FakeWindow {
    constructor() {
        this.listeners = new Map();
    }

    addEventListener(name, listener) {
        this.listeners.set(name, listener);
    }

    removeEventListener(name, listener) {
        if (this.listeners.get(name) === listener) this.listeners.delete(name);
    }

    dispatchMessage(source, data) {
        this.listeners.get("message")?.({ source, data });
    }
}

class FakeElement {
    constructor() {
        this.attributes = {};
        this.listeners = new Map();
        this.contentWindow = {};
    }

    setAttribute(name, value) {
        this.attributes[name] = value;
    }

    removeAttribute(name) {
        delete this.attributes[name];
        if (name === "src") delete this.src;
    }

    addEventListener(name, listener) {
        this.listeners.set(name, listener);
    }

    dispatch(name) {
        this.listeners.get(name)?.();
    }
}

class FakeDocument {
    createElement() {
        return new FakeElement();
    }
}

function mount(plugin) {
    const documentRoot = new FakeDocument();
    const windowRoot = new FakeWindow();
    let loaded = false;
    const session = plugin.mount({
        documentRoot,
        windowRoot,
        command: { title: "Lunch picker", alt: "" },
        source: { url: "https://example.com/app" },
        view: { loaded: () => { loaded = true; } },
    });
    return { windowRoot, session, wasLoaded: () => loaded };
}

test("web app plugin parses URLs and mounts a sandboxed iframe", () => {
    const plugin = new WebAppArtifactPlugin();
    const command = plugin.parse({
        type: "webapp",
        src: "https://example.com/app",
        title: "Lunch picker",
    });
    assert.equal(command.type, "webapp");
    assert.equal(plugin.resolveSource(command).url, "https://example.com/app");
    assert.deepEqual(plugin.getDefaults(), { aspect: "16:9" });

    const { session, wasLoaded } = mount(plugin);
    assert.equal(session.element.attributes.sandbox, "allow-scripts");
    assert.equal(session.element.referrerPolicy, "no-referrer");
    session.element.dispatch("load");
    assert.equal(wasLoaded(), true);
});

test("web app plugin forwards only valid messages from its iframe", () => {
    const invocations = [];
    const plugin = new WebAppArtifactPlugin({
        onInvoke: (message) => invocations.push(message),
        invokeCooldownMs: 0,
    });
    const { windowRoot, session } = mount(plugin);
    const iframeWindow = session.element.contentWindow;

    windowRoot.dispatchMessage({}, {
        type: "aiavatar.webapp.invoke",
        version: 1,
        text: "Ignored source",
    });
    windowRoot.dispatchMessage(iframeWindow, {
        type: "aiavatar.webapp.invoke",
        version: 2,
        text: "Ignored version",
    });
    windowRoot.dispatchMessage(iframeWindow, {
        type: "aiavatar.webapp.invoke",
        version: 1,
        text: "  Find lunch in Shinjuku.  ",
        imageDataUrl: "data:image/png;base64,YQ==",
    });

    assert.deepEqual(invocations, [{
        text: "Find lunch in Shinjuku.",
        imageDataUrl: "data:image/png;base64,YQ==",
    }]);

    session.dispose();
    assert.equal(windowRoot.listeners.has("message"), false);
    windowRoot.dispatchMessage(iframeWindow, {
        type: "aiavatar.webapp.invoke",
        version: 1,
        text: "Ignored after dispose",
    });
    assert.equal(invocations.length, 1);
});

test("web app plugin rejects oversized or unexpected invoke payloads", () => {
    const invocations = [];
    const plugin = new WebAppArtifactPlugin({
        onInvoke: (message) => invocations.push(message),
        maxTextLength: 4,
        maxImageBytes: 1,
        invokeCooldownMs: 0,
    });
    const { windowRoot, session } = mount(plugin);
    const iframeWindow = session.element.contentWindow;
    const send = (data) => windowRoot.dispatchMessage(iframeWindow, data);

    send({ type: "aiavatar.webapp.invoke", version: 1, text: "12345" });
    send({ type: "aiavatar.webapp.invoke", version: 1, text: "okay", extra: true });
    send({
        type: "aiavatar.webapp.invoke",
        version: 1,
        text: "okay",
        imageDataUrl: "https://example.com/image.png",
    });
    send({
        type: "aiavatar.webapp.invoke",
        version: 1,
        text: "okay",
        imageDataUrl: "data:image/png;base64,YWI=",
    });

    assert.deepEqual(invocations, []);
});
