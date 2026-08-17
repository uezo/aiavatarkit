import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const driverDirectory = new URL("../../../examples/websocket/html/artifact/", import.meta.url);

function sourceDataUrl(source) {
    return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}

const videoDataUrl = sourceDataUrl(
    await readFile(new URL("video-driver.js", driverDirectory), "utf8"),
);
const loaderSource = await readFile(new URL("youtube-iframe-api-loader.js", driverDirectory), "utf8");
const loaderDataUrl = sourceDataUrl(loaderSource);
const youtubeDataUrl = sourceDataUrl(
    (await readFile(new URL("youtube-driver.js", driverDirectory), "utf8"))
        .replace('"./video-driver.js"', `"${videoDataUrl}"`)
        .replace('"./youtube-iframe-api-loader.js"', `"${loaderDataUrl}"`),
);

const { loadYouTubeIframeApi } = await import(loaderDataUrl);
const { YouTubeDriver } = await import(youtubeDataUrl);

function fakeIframe() {
    return {
        allow: "fullscreen",
        attributes: {},
        setAttribute(name, value) {
            this.attributes[name] = value;
        },
    };
}

test("YouTube driver validates sources and schedules fixed-delay autoplay once", async () => {
    const resolved = new URL(YouTubeDriver.resolveUrl("https://youtu.be/dQw4w9WgXcQ?t=1m2s"));
    assert.equal(resolved.pathname, "/embed/dQw4w9WgXcQ");
    assert.equal(resolved.searchParams.get("start"), "62");
    assert.throws(
        () => YouTubeDriver.resolveUrl("https://youtube.com.example/watch?v=dQw4w9WgXcQ"),
        /Invalid YouTube host/,
    );

    const players = [];
    class FakePlayer {
        constructor(iframe, options) {
            this.options = options;
            this.playCalls = 0;
            this.destroyCalls = 0;
            players.push(this);
            queueMicrotask(() => options.events.onReady({ target: this }));
        }

        playVideo() {
            this.playCalls += 1;
            this.options.events.onStateChange({ data: 1 });
        }

        destroy() {
            this.destroyCalls += 1;
        }
    }

    const windowRoot = {
        location: { origin: "http://localhost:8000" },
        setTimeout,
        clearTimeout,
    };
    const events = [];
    const driver = new YouTubeDriver({
        iframe: fakeIframe(),
        url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        windowRoot,
        documentRoot: {},
        apiLoader: async () => ({ Player: FakePlayer }),
        autoplay: true,
        autoplayDelaySeconds: 0.03,
        onEvent: (event) => events.push(event.type),
    });

    await driver.initialize();
    assert.equal(players[0].playCalls, 0);
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(players[0].playCalls, 0);
    await new Promise((resolve) => setTimeout(resolve, 35));
    assert.equal(players[0].playCalls, 1);
    assert.equal(events.filter((type) => type === "autoplayrequested").length, 1);

    driver.dispose();
    assert.equal(players[0].destroyCalls, 1);
});

test("YouTube API loader shares one request and preserves an existing callback", async () => {
    let existingCallbackCalls = 0;
    let appendCalls = 0;
    const listeners = new Map();
    const script = {
        addEventListener(type, listener) {
            listeners.set(type, listener);
        },
        removeEventListener(type) {
            listeners.delete(type);
        },
    };
    const windowRoot = {
        setTimeout,
        clearTimeout,
        setInterval,
        clearInterval,
        onYouTubeIframeAPIReady() {
            existingCallbackCalls += 1;
        },
    };
    const originalCallback = windowRoot.onYouTubeIframeAPIReady;
    const documentRoot = {
        querySelector() {
            return null;
        },
        createElement() {
            return script;
        },
        head: {
            appendChild() {
                appendCalls += 1;
            },
        },
    };

    const first = loadYouTubeIframeApi({ windowRoot, documentRoot, timeoutMs: 500 });
    const second = loadYouTubeIframeApi({ windowRoot, documentRoot, timeoutMs: 500 });
    assert.equal(first, second);
    assert.equal(appendCalls, 1);

    windowRoot.YT = { Player: function Player() {} };
    windowRoot.onYouTubeIframeAPIReady();
    assert.equal(await first, windowRoot.YT);
    assert.equal(existingCallbackCalls, 1);
    assert.equal(windowRoot.onYouTubeIframeAPIReady, originalCallback);
});
