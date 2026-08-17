import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const driverDirectory = new URL("../../../examples/websocket/html/artifact/", import.meta.url);

function sourceDataUrl(source) {
    return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}

const baseSource = await readFile(new URL("presentation-driver.js", driverDirectory), "utf8");
const baseDataUrl = sourceDataUrl(baseSource);
const docswellDataUrl = sourceDataUrl(
    (await readFile(new URL("docswell-driver.js", driverDirectory), "utf8"))
        .replace('"./presentation-driver.js"', `"${baseDataUrl}"`),
);
const speakerDeckDataUrl = sourceDataUrl(
    (await readFile(new URL("speakerdeck-driver.js", driverDirectory), "utf8"))
        .replace('"./presentation-driver.js"', `"${baseDataUrl}"`),
);

const { PresentationDriver } = await import(baseDataUrl);
const { DocswellDriver } = await import(docswellDataUrl);
const { SpeakerDeckDriver } = await import(speakerDeckDataUrl);

class FakeWindow {
    constructor() {
        this.listeners = new Map();
    }

    addEventListener(name, listener) {
        this.listeners.set(name, listener);
    }

    removeEventListener(name) {
        this.listeners.delete(name);
    }

    dispatchMessage(event) {
        this.listeners.get("message")?.(event);
    }
}

function fakeIframe() {
    return {
        src: "",
        contentWindow: {
            messages: [],
            postMessage(message, origin) {
                this.messages.push({ message, origin });
            },
        },
    };
}

test("presentation driver remains abstract", () => {
    assert.throws(() => new PresentationDriver(), /abstract/);
    assert.throws(() => PresentationDriver.provider, /must be implemented/);
});

test("presentation providers validate and normalize their URLs", () => {
    assert.equal(
        SpeakerDeckDriver.resolveUrl(new URL("https://speakerdeck.com/player/deck_1"), "7"),
        "https://speakerdeck.com/player/deck_1?slide=7",
    );
    assert.equal(
        DocswellDriver.resolveUrl(
            new URL("https://www.docswell.com/s/harinezumi/KDMGQW-2026-05-13-130255"),
            "10",
        ),
        "https://www.docswell.com/slide/KDMGQW/embed#p10",
    );
    assert.throws(
        () => SpeakerDeckDriver.resolveUrl(new URL("https://speakerdeck.com/not-a-player/deck")),
        /player embed URL/,
    );
    assert.throws(
        () => DocswellDriver.resolveUrl(new URL("https://www.docswell.com/not-a-viewer")),
        /viewer or embed URLs/,
    );
});

test("Docswell driver initializes, navigates, tracks manual moves, and disposes", () => {
    const iframe = fakeIframe();
    const windowRoot = new FakeWindow();
    const changes = [];
    const driver = new DocswellDriver({
        iframe,
        url: "https://www.docswell.com/slide/DECK_1/embed#p1",
        windowRoot,
        onSlideChange: (slide) => changes.push(slide),
    });

    assert.equal(driver.initialize(), true);
    const initialization = iframe.contentWindow.messages[0];
    windowRoot.dispatchMessage({
        source: iframe.contentWindow,
        origin: "https://www.docswell.com",
        data: { type: "docswell:initialized", id: initialization.message.id, total: 24 },
    });
    assert.equal(driver.ready, true);
    assert.equal(driver.totalSlides, 24);

    assert.deepEqual(driver.navigateBy(2), { slide: 3 });
    assert.equal(changes.at(-1), 3);
    windowRoot.dispatchMessage({
        source: iframe.contentWindow,
        origin: "https://www.docswell.com",
        data: { type: "docswell:move", id: initialization.message.id, index: 6 },
    });
    assert.equal(driver.currentSlide, 7);
    assert.equal(changes.at(-1), 7);

    driver.dispose();
    assert.equal(windowRoot.listeners.has("message"), false);
});

test("Speaker Deck driver updates the existing iframe URL", () => {
    const iframe = fakeIframe();
    const driver = new SpeakerDeckDriver({
        iframe,
        url: "https://speakerdeck.com/player/DECK_1?slide=2",
    });
    assert.equal(driver.initialize(), true);
    assert.deepEqual(driver.navigateBy(3), {
        slide: 5,
        url: "https://speakerdeck.com/player/DECK_1?slide=5",
    });
    assert.equal(iframe.src, "https://speakerdeck.com/player/DECK_1?slide=5");
});
