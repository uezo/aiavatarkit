import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactDirectory = new URL("../../../examples/websocket/html/artifact/", import.meta.url);

function sourceDataUrl(source) {
    return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}

const registrySource = await readFile(new URL("artifact-registry.js", artifactDirectory), "utf8");
const parserSource = await readFile(new URL("artifact-parser.js", artifactDirectory), "utf8");
const urlSource = await readFile(new URL("artifact-url.js", artifactDirectory), "utf8");
const registryDataUrl = sourceDataUrl(registrySource);
const parserDataUrl = sourceDataUrl(parserSource);
const urlDataUrl = sourceDataUrl(urlSource);
const imagePluginDataUrl = sourceDataUrl(
    (await readFile(new URL("image-artifact-plugin.js", artifactDirectory), "utf8"))
        .replace('"./artifact-parser.js"', `"${parserDataUrl}"`)
        .replace('"./artifact-url.js"', `"${urlDataUrl}"`),
);
const presentationPluginDataUrl = sourceDataUrl(
    (await readFile(new URL("presentation-artifact-plugin.js", artifactDirectory), "utf8"))
        .replace('"./artifact-parser.js"', `"${parserDataUrl}"`)
        .replace('"./artifact-url.js"', `"${urlDataUrl}"`),
);
const videoPluginDataUrl = sourceDataUrl(
    (await readFile(new URL("video-artifact-plugin.js", artifactDirectory), "utf8"))
        .replace('"./artifact-parser.js"', `"${parserDataUrl}"`)
        .replace('"./artifact-url.js"', `"${urlDataUrl}"`),
);
const controllerDataUrl = sourceDataUrl(
    (await readFile(new URL("artifact-controller.js", artifactDirectory), "utf8"))
        .replace('"./artifact-parser.js"', `"${parserDataUrl}"`)
        .replace('"./artifact-registry.js"', `"${registryDataUrl}"`),
);

const { ArtifactRegistry } = await import(registryDataUrl);
const { parseArtifactControlTags, parseArtifactTags } = await import(parserDataUrl);
const { ArtifactController } = await import(controllerDataUrl);
const { ImageArtifactPlugin } = await import(imagePluginDataUrl);
const { PresentationArtifactPlugin } = await import(presentationPluginDataUrl);
const { VideoArtifactPlugin } = await import(videoPluginDataUrl);

class FakeClassList {
    constructor() {
        this.values = new Set();
    }

    add(...names) {
        for (const name of names) this.values.add(name);
    }

    remove(...names) {
        for (const name of names) this.values.delete(name);
    }

    contains(name) {
        return this.values.has(name);
    }
}

class FakeElement {
    constructor(tagName) {
        this.tagName = tagName.toUpperCase();
        this.children = [];
        this.dataset = {};
        this.attributes = {};
        this.classList = new FakeClassList();
        this.listeners = new Map();
        this.style = { setProperty: (name, value) => { this.style[name] = value; } };
        this.hidden = false;
        this.parentNode = null;
    }

    append(...children) {
        this.children.push(...children);
        for (const child of children) child.parentNode = this;
    }

    appendChild(child) {
        this.append(child);
        return child;
    }

    replaceChildren(...children) {
        for (const child of this.children) child.parentNode = null;
        this.children = [];
        this.append(...children);
    }

    contains(child) {
        return this.children.includes(child);
    }

    setAttribute(name, value) {
        this.attributes[name] = value;
    }

    addEventListener(name, listener) {
        this.listeners.set(name, listener);
    }

    dispatch(name) {
        this.listeners.get(name)?.({ target: this });
    }

    remove() {
        if (this.parentNode) {
            this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
        }
        this.parentNode = null;
    }
}

class FakeDocument {
    constructor() {
        this.documentElement = new FakeElement("html");
        this.body = new FakeElement("body");
    }

    createElement(tagName) {
        const element = new FakeElement(tagName);
        if (tagName.toLowerCase() === "iframe") element.contentWindow = {};
        return element;
    }
}

class FakePresentationDriver {
    static provider = "fake-slides";

    static supports(url) {
        return url.hostname === "slides.example.com";
    }

    static resolveUrl(source, slide = null) {
        const url = new URL(source.href);
        if (slide !== null) url.searchParams.set("slide", slide);
        return url.href;
    }

    constructor({ iframe, url, onSlideChange }) {
        this.iframe = iframe;
        this.url = url;
        this.onSlideChange = onSlideChange;
        this.ready = false;
        this.currentSlide = Number(new URL(url).searchParams.get("slide") || 1);
        this.totalSlides = 20;
        this.disposed = false;
    }

    initialize() {
        this.ready = true;
    }

    navigate(url) {
        if (!this.ready) return false;
        this.url = url;
        this.currentSlide = Number(new URL(url).searchParams.get("slide") || 1);
        this.iframe.src = url;
        return this.currentSlide;
    }

    navigateBy(offset) {
        if (!this.ready) return false;
        this.currentSlide = Math.max(1, this.currentSlide + offset);
        const url = new URL(this.url);
        url.searchParams.set("slide", this.currentSlide);
        this.url = url.href;
        this.iframe.src = this.url;
        this.onSlideChange?.(this.currentSlide);
        return { slide: this.currentSlide, url: this.url };
    }

    dispose() {
        this.disposed = true;
        this.ready = false;
    }
}

const videoDrivers = [];
class FakeVideoDriver {
    static provider = "fake-video";

    static supports(url) {
        return url.hostname === "video.example.com";
    }

    static resolveUrl(source) {
        return new URL(source.href).href;
    }

    constructor(options) {
        this.options = options;
        this.state = "idle";
        this.disposed = false;
        options.iframe.src = options.url;
        videoDrivers.push(this);
    }

    async initialize() {
        this.state = "ready";
        this.options.onEvent({ type: "ready", provider: this.constructor.provider });
        return true;
    }

    dispose() {
        this.disposed = true;
    }
}

function plugins() {
    return [
        new ImageArtifactPlugin({ type: "image" }),
        new ImageArtifactPlugin({ type: "chart" }),
        new PresentationArtifactPlugin({ drivers: [FakePresentationDriver] }),
        new VideoArtifactPlugin({ drivers: [FakeVideoDriver], autoplay: true }),
    ];
}

test("registry owns type aliases and rejects duplicate registrations", () => {
    const registry = new ArtifactRegistry(plugins());
    assert.equal(registry.get("slides").type, "presentation");
    assert.equal(registry.get("VIDEO").type, "video");
    assert.throws(() => registry.register(new ImageArtifactPlugin({ type: "image" })), /already registered/);
});

test("registered plugins parse text and structured artifact commands", () => {
    const registry = new ArtifactRegistry(plugins());
    const { commands, errors } = parseArtifactTags([
        '<artifact type="image" src="https://example.com/result.png" />',
        '<artifact type="chart" src="https://example.com/chart.svg" aspect="4:3" />',
        '<artifact type="slides" src="https://slides.example.com/deck" slide="4" />',
        '<artifact type="presentation" offset="+2" />',
        '<artifact type="video" src="https://video.example.com/watch/1" autoplay-delay="1.5" />',
    ].join(""), { registry });

    assert.equal(errors.length, 0);
    assert.deepEqual(commands.map((command) => command.type), [
        "image", "chart", "presentation", "presentation", "video",
    ]);
    assert.equal(commands[2].slide, "4");
    assert.equal(commands[3].action, "move");
    assert.equal(commands[4].autoplayDelaySeconds, 1.5);

    const structured = parseArtifactControlTags([{
        name: "artifact",
        attributes: { type: "video", src: "https://video.example.com/watch/2", "autoplay-delay": 3 },
    }], { registry });
    assert.equal(structured.errors.length, 0);
    assert.equal(structured.commands[0].autoplayDelaySeconds, 3);

    const invalid = parseArtifactTags(
        '<artifact type="image" src="https://example.com/x.png" autoplay-delay="2" />',
        { registry },
    );
    assert.equal(invalid.commands.length, 0);
    assert.match(invalid.errors[0].error.message, /Unsupported artifact attribute/);
});

test("controller delegates image loading and presentation updates to registered plugins", () => {
    const documentRoot = new FakeDocument();
    const visibility = [];
    const controller = new ArtifactController({
        documentRoot,
        plugins: plugins(),
        onVisibilityChange: (active) => visibility.push(active),
    });

    controller.apply({
        action: "show",
        type: "image",
        src: "https://example.com/image.png",
        size: null,
        aspect: null,
        alt: "",
        title: "",
    });
    const imageSurface = controller.root.children[0];
    const image = imageSurface.children[0];
    image.naturalWidth = 1200;
    image.naturalHeight = 800;
    image.dispatch("load");
    assert.equal(imageSurface.classList.contains("loaded"), true);
    assert.equal(imageSurface.style["--artifact-ratio"], "1.5");

    controller.apply({
        action: "show",
        type: "presentation",
        src: "https://slides.example.com/deck",
        slide: "2",
        size: null,
        aspect: null,
        alt: "",
        title: "",
    });
    const presentationSurface = controller.root.children[0];
    const iframe = presentationSurface.children[0];
    iframe.dispatch("load");
    assert.equal(controller.currentSlide, 2);
    assert.equal(controller.totalSlides, 20);

    controller.apply({
        action: "show",
        type: "presentation",
        src: "https://slides.example.com/deck",
        slide: "7",
        size: null,
        aspect: null,
        alt: "",
        title: "",
    });
    assert.equal(controller.root.children[0], presentationSurface);
    assert.equal(controller.current.slide, "7");

    controller.apply({ action: "move", type: "presentation", offset: 2 });
    assert.equal(controller.current.slide, "9");
    controller.clear();
    assert.deepEqual(visibility, [true, false]);
});

test("controller handles a tagged chunk once and uses final as fallback", () => {
    const documentRoot = new FakeDocument();
    const controller = new ArtifactController({ documentRoot, plugins: plugins() });
    const tag = '<artifact type="image" src="https://example.com/image.png" />';

    controller.handleResponse({ type: "start", text: null });
    controller.handleResponse({
        type: "chunk",
        text: tag,
        control_tags: [{
            name: "artifact",
            attributes: { type: "image", src: "https://example.com/image.png" },
        }],
    });
    const chunkSurface = controller.root.children[0];
    controller.handleResponse({ type: "final", text: tag });
    assert.equal(controller.root.children[0], chunkSurface);

    controller.handleResponse({ type: "start", text: null });
    controller.handleResponse({
        type: "final",
        text: '<artifact type="chart" src="https://example.com/chart.svg" />',
    });
    assert.equal(controller.current.type, "chart");
    assert.notEqual(controller.root.children[0], chunkSurface);
});

test("video plugin passes fixed autoplay delay and disposes its driver", async () => {
    videoDrivers.length = 0;
    const documentRoot = new FakeDocument();
    const controller = new ArtifactController({ documentRoot, plugins: plugins() });
    controller.apply({
        action: "show",
        type: "video",
        src: "https://video.example.com/watch/1",
        autoplayDelaySeconds: 2.5,
        size: null,
        aspect: null,
        alt: "",
        title: "",
    });

    const surface = controller.root.children[0];
    const iframe = surface.children[0];
    assert.equal(videoDrivers[0].options.autoplay, true);
    assert.equal(videoDrivers[0].options.autoplayDelaySeconds, 2.5);
    iframe.dispatch("load");
    await Promise.resolve();
    assert.equal(surface.classList.contains("loaded"), true);

    controller.clear();
    assert.equal(videoDrivers[0].disposed, true);
});
