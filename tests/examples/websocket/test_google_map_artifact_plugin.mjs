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
const pluginDataUrl = sourceDataUrl(
    (await readFile(new URL("google-map-artifact-plugin.js", artifactDirectory), "utf8"))
        .replace('"./artifact-parser.js"', `"${parserDataUrl}"`),
);
const { GoogleMapArtifactPlugin } = await import(pluginDataUrl);

class FakeElement {
    constructor() {
        this.attributes = {};
        this.listeners = new Map();
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

test("Google map plugin builds a Maps Embed view URL", () => {
    const plugin = new GoogleMapArtifactPlugin({
        apiKey: "test-api-key",
        language: "ja",
        region: "jp",
    });
    const command = plugin.parse({
        type: "map",
        latitude: "35.681236",
        longitude: "139.767125",
        zoom: "16",
        maptype: "satellite",
        title: "Tokyo Station",
    });
    const source = plugin.resolveSource(command);
    const url = new URL(source.url);

    assert.equal(command.latitude, 35.681236);
    assert.equal(command.longitude, 139.767125);
    assert.equal(command.zoom, 16);
    assert.equal(command.mapType, "satellite");
    assert.equal(source.provider, "google-maps-embed");
    assert.equal(url.origin, "https://www.google.com");
    assert.equal(url.pathname, "/maps/embed/v1/view");
    assert.equal(url.searchParams.get("key"), "test-api-key");
    assert.equal(url.searchParams.get("center"), "35.681236,139.767125");
    assert.equal(url.searchParams.get("zoom"), "16");
    assert.equal(url.searchParams.get("maptype"), "satellite");
    assert.equal(url.searchParams.get("language"), "ja");
    assert.equal(url.searchParams.get("region"), "JP");
    assert.deepEqual(plugin.getDefaults(), { aspect: "16:9" });
});

test("Google map plugin builds a Maps Embed place URL from a location name", () => {
    const plugin = new GoogleMapArtifactPlugin({ apiKey: "test-api-key" });
    const command = plugin.parse({
        type: "map",
        location: "  東京駅  ",
        zoom: "17",
    });
    const source = plugin.resolveSource(command);
    const url = new URL(source.url);

    assert.equal(command.mode, "place");
    assert.equal(command.location, "東京駅");
    assert.equal(command.latitude, null);
    assert.equal(command.longitude, null);
    assert.equal(url.pathname, "/maps/embed/v1/place");
    assert.equal(url.searchParams.get("q"), "東京駅");
    assert.equal(url.searchParams.get("zoom"), "17");
    assert.equal(url.searchParams.has("center"), false);
});

test("Google map plugin builds a Maps Embed directions URL", () => {
    const plugin = new GoogleMapArtifactPlugin({
        apiKey: "test-api-key",
        language: "ja",
        region: "JP",
    });
    const command = plugin.parse({
        type: "map",
        origin: "東京駅",
        destination: "東京タワー",
        "travel-mode": "walking",
    });
    const source = plugin.resolveSource(command);
    const url = new URL(source.url);

    assert.equal(command.mode, "directions");
    assert.equal(command.origin, "東京駅");
    assert.equal(command.destination, "東京タワー");
    assert.equal(command.travelMode, "walking");
    assert.equal(command.zoom, null);
    assert.equal(url.pathname, "/maps/embed/v1/directions");
    assert.equal(url.searchParams.get("origin"), "東京駅");
    assert.equal(url.searchParams.get("destination"), "東京タワー");
    assert.equal(url.searchParams.get("mode"), "walking");
    assert.equal(url.searchParams.has("zoom"), false);
});

test("Google map plugin validates configuration and artifact attributes", () => {
    assert.throws(
        () => new GoogleMapArtifactPlugin(),
        /API key is required/,
    );
    assert.throws(
        () => new GoogleMapArtifactPlugin({ apiKey: "has whitespace" }),
        /API key is invalid/,
    );

    const plugin = new GoogleMapArtifactPlugin({ apiKey: "test-api-key" });
    assert.throws(
        () => plugin.parse({ type: "map" }),
        /location, coordinates, or route are required/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", longitude: "139" }),
        /latitude and longitude must be specified together/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", location: "Tokyo", latitude: "35", longitude: "139" }),
        /either a location, coordinates, or route/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", origin: "東京駅" }),
        /origin and destination must be specified together/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", location: "東京駅", origin: "上野駅", destination: "東京駅" }),
        /either a location, coordinates, or route/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", location: "東京駅", "travel-mode": "walking" }),
        /travel-mode is available only for routes/,
    );
    assert.throws(
        () => plugin.parse({
            type: "map",
            origin: "東京駅",
            destination: "東京タワー",
            "travel-mode": "swimming",
        }),
        /Unsupported map travel mode/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", latitude: "91", longitude: "139" }),
        /latitude must be between -90 and 90/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", latitude: "35", longitude: "181" }),
        /longitude must be between -180 and 180/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", latitude: "35", longitude: "139", zoom: "1.5" }),
        /zoom must be an integer/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", latitude: "35", longitude: "139", maptype: "terrain" }),
        /Unsupported map type/,
    );
    assert.throws(
        () => plugin.parse({ type: "map", src: "https://example.com/map" }),
        /location, coordinates, or route instead/,
    );
});

test("Google map plugin gives only its iframe a Google-compatible referrer policy", () => {
    const plugin = new GoogleMapArtifactPlugin({ apiKey: "test-api-key" });
    const command = plugin.parse({
        type: "map",
        latitude: "35.681236",
        longitude: "139.767125",
    });
    const source = plugin.resolveSource(command);
    let loaded = false;
    let errorMessage = null;
    const session = plugin.mount({
        documentRoot: new FakeDocument(),
        command,
        source,
        view: {
            loaded: () => { loaded = true; },
            error: (message) => { errorMessage = message; },
        },
    });

    assert.equal(session.element.referrerPolicy, "strict-origin-when-cross-origin");
    assert.equal(session.element.attributes.allowfullscreen, "");
    assert.equal(Object.hasOwn(session.element.attributes, "sandbox"), false);
    assert.equal(session.element.title, "Google Map");

    session.element.dispatch("load");
    assert.equal(loaded, true);
    assert.equal(errorMessage, null);

    session.dispose();
    assert.equal(session.element.src, undefined);
});

test("Google map plugin uses the location as the default iframe title", () => {
    const plugin = new GoogleMapArtifactPlugin({ apiKey: "test-api-key" });
    const command = plugin.parse({ type: "map", location: "東京駅" });
    const session = plugin.mount({
        documentRoot: new FakeDocument(),
        command,
        source: plugin.resolveSource(command),
        view: { loaded() {}, error() {} },
    });

    assert.equal(session.element.title, "東京駅");
});

test("Google map plugin uses route endpoints as the default iframe title", () => {
    const plugin = new GoogleMapArtifactPlugin({ apiKey: "test-api-key" });
    const command = plugin.parse({
        type: "map",
        origin: "東京駅",
        destination: "東京タワー",
    });
    const session = plugin.mount({
        documentRoot: new FakeDocument(),
        command,
        source: plugin.resolveSource(command),
        view: { loaded() {}, error() {} },
    });

    assert.equal(session.element.title, "東京駅 → 東京タワー");
});

test("web viewers register the Google map plugin with a replaceable API key", async () => {
    for (const pageName of ["index.html", "3d.html"]) {
        const page = await readFile(new URL(`../${pageName}`, artifactDirectory), "utf8");
        assert.match(page, /import \{ GoogleMapArtifactPlugin \} from "\.\/artifact\/google-map-artifact-plugin\.js";/);
        assert.match(page, /const GOOGLE_MAPS_EMBED_API_KEY = "YOUR_GOOGLE_MAPS_EMBED_API_KEY";/);
        assert.match(page, /new GoogleMapArtifactPlugin\(\{[\s\S]*?apiKey: GOOGLE_MAPS_EMBED_API_KEY,/);
    }
});
