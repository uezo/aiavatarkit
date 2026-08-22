import {
    assertArtifactAttributes,
    parseArtifactDisplayAttributes,
    parseArtifactSourceAttributes,
} from "./artifact-parser.js";
import { parseArtifactHttpUrl } from "./artifact-url.js";

const INVOKE_MESSAGE_TYPE = "aiavatar.webapp.invoke";
const INVOKE_MESSAGE_VERSION = 1;
const DEFAULT_MAX_TEXT_LENGTH = 10_000;
const DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const DEFAULT_INVOKE_COOLDOWN_MS = 1000;
const MESSAGE_FIELDS = new Set(["type", "version", "text", "imageDataUrl"]);
const IMAGE_DATA_URL_PATTERN = /^data:image\/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})$/i;

function isObject(value) {
    return value && typeof value === "object" && !Array.isArray(value);
}

function normalizePositiveInteger(value, name) {
    if (!Number.isSafeInteger(value) || value < 1) {
        throw new TypeError(`${name} must be a positive integer`);
    }
    return value;
}

function normalizeCooldown(value) {
    if (!Number.isSafeInteger(value) || value < 0) {
        throw new TypeError("invokeCooldownMs must be a non-negative integer");
    }
    return value;
}

function parseImageDataUrl(value, maxImageBytes) {
    if (value === undefined || value === null) return null;
    if (typeof value !== "string") return false;

    const match = IMAGE_DATA_URL_PATTERN.exec(value);
    if (!match || match[1].length % 4 !== 0) return false;

    const padding = match[1].endsWith("==") ? 2 : match[1].endsWith("=") ? 1 : 0;
    const decodedBytes = (match[1].length * 3 / 4) - padding;
    return decodedBytes <= maxImageBytes ? value : false;
}

export class WebAppArtifactPlugin {
    constructor({
        type = "webapp",
        onInvoke = () => false,
        maxTextLength = DEFAULT_MAX_TEXT_LENGTH,
        maxImageBytes = DEFAULT_MAX_IMAGE_BYTES,
        invokeCooldownMs = DEFAULT_INVOKE_COOLDOWN_MS,
    } = {}) {
        if (typeof onInvoke !== "function") throw new TypeError("onInvoke must be a function");
        this.type = type;
        this.aliases = [];
        this.onInvoke = onInvoke;
        this.maxTextLength = normalizePositiveInteger(maxTextLength, "maxTextLength");
        this.maxImageBytes = normalizePositiveInteger(maxImageBytes, "maxImageBytes");
        this.invokeCooldownMs = normalizeCooldown(invokeCooldownMs);
    }

    parse(attributes) {
        assertArtifactAttributes(attributes);
        const src = parseArtifactSourceAttributes(attributes);
        if (!src) throw new Error("Artifact src is required");
        return {
            action: "show",
            type: this.type,
            src,
            ...parseArtifactDisplayAttributes(attributes),
        };
    }

    resolveSource(command) {
        return {
            provider: "webapp",
            url: parseArtifactHttpUrl(command.src).href,
        };
    }

    getDefaults() {
        return { aspect: "16:9" };
    }

    parseInvokeMessage(value) {
        if (!isObject(value)) return null;
        if (Object.keys(value).some((name) => !MESSAGE_FIELDS.has(name))) return null;
        if (value.type !== INVOKE_MESSAGE_TYPE || value.version !== INVOKE_MESSAGE_VERSION) return null;
        if (typeof value.text !== "string") return null;

        const text = value.text.trim();
        if (!text || text.length > this.maxTextLength) return null;

        const imageDataUrl = parseImageDataUrl(value.imageDataUrl, this.maxImageBytes);
        if (imageDataUrl === false) return null;
        return { text, imageDataUrl };
    }

    mount({ documentRoot, windowRoot, command, source, view }) {
        const iframe = documentRoot.createElement("iframe");
        iframe.className = "artifact-media";
        iframe.src = source.url;
        iframe.title = command.title || command.alt || "Web app";
        iframe.loading = "eager";
        iframe.referrerPolicy = "no-referrer";
        iframe.setAttribute("sandbox", "allow-scripts");
        iframe.addEventListener("load", () => view.loaded(), { once: true });

        let disposed = false;
        let invokeInFlight = false;
        let nextInvokeAt = 0;
        const handleMessage = (event) => {
            if (disposed || event.source !== iframe.contentWindow) return;
            const message = this.parseInvokeMessage(event.data);
            if (!message || invokeInFlight || Date.now() < nextInvokeAt) return;

            invokeInFlight = true;
            nextInvokeAt = Date.now() + this.invokeCooldownMs;
            try {
                const result = this.onInvoke(message);
                if (result?.then) {
                    Promise.resolve(result)
                        .catch((error) => console.warn("Could not invoke from web app artifact:", error))
                        .finally(() => { invokeInFlight = false; });
                } else {
                    invokeInFlight = false;
                }
            } catch (error) {
                invokeInFlight = false;
                console.warn("Could not invoke from web app artifact:", error);
            }
        };
        windowRoot.addEventListener("message", handleMessage);

        return {
            element: iframe,
            dispose: () => {
                disposed = true;
                windowRoot.removeEventListener("message", handleMessage);
                iframe.removeAttribute("src");
            },
        };
    }
}
