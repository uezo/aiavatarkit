import {
    assertArtifactAttributes,
    parseArtifactDisplayAttributes,
    parseArtifactSourceAttributes,
} from "./artifact-parser.js";
import { parseArtifactHttpUrl } from "./artifact-url.js";

const MAX_AUTOPLAY_DELAY_SECONDS = 3600;

function parseAutoplayDelay(value) {
    if (value === undefined || value === "") return 0;
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0 || seconds > MAX_AUTOPLAY_DELAY_SECONDS) {
        throw new RangeError(`Video autoplay-delay must be between 0 and ${MAX_AUTOPLAY_DELAY_SECONDS} seconds`);
    }
    return seconds;
}

export class VideoArtifactPlugin {
    constructor({ drivers = [], autoplay = true, defaultAutoplayDelaySeconds = 0 } = {}) {
        if (!Array.isArray(drivers) || !drivers.length) {
            throw new TypeError("VideoArtifactPlugin requires at least one driver");
        }
        this.type = "video";
        this.aliases = [];
        this.drivers = [...drivers];
        this.autoplay = Boolean(autoplay);
        this.defaultAutoplayDelaySeconds = parseAutoplayDelay(defaultAutoplayDelaySeconds);
    }

    parse(attributes) {
        assertArtifactAttributes(attributes, ["autoplay-delay"]);
        const src = parseArtifactSourceAttributes(attributes);
        if (!src) throw new Error("Artifact src is required");
        return {
            action: "show",
            type: this.type,
            src,
            autoplayDelaySeconds: attributes["autoplay-delay"] === undefined
                ? this.defaultAutoplayDelaySeconds
                : parseAutoplayDelay(attributes["autoplay-delay"]),
            ...parseArtifactDisplayAttributes(attributes),
        };
    }

    resolveSource(command) {
        const url = parseArtifactHttpUrl(command.src);
        const Driver = this.drivers.find((candidate) => candidate.supports(url));
        if (!Driver) throw new Error("Video URL does not match a registered provider");
        const source = { provider: Driver.provider, url: Driver.resolveUrl(url) };
        Object.defineProperty(source, "Driver", { value: Driver });
        return source;
    }

    getDefaults() {
        return { aspect: "16:9" };
    }

    mount({ documentRoot, windowRoot, command, source, view }) {
        const iframe = documentRoot.createElement("iframe");
        iframe.className = "artifact-media";
        iframe.title = command.title || "Video";
        iframe.loading = "eager";
        iframe.style.background = "#000";
        iframe.setAttribute(
            "sandbox",
            "allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox",
        );

        const driver = new source.Driver({
            iframe,
            url: source.url,
            windowRoot,
            documentRoot,
            autoplay: this.autoplay,
            autoplayDelaySeconds: command.autoplayDelaySeconds,
            onEvent: (event) => {
                if (event.type === "ready") view.loaded();
                else if (event.type === "error") view.error(event.error?.message);
                else if (event.type === "autoplayblocked") {
                    console.warn("Video autoplay was blocked; use the embedded player controls");
                }
            },
        });

        iframe.addEventListener("load", () => {
            driver.initialize().catch((error) => view.error(error.message));
        }, { once: true });

        return {
            element: iframe,
            getState: () => ({ state: driver.state }),
            dispose: () => driver.dispose(),
        };
    }
}
