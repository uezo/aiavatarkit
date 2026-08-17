import {
    assertArtifactAttributes,
    parseArtifactDisplayAttributes,
    parseArtifactSourceAttributes,
} from "./artifact-parser.js";
import { normalizeArtifactSlide, parseArtifactHttpUrl } from "./artifact-url.js";

export class PresentationArtifactPlugin {
    constructor({ drivers = [] } = {}) {
        if (!Array.isArray(drivers) || !drivers.length) {
            throw new TypeError("PresentationArtifactPlugin requires at least one driver");
        }
        this.type = "presentation";
        this.aliases = ["slide", "slides"];
        this.drivers = [...drivers];
    }

    parse(attributes) {
        assertArtifactAttributes(attributes, ["slide", "offset"]);

        if (attributes.offset !== undefined) {
            if (Object.keys(attributes).some((name) => name !== "type" && name !== "offset")) {
                throw new Error("Relative presentation navigation accepts only type and offset");
            }
            if (!/^[+-][1-9]\d*$/.test(attributes.offset)) {
                throw new Error("Artifact offset must be a signed non-zero integer");
            }
            return { action: "move", type: this.type, offset: Number(attributes.offset) };
        }

        const slide = normalizeArtifactSlide(attributes.slide);
        const src = parseArtifactSourceAttributes(attributes);
        if (!src) {
            if (slide !== null) {
                if (Object.keys(attributes).some((name) => name !== "type" && name !== "slide")) {
                    throw new Error("Current presentation navigation accepts only type and slide");
                }
                return { action: "go", type: this.type, slide };
            }
            throw new Error("Artifact src is required");
        }

        return {
            action: "show",
            type: this.type,
            src,
            slide,
            ...parseArtifactDisplayAttributes(attributes),
        };
    }

    findDriver(url) {
        return this.drivers.find((Driver) => Driver.supports(url)) || null;
    }

    resolveSource(command) {
        const url = parseArtifactHttpUrl(command.src);
        const Driver = this.findDriver(url);
        if (!Driver) throw new Error("Presentation URL does not match a registered provider");
        const source = {
            provider: Driver.provider,
            url: Driver.resolveUrl(url, normalizeArtifactSlide(command.slide)),
        };
        Object.defineProperty(source, "Driver", { value: Driver });
        return source;
    }

    getDefaults() {
        return { aspect: "16:9" };
    }

    mount({ documentRoot, windowRoot, command, source, view }) {
        const iframe = documentRoot.createElement("iframe");
        iframe.className = "artifact-media";
        iframe.src = source.url;
        iframe.title = command.title || "Presentation";
        iframe.loading = "eager";
        iframe.allow = "fullscreen; encrypted-media";
        iframe.style.background = "#fff";
        iframe.setAttribute("allowfullscreen", "");
        iframe.setAttribute(
            "sandbox",
            "allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox",
        );

        let currentUrl = source.url;
        const driver = new source.Driver({
            iframe,
            url: source.url,
            windowRoot,
            onSlideChange: (slide) => {
                try {
                    currentUrl = source.Driver.resolveUrl(new URL(currentUrl), String(slide));
                } catch {
                    // Keep the last provider URL if it cannot represent manual navigation.
                }
                view.updateCurrent({ slide: String(slide), url: currentUrl });
            },
        });

        iframe.addEventListener("load", () => {
            view.loaded();
            try {
                driver.initialize();
            } catch (error) {
                view.error(error.message);
            }
        }, { once: true });

        return {
            element: iframe,
            getState: () => ({
                slide: driver.currentSlide,
                totalSlides: driver.totalSlides,
            }),
            update: ({ source: nextSource }) => {
                if (nextSource.provider !== source.provider) return false;
                const slide = driver.navigate(nextSource.url);
                if (slide === false) return false;
                currentUrl = nextSource.url;
                return { slide: String(slide), url: currentUrl };
            },
            execute: (nextCommand) => {
                if (!driver.ready) return false;
                if (nextCommand.action === "move") {
                    const result = driver.navigateBy(nextCommand.offset);
                    if (result === false) return false;
                    if (result.url) currentUrl = result.url;
                    return { slide: String(result.slide), url: currentUrl };
                }
                if (nextCommand.action === "go") {
                    const nextSource = this.resolveSource({
                        action: "show",
                        type: this.type,
                        src: currentUrl,
                        slide: nextCommand.slide,
                    });
                    const slide = driver.navigate(nextSource.url);
                    if (slide === false) return false;
                    currentUrl = nextSource.url;
                    return { slide: String(slide), url: currentUrl };
                }
                return false;
            },
            dispose: () => driver.dispose(),
        };
    }
}
