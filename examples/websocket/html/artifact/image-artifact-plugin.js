import {
    assertArtifactAttributes,
    parseArtifactDisplayAttributes,
    parseArtifactSourceAttributes,
} from "./artifact-parser.js";
import { parseArtifactHttpUrl } from "./artifact-url.js";

export class ImageArtifactPlugin {
    constructor({ type = "image" } = {}) {
        this.type = type;
        this.aliases = [];
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
        return { provider: "image", url: parseArtifactHttpUrl(command.src).href };
    }

    getDefaults() {
        return { aspect: "auto" };
    }

    mount({ documentRoot, command, source, defaults, view }) {
        const image = documentRoot.createElement("img");
        image.className = "artifact-media";
        image.src = source.url;
        image.alt = command.alt || command.title || "";
        image.decoding = "async";
        image.draggable = false;
        image.style.objectFit = "contain";
        image.style.height = defaults.aspect === "auto" ? "auto" : "100%";
        image.addEventListener("load", () => {
            view.setIntrinsicAspect(image.naturalWidth, image.naturalHeight);
            view.loaded();
        }, { once: true });
        image.addEventListener("error", () => view.error(), { once: true });
        return { element: image, dispose() {} };
    }
}
