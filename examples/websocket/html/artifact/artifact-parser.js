const COMMON_DISPLAY_ATTRIBUTES = new Set(["type", "src", "href", "size", "aspect", "alt", "title"]);
const SIZE_PRESETS = new Set(["small", "medium", "large", "full"]);
const ASPECT_PRESETS = new Set(["auto", "16:9", "4:3", "3:2", "1:1", "9:16"]);
const STRUCTURED_ATTRIBUTE_TYPES = new Set(["string", "number", "boolean"]);

export function assertArtifactAttributes(attributes, additionalAttributes = []) {
    const allowed = new Set([...COMMON_DISPLAY_ATTRIBUTES, ...additionalAttributes]);
    for (const name of Object.keys(attributes)) {
        if (!allowed.has(name)) throw new Error(`Unsupported artifact attribute: ${name}`);
    }
}

export function parseArtifactSourceAttributes(attributes) {
    if (attributes.src !== undefined && attributes.href !== undefined) {
        throw new Error("Use either src or href, not both");
    }
    return attributes.src ?? attributes.href ?? null;
}

export function parseArtifactDisplayAttributes(attributes) {
    const size = (attributes.size || "").toLowerCase() || null;
    if (size && !SIZE_PRESETS.has(size)) throw new Error(`Unsupported artifact size: ${size}`);

    const aspect = (attributes.aspect || "").toLowerCase() || null;
    if (aspect && !ASPECT_PRESETS.has(aspect)) throw new Error(`Unsupported artifact aspect: ${aspect}`);

    return {
        size,
        aspect,
        alt: attributes.alt || "",
        title: attributes.title || "",
    };
}

function parseCommand(attributes, registry) {
    if (attributes.action !== undefined) {
        const action = attributes.action.toLowerCase();
        if (action !== "clear") throw new Error(`Unsupported artifact action: ${attributes.action}`);
        if (Object.keys(attributes).some((name) => name !== "action")) {
            throw new Error("The clear artifact action cannot have additional attributes");
        }
        return { action: "clear" };
    }

    const requestedType = (attributes.type || "").toLowerCase();
    const plugin = registry?.get(requestedType);
    if (!plugin) throw new Error(`Unsupported artifact type: ${attributes.type || "(missing)"}`);
    return plugin.parse(attributes, { requestedType });
}

function normalizeStructuredAttributes(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("Artifact attributes must be an object");
    }
    const attributes = {};
    for (const [rawName, rawValue] of Object.entries(value)) {
        const name = rawName.toLowerCase();
        if (Object.prototype.hasOwnProperty.call(attributes, name)) {
            throw new Error(`Duplicate artifact attribute: ${name}`);
        }
        if (!STRUCTURED_ATTRIBUTE_TYPES.has(typeof rawValue)) {
            throw new Error(`Artifact attribute must be a scalar value: ${name}`);
        }
        attributes[name] = String(rawValue);
    }
    return attributes;
}

export function parseArtifactControlTags(controlTags, { registry = null } = {}) {
    const commands = [];
    const errors = [];
    if (!Array.isArray(controlTags)) return { commands, errors };

    for (const controlTag of controlTags) {
        if (String(controlTag?.name || "").toLowerCase() !== "artifact") continue;
        try {
            commands.push(parseCommand(normalizeStructuredAttributes(controlTag.attributes), registry));
        } catch (error) {
            errors.push({ tag: controlTag, error });
        }
    }
    return { commands, errors };
}
