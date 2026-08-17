const TYPE_PATTERN = /^[a-z][a-z0-9-]*$/;

function normalizeType(value, label = "Artifact type") {
    const type = String(value || "").toLowerCase();
    if (!TYPE_PATTERN.test(type)) throw new TypeError(`${label} is invalid: ${value}`);
    return type;
}

export class ArtifactRegistry {
    constructor(plugins = []) {
        this.plugins = new Map();
        for (const plugin of plugins) this.register(plugin);
    }

    register(plugin) {
        if (!plugin || typeof plugin !== "object") {
            throw new TypeError("Artifact plugin must be an object");
        }
        const type = normalizeType(plugin.type);
        for (const method of ["parse", "resolveSource", "mount"]) {
            if (typeof plugin[method] !== "function") {
                throw new TypeError(`Artifact plugin ${type} must implement ${method}()`);
            }
        }

        const names = [type, ...(plugin.aliases || []).map((alias) => normalizeType(alias, "Artifact alias"))];
        for (const name of names) {
            if (this.plugins.has(name)) throw new Error(`Artifact type is already registered: ${name}`);
        }
        for (const name of names) this.plugins.set(name, plugin);
        return plugin;
    }

    get(type) {
        if (typeof type !== "string") return null;
        return this.plugins.get(type.toLowerCase()) || null;
    }

    has(type) {
        return this.get(type) !== null;
    }
}
