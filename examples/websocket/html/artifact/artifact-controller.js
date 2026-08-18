import { parseArtifactControlTags } from "./artifact-parser.js";
import { ArtifactRegistry } from "./artifact-registry.js";

const ASPECT_VALUES = {
    "16:9": ["16 / 9", 16 / 9],
    "4:3": ["4 / 3", 4 / 3],
    "3:2": ["3 / 2", 3 / 2],
    "1:1": ["1 / 1", 1],
    "9:16": ["9 / 16", 9 / 16],
};

function isObject(value) {
    return value && typeof value === "object" && !Array.isArray(value);
}

export class ArtifactController {
    constructor({
        documentRoot = document,
        windowRoot = globalThis.window,
        host = null,
        plugins = [],
        onVisibilityChange = () => {},
    } = {}) {
        this.document = documentRoot;
        this.window = windowRoot;
        this.host = host || this.document.body;
        this.registry = new ArtifactRegistry(plugins);
        this.onVisibilityChange = onVisibilityChange;
        this.active = false;
        this.root = this.createRoot();
        this.current = null;
        this.currentPlugin = null;
        this.currentSession = null;
        this.generation = 0;
    }

    register(plugin) {
        return this.registry.register(plugin);
    }

    get currentState() {
        return this.currentSession?.getState?.() || null;
    }

    get currentSlide() {
        return this.currentState?.slide ?? null;
    }

    get totalSlides() {
        return this.currentState?.totalSlides ?? null;
    }

    createRoot() {
        const root = this.document.createElement("div");
        root.id = "artifactLayer";
        root.className = "artifact-layer";
        root.hidden = true;
        root.setAttribute("aria-live", "polite");
        this.host.appendChild(root);
        return root;
    }

    handleResponse(response) {
        if (!response || typeof response !== "object") return;

        if (!Array.isArray(response.control_tags)) return;
        const hasStructuredArtifact = response.control_tags.some(
            (tag) => String(tag?.name || "").toLowerCase() === "artifact",
        );
        if (!hasStructuredArtifact) return;

        const options = { registry: this.registry };
        const { commands, errors } = parseArtifactControlTags(response.control_tags, options);
        for (const item of errors) console.warn("Ignored invalid artifact tag:", item.error.message, item.tag);
        if (!commands.length) return;

        for (const command of commands) {
            try {
                this.apply(command);
            } catch (error) {
                console.warn("Could not display artifact:", error.message);
            }
        }
    }

    apply(command) {
        if (command.action === "clear") {
            this.clear();
            return;
        }

        const plugin = this.registry.get(command.type);
        if (!plugin) throw new Error(`No artifact plugin is registered for ${command.type}`);
        if (command.action !== "show") {
            this.executeCurrent(plugin, command);
            return;
        }

        const source = plugin.resolveSource(command);
        const pluginDefaults = plugin.getDefaults?.(command) || {};
        const defaults = {
            size: command.size || pluginDefaults.size || "large",
            aspect: command.aspect || pluginDefaults.aspect || "auto",
        };
        if (this.tryUpdateCurrent(plugin, command, source, defaults)) return;

        const surface = this.document.createElement("section");
        surface.className = "artifact-surface";
        surface.dataset.type = command.type;
        surface.dataset.provider = source.provider;
        surface.dataset.size = defaults.size;
        surface.dataset.aspect = defaults.aspect;
        surface.setAttribute("aria-label", command.title || command.alt || "AI artifact");

        if (defaults.aspect !== "auto") {
            const [aspect, ratio] = ASPECT_VALUES[defaults.aspect];
            surface.style.setProperty("--artifact-aspect", aspect);
            surface.style.setProperty("--artifact-ratio", String(ratio));
        }

        const closeButton = this.document.createElement("button");
        closeButton.type = "button";
        closeButton.className = "artifact-close";
        closeButton.setAttribute("aria-label", "Close artifact");
        closeButton.textContent = "×";
        closeButton.addEventListener("click", () => this.clear());

        const status = this.document.createElement("div");
        status.className = "artifact-status";
        status.textContent = "Loading…";

        const generation = this.generation + 1;
        const view = this.createView(surface, status, generation);
        const session = plugin.mount({
            documentRoot: this.document,
            windowRoot: this.window,
            command,
            source,
            defaults,
            view,
        });
        if (!session || typeof session !== "object" || !session.element) {
            session?.dispose?.();
            throw new Error(`Artifact plugin ${plugin.type} did not return a media element`);
        }

        this.disposeCurrentSession();
        this.generation = generation;
        this.currentPlugin = plugin;
        this.currentSession = session;
        this.current = { ...command, ...defaults, ...source };

        surface.append(session.element, status, closeButton);
        this.root.replaceChildren(surface);
        this.root.hidden = false;
        this.setActive(true);
    }

    tryUpdateCurrent(plugin, command, source, defaults) {
        if (plugin !== this.currentPlugin || typeof this.currentSession?.update !== "function") return false;
        if (this.current.size !== defaults.size
            || this.current.aspect !== defaults.aspect
            || this.current.title !== command.title
            || this.current.alt !== command.alt) return false;

        const result = this.currentSession.update({ command, source, defaults });
        if (result === false) return false;
        if (result?.then) throw new Error("Artifact session update() must be synchronous");
        this.current = { ...command, ...defaults, ...source, ...(isObject(result) ? result : {}) };
        return true;
    }

    executeCurrent(plugin, command) {
        if (plugin !== this.currentPlugin || typeof this.currentSession?.execute !== "function") {
            throw new Error(`No compatible ${command.type} artifact is displayed`);
        }
        const result = this.currentSession.execute(command);
        if (result === false) throw new Error(`Artifact does not support ${command.action}`);
        if (result?.then) throw new Error("Artifact session execute() must be synchronous");
        if (isObject(result)) this.current = { ...this.current, ...result };
    }

    createView(surface, status, generation) {
        const isCurrent = () => this.generation === generation;
        return {
            loaded: () => {
                if (!isCurrent()) return;
                surface.classList.add("loaded");
                status.remove();
            },
            error: (message = "表示できませんでした") => {
                if (!isCurrent()) return;
                surface.classList.add("artifact-load-error");
                status.textContent = message;
                if (typeof surface.contains === "function" && !surface.contains(status)) surface.append(status);
            },
            setIntrinsicAspect: (width, height) => {
                if (!isCurrent() || surface.dataset.aspect !== "auto" || !width || !height) return;
                surface.style.setProperty("--artifact-aspect", `${width} / ${height}`);
                surface.style.setProperty("--artifact-ratio", String(width / height));
            },
            updateCurrent: (patch) => {
                if (!isCurrent() || !this.current || !isObject(patch)) return;
                this.current = { ...this.current, ...patch };
            },
        };
    }

    disposeCurrentSession() {
        this.generation += 1;
        try {
            this.currentSession?.dispose?.();
        } catch (error) {
            console.warn("Could not dispose artifact:", error.message);
        }
        this.currentSession = null;
        this.currentPlugin = null;
    }

    setActive(active) {
        if (active === this.active) return;
        this.active = active;
        if (active) this.document.documentElement.classList.add("artifact-active");
        else this.document.documentElement.classList.remove("artifact-active");
        this.onVisibilityChange(active);
    }

    clear() {
        this.disposeCurrentSession();
        this.root.replaceChildren();
        this.root.hidden = true;
        this.setActive(false);
        this.current = null;
    }

    dispose() {
        this.clear();
        this.root.remove();
    }
}
