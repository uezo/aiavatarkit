import { startAvatarApp } from "./common/app.js";
import { createBlobStore } from "./common/blob-store.js";

function selectedAdapter(modelConfig) {
    const queryName = modelConfig.queryParameter;
    const requested = queryName
        ? new URLSearchParams(window.location.search).get(queryName)
        : null;
    const type = requested || modelConfig.type;
    const adapter = modelConfig.adapters?.[type];
    if (!adapter) {
        const available = Object.keys(modelConfig.adapters || {}).join(", ");
        throw new Error(`Unknown model type "${type}". Available types: ${available}`);
    }
    return { type, adapter };
}

function loadStylesheet(url) {
    if ([...document.styleSheets].some((sheet) => sheet.href === new URL(url, document.baseURI).href)) {
        return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = url;
        link.addEventListener("load", resolve, { once: true });
        link.addEventListener("error", () => reject(new Error(`Could not load stylesheet: ${url}`)), { once: true });
        document.head.appendChild(link);
    });
}

function loadScript(url) {
    const absoluteUrl = new URL(url, document.baseURI).href;
    if ([...document.scripts].some((script) => script.src === absoluteUrl)) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.addEventListener("load", resolve, { once: true });
        script.addEventListener("error", () => reject(new Error(`Could not load script: ${url}`)), { once: true });
        document.head.appendChild(script);
    });
}

async function loadDependencies({ stylesheets = [], scripts = [] } = {}) {
    await Promise.all(stylesheets.map(loadStylesheet));
    for (const script of scripts) await loadScript(script);
}

function applyLabels(labels = {}) {
    if (labels.title) document.title = labels.title;
    const placeholder = document.getElementById("avatarPlaceholder");
    const dropText = document.querySelector(".drop-overlay-text");
    if (labels.placeholder && placeholder) placeholder.textContent = labels.placeholder;
    if (labels.drop && dropText) dropText.textContent = labels.drop;
}

export async function start3dPage(config) {
    if (!config?.model || typeof config.model !== "object") {
        throw new TypeError("model configuration is required");
    }
    const { type, adapter: adapterConfig } = selectedAdapter(config.model);
    if (!adapterConfig.module || !adapterConfig.exportName) {
        throw new Error(`model.adapters.${type} must define module and exportName`);
    }

    document.documentElement.dataset.modelType = type;
    applyLabels(adapterConfig.labels);
    await loadDependencies(adapterConfig.dependencies);

    const persistence = {
        ...config.persistence,
        ...adapterConfig.persistence,
    };
    const blobStore = createBlobStore(persistence);
    const adapterModule = await import(new URL(adapterConfig.module, document.baseURI).href);
    const Adapter = adapterModule[adapterConfig.exportName];
    if (typeof Adapter !== "function") {
        throw new Error(`${adapterConfig.exportName} is not exported by ${adapterConfig.module}`);
    }
    const modelAdapter = new Adapter({
        config: adapterConfig.options,
        persistence,
        blobStore,
    });
    const { model, ...commonConfig } = config;
    return startAvatarApp({
        config: { ...commonConfig, persistence },
        modelAdapter,
        blobStore,
    });
}
