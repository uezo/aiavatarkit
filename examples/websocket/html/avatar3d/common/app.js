import { DisplayController } from "./display-controller.js";
import { assertAvatarAdapter } from "./avatar-adapter.js";
import { installBacklog } from "./backlog-controller.js";
import { createBacklogStore } from "./backlog-store.js";
import { installMessageController } from "./message-controller.js";
import { installPageControls } from "./page-controls.js";
import { installRequestInput } from "./request-input-controller.js";
import { installToolToasts } from "./tool-toast.js";
import { VisionController } from "./vision-controller.js";
import { ArtifactController } from "../../artifact/artifact-controller.js";

function requireObject(value, name) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new TypeError(`${name} must be an object`);
    }
}

function validateConfig(config) {
    for (const key of ["connection", "audio", "ui", "vision", "persistence"]) {
        requireObject(config[key], key);
    }
    if (!config.connection.webSocketUrl) throw new Error("connection.webSocketUrl is required");
    if (!Array.isArray(config.vision.availableModes) || !config.vision.availableModes.length) {
        throw new Error("vision.availableModes must be a non-empty array");
    }
    if (config.ui.messageSpeed < 1 || config.ui.messageSpeed > 100) {
        throw new RangeError("ui.messageSpeed must be between 1 and 100");
    }
    if (config.ui.messageBoxOpacity < 0 || config.ui.messageBoxOpacity > 100) {
        throw new RangeError("ui.messageBoxOpacity must be between 0 and 100");
    }
    if (config.backlog != null) requireObject(config.backlog, "backlog");
}

const IMAGE_FILE_EXTENSION = /\.(?:avif|bmp|gif|heic|heif|ico|jfif|jpe?g|png|svg|tiff?|webp)$/i;

function isImageFile(file) {
    return String(file?.type || "").toLowerCase().startsWith("image/")
        || IMAGE_FILE_EXTENSION.test(String(file?.name || ""));
}

export async function importDroppedFiles(files, { display, modelAdapter }) {
    const modelFiles = [];
    let backgroundFile = null;

    for (const file of files) {
        if (isImageFile(file)) backgroundFile = file;
        else modelFiles.push(file);
    }

    const imports = [];
    if (backgroundFile) imports.push(display.storeBackground(backgroundFile));
    if (modelFiles.length) imports.push(modelAdapter.importFiles(modelFiles));
    await Promise.all(imports);
}

export async function startAvatarApp({ config, modelAdapter, blobStore, artifactPlugins = [] }) {
    validateConfig(config);
    assertAvatarAdapter(modelAdapter);
    const aiavatar = new AIAvatarClient({
        webSocketUrl: config.connection.webSocketUrl,
        apiKey: config.connection.apiKey,
        sampleRate: config.audio.sampleRate,
        playbackAudioHz: config.audio.playbackAudioHz,
        faceImage: null,
        faceImagePaths: null,
    });
    aiavatar.setVolume(config.audio.initialVolume);
    aiavatar.setMicrophoneVolume(config.audio.initialMicrophoneVolume ?? 1.0);

    const ui = new AvatarUI({
        aiavatar,
        voiceDetectMode: config.audio.voiceDetectMode,
        voiceHoldDuration: config.audio.voiceHoldDurationMs,
        toolLabels: config.ui.toolLabels,
        onStop: () => modelAdapter.stop(),
    });
    const interruptToggle = document.getElementById("interruptToggle");
    interruptToggle.checked = config.audio.bargeInEnabled;
    ui.interruptEnabled = config.audio.bargeInEnabled;

    const settingsHost = await modelAdapter.initialize({ aiavatar, ui });
    const display = new DisplayController({
        aiavatar,
        ui,
        settingsHost,
        blobStore,
        config: config.ui,
        persistence: config.persistence,
    });
    const messages = installMessageController({
        aiavatar,
        ui,
        state: display.state,
        autoHideDelayMs: config.ui.autoHideDelayMs,
    });
    const backlogConfig = {
        enabled: true,
        maxEntries: 100,
        ...config.backlog,
    };
    const backlogStore = createBacklogStore({
        enabled: config.persistence.enabled && backlogConfig.enabled,
        databaseName: `${config.persistence.databaseName}_backlog`,
        maxEntries: backlogConfig.maxEntries,
    });
    const backlog = installBacklog({
        aiavatar,
        ui,
        store: backlogStore,
        maxEntries: backlogConfig.maxEntries,
    });
    await backlog.ready;
    const requestInput = installRequestInput({
        aiavatar,
        ui,
        imageOptions: {
            maxLongEdge: config.vision.maxLongEdge,
            jpegQuality: config.vision.jpegQuality,
        },
        onSent: (request) => backlog.stageUser(request),
    });
    const vision = new VisionController({ aiavatar, ui, config: config.vision });
    const artifacts = new ArtifactController({
        plugins: artifactPlugins,
        onVisibilityChange: (active) => modelAdapter.setArtifactMode?.(active),
    });
    const controls = installPageControls({
        ui,
        state: display.state,
        settingsHost,
        labels: config.ui.controls,
        onDisconnected: () => display.updateConnection(false),
    });
    const toasts = installToolToasts({ durationMs: config.ui.toastDurationMs });

    const volumePercent = Math.round(config.audio.initialVolume * 100);
    document.getElementById("volumeSlider").value = volumePercent;
    document.getElementById("volumeValue").textContent = volumePercent;
    const microphoneVolumePercent = Math.round(
        (config.audio.initialMicrophoneVolume ?? 1.0) * 100,
    );
    document.getElementById("microphoneVolumeSlider").value = microphoneVolumePercent;
    document.getElementById("microphoneVolumeValue").textContent = microphoneVolumePercent;

    const dropOverlay = document.getElementById("dropOverlay");
    const onDragOver = (event) => {
        event.preventDefault();
        dropOverlay.classList.add("show");
    };
    const onDragLeave = (event) => {
        if (event.relatedTarget === null) dropOverlay.classList.remove("show");
    };
    const onDrop = async (event) => {
        event.preventDefault();
        dropOverlay.classList.remove("show");
        try {
            await importDroppedFiles(Array.from(event.dataTransfer.files || []), { display, modelAdapter });
        } catch (error) {
            console.error("Failed to import dropped files:", error);
        }
    };
    document.addEventListener("dragover", onDragOver);
    document.addEventListener("dragleave", onDragLeave);
    document.addEventListener("drop", onDrop);

    aiavatar.onResponseReceived = (response) => {
        backlog.handleResponse(response);
        artifacts.handleResponse(response);
        modelAdapter.handleResponse(response);
        vision.handleResponse(response);
        if (response.type === "connected") display.updateConnection(true, response);
        if (response.type === "tool_call") console.log(response.metadata);
        ui.handleResponse(response);
    };

    globalThis.chat = (text, imageDataUrl) => aiavatar.chat(ui.sessionId, ui.userId, text, imageDataUrl);

    if (config.vision.defaultMode !== "off") await vision.setMode(config.vision.defaultMode);

    const dispose = () => {
        document.removeEventListener("dragover", onDragOver);
        document.removeEventListener("dragleave", onDragLeave);
        document.removeEventListener("drop", onDrop);
        controls.dispose();
        backlog.dispose();
        messages.dispose();
        requestInput.dispose();
        toasts.dispose();
        artifacts.dispose();
        vision.dispose();
        display.dispose();
        modelAdapter.dispose();
    };
    window.addEventListener("pagehide", dispose, { once: true });

    const app = { aiavatar, ui, modelAdapter, display, vision, artifacts, backlog, dispose };
    globalThis.avatar3d = app;
    return app;
}
