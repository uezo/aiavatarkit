import { DisplayController } from "./display-controller.js";
import { assertAvatarAdapter } from "./avatar-adapter.js";
import { installMessageController } from "./message-controller.js";
import { installPageControls } from "./page-controls.js";
import { installToolToasts } from "./tool-toast.js";
import { VisionController } from "./vision-controller.js";

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
}

export async function startAvatarApp({ config, modelAdapter, blobStore }) {
    validateConfig(config);
    assertAvatarAdapter(modelAdapter);
    const aiavatar = new AIAvatarClient({
        webSocketUrl: config.connection.webSocketUrl,
        apiKey: config.connection.apiKey,
        sampleRate: config.audio.sampleRate,
        playbackAnalyzeHz: config.audio.playbackAnalyzeHz,
        faceImage: null,
        faceImagePaths: null,
    });
    aiavatar.setVolume(config.audio.initialVolume);

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
    const vision = new VisionController({ aiavatar, ui, config: config.vision });
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
            await modelAdapter.importFiles(Array.from(event.dataTransfer.files || []));
        } catch (error) {
            console.error("Failed to import model files:", error);
        }
    };
    document.addEventListener("dragover", onDragOver);
    document.addEventListener("dragleave", onDragLeave);
    document.addEventListener("drop", onDrop);

    aiavatar.onResponseReceived = (response) => {
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
        messages.dispose();
        toasts.dispose();
        vision.dispose();
        display.dispose();
        modelAdapter.dispose();
    };
    window.addEventListener("pagehide", dispose, { once: true });

    const app = { aiavatar, ui, modelAdapter, display, vision, dispose };
    globalThis.avatar3d = app;
    return app;
}
