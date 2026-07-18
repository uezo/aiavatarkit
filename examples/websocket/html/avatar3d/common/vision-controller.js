const MODES = new Set(["off", "capture", "stream"]);

export class VisionController {
    constructor({ aiavatar, ui, config }) {
        this.aiavatar = aiavatar;
        this.ui = ui;
        this.config = config;
        this.preview = document.getElementById("visionPreview");
        this.button = document.getElementById("visionBtn");
        this.client = null;
        this.mode = "off";
        this.availableModes = config.availableModes.filter((mode) => MODES.has(mode));
        if (!this.availableModes.includes("off")) this.availableModes.unshift("off");

        this.onButtonClick = () => this.cycleMode();
        this.onPreviewClick = () => this.client?.switchCamera();
        this.button.addEventListener("click", this.onButtonClick);
        this.preview.addEventListener("click", this.onPreviewClick);
        this.ui.onVisionRequested = () => this.captureOnDemand();
        this.renderState();
    }

    async cycleMode() {
        const currentIndex = this.availableModes.indexOf(this.mode);
        const nextMode = this.availableModes[(currentIndex + 1) % this.availableModes.length];
        await this.setMode(nextMode);
    }

    async setMode(mode) {
        if (!MODES.has(mode)) throw new Error(`Unknown vision mode: ${mode}`);
        if (!this.config.enabled) mode = "off";

        if (mode !== "off" && !this.client) {
            try {
                await this.startCamera();
            } catch (error) {
                console.error("[Vision] Failed to start camera:", error);
                this.stopCamera();
                this.mode = "off";
                this.renderState();
                return;
            }
        }

        this.mode = mode;
        if (mode === "off") this.stopCamera();
        this.renderState();
    }

    async startCamera() {
        this.preview.style.display = "block";
        this.client = new VisionStreamClient({
            serverUrl: this.config.serverUrl,
            contextId: this.aiavatar.chatContextId || null,
            userId: this.ui.userId,
            interval: this.config.intervalSeconds,
            maxLongEdge: this.config.maxLongEdge,
            jpegQuality: this.config.jpegQuality,
            videoIdealSize: this.config.camera.idealSize,
            facingMode: this.config.camera.facingMode,
            videoElement: this.preview,
            pauseWhen: () => this.mode !== "stream" || this.ui.isUserSpeaking,
            onResult: (text, attentionLevel, imageUrl) => this.handleStreamResult(text, attentionLevel, imageUrl),
            onError: (_sequence, error) => console.error("[Vision] Error:", error),
        });
        await this.client.start();
    }

    stopCamera() {
        this.client?.stop();
        this.client = null;
        this.preview.style.display = "none";
    }

    async handleStreamResult(text, attentionLevel, imageUrl) {
        console.log(`[Vision] (${attentionLevel}) ${text}`);
        if (attentionLevel < this.config.attentionThreshold) return undefined;
        this.sendImage(this.config.streamPrompt, imageUrl);
        return this.config.cooldownMs;
    }

    async captureOnDemand() {
        if (this.mode === "off" || !this.client) return;
        const imageDataUrl = await this.client.capture();
        if (imageDataUrl) this.sendImage(this.config.capturePrompt, imageDataUrl);
    }

    sendImage(text, imageUrl) {
        const socket = this.aiavatar.ws;
        if (!socket || socket.readyState !== WebSocket.OPEN) return false;
        socket.send(JSON.stringify({
            type: "invoke",
            session_id: this.ui.sessionId,
            user_id: this.ui.userId,
            text,
            files: [{ url: imageUrl }],
            allow_merge: false,
            wait_in_queue: true,
        }));
        return true;
    }

    handleResponse(response) {
        if (this.client && response.context_id) this.client.contextId = response.context_id;
    }

    renderState() {
        const labels = this.config.labels;
        this.button.classList.toggle("active", this.mode !== "off");
        this.button.dataset.state = this.mode;
        this.button.title = labels[this.mode];
        this.button.setAttribute("aria-label", labels[this.mode]);
    }

    dispose() {
        this.stopCamera();
        this.button.removeEventListener("click", this.onButtonClick);
        this.preview.removeEventListener("click", this.onPreviewClick);
        this.ui.onVisionRequested = null;
    }
}
