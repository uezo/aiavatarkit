class BlinkController {
    constructor({
        minIntervalMs = 3000,
        maxIntervalMs = 6000,
        blinkDurationMs = 150,
        shouldBlink = null,
        onBlinkStart = () => {},
        onBlinkEnd = () => {},
        targetImage = null,
        facePathTemplate = "images/{face}.png",
        blinkFaceName = "eyes_closed",
        stateProvider = null,
        statePollIntervalMs = 200,
        autoStart = true,
    } = {}) {
        this.minIntervalMs = minIntervalMs;
        this.maxIntervalMs = maxIntervalMs;
        this.blinkDurationMs = blinkDurationMs;
        this.shouldBlink = shouldBlink;
        this.onBlinkStart = onBlinkStart;
        this.onBlinkEnd = onBlinkEnd;
        this.targetImage = targetImage;
        this.facePathTemplate = facePathTemplate;
        this.blinkFaceName = blinkFaceName;
        this.stateProvider = stateProvider;
        this.statePollIntervalMs = statePollIntervalMs;
        this.autoStart = autoStart;

        this.isRunning = false;
        this.state = { isSpeaking: false, currentFace: "neutral" };
        this.timer = null;
        this.blinkTimer = null;
        this.statePollTimer = null;
        if (this.autoStart) this.start();
    }

    setState({ isSpeaking, currentFace }) {
        if (typeof isSpeaking === "boolean") this.state.isSpeaking = isSpeaking;
        if (typeof currentFace === "string") this.state.currentFace = currentFace;
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        if (typeof this.stateProvider === "function") {
            this.statePollTimer = setInterval(() => {
                try {
                    const s = this.stateProvider();
                    if (s) this.setState(s);
                } catch (e) {
                    // ignore provider errors
                }
            }, this.statePollIntervalMs);
        }
        this.scheduleNext();
    }

    stop() {
        this.isRunning = false;
        if (this.timer) clearTimeout(this.timer);
        if (this.blinkTimer) clearTimeout(this.blinkTimer);
        if (this.statePollTimer) clearInterval(this.statePollTimer);
        this.timer = null;
        this.blinkTimer = null;
        this.statePollTimer = null;
    }

    scheduleNext() {
        if (!this.isRunning) return;
        if (this.timer) clearTimeout(this.timer);
        const interval = this.randomInterval();
        this.timer = setTimeout(() => this.tryBlink(), interval);
    }

    tryBlink() {
        if (!this.isRunning) return;
        if (!this.canBlink()) {
            this.scheduleNext();
            return;
        }
        this.applyBlink();
        if (this.blinkTimer) clearTimeout(this.blinkTimer);
        this.blinkTimer = setTimeout(() => {
            this.applyUnblink();
            this.scheduleNext();
        }, this.blinkDurationMs);
    }

    canBlink() {
        if (typeof this.shouldBlink === "function") {
            return this.shouldBlink({ ...this.state });
        }
        // Default rule: no blink while speaking or when face is non-neutral
        return !this.state.isSpeaking && (!this.state.currentFace || this.state.currentFace === "neutral");
    }

    randomInterval() {
        const { minIntervalMs, maxIntervalMs } = this;
        return minIntervalMs + Math.random() * Math.max(0, maxIntervalMs - minIntervalMs);
    }

    applyBlink() {
        this.onBlinkStart();
        if (this.targetImage) {
            this.targetImage.src = this.buildFacePath(this.blinkFaceName);
        }
    }

    applyUnblink() {
        this.onBlinkEnd();
        if (this.targetImage) {
            const face = this.state.currentFace || "neutral";
            this.targetImage.src = this.buildFacePath(face);
        }
    }

    buildFacePath(faceName) {
        return this.facePathTemplate.replace("{face}", faceName);
    }
}
