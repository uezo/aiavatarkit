class ImageAvatar {
    constructor({
        faceImage,
        mouthImage,
        faceImagePaths,
        lipsyncEngine = null,
        mouthPathTemplate = "images/mouth_{mouth}.png",
        mouthOpenThreshold = 0.52,
        visemeConfidenceThreshold = 0.55,
        rmsScale = 1.0,
        lipsyncEnabled = true,
        blinkEnabled = true,
    }) {
        this.faceImage = faceImage;
        this.mouthImage = mouthImage;
        this.faceImagePaths = faceImagePaths;
        this.lipsyncEngine = lipsyncEngine;
        this.mouthPathTemplate = mouthPathTemplate;
        this.mouthOpenThreshold = mouthOpenThreshold;
        this.visemeConfidenceThreshold = visemeConfidenceThreshold;
        this.rmsScale = rmsScale;
        this.lipsyncEnabled = lipsyncEnabled;
        this.blinkEnabled = blinkEnabled;
        this.mouthCache = new Map();
        this.mouthPreloaded = false;
        this.blinker = null;

        this.faceImage.hidden = false;
        this.mouthImage.hidden = false;
    }

    get aiAvatarOptions() {
        return {
            faceImage: this.faceImage,
            faceImagePaths: this.faceImagePaths,
        };
    }

    async bind(aiavatar) {
        if (this.lipsyncEnabled) {
            this.preloadMouths();
            this.lipsyncEngine ||= new LipSyncEngine();
            await this.lipsyncEngine.initialize();
            aiavatar.onPlaybackAudio = (audio) => {
                const result = this.lipsyncEngine.processAudioData({
                    ...audio,
                    gain: this.rmsScale,
                });
                this.applyLipSyncResult(result);
            };
            aiavatar.onResetFace = () => this.reset();
            aiavatar.onPlaybackEnd = () => this.stop();
        }

        if (this.blinkEnabled) {
            this.blinker = new BlinkController({
                stateProvider: () => ({
                    isSpeaking: aiavatar.isAudioPlaying,
                    currentFace: aiavatar.getCurrentFace(),
                }),
                onBlinkStart: () => {
                    this.faceImage.src = this.faceImagePaths.eyes_closed;
                },
                onBlinkEnd: () => {
                    const currentFace = aiavatar.getCurrentFace() || "neutral";
                    this.faceImage.src = this.faceImagePaths[currentFace] || this.faceImagePaths.neutral;
                },
            });
        }
    }

    applyLipSyncResult(result) {
        this.applyMouthShape(this.selectMouthShape(result));
    }

    selectMouthShape({ visemes = {}, mainViseme, mainVisemeWeight } = {}) {
        const weight = Number.isFinite(mainVisemeWeight) ? mainVisemeWeight : 0;
        if (!mainViseme || weight <= 0) return "closed";

        let total = 0;
        for (const value of Object.values(visemes)) {
            if (Number.isFinite(value) && value > 0) total += value;
        }
        const mainWeight = Number(visemes[mainViseme]) || 0;
        const confidence = total > 0 ? mainWeight / total : 0;
        if (confidence < this.visemeConfidenceThreshold) return "half";

        const isSmall = weight < this.mouthOpenThreshold;
        if (isSmall && (mainViseme === "A" || mainViseme === "O")) return "half";
        if (mainViseme === "U") return "u";
        if (mainViseme === "I" || mainViseme === "E") return "e";
        if (mainViseme === "A" || mainViseme === "O") return "open";
        return "half";
    }

    applyMouthShape(mouthShape) {
        if (mouthShape === "closed") {
            this.reset();
            return;
        }
        if (!this.mouthPreloaded) return;
        const cached = this.mouthCache.get(mouthShape);
        if (!cached) return;
        this.mouthImage.src = cached;
        this.mouthImage.style.display = "block";
    }

    async preloadMouths() {
        const tasks = ["half", "open", "u", "e"].map((mouthShape) => (
            this.fetchMouth(mouthShape).catch((error) => {
                console.warn(`ImageAvatar: failed to preload mouth "${mouthShape}"`, error);
            })
        ));
        try {
            await Promise.all(tasks);
        } finally {
            this.mouthPreloaded = true;
        }
    }

    async fetchMouth(mouthShape) {
        if (this.mouthCache.has(mouthShape)) return this.mouthCache.get(mouthShape);
        const response = await fetch(this.buildMouthPath(mouthShape));
        if (!response.ok) throw new Error(`Failed to load mouth: ${mouthShape}`);
        const url = URL.createObjectURL(await response.blob());
        this.mouthCache.set(mouthShape, url);
        return url;
    }

    buildMouthPath(mouthShape) {
        return this.mouthPathTemplate.replace("{mouth}", mouthShape);
    }

    reset() {
        this.mouthImage.src = "";
        this.mouthImage.style.display = "none";
    }

    stop() {
        this.reset();
    }
}
