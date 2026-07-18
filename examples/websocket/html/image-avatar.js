class ImageAvatar {
    constructor({
        faceImage,
        mouthImage,
        faceImagePaths,
        mouthPathTemplate = "images/mouth_{mouth}.png",
        rmsScale = 1.0,
        lipsyncEnabled = true,
        blinkEnabled = true,
    }) {
        this.faceImage = faceImage;
        this.mouthImage = mouthImage;
        this.faceImagePaths = faceImagePaths;
        this.mouthPathTemplate = mouthPathTemplate;
        this.rmsScale = rmsScale;
        this.lipsyncEnabled = lipsyncEnabled;
        this.blinkEnabled = blinkEnabled;
        this.lipsyncEngine = null;
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

    bind(aiavatar) {
        if (this.lipsyncEnabled) {
            this.lipsyncEngine = new LipSyncEngine({
                applyTarget: this.mouthImage,
                mouthPathTemplate: this.mouthPathTemplate,
                hideOnClosed: true,
            });
            aiavatar.onPlaybackAnalyze = ({ rms, centroid01, tSec }) => {
                this.lipsyncEngine.apply({
                    rms: rms * this.rmsScale,
                    centroid01,
                    tSec,
                });
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

    reset() {
        if (this.lipsyncEngine) this.lipsyncEngine.reset();
    }

    stop() {
        this.reset();
    }
}
