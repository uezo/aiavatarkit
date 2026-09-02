class MPTAvatar {
    constructor({
        video,
        mouthCanvas,
        stage,
        assets,
        rmsScale = 1.0,
        options = {
            hqAudioEnabled: true,
            sensitivity: 0,
            debug: false,
        },
    }) {
        video.hidden = false;
        mouthCanvas.hidden = false;
        this.rmsScale = rmsScale;

        this.lipsyncEngine = new LipsyncEngine({
            elements: { video, mouthCanvas, stage },
            assets,
            options,
        });
    }

    get aiAvatarOptions() {
        return {};
    }

    bind(aiavatar) {
        aiavatar.onPlaybackAudio = (audio) => {
            this.lipsyncEngine.processAudioData({
                ...audio,
                gain: this.rmsScale,
            });
        };
        aiavatar.onResetFace = () => this.reset();
        aiavatar.onPlaybackEnd = () => this.stop();
    }

    reset() {
        this.lipsyncEngine.resetAudioStats();
    }

    stop() {
        this.lipsyncEngine.volume = 0;
        this.lipsyncEngine.smoothedHighRatio = 0;
        this.lipsyncEngine.setMouthState("closed", true);
    }
}
