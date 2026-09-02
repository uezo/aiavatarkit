class AIAvatarClient {
    constructor({ webSocketUrl, faceImage, faceImagePaths, sampleRate = 16000, playbackAudioHz = 30, apiKey = null }) {
        this.webSocketUrl = webSocketUrl;
        this.faceImage = faceImage;
        this.faceImagePaths = faceImagePaths;
        this.sampleRate = sampleRate;
        this.playbackAudioHz = playbackAudioHz;
        this.apiKey = apiKey;

        this.ws = null;
        this.audioContext = null;
        this.scriptNode = null;
        this.micStream = null;
        this.isAudioPlaying = false;
        this.messageQueue = [];
        this.processingQueue = false;
        this.queueGeneration = 0;
        this.currentAudioSource = null;
        this.currentAudioFinalize = null;
        this.playbackGeneration = 0;
        this.latestFaceUpdate = null;
        this.faceTimeout = null;
        this.currentFaceName = null;
        this.onResetFace = null;
        this.onMicrophoneDataSend = () => { };
        this.onResponseReceived = () => { };
        this.onPlaybackAudio = null;
        this.isMicrophoneMuted = () => this.isAudioPlaying;
        this.getStartMetadata = () => null;
        this._userMuted = false;
        this.volume = 1.0;
        this.gainNode = null;
        this.chatContextId = null;
    }

    async startListening(sessionId, userId) {
        const queueGeneration = ++this.queueGeneration;
        this.messageQueue.length = 0;
        this.processingQueue = false;
        const protocols = this.apiKey
            ? ["Authorization." + btoa(this.apiKey)]
            : undefined;
        this.ws = new WebSocket(this.webSocketUrl, protocols);
        this.ws.onopen = () => {
            if (queueGeneration !== this.queueGeneration) return;
            console.log(`Connected to server: ${this.webSocketUrl}`);
            const metadata = this.getStartMetadata?.() || null;
            const startMessage = {
                type: "start",
                session_id: sessionId,
                user_id: userId,
                // Do not send context_id here; the server manages it via the session for voice conversations
                context_id: null,
                metadata
            };
            this.ws.send(JSON.stringify(startMessage));
        };

        this.ws.onmessage = (event) => {
            if (queueGeneration !== this.queueGeneration) return;
            try {
                const msg = JSON.parse(event.data);
                this.onResponseReceived(msg);
                if (msg.type === "start" || msg.type === "chunk") {
                    if (msg.type === "start" && msg.context_id) {
                        this.chatContextId = msg.context_id;
                    }
                    this.messageQueue.push(msg);
                    if (!this.processingQueue) this.processQueue(queueGeneration);
                } else if (msg.type === "connected") {
                    userId = msg.user_id;   // Update userId (Created on server if not exists)
                    console.log(`Session: sessionId=${msg.session_id}, userId=${msg.user_id}, contextId=${msg.context_id}`);
                } else if (msg.type === "stop") {
                    this.messageQueue.length = 0;
                    this.stopAudio();
                    this.resetFace();
                } else if (msg.type === "final") {
                    console.log("Final response:", msg);
                }
            } catch (e) {
                console.error("Error parsing message:", e);
            }
        };

        this.ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        // Create new AudioContext if needed
        if (!this.audioContext || this.audioContext.state === "closed") {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
        }
        await this.audioContext.resume();
        console.log("AudioContext state:", this.audioContext.state);

        try {
            this.micStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
            });
            console.log("Microphone accepted.");
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            this.scriptNode = this.audioContext.createScriptProcessor(256, 1, 1);
            this.scriptNode.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    if (!this._userMuted && !this.isMicrophoneMuted()) {
                        let sum = 0;
                        for (let i = 0; i < inputData.length; i++) {
                            sum += inputData[i] * inputData[i];
                        }
                        const rms = Math.sqrt(sum / inputData.length);
                        this.onMicrophoneDataSend(rms);
                        const pcmBuffer = this.float32To16BitPCMBuffer(inputData);
                        const base64Data = this.arrayBufferToBase64(pcmBuffer);
                        this.ws.send(JSON.stringify({ type: "data", session_id: sessionId, audio_data: base64Data }));
                    } else {
                        const silentBuffer = new ArrayBuffer(inputData.length * 2);
                        const base64Data = this.arrayBufferToBase64(silentBuffer);
                        this.ws.send(JSON.stringify({ type: "data", session_id: sessionId, audio_data: base64Data }));
                    }
                }
            };

            source.connect(this.scriptNode);
            // Connect to dest to fire onaudioprocess event
            this.scriptNode.connect(this.audioContext.destination);

            // Setup gain node for volume control
            if (!this.gainNode) {
                this.gainNode = this.audioContext.createGain();
                this.gainNode.gain.value = this.volume;
                this.gainNode.connect(this.audioContext.destination);
            }

        } catch (err) {
            console.error("Error during microphone activation:", err);
        }
    }

    async processQueue(queueGeneration = this.queueGeneration) {
        if (queueGeneration !== this.queueGeneration) return;
        this.processingQueue = true;
        while (this.messageQueue.length > 0
            && queueGeneration === this.queueGeneration) {
            const msg = this.messageQueue.shift();
            if (msg.metadata && msg.metadata.request_text) {
                console.log("User:", msg.metadata.request_text);
            } else {
                if (msg.text != null && msg.text !== "") {
                    console.log("AI:", msg.text);
                }
            }
            if (msg.avatar_control_request && msg.avatar_control_request.face_name) {
                this.updateFace(msg.avatar_control_request.face_name, msg.avatar_control_request.face_duration);
            }
            if (msg.audio_data) {
                try {
                    this.isAudioPlaying = true;
                    await this.playAudioSync(msg.audio_data);
                } catch (e) {
                    console.error("Error during audio playback:", e);
                } finally {
                    if (queueGeneration === this.queueGeneration) {
                        this.isAudioPlaying = false;
                    }
                }
            }
        }
        if (queueGeneration === this.queueGeneration) this.processingQueue = false;
    }

    playAudioSync(audioDataBase64) {
        this.stopAudio();
        const playbackGeneration = this.playbackGeneration;
        return new Promise((resolve, reject) => {
            try {
                const binaryString = atob(audioDataBase64);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const buffer = bytes.buffer;
                this.audioContext.decodeAudioData(
                    buffer,
                    (decodedData) => {
                        if (playbackGeneration !== this.playbackGeneration
                            || this.audioContext?.state === "closed") {
                            resolve();
                            return;
                        }
                        const source = this.audioContext.createBufferSource();
                        source.buffer = decodedData;

                        const dest = this.gainNode || this.audioContext.destination;
                        const playbackAudioCallback = typeof this.onPlaybackAudio === "function"
                            ? this.onPlaybackAudio
                            : null;
                        const playbackPcm = decodedData.getChannelData(0);
                        const playbackSampleRate = decodedData.sampleRate;
                        source.connect(dest);

                        const startedAt = this.audioContext.currentTime;
                        this.currentAudioSource = source;
                        source.start(0);

                        const playbackFrame = () => {
                            const tSec = this.audioContext.currentTime;
                            const playbackTimeSec = Math.max(0, tSec - startedAt);
                            const samplePosition = Math.min(
                                playbackPcm.length,
                                Math.floor(playbackTimeSec * playbackSampleRate),
                            );
                            return {
                                pcm: playbackPcm,
                                sampleRate: playbackSampleRate,
                                samplePosition,
                                tSec,
                            };
                        };
                        let playbackFinalized = false;
                        const finalizePlayback = () => {
                            if (playbackFinalized) return;
                            playbackFinalized = true;
                            const currentSource = this.currentAudioSource;
                            const superseded = currentSource != null && currentSource !== source;
                            if (currentSource === source) {
                                this.currentAudioSource = null;
                                this.currentAudioFinalize = null;
                            }
                            try {
                                if (!superseded) this.onPlaybackEnd?.();
                            } catch (error) {
                                console.error("Error handling playback end:", error);
                            } finally {
                                resolve();
                            }
                        };
                        this.currentAudioFinalize = finalizePlayback;

                        if (playbackAudioCallback) {
                            // Preserve the deadline phase across rounded rAF timestamps.
                            const callbackIntervalMs = 1000 / (this.playbackAudioHz || 30);
                            const timestampToleranceMs = 1;
                            let nextCallbackT = null;
                            const tick = (ts) => {
                                if (this.currentAudioSource !== source) return;
                                if (nextCallbackT == null
                                    || ts + timestampToleranceMs >= nextCallbackT) {
                                    if (nextCallbackT == null) {
                                        nextCallbackT = ts + callbackIntervalMs;
                                    } else {
                                        const intervalsToNextDeadline = Math.max(
                                            1,
                                            Math.ceil(
                                                (ts + timestampToleranceMs - nextCallbackT)
                                                / callbackIntervalMs,
                                            ),
                                        );
                                        nextCallbackT += intervalsToNextDeadline
                                            * callbackIntervalMs;
                                    }
                                    playbackAudioCallback(playbackFrame());
                                }
                                requestAnimationFrame(tick);
                            };
                            requestAnimationFrame(tick);
                        }

                        source.onended = finalizePlayback;
                    },
                    (error) => {
                        if (playbackGeneration !== this.playbackGeneration) resolve();
                        else reject(error);
                    }
                );
            } catch (e) {
                reject(e);
            }
        });
    }

    mute() {
        this._userMuted = true;
    }

    unmute() {
        this._userMuted = false;
    }

    toggleMute() {
        this._userMuted = !this._userMuted;
        return this._userMuted;
    }

    get isMuted() {
        return this._userMuted;
    }

    setVolume(value) {
        this.volume = Math.max(0, Math.min(1, value));
        if (this.gainNode) {
            this.gainNode.gain.value = this.volume;
        }
    }

    chat(sessionId, userId, text, imageDataUrl) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
        const msg = {
            type: "invoke",
            session_id: sessionId,
            user_id: userId,
            context_id: this.chatContextId,
            text: text,
        };
        if (imageDataUrl) {
            msg.files = [{ url: imageDataUrl }];
        }
        this.ws.send(JSON.stringify(msg));
        return true;
    }

    sendConfig(sessionId, metadata) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
        this.ws.send(JSON.stringify({
            type: "config",
            session_id: sessionId,
            metadata
        }));
        return true;
    }

    stopAudio() {
        this.playbackGeneration++;
        if (this.currentAudioSource) {
            const source = this.currentAudioSource;
            this.currentAudioFinalize?.();
            try {
                source.stop();
            } catch (error) {
                console.error("Error stopping audio:", error);
            }
            if (this.currentAudioSource === source) this.currentAudioSource = null;
            this.currentAudioFinalize = null;
        }
    }

    updateFace(faceName, faceDuration) {
        if (this.faceImagePaths === undefined || this.faceImagePaths === null) {
            return;
        }

        faceName = faceName.toLowerCase();
        const faceImagePath = this.faceImagePaths[faceName];
        if (faceImagePath === undefined || faceImagePath === null || faceImagePath === "") {
            return;
        }
        this.currentFaceName = faceName;
        this.faceImage.src = faceImagePath;
        const currentUpdate = Date.now();
        this.latestFaceUpdate = currentUpdate;

        if (this.faceTimeout) clearTimeout(this.faceTimeout);
        this.faceTimeout = setTimeout(() => {
            if (this.latestFaceUpdate === currentUpdate) {
                this.currentFaceName = "neutral";
                this.faceImage.src = this.faceImagePaths["neutral"];
            }
        }, (faceDuration || 2) * 1000);
    }

    resetFace() {
        this.updateFace("neutral", 0);
        this.onResetFace?.();
    }

    getCurrentFace() {
        return this.currentFaceName;
    }

    float32To16BitPCMBuffer(floatBuffer) {
        const len = floatBuffer.length;
        const buffer = new ArrayBuffer(len * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < len; i++) {
            let sample = floatBuffer[i];
            sample = Math.max(-1, Math.min(1, sample));
            const intSample = sample < 0 ? sample * 32768 : sample * 32767;
            view.setInt16(i * 2, intSample, true);
        }
        return buffer;
    }

    arrayBufferToBase64(buffer) {
        let binary = "";
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    async stopListening(sessionId) {
        this.resetFace();
        this.queueGeneration++;
        this.processingQueue = false;
        this.messageQueue.length = 0;
        const ws = this.ws;
        this.ws = null;
        if (ws) {
            ws.onopen = null;
            ws.onmessage = null;
            ws.onerror = null;
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "stop", session_id: sessionId }));
            }
            if (ws.readyState === WebSocket.CONNECTING
                || ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        }
        this.stopAudio();
        if (this.scriptNode) {
            this.scriptNode.disconnect();
        }
        if (this.audioContext) {
            await this.audioContext.close();
            this.gainNode = null;
            this.isAudioPlaying = false;
        }
        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
        }
    }
}
