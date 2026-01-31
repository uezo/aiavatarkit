class AIAvatarClient {
    constructor({ webSocketUrl, faceImage, faceImagePaths, sampleRate = 16000, playbackAnalyzeHz = 60 }) {
        this.webSocketUrl = webSocketUrl;
        this.faceImage = faceImage;
        this.faceImagePaths = faceImagePaths;
        this.sampleRate = sampleRate;
        this.playbackAnalyzeHz = playbackAnalyzeHz;

        this.ws = null;
        this.audioContext = null;
        this.scriptNode = null;
        this.micStream = null;
        this.isAudioPlaying = false;
        this.messageQueue = [];
        this.processingQueue = false;
        this.currentAudioSource = null;
        this.latestFaceUpdate = null;
        this.faceTimeout = null;
        this.currentFaceName = null;
        this.onResetFace = null;
        this.onMicrophoneDataSend = () => { };
        this.onResponseReceived = () => { };
        this.analyser = null;
        this.onPlaybackAnalyze = null;
        this.isMicrophoneMuted = () => this.isAudioPlaying;
        this.volume = 1.0;
        this.gainNode = null;
    }

    async startListening(sessionId, userId) {
        this.ws = new WebSocket(this.webSocketUrl);
        this.ws.onopen = () => {
            console.log(`Connected to server: ${this.webSocketUrl}`);
            const startMessage = {
                type: "start",
                session_id: sessionId,
                user_id: userId,
                context_id: null
            };
            this.ws.send(JSON.stringify(startMessage));
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this.onResponseReceived(msg);
                if (msg.type === "start" || msg.type === "chunk") {
                    this.messageQueue.push(msg);
                    if (!this.processingQueue) this.processQueue();
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
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) {
                    sum += inputData[i] * inputData[i];
                }
                const rms = Math.sqrt(sum / inputData.length);

                const pcmBuffer = this.float32To16BitPCMBuffer(inputData);
                const base64Data = this.arrayBufferToBase64(pcmBuffer);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    if (!this.isMicrophoneMuted()) {
                        this.onMicrophoneDataSend(rms);
                        const dataMessage = {
                            type: "data",
                            session_id: sessionId,
                            audio_data: base64Data
                        };
                        this.ws.send(JSON.stringify(dataMessage));
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

            // Setup analyzer
            if (!this.analyser) {
                const a = this.audioContext.createAnalyser();
                a.fftSize = 256;
                a.smoothingTimeConstant = 0.6;
                this.analyser = a;
            }
        } catch (err) {
            console.error("Error during microphone activation:", err);
        }
    }

    async processQueue() {
        this.processingQueue = true;
        while (this.messageQueue.length > 0) {
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
                    this.isAudioPlaying = false;
                }
            }
        }
        this.processingQueue = false;
    }

    playAudioSync(audioDataBase64) {
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
                        const source = this.audioContext.createBufferSource();
                        source.buffer = decodedData;

                        const dest = this.gainNode || this.audioContext.destination;
                        const canAnalyze = this.analyser && typeof this.onPlaybackAnalyze === "function";
                        if (canAnalyze) {
                            source.connect(this.analyser);
                            this.analyser.connect(dest);  // analyser -> gainNode -> destination

                            // Analyze audio
                            const freqData = new Float32Array(this.analyser.frequencyBinCount);
                            const timeData = new Float32Array(this.analyser.fftSize);
                            let lastAnalyzeT = 0;
                            const tick = (ts) => {
                                if (!this.currentAudioSource) return; // Exit on playback ends
                                if (!this.analyser) return; // Analyzer disposed after stopListening
                                // Throttle by playbackAnalyzeHz
                                const analyzeIntervalMs = 1000 / (this.playbackAnalyzeHz || 60);
                                if (ts - lastAnalyzeT >= analyzeIntervalMs) {
                                    lastAnalyzeT = ts;
                                    // RMS (time domain)
                                    this.analyser.getFloatTimeDomainData(timeData);
                                    let sum = 0;
                                    for (let i = 0; i < timeData.length; i++) sum += timeData[i] * timeData[i];
                                    const rms = Math.sqrt(sum / timeData.length);

                                    // Spectral centroid (0..1)
                                    this.analyser.getFloatFrequencyData(freqData); // dB
                                    const nyq = this.audioContext.sampleRate / 2;
                                    let num = 0, den = 0;
                                    for (let i = 0; i < freqData.length; i++) {
                                        const mag = Math.pow(10, freqData[i] / 20); // dB->amp
                                        const freq = (i / freqData.length) * nyq;
                                        num += mag * freq;
                                        den += mag;
                                    }
                                    const centroid01 = den > 0 ? Math.min(1, (num / den) / nyq) : 0;

                                    this.onPlaybackAnalyze?.({ rms, centroid01, tSec: ts / 1000 });
                                }
                                requestAnimationFrame(tick);
                            };
                            requestAnimationFrame(tick);
                        } else {
                            // No analyzer or callback: connect directly and skip analysis loop
                            source.connect(dest);
                        }

                        this.currentAudioSource = source;
                        source.start(0);
                        source.onended = () => {
                            this.currentAudioSource = null;
                            this.onPlaybackEnd?.();     // e.g. reset current volume
                            resolve();
                        };
                    },
                    (error) => {
                        reject(error);
                    }
                );
            } catch (e) {
                reject(e);
            }
        });
    }

    setVolume(value) {
        this.volume = Math.max(0, Math.min(1, value));
        if (this.gainNode) {
            this.gainNode.gain.value = this.volume;
        }
    }

    stopAudio() {
        if (this.currentAudioSource) {
            try {
                this.currentAudioSource.stop();
            } catch (error) {
                console.error("Error stopping audio:", error);
            }
            this.currentAudioSource = null;
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
        this.processingQueue = false;
        if (this.scriptNode) {
            this.scriptNode.disconnect();
        }
        if (this.audioContext) {
            await this.audioContext.close();
            this.analyser = null;
            this.gainNode = null;
            this.isAudioPlaying = false;
        }
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "stop", session_id: sessionId }));
            this.ws.close();
        }
        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
        }
    }
}
