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
        this.isBacklogAudioPlaying = false;
        this.messageQueue = [];
        this.processingQueue = false;
        this.queueGeneration = 0;
        this.currentAudioSource = null;
        this.currentAudioFinalize = null;
        this.currentAudioMessage = null;
        this.playbackGeneration = 0;
        this.pcmAudio = new Map();
        this.currentPcmAudio = null;
        this.latestFaceUpdate = null;
        this.faceTimeout = null;
        this.currentFaceName = null;
        this.onResetFace = null;
        this.onMicrophoneDataSend = () => { };
        this.onResponseReceived = () => { };
        this.onPlaybackStart = null;
        this.onPlaybackAudio = null;
        this.isMicrophoneMuted = () => this.isAudioPlaying;
        this.getStartMetadata = () => null;
        this._userMuted = false;
        this.volume = 1.0;
        this.microphoneVolume = 1.0;
        this.gainNode = null;
        this.chatContextId = null;
    }

    async startListening(sessionId, userId) {
        const queueGeneration = ++this.queueGeneration;
        this.messageQueue.length = 0;
        this.pcmAudio.clear();
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
                const isPcm = msg.type === "chunk" && msg.metadata?.pcm_format;
                if (isPcm && msg.audio_data) {
                    const audio = this.pcmAudio.get(msg.metadata.audio_id);
                    if (!audio) return; // Its descriptor was canceled or never received.
                    this.onResponseReceived(msg);
                    this.appendPcmAudio(audio, msg.audio_data);
                    return;
                }
                this.onResponseReceived(msg);
                const isNod = message => message?.metadata?.nod === true;
                if (msg.type === "accepted"
                    || ((msg.type === "start" || msg.type === "chunk") && !isNod(msg))) {
                    this.filterQueue(message => !isNod(message));
                    // Cancel pending nods; audio that has started may finish.
                    const hasStarted = this.currentPcmAudio
                        ? this.currentPcmAudio.startedAt !== null
                            && this.audioContext.currentTime >= this.currentPcmAudio.startedAt
                        : !!this.currentAudioSource;
                    if (isNod(this.currentAudioMessage) && !hasStarted) {
                        this.stopAudio();
                    } else if (isNod(this.currentAudioMessage)) {
                        this.endPcmInput(isNod);
                    }
                }
                if (msg.type === "start" || msg.type === "chunk") {
                    if (msg.type === "start" && msg.context_id) {
                        this.chatContextId = msg.context_id;
                    }
                    if (isPcm) {
                        this.pcmAudio.set(msg.metadata.audio_id, {
                            message: msg, pending: [], sources: new Set(), receivedFrames: 0,
                            nextStartTime: 0, startedAt: null, started: false,
                            history: new Float32Array(0), animationFrame: null, startTimer: null,
                        });
                    }
                    this.messageQueue.push(msg);
                    if (!this.processingQueue) this.processQueue(queueGeneration);
                } else if (msg.type === "connected") {
                    userId = msg.user_id;   // Update userId (Created on server if not exists)
                    console.log(`Session: sessionId=${msg.session_id}, userId=${msg.user_id}, contextId=${msg.context_id}`);
                } else if (msg.type === "stop") {
                    // Response stops keep nods; Stop/disconnect ends all playback.
                    this.filterQueue(isNod);
                    const playingNod = isNod(this.currentAudioMessage);
                    this.endPcmInput(isNod);
                    if (playingNod) return;
                    this.stopAudio();
                    this.resetFace();
                } else if (msg.type === "final") {
                    if (msg.metadata?.interrupted) {
                        this.endPcmInput(message => message.transaction_id === msg.transaction_id);
                    }
                    console.log("Final response:", msg);
                }
            } catch (e) {
                console.error("Error parsing message:", e);
            }
        };

        this.ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
        this.ws.onclose = () => {
            if (queueGeneration !== this.queueGeneration) return;
            this.stopListening(sessionId).catch(error => {
                console.error("Error closing client:", error);
            });
        };

        // Create new AudioContext if needed
        if (!this.audioContext || this.audioContext.state === "closed") {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
        }
        await this.audioContext.resume();
        if (queueGeneration !== this.queueGeneration) return;
        console.log("AudioContext state:", this.audioContext.state);

        try {
            const micStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
            });
            if (queueGeneration !== this.queueGeneration) {
                micStream.getTracks().forEach(track => track.stop());
                return;
            }
            this.micStream = micStream;
            console.log("Microphone accepted.");
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            this.scriptNode = this.audioContext.createScriptProcessor(256, 1, 1);
            this.scriptNode.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    if (!this._userMuted && !this.isMicrophoneMuted()) {
                        let sum = 0;
                        for (let i = 0; i < inputData.length; i++) {
                            const sample = Math.max(
                                -1,
                                Math.min(1, inputData[i] * this.microphoneVolume),
                            );
                            sum += sample * sample;
                        }
                        const rms = Math.sqrt(sum / inputData.length);
                        this.onMicrophoneDataSend(rms);
                        const pcmBuffer = this.float32To16BitPCMBuffer(
                            inputData,
                            this.microphoneVolume,
                        );
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
            if (msg.audio_data || msg.metadata?.pcm_format) {
                try {
                    this.isAudioPlaying = true;
                    if (msg.metadata?.pcm_format) {
                        await this.playPcmAudio(this.pcmAudio.get(msg.metadata.audio_id));
                    } else {
                        await this.playAudioSync(msg.audio_data, msg);
                    }
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

    filterQueue(keep) {
        this.messageQueue = this.messageQueue.filter(message => {
            if (keep(message)) return true;
            this.pcmAudio.delete(message.metadata?.audio_id);
            return false;
        });
    }

    endPcmInput(matches) {
        // An interrupted sender may leave a partial audio. Drain only what arrived.
        for (const audio of this.pcmAudio.values()) {
            if (!matches(audio.message)) continue;
            audio.inputEnded = true;
            if (this.currentPcmAudio === audio && !audio.sources.size) {
                audio.finish(audio.receivedFrames === audio.message.metadata.audio_frame_count);
            }
        }
    }

    appendPcmAudio(audio, base64) {
        const { channels, sample_rate: rate } = audio.message.metadata.pcm_format;
        const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
        // The protocol supplies signed 16-bit LE PCM, split on whole frames.
        const buffer = this.audioContext.createBuffer(channels, bytes.length / (channels * 2), rate);
        const view = new DataView(bytes.buffer);
        for (let channel = 0; channel < channels; channel++) {
            const samples = buffer.getChannelData(channel);
            for (let i = 0; i < samples.length; i++) {
                samples[i] = view.getInt16((i * channels + channel) * 2, true) / 32768;
            }
        }
        audio.receivedFrames += buffer.length;
        if (this.currentPcmAudio === audio) this.schedulePcmBuffer(audio, buffer);
        else audio.pending.push(buffer);
    }

    playPcmAudio(audio) {
        this.stopAudio();
        this.currentPcmAudio = audio;
        this.currentAudioMessage = audio.message;
        audio.state = {
            message: audio.message,
            playbackId: audio.message.metadata.audio_id,
            durationSeconds: audio.message.metadata.audio_frame_count
                / audio.message.metadata.pcm_format.sample_rate,
        };
        return new Promise(resolve => {
            audio.finish = completed => {
                if (this.currentPcmAudio !== audio) return;
                clearTimeout(audio.startTimer);
                if (audio.animationFrame !== null) cancelAnimationFrame(audio.animationFrame);
                this.currentPcmAudio = null;
                this.currentAudioMessage = null;
                this.currentAudioFinalize = null;
                this.pcmAudio.delete(audio.message.metadata.audio_id);
                for (const chunk of audio.sources) {
                    chunk.source.onended = null;
                    chunk.source.stop();
                    chunk.source.disconnect();
                }
                audio.sources.clear();
                try {
                    if (audio.started) this.onPlaybackEnd?.({ ...audio.state, completed });
                } catch (error) {
                    console.error("Error handling playback end:", error);
                }
                resolve();
            };
            this.currentAudioFinalize = () => audio.finish(false);
            // Already-buffered audio can follow the preceding queue item immediately.
            audio.nextStartTime = this.audioContext.currentTime + (audio.pending.length ? 0 : 0.05);
            for (const buffer of audio.pending) this.schedulePcmBuffer(audio, buffer);
            audio.pending.length = 0;
            if (!audio.sources.size && (audio.inputEnded || audio.message.metadata.audio_frame_count === 0)) {
                audio.finish(audio.message.metadata.audio_frame_count === 0);
            }
        });
    }

    notifyPcmStart(audio) {
        if (audio.started || this.currentPcmAudio !== audio) return;
        audio.started = true;
        try {
            this.onPlaybackStart?.({ ...audio.state });
        } catch (error) {
            console.error("Error handling playback start:", error);
        }
    }

    schedulePcmBuffer(audio, buffer) {
        const now = this.audioContext.currentTime;
        const startedAt = audio.nextStartTime >= now ? audio.nextStartTime : now + 0.05;
        const history = audio.nextStartTime >= now ? audio.history : new Float32Array(0);
        const pcm = new Float32Array(history.length + buffer.length);
        pcm.set(history);
        pcm.set(buffer.getChannelData(0), history.length);
        audio.history = pcm.slice(-Math.ceil(buffer.sampleRate * 0.1));
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.gainNode || this.audioContext.destination);
        const chunk = { source, buffer, pcm, sampleOffset: history.length, startedAt };
        audio.sources.add(chunk);
        audio.nextStartTime = startedAt + buffer.duration;
        source.onended = () => {
            if (this.currentPcmAudio !== audio) return;
            this.notifyPcmStart(audio);
            audio.sources.delete(chunk);
            source.disconnect();
            if (!audio.sources.size
                && (audio.inputEnded
                    || audio.receivedFrames === audio.message.metadata.audio_frame_count)) {
                audio.finish(audio.receivedFrames === audio.message.metadata.audio_frame_count);
            }
        };
        source.start(startedAt);
        if (audio.startedAt === null) {
            audio.startedAt = startedAt;
            audio.startTimer = setTimeout(() => this.notifyPcmStart(audio), (startedAt - now) * 1000);
            this.startPcmPlaybackFrames(audio);
        }
    }

    startPcmPlaybackFrames(audio) {
        if (typeof this.onPlaybackAudio !== "function") return;
        const interval = 1000 / (this.playbackAudioHz || 30);
        let nextCallbackAt = null;
        const tick = timestamp => {
            if (this.currentPcmAudio !== audio) return;
            const tSec = this.audioContext.currentTime;
            if (nextCallbackAt === null || timestamp + 1 >= nextCallbackAt) {
                for (const chunk of audio.sources) {
                    if (tSec < chunk.startedAt || tSec >= chunk.startedAt + chunk.buffer.duration) continue;
                    this.onPlaybackAudio?.({
                        pcm: chunk.pcm, sampleRate: chunk.buffer.sampleRate,
                        samplePosition: chunk.sampleOffset + Math.floor((tSec - chunk.startedAt) * chunk.buffer.sampleRate),
                        tSec,
                    });
                    nextCallbackAt = nextCallbackAt === null ? timestamp + interval
                        : nextCallbackAt + Math.max(1, Math.ceil((timestamp + 1 - nextCallbackAt) / interval)) * interval;
                    break;
                }
            }
            audio.animationFrame = requestAnimationFrame(tick);
        };
        audio.animationFrame = requestAnimationFrame(tick);
    }

    playAudioSync(audioDataBase64, message = null) {
        this.stopAudio();
        const playbackGeneration = this.playbackGeneration;
        this.currentAudioMessage = message;
        return new Promise((resolve, reject) => {
            const playbackState = {
                message,
                playbackId: globalThis.crypto?.randomUUID?.()
                    || `${Date.now()}-${playbackGeneration}-${Math.random().toString(36).slice(2)}`,
                durationSeconds: 0,
            };
            let source = null;
            let playbackFinalized = false;
            const finishPlayback = (error = null, completed = false) => {
                if (playbackFinalized) return;
                playbackFinalized = true;
                const superseded = this.currentAudioFinalize !== finalizePlayback;
                if (!superseded) {
                    this.currentAudioSource = null;
                    this.currentAudioFinalize = null;
                    this.currentAudioMessage = null;
                }
                try {
                    if (source && !superseded) this.onPlaybackEnd?.({
                        ...playbackState,
                        completed,
                    });
                } catch (callbackError) {
                    console.error("Error handling playback end:", callbackError);
                } finally {
                    if (error) reject(error);
                    else resolve();
                }
            };
            const finalizePlayback = () => finishPlayback();
            // A canceled decode must release the queue before its callback returns.
            this.currentAudioFinalize = finalizePlayback;
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
                            finalizePlayback();
                            return;
                        }
                        source = this.audioContext.createBufferSource();
                        source.buffer = decodedData;

                        const dest = this.gainNode || this.audioContext.destination;
                        const playbackAudioCallback = typeof this.onPlaybackAudio === "function"
                            ? this.onPlaybackAudio
                            : null;
                        const playbackPcm = decodedData.getChannelData(0);
                        const playbackSampleRate = decodedData.sampleRate;
                        source.connect(dest);

                        this.currentAudioSource = source;
                        playbackState.durationSeconds = playbackPcm.length / playbackSampleRate;
                        source.onended = () => finishPlayback(null, true);
                        try {
                            this.onPlaybackStart?.({ ...playbackState });
                        } catch (callbackError) {
                            console.error("Error handling playback start:", callbackError);
                        }
                        if (playbackFinalized || playbackGeneration !== this.playbackGeneration) {
                            finalizePlayback();
                            return;
                        }
                        const startedAt = this.audioContext.currentTime;
                        try {
                            source.start(0);
                        } catch (error) {
                            finishPlayback(error);
                            return;
                        }

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
                    },
                    (error) => {
                        finishPlayback(playbackGeneration !== this.playbackGeneration ? null : error);
                    }
                );
            } catch (e) {
                finishPlayback(e);
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

    setMicrophoneVolume(value) {
        this.microphoneVolume = Math.max(0, Math.min(2, value));
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
        const source = this.currentAudioSource;
        this.currentAudioFinalize?.();
        if (source) {
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

    float32To16BitPCMBuffer(floatBuffer, gain = 1) {
        const len = floatBuffer.length;
        const buffer = new ArrayBuffer(len * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < len; i++) {
            let sample = floatBuffer[i] * gain;
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
        this.stopAudio();
        this.pcmAudio.clear();
        this.ws = null;
        if (ws) {
            ws.onopen = null;
            ws.onmessage = null;
            ws.onerror = null;
            ws.onclose = null;
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "stop", session_id: sessionId }));
            }
            if (ws.readyState === WebSocket.CONNECTING
                || ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        }
        if (this.scriptNode) {
            this.scriptNode.disconnect();
            this.scriptNode = null;
        }
        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
            this.micStream = null;
        }
        const audioContext = this.audioContext;
        this.audioContext = null;
        this.gainNode = null;
        this.isAudioPlaying = false;
        if (audioContext && audioContext.state !== "closed") await audioContext.close();
    }
}
