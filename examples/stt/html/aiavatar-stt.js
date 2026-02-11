/**
 * SpeechRecognitionClient - WebSocket client for streaming speech recognition
 *
 * Usage:
 *   const client = new SpeechRecognitionClient({
 *     webSocketUrl: "ws://localhost:8000/ws/stt",
 *     sampleRate: 16000
 *   });
 *
 *   client.onPartialResult = (text) => console.log("Partial:", text);
 *   client.onFinalResult = (text, metadata) => console.log("Final:", text);
 *
 *   await client.start("session-123");
 *   // ... later
 *   await client.stop();
 */
class SpeechRecognitionClient {
    /**
     * @param {Object} options
     * @param {string} options.webSocketUrl - WebSocket server URL
     * @param {number} [options.sampleRate=16000] - Audio sample rate
     * @param {number} [options.bufferSize=512] - Audio buffer size (matches Silero VAD chunk size)
     * @param {string} [options.workletUrl="audio-processor.js"] - AudioWorklet processor URL
     */
    constructor({ webSocketUrl, sampleRate = 16000, bufferSize = 512, workletUrl = "audio-processor.js" }) {
        this.webSocketUrl = webSocketUrl;
        this.sampleRate = sampleRate;
        this.bufferSize = bufferSize;
        this.workletUrl = workletUrl;

        // Internal state
        this.ws = null;
        this.audioContext = null;
        this.workletNode = null;
        this.micStream = null;
        this.sessionId = null;
        this.isConnected = false;

        // Callbacks (override these)
        this.onConnected = (sessionId) => {};
        this.onPartialResult = (text, metadata) => {};
        this.onFinalResult = (text, metadata) => {};
        this.onVoiced = () => {};  // Called when voice activity is detected
        this.onError = (error, metadata) => {};
        this.onDisconnected = () => {};
        this.onMicrophoneData = (rms) => {};  // Called when mic data is sent
    }

    /**
     * Start listening and connect to WebSocket server
     * @param {string} sessionId - Unique session identifier
     * @returns {Promise<void>}
     */
    async start(sessionId) {
        this.sessionId = sessionId;

        // Setup WebSocket
        this.ws = new WebSocket(this.webSocketUrl);

        this.ws.onopen = () => {
            console.log(`[SpeechRecognitionClient] Connected to ${this.webSocketUrl}`);
            this._sendMessage({
                type: "start",
                session_id: sessionId
            });
        };

        this.ws.onmessage = (event) => {
            this._handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
            console.error("[SpeechRecognitionClient] WebSocket error:", error);
            this.onError(error, { type: "websocket" });
        };

        this.ws.onclose = () => {
            console.log("[SpeechRecognitionClient] WebSocket closed");
            this.isConnected = false;
            this.onDisconnected();
        };

        // Setup AudioContext
        await this._setupAudio();
    }

    /**
     * Stop listening and disconnect
     * @returns {Promise<void>}
     */
    async stop() {
        // Stop AudioWorklet
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
            this.micStream = null;
        }

        // Close AudioContext
        if (this.audioContext && this.audioContext.state !== "closed") {
            await this.audioContext.close();
            this.audioContext = null;
        }

        // Close WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this._sendMessage({
                type: "stop",
                session_id: this.sessionId
            });
            this.ws.close();
        }
        this.ws = null;

        this.isConnected = false;
        this.sessionId = null;
    }

    /**
     * Check if client is currently connected
     * @returns {boolean}
     */
    get connected() {
        return this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    // ========== Private Methods ==========

    async _setupAudio() {
        // Create AudioContext
        if (!this.audioContext || this.audioContext.state === "closed") {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
        }
        await this.audioContext.resume();
        console.log("[SpeechRecognitionClient] AudioContext state:", this.audioContext.state);

        // Load AudioWorklet module
        try {
            await this.audioContext.audioWorklet.addModule(this.workletUrl);
            console.log("[SpeechRecognitionClient] AudioWorklet module loaded");
        } catch (err) {
            console.error("[SpeechRecognitionClient] Failed to load AudioWorklet:", err);
            this.onError(err, { type: "worklet" });
            throw err;
        }

        // Get microphone access
        try {
            this.micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    channelCount: 1,
                    sampleRate: this.sampleRate
                }
            });
            console.log("[SpeechRecognitionClient] Microphone access granted");

            // Create audio processing pipeline with AudioWorklet
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            this.workletNode = new AudioWorkletNode(this.audioContext, "audio-capture-processor", {
                processorOptions: { bufferSize: this.bufferSize }
            });

            // Handle messages from AudioWorklet
            this.workletNode.port.onmessage = (event) => {
                if (!this.connected) return;

                const { audioData, rms } = event.data;
                this.onMicrophoneData(rms);

                // Convert to 16-bit PCM and send
                const pcmBuffer = this._float32ToPCM16(audioData);
                const base64Data = this._arrayBufferToBase64(pcmBuffer);

                this._sendMessage({
                    type: "data",
                    session_id: this.sessionId,
                    audio_data: base64Data
                });
            };

            source.connect(this.workletNode);
            // Connect to destination (required for processing to run)
            this.workletNode.connect(this.audioContext.destination);

        } catch (err) {
            console.error("[SpeechRecognitionClient] Microphone error:", err);
            this.onError(err, { type: "microphone" });
            throw err;
        }
    }

    _handleMessage(data) {
        try {
            const msg = JSON.parse(data);

            switch (msg.type) {
                case "connected":
                    this.isConnected = true;
                    console.log(`[SpeechRecognitionClient] Session connected: ${msg.session_id}`);
                    this.onConnected(msg.session_id);
                    break;

                case "partial":
                    this.onPartialResult(msg.text, msg.metadata);
                    break;

                case "final":
                    this.onFinalResult(msg.text, msg.metadata);
                    break;

                case "voiced":
                    this.onVoiced();
                    break;

                case "error":
                    console.error("[SpeechRecognitionClient] Server error:", msg.metadata);
                    this.onError(new Error(msg.metadata?.error || "Unknown error"), msg.metadata);
                    break;

                default:
                    console.log("[SpeechRecognitionClient] Unknown message type:", msg.type);
            }
        } catch (e) {
            console.error("[SpeechRecognitionClient] Error parsing message:", e);
        }
    }

    _sendMessage(msg) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(msg));
        }
    }

    _float32ToPCM16(floatBuffer) {
        const len = floatBuffer.length;
        const buffer = new ArrayBuffer(len * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < len; i++) {
            let sample = Math.max(-1, Math.min(1, floatBuffer[i]));
            const intSample = sample < 0 ? sample * 32768 : sample * 32767;
            view.setInt16(i * 2, intSample, true);  // little-endian
        }
        return buffer;
    }

    _arrayBufferToBase64(buffer) {
        let binary = "";
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}

// Export for module systems (optional)
if (typeof module !== "undefined" && module.exports) {
    module.exports = SpeechRecognitionClient;
}
