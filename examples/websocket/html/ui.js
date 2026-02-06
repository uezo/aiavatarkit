/**
 * Shared UI helpers for AIAvatarKit WebSocket demo pages.
 *
 * Usage:
 *   const ui = new AvatarUI({ aiavatar, sessionId, camera });
 *   // Then wire up page-specific logic (onResponseReceived, lipsync, etc.)
 */
class AvatarUI {
    constructor({ aiavatar, userId, camera, onStop }) {
        this.aiavatar = aiavatar;
        this.sessionId = crypto.randomUUID();
        this.userId = userId || localStorage.getItem("userId") || "user01";
        this.camera = camera;
        this.onStop = onStop || (() => {});

        // State
        this.isChatActive = false;
        this.interruptEnabled = false;
        this.cameraEnabled = false;
        this.isServerProcessing = false;
        this.isBargeInBlocked = false;

        // Input level averaging
        this.rmsSum = 0;
        this.rmsCount = 0;
        this.lastUpdateTime = Date.now();

        // Message box
        this.currentAIText = "";
        this.currentUserText = "";
        this.speakerLabelUser = "User";
        this.speakerLabelAI = "AI";

        // DOM elements
        this.avatarFrame = document.getElementById("avatarFrame");
        this.inputLevelElement = document.getElementById("inputLevel");
        this.chatBtn = document.getElementById("chatBtn");
        this.interruptToggle = document.getElementById("interruptToggle");
        this.cameraToggle = document.getElementById("cameraToggle");
        this.volumeBtn = document.getElementById("volumeBtn");
        this.volumePopup = document.getElementById("volumePopup");
        this.volumeSlider = document.getElementById("volumeSlider");
        this.volumeValue = document.getElementById("volumeValue");
        this.volumeControl = document.getElementById("volumeControl");
        this.messageBox = document.getElementById("messageBox");
        this.messageSpeaker = document.getElementById("messageSpeaker");
        this.messageText = document.getElementById("messageText");
        this.toolStatus = document.getElementById("toolStatus");

        this._setupMicrophoneCallback();
        this._setupMicrophoneMute();
        this._setupChatButton();
        this._setupInterruptToggle();
        this._setupCameraToggle();
        this._setupVolumeControl();
    }

    // --- Microphone frame color & input level ---

    _setupMicrophoneCallback() {
        this.aiavatar.onMicrophoneDataSend = (rms) => {
            if (rms > 0.01) {
                this.avatarFrame.classList.add("mic-active");
            } else {
                this.avatarFrame.classList.remove("mic-active");
            }

            this.rmsSum += rms;
            this.rmsCount++;

            const currentTime = Date.now();
            if (currentTime - this.lastUpdateTime >= 500) {
                const avgRms = this.rmsSum / this.rmsCount;
                const dbLevel = avgRms > 0 ? 20 * Math.log10(avgRms) : -80;
                this.inputLevelElement.textContent = `${dbLevel.toFixed(1)} dB`;

                this.rmsSum = 0;
                this.rmsCount = 0;
                this.lastUpdateTime = currentTime;
            }
        };
    }

    // --- Microphone mute logic ---

    _setupMicrophoneMute() {
        this.aiavatar.isMicrophoneMuted = () => {
            if (this.aiavatar.isAudioPlaying) {
                this.isServerProcessing = false;
            }
            if (!this.aiavatar.isAudioPlaying && !this.isServerProcessing) {
                this.isBargeInBlocked = false;
            }

            if (this.isBargeInBlocked) {
                return true;
            }
            if (this.interruptEnabled) {
                return false;
            }
            return this.aiavatar.isAudioPlaying || this.isServerProcessing;
        };
    }

    // --- Start / Stop button ---

    _setupChatButton() {
        this.chatBtn.addEventListener("click", () => {
            if (this.isChatActive) {
                this.aiavatar.stopListening(this.sessionId);
                this.onStop();
                this.chatBtn.textContent = "Start";
                this.chatBtn.classList.remove("active");
            } else {
                this.aiavatar.startListening(this.sessionId, this.userId);
                this.chatBtn.textContent = "Stop";
                this.chatBtn.classList.add("active");
            }
            this.isChatActive = !this.isChatActive;
        });
    }

    // --- Barge-in toggle ---

    _setupInterruptToggle() {
        this.interruptToggle.addEventListener("change", () => {
            this.interruptEnabled = this.interruptToggle.checked;
        });
    }

    // --- Camera toggle ---

    _setupCameraToggle() {
        this.cameraToggle.addEventListener("change", async () => {
            if (this.cameraToggle.checked) {
                const initialized = await this.camera.init();
                if (initialized) {
                    this.cameraEnabled = true;
                } else {
                    this.cameraToggle.checked = false;
                }
            } else {
                this.camera.stop();
                this.cameraEnabled = false;
            }
        });
    }

    // --- Volume control ---

    _setupVolumeControl() {
        this.volumeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.volumePopup.classList.toggle("open");
        });

        document.addEventListener("click", (e) => {
            if (!this.volumeControl.contains(e.target)) {
                this.volumePopup.classList.remove("open");
            }
        });

        this.volumeSlider.addEventListener("input", () => {
            const val = parseInt(this.volumeSlider.value) / 100;
            this.aiavatar.setVolume(val);
            this.volumeValue.textContent = this.volumeSlider.value;
            this.volumeBtn.textContent = val === 0 ? "\u{1f507}" : val < 0.5 ? "\u{1f509}" : "\u{1f50a}";
        });
    }

    // --- Message box ---

    showMessage(speaker, text) {
        this.messageSpeaker.className = "message-speaker " + speaker;
        this.messageSpeaker.textContent = speaker === "user" ? this.speakerLabelUser : this.speakerLabelAI;
        this.messageText.textContent = text;
        this.messageBox.classList.remove("hidden");
        this.messageBox.classList.add("visible");
    }

    updateMessage(speaker, text, isPartial) {
        this.messageSpeaker.className = "message-speaker " + speaker;
        this.messageSpeaker.textContent = speaker === "user" ? this.speakerLabelUser : this.speakerLabelAI;

        if (isPartial) {
            if (speaker === "user") {
                this.currentUserText = text;
                this.messageText.textContent = this.currentUserText;
            } else {
                this.currentAIText += text;
                this.messageText.textContent = this.currentAIText;
            }
        } else {
            if (speaker === "user") {
                this.currentUserText = "";
            } else {
                this.currentAIText = "";
            }
            this.messageText.textContent = text;
        }

        this.messageBox.classList.remove("hidden");
        this.messageBox.classList.add("visible");
    }

    // --- Common response handling helpers ---

    /**
     * Handle response patterns (accepted, vision, hangup).
     * Call this from your page-specific onResponseReceived handler.
     */
    handleResponse(response) {
        // Save userId
        if (response.type == "connected") {
            localStorage.setItem("userId", response.user_id);
            this.userId = response.user_id;
        }

        // Update speaker labels from server
        if ((response.type === "connected" || response.type === "tool_call") && response.metadata) {
            if (response.metadata.username) {
                this.speakerLabelUser = response.metadata.username;
            }
            if (response.metadata.charactername) {
                this.speakerLabelAI = response.metadata.charactername;
            }
        }

        // Track server processing state
        if (response.type === "accepted") {
            this.isServerProcessing = true;
            if (response.metadata && response.metadata.block_barge_in) {
                this.isBargeInBlocked = true;
            }
        } else if (response.type === "final" || response.type === "canceled" || response.type === "error") {
            this.isServerProcessing = false;
        }

        // Vision / camera capture
        if (response.type === "vision" && response.metadata !== null) {
            if (response.metadata.source === "camera" && this.cameraEnabled) {
                this.camera.capture();
            }
        }

        // Disconnect
        if (response.text !== null && response.text.indexOf("[operation:hangup]") >= 0) {
            const waitForAudioEnd = () => {
                if (this.aiavatar.isAudioPlaying) {
                    setTimeout(waitForAudioEnd, 100);
                } else {
                    this.aiavatar.stopListening(this.sessionId);
                    this.inputLevelElement.textContent = "0.0 dB";
                }
            };
            setTimeout(waitForAudioEnd, 0);
        }

        // Message display
        if (response.type === "info" && response.metadata && response.metadata.partial_request_text) {
            this.updateMessage("user", response.metadata.partial_request_text, true);
        }
        if (response.type === "start" && response.metadata && response.metadata.recognized_text) {
            this.updateMessage("user", response.metadata.recognized_text, false);
        }
        if (response.type === "chunk" && response.voice_text) {
            this.updateMessage("ai", response.voice_text, true);
        }
        if (response.type === "final" && response.voice_text) {
            this.updateMessage("ai", response.voice_text, false);
        }
        if (response.type === "error") {
            this.updateMessage("ai", response.voice_text, false);
        }

        // Show tool status
        if (response.type == "tool_call") {
            console.log(`Tool Call: ${JSON.stringify(response.metadata.tool_call, null, 2)}`);
            this.toolStatus.textContent = `Tool Call: ${response.metadata.tool_call.name}`;
        } else if (response.type == "final" || response.type == "error") {
            this.toolStatus.textContent = "";
        }
    }
}
