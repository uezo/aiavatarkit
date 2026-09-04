function bytesFromBase64(value) {
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    return bytes;
}

function dataUrlToBlob(dataUrl) {
    if (!dataUrl) return null;
    const comma = dataUrl.indexOf(",");
    if (comma < 0) return null;
    const metadata = dataUrl.slice(0, comma);
    const mimeType = metadata.match(/^data:([^;,]+)/i)?.[1] || "application/octet-stream";
    const payload = dataUrl.slice(comma + 1);
    const bytes = metadata.includes(";base64")
        ? bytesFromBase64(payload)
        : new TextEncoder().encode(decodeURIComponent(payload));
    return new Blob([bytes], { type: mimeType });
}

function writeAscii(view, offset, value) {
    for (let index = 0; index < value.length; index++) view.setUint8(offset + index, value.charCodeAt(index));
}

function pcmToWav(bytes, { sample_rate, channels, sample_width }) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    const channelCount = Number(channels) || 1;
    const sampleWidth = Number(sample_width) || 2;
    const sampleRate = Number(sample_rate) || 16000;
    writeAscii(view, 0, "RIFF");
    view.setUint32(4, 36 + bytes.byteLength, true);
    writeAscii(view, 8, "WAVEfmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channelCount, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * channelCount * sampleWidth, true);
    view.setUint16(32, channelCount * sampleWidth, true);
    view.setUint16(34, sampleWidth * 8, true);
    writeAscii(view, 36, "data");
    view.setUint32(40, bytes.byteLength, true);
    return new Blob([header, bytes], { type: "audio/wav" });
}

function responseAudioToBlob(response) {
    if (!response.audio_data) return null;
    const bytes = bytesFromBase64(response.audio_data);
    const isWav = bytes.length >= 12
        && String.fromCharCode(...bytes.slice(0, 4)) === "RIFF"
        && String.fromCharCode(...bytes.slice(8, 12)) === "WAVE";
    if (isWav) return new Blob([bytes], { type: "audio/wav" });
    const pcmFormat = response.metadata?.pcm_format;
    return pcmFormat ? pcmToWav(bytes, pcmFormat) : new Blob([bytes], { type: "audio/wav" });
}

function entryId() {
    const suffix = typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).slice(2);
    return `${Date.now()}-${suffix}`;
}

function visibleRequestText(response) {
    const metadata = response.metadata || {};
    const text = String(metadata.recognized_text || metadata.request_text || "").trim();
    return text.startsWith("$") ? "" : text;
}

function isQuotaError(error) {
    return error?.name === "QuotaExceededError" || error?.cause?.name === "QuotaExceededError";
}

class BacklogView {
    constructor({ maxEntries }) {
        this.maxEntries = maxEntries;
        this.button = document.getElementById("backlogBtn");
        this.overlay = document.getElementById("backlogOverlay");
        this.closeButton = document.getElementById("backlogClose");
        this.list = document.getElementById("backlogList");
        this.empty = document.getElementById("backlogEmpty");
        this.count = document.getElementById("backlogCount");
        this.imageUrls = [];
        this.playButtons = new Map();
        this.previouslyFocused = null;
        this.handlers = {};

        this.onButtonClick = () => this.handlers.open?.();
        this.onCloseClick = () => this.handlers.close?.();
        this.onOverlayClick = (event) => {
            if (event.target === this.overlay) this.handlers.close?.();
        };
        this.onKeyDown = (event) => {
            if (event.key === "Escape" && !this.overlay?.hidden) this.handlers.close?.();
        };
        this.button?.addEventListener("click", this.onButtonClick);
        this.closeButton?.addEventListener("click", this.onCloseClick);
        this.overlay?.addEventListener("click", this.onOverlayClick);
        document.addEventListener("keydown", this.onKeyDown);
    }

    bind(handlers) {
        this.handlers = handlers;
    }

    render(entries) {
        if (!this.list) return;
        for (const url of this.imageUrls) URL.revokeObjectURL(url);
        this.imageUrls = [];
        this.playButtons.clear();
        this.list.replaceChildren();

        if (this.count) this.count.textContent = `${entries.length} / ${this.maxEntries}`;
        if (!entries.length) {
            if (this.empty) this.list.appendChild(this.empty);
            return;
        }

        for (const entry of entries) {
            const item = document.createElement("article");
            item.className = `backlog-entry ${entry.role}`;

            const header = document.createElement("header");
            header.className = "backlog-entry-header";
            const speaker = document.createElement("span");
            speaker.className = "backlog-speaker";
            speaker.textContent = entry.speaker || (entry.role === "user" ? "User" : "AI");
            const time = document.createElement("time");
            time.className = "backlog-time";
            time.dateTime = new Date(entry.createdAt).toISOString();
            time.textContent = new Date(entry.createdAt).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
            });
            header.append(speaker, time);

            const body = document.createElement("div");
            body.className = "backlog-entry-body message-inner";
            if (entry.text) {
                const text = document.createElement("p");
                text.className = "backlog-text";
                text.textContent = entry.text;
                body.appendChild(text);
            }
            if (entry.image instanceof Blob) {
                const image = document.createElement("img");
                const url = URL.createObjectURL(entry.image);
                this.imageUrls.push(url);
                image.className = "backlog-image";
                image.src = url;
                image.alt = "Attached image";
                image.loading = "lazy";
                body.appendChild(image);
            }
            if (entry.role === "ai" && entry.audioChunks?.length) {
                const play = document.createElement("button");
                play.type = "button";
                play.className = "backlog-play";
                play.textContent = "▶ PLAY";
                play.setAttribute("aria-label", `Play audio from ${speaker.textContent}`);
                play.addEventListener("click", () => this.handlers.play?.(entry.id));
                this.playButtons.set(entry.id, play);
                body.appendChild(play);
            }
            item.append(header, body);
            this.list.appendChild(item);
        }
    }

    setPlaying(entryId) {
        for (const [id, button] of this.playButtons) {
            const active = id === entryId;
            button.classList.toggle("active", active);
            button.textContent = active ? "■ STOP" : "▶ PLAY";
        }
    }

    open() {
        if (!this.overlay) return;
        this.previouslyFocused = document.activeElement;
        this.overlay.hidden = false;
        this.button?.classList.add("active");
        this.closeButton?.focus();
        if (this.list) this.list.scrollTop = this.list.scrollHeight;
    }

    close() {
        if (!this.overlay) return;
        this.overlay.hidden = true;
        this.button?.classList.remove("active");
        this.previouslyFocused?.focus?.();
        this.previouslyFocused = null;
    }

    dispose() {
        this.button?.removeEventListener("click", this.onButtonClick);
        this.closeButton?.removeEventListener("click", this.onCloseClick);
        this.overlay?.removeEventListener("click", this.onOverlayClick);
        document.removeEventListener("keydown", this.onKeyDown);
        for (const url of this.imageUrls) URL.revokeObjectURL(url);
        this.imageUrls = [];
    }
}

export class BacklogController {
    constructor({ aiavatar, ui, store, view, maxEntries = 100, audioFactory = (url) => new Audio(url) }) {
        this.aiavatar = aiavatar;
        this.ui = ui;
        this.store = store;
        this.view = view;
        this.maxEntries = Number.isInteger(maxEntries) && maxEntries > 0 ? maxEntries : 100;
        this.audioFactory = audioFactory;
        this.entries = [];
        this.contextId = null;
        this.pendingTurn = null;
        this.playbackGeneration = 0;
        this.playingEntryId = null;
        this.currentAudio = null;
        this.currentAudioFinish = null;
        this.currentAudioUrl = null;
        this.view?.bind({
            open: () => this.open(),
            close: () => this.close(),
            play: (id) => void this.togglePlayback(id),
        });
        this.ready = this.initialize();
    }

    async initialize() {
        try {
            const saved = await this.store.load();
            this.contextId = saved.contextId;
            this.entries = saved.entries.slice(-this.maxEntries);
        } catch (error) {
            console.warn("Could not restore backlog:", error);
        }
        this.render();
    }

    render() {
        this.view?.render(this.entries);
        this.view?.setPlaying(this.playingEntryId);
    }

    stageUser({ text = "", imageDataUrl = null }) {
        this.pendingTurn = {
            started: false,
            contextId: this.aiavatar.chatContextId || null,
            user: {
                text: String(text || "").trim(),
                image: dataUrlToBlob(imageDataUrl),
                speaker: this.ui.speakerLabelUser || "User",
            },
            aiText: "",
            audioChunks: [],
            aiSpeaker: this.ui.speakerLabelAI || "AI",
            startedAt: Date.now(),
        };
    }

    handleResponse(response) {
        if (response.type === "accepted") this.stopPlayback();

        if (response.type === "start") {
            const requestText = visibleRequestText(response);
            if (!this.pendingTurn || this.pendingTurn.started) {
                this.pendingTurn = {
                    started: true,
                    contextId: response.context_id || null,
                    user: {
                        text: requestText,
                        image: null,
                        speaker: this.ui.speakerLabelUser || "User",
                    },
                    aiText: "",
                    audioChunks: [],
                    aiSpeaker: this.ui.speakerLabelAI || "AI",
                    startedAt: Date.now(),
                };
            } else {
                this.pendingTurn.started = true;
                this.pendingTurn.contextId = response.context_id || this.pendingTurn.contextId;
                if (!this.pendingTurn.user.text && requestText) this.pendingTurn.user.text = requestText;
            }
            return;
        }

        if (response.type === "chunk") {
            if (!this.pendingTurn) {
                this.pendingTurn = {
                    started: true,
                    contextId: response.context_id || null,
                    user: { text: "", image: null, speaker: this.ui.speakerLabelUser || "User" },
                    aiText: "",
                    audioChunks: [],
                    aiSpeaker: this.ui.speakerLabelAI || "AI",
                    startedAt: Date.now(),
                };
            }
            if (response.context_id) this.pendingTurn.contextId = response.context_id;
            if (response.voice_text) this.pendingTurn.aiText += response.voice_text;
            try {
                const audio = responseAudioToBlob(response);
                if (audio) this.pendingTurn.audioChunks.push(audio);
            } catch (error) {
                console.warn("Could not retain response audio:", error);
            }
            return;
        }

        if (response.type === "final") return this.commitFinal(response);
        if (["error", "canceled", "cancelled"].includes(response.type)) this.pendingTurn = null;
    }

    async commitFinal(response) {
        const turn = this.pendingTurn;
        this.pendingTurn = null;
        if (!turn) return;

        const contextId = response.context_id || turn.contextId || this.aiavatar.chatContextId;
        if (!contextId) return;
        const aiText = String(response.voice_text || turn.aiText || "").trim();
        const createdAt = Date.now();
        const newEntries = [];
        if (turn.user.text || turn.user.image) {
            newEntries.push({
                id: entryId(),
                contextId,
                role: "user",
                speaker: turn.user.speaker,
                text: turn.user.text,
                image: turn.user.image,
                audioChunks: [],
                createdAt: turn.startedAt,
            });
        }
        if (aiText || turn.audioChunks.length) {
            newEntries.push({
                id: entryId(),
                contextId,
                role: "ai",
                speaker: turn.aiSpeaker,
                text: aiText,
                image: null,
                audioChunks: turn.audioChunks,
                createdAt,
            });
        }
        if (!newEntries.length) return;

        if (this.contextId && this.contextId !== contextId) this.entries = [];
        this.contextId = contextId;
        this.entries = [...this.entries, ...newEntries].slice(-this.maxEntries);
        this.render();

        try {
            await this.store.appendTurn(contextId, newEntries);
        } catch (error) {
            if (!isQuotaError(error)) {
                console.warn("Could not save backlog:", error);
                return;
            }
            try {
                await this.store.removeOldest(Math.max(10, Math.ceil(this.maxEntries / 5)));
                await this.store.appendTurn(contextId, newEntries);
                const saved = await this.store.load();
                this.entries = saved.entries.slice(-this.maxEntries);
                this.render();
            } catch (retryError) {
                console.warn("Could not save backlog after freeing storage:", retryError);
            }
        }
    }

    open() {
        this.view?.open();
    }

    close() {
        this.stopPlayback();
        this.view?.close();
    }

    async togglePlayback(id) {
        if (this.playingEntryId === id) {
            this.stopPlayback();
            return;
        }
        const entry = this.entries.find((candidate) => candidate.id === id);
        if (!entry?.audioChunks?.length || this.aiavatar.isAudioPlaying || this.ui.isServerProcessing) return;

        this.stopPlayback();
        const generation = ++this.playbackGeneration;
        this.playingEntryId = id;
        this.aiavatar.isBacklogAudioPlaying = true;
        this.view?.setPlaying(id);
        try {
            for (const blob of entry.audioChunks) {
                if (generation !== this.playbackGeneration) break;
                await this.playBlob(blob, generation);
            }
        } catch (error) {
            if (generation === this.playbackGeneration) console.warn("Could not play backlog audio:", error);
        } finally {
            if (generation === this.playbackGeneration) this.stopPlayback();
        }
    }

    playBlob(blob, generation) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(blob);
            const audio = this.audioFactory(url);
            let settled = false;
            const finish = (error = null) => {
                if (settled) return;
                settled = true;
                audio.removeEventListener?.("ended", onEnded);
                audio.removeEventListener?.("error", onError);
                if (this.currentAudio === audio) {
                    this.currentAudio = null;
                    this.currentAudioFinish = null;
                    this.currentAudioUrl = null;
                }
                URL.revokeObjectURL(url);
                if (error) reject(error);
                else resolve();
            };
            const onEnded = () => finish();
            const onError = () => finish(new Error("Audio playback failed"));
            this.currentAudio = audio;
            this.currentAudioFinish = () => finish();
            this.currentAudioUrl = url;
            audio.volume = this.aiavatar.volume;
            audio.addEventListener("ended", onEnded, { once: true });
            audio.addEventListener("error", onError, { once: true });
            if (generation !== this.playbackGeneration) {
                finish();
                return;
            }
            Promise.resolve(audio.play()).catch((error) => finish(error));
        });
    }

    stopPlayback() {
        this.playbackGeneration += 1;
        this.currentAudio?.pause?.();
        this.currentAudioFinish?.();
        this.currentAudio = null;
        this.currentAudioFinish = null;
        this.currentAudioUrl = null;
        this.playingEntryId = null;
        this.aiavatar.isBacklogAudioPlaying = false;
        this.view?.setPlaying(null);
    }

    dispose() {
        this.stopPlayback();
        this.view?.dispose();
    }
}

export function installBacklog({ aiavatar, ui, store, maxEntries = 100 }) {
    const view = new BacklogView({ maxEntries });
    return new BacklogController({ aiavatar, ui, store, view, maxEntries });
}
