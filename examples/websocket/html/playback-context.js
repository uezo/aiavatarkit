// Notify the server at audible chunk boundaries; microphone packets stay unchanged.
export function installPlaybackContext(client) {
    const previousOnPlaybackStart = client.onPlaybackStart;
    const previousOnPlaybackEnd = client.onPlaybackEnd;
    let active = null;
    let latestPlayback = null;
    let finalChunks = new WeakSet();
    let socket = client.ws;
    let queueGeneration = client.queueGeneration;
    let disposed = false;

    function syncConnection() {
        if (socket === client.ws && queueGeneration === client.queueGeneration) return;
        socket = client.ws;
        queueGeneration = client.queueGeneration;
        latestPlayback = null;
        finalChunks = new WeakSet();
    }

    function send(socket, sessionId, metadata) {
        if (client.ws !== socket || socket?.readyState !== 1) return false;
        try {
            socket.send(JSON.stringify({ type: "playback", session_id: sessionId, metadata }));
            return true;
        } catch (error) {
            console.error("Error sending playback context:", error);
            return false;
        }
    }

    function endPlayback(completed) {
        if (!active) return;
        const ended = active;
        active = null;
        if (!completed && latestPlayback === ended) latestPlayback = null;
        if (ended.reported) {
            send(ended.socket, ended.sessionId, {
                event: "end",
                playback_id: ended.playbackId,
                completed,
            });
        }
    }

    function notifyFinal(playback) {
        if (!playback?.reported || playback.finalReported
            || playback.queueGeneration !== client.queueGeneration) return;
        if (send(playback.socket, playback.sessionId, {
            event: "final",
            playback_id: playback.playbackId,
            transaction_id: playback.message.transaction_id,
        })) playback.finalReported = true;
    }

    function handleResponse(response) {
        if (disposed) return;
        syncConnection();
        if (response?.type === "stop") {
            latestPlayback = null;
            finalChunks = new WeakSet();
            return;
        }
        if (response?.type !== "final" || response.metadata?.interrupted
            || response.metadata?.error || response.metadata?.nod === true
            || typeof response.session_id !== "string" || !response.session_id
            || typeof response.transaction_id !== "string" || !response.transaction_id) return;

        const matches = message => message?.session_id === response.session_id
            && message.transaction_id === response.transaction_id
            && message.metadata?.nod !== true
            && Boolean(message.audio_data || message.metadata?.audio_frame_count > 0);
        // Include non-text audio: it must not make an earlier spoken chunk look final.
        const queued = client.messageQueue?.slice().reverse().find(matches);
        const current = !client.isBacklogAudioPlaying && matches(client.currentAudioMessage)
            ? client.currentAudioMessage : null;
        const latest = matches(latestPlayback?.message) ? latestPlayback.message : null;
        const message = queued || current || latest;
        if (!message) return;
        finalChunks.add(message);
        if (latestPlayback?.message === message) notifyFinal(latestPlayback);
    }

    function startPlayback(state) {
        if (disposed || active?.playbackId === state?.playbackId) return;
        syncConnection();
        endPlayback(false);
        latestPlayback = null;
        const message = state?.message;
        if (client.isBacklogAudioPlaying || message?.metadata?.nod === true
            || typeof message?.session_id !== "string" || !message.session_id
            || typeof state?.playbackId !== "string" || !state.playbackId
            || !Number.isFinite(state.durationSeconds) || state.durationSeconds <= 0) return;

        active = latestPlayback = {
            socket: client.ws, queueGeneration: client.queueGeneration,
            sessionId: message.session_id, playbackId: state.playbackId,
            message, reported: false, finalReported: false,
        };
        if (typeof message.voice_text === "string" && message.voice_text.trim()) {
            active.reported = send(active.socket, message.session_id, {
                event: "start",
                playback_id: state.playbackId,
                text: message.voice_text,
                duration_seconds: state.durationSeconds,
                ...(message.transaction_id != null ? { transaction_id: message.transaction_id } : {}),
            });
            if (finalChunks.has(message)) notifyFinal(active);
        }
    }

    const onPlaybackStart = function (state) {
        try {
            startPlayback(state);
        } finally {
            previousOnPlaybackStart?.call(client, state);
        }
    };
    const onPlaybackEnd = function (state) {
        try {
            if (active && active.playbackId === state?.playbackId) {
                endPlayback(state.completed === true);
            }
        } finally {
            previousOnPlaybackEnd?.call(client, state);
        }
    };
    client.onPlaybackStart = onPlaybackStart;
    client.onPlaybackEnd = onPlaybackEnd;

    return {
        handleResponse,
        dispose() {
            if (disposed) return;
            disposed = true;
            endPlayback(false);
            latestPlayback = null;
            finalChunks = new WeakSet();
            if (client.onPlaybackStart === onPlaybackStart) client.onPlaybackStart = previousOnPlaybackStart;
            if (client.onPlaybackEnd === onPlaybackEnd) client.onPlaybackEnd = previousOnPlaybackEnd;
        },
    };
}
