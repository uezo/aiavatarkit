export function installMessageController({ aiavatar, ui, state, autoHideDelayMs = 10000 }) {
    let fullText = "";
    let displayedLength = 0;
    let timer = null;
    let responseComplete = false;
    let lastActivityAt = Date.now();
    let autoHidden = false;
    const messageBox = document.getElementById("messageBox");

    const delay = () => Math.max(1, 101 - state.messageSpeed);

    function stopTypewriter() {
        if (timer) clearTimeout(timer);
        timer = null;
    }

    function tick() {
        if (displayedLength >= fullText.length) {
            timer = null;
            return;
        }
        displayedLength += 1;
        ui.messageText.textContent = fullText.substring(0, displayedLength);
        if (displayedLength === 1) {
            ui.messageBox.classList.remove("hidden", "auto-hidden");
            ui.messageBox.classList.add("visible");
        }
        timer = setTimeout(tick, delay());
    }

    function startTypewriter() {
        if (!timer) timer = setTimeout(tick, delay());
    }

    function resetActivity() {
        lastActivityAt = Date.now();
        if (!autoHidden) return;
        autoHidden = false;
        ui.messageText.textContent = "";
        stopTypewriter();
        fullText = "";
        displayedLength = 0;
        responseComplete = false;
    }

    const originalUpdateMessage = ui.updateMessage.bind(ui);
    ui.updateMessage = (speaker, text, isPartial) => {
        resetActivity();
        if (speaker === "user" && !state.showUserText) return;
        if (speaker === "ai" && !state.showAIText) return;

        if (speaker === "ai" && isPartial) {
            if (responseComplete) {
                stopTypewriter();
                fullText = "";
                displayedLength = 0;
                responseComplete = false;
            }
            fullText += text;
            ui.currentAIText = fullText;
            ui.messageSpeaker.className = "message-speaker ai";
            ui.messageSpeaker.textContent = ui.speakerLabelAI;
            startTypewriter();
            return;
        }

        if (speaker === "ai") {
            responseComplete = true;
            if (!timer && fullText === "") originalUpdateMessage(speaker, text, isPartial);
            return;
        }

        stopTypewriter();
        fullText = "";
        displayedLength = 0;
        responseComplete = false;
        originalUpdateMessage(speaker, text, isPartial);
    };

    const originalShowMessage = ui.showMessage.bind(ui);
    ui.showMessage = (speaker, text) => {
        resetActivity();
        if (speaker === "user" && !state.showUserText) return;
        if (speaker === "ai" && !state.showAIText) return;
        originalShowMessage(speaker, text);
    };

    const interval = setInterval(() => {
        if (!state.autoHide || autoHidden) return;
        if (aiavatar.isAudioPlaying || timer) {
            lastActivityAt = Date.now();
            return;
        }
        if (Date.now() - lastActivityAt > autoHideDelayMs) {
            autoHidden = true;
            messageBox.classList.add("auto-hidden");
        }
    }, 1000);

    return {
        resetActivity,
        dispose() {
            clearInterval(interval);
            stopTypewriter();
            ui.updateMessage = originalUpdateMessage;
            ui.showMessage = originalShowMessage;
        },
    };
}
