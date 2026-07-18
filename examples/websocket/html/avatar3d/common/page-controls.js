export function installPageControls({ ui, state, settingsHost, labels, onDisconnected }) {
    const micGlow = document.getElementById("micGlow");
    const chatButton = document.getElementById("chatBtn");
    const bargeInButton = document.getElementById("bargeInBtn");
    const interruptToggle = document.getElementById("interruptToggle");
    const volumeSlider = document.getElementById("volumeSlider");
    const volumeButton = document.getElementById("volumeBtn");
    const configButton = document.getElementById("configBtn");
    const settingsPanel = settingsHost?.element;

    chatButton.textContent = ui.isChatActive ? labels.stop : labels.start;
    bargeInButton.classList.toggle("active", interruptToggle.checked);
    volumeButton.textContent = labels.volume;

    ui.setMicGlow = (active) => {
        micGlow.classList.toggle("active", state.showMicGlow && active);
    };

    const onChatClick = () => {
        chatButton.textContent = ui.isChatActive ? labels.stop : labels.start;
        chatButton.classList.toggle("active", ui.isChatActive);
        if (!ui.isChatActive) onDisconnected();
    };
    chatButton.addEventListener("click", onChatClick);

    const onBargeInClick = () => {
        interruptToggle.checked = !interruptToggle.checked;
        interruptToggle.dispatchEvent(new Event("change"));
        bargeInButton.classList.toggle("active", interruptToggle.checked);
    };
    bargeInButton.addEventListener("click", onBargeInClick);

    const keepVolumeLabel = () => { volumeButton.textContent = labels.volume; };
    volumeSlider.addEventListener("input", keepVolumeLabel);

    const toggleSettings = () => settingsPanel?.classList.toggle("open");
    configButton.addEventListener("click", toggleSettings);

    const onKeyDown = (event) => {
        if (event.key === "Escape") toggleSettings();
    };
    document.addEventListener("keydown", onKeyDown);

    return {
        dispose() {
            chatButton.removeEventListener("click", onChatClick);
            bargeInButton.removeEventListener("click", onBargeInClick);
            volumeSlider.removeEventListener("input", keepVolumeLabel);
            configButton.removeEventListener("click", toggleSettings);
            document.removeEventListener("keydown", onKeyDown);
        },
    };
}
