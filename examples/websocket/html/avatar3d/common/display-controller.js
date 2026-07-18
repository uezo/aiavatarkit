function sectionHeading(text) {
    const heading = document.createElement("div");
    heading.style.cssText = "font-size:11px;color:#999;margin-bottom:6px";
    heading.textContent = text;
    return heading;
}

function createToggle(labelText, checked, onChange) {
    const row = document.createElement("label");
    row.style.cssText = "display:flex;align-items:center;gap:10px;cursor:pointer";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.className = "toggle-switch";
    input.checked = checked;
    input.addEventListener("change", () => onChange(input.checked));
    const label = document.createElement("span");
    label.style.cssText = "font-size:12px;color:#666";
    label.textContent = labelText;
    row.append(input, label);
    return { row, input };
}

export class DisplayController {
    constructor({ aiavatar, ui, settingsHost, blobStore, config, persistence }) {
        this.aiavatar = aiavatar;
        this.ui = ui;
        this.settingsHost = settingsHost;
        this.blobStore = blobStore;
        this.config = config;
        this.persistence = persistence;
        this.defaults = {
            messageBoxOpacity: config.messageBoxOpacity,
            characterName: config.characterName,
            userName: config.userName,
            showUserText: config.showUserText,
            showAIText: config.showAIText,
            showMicGlow: config.showMicGlow,
            showMenu: config.showMenu,
            autoHide: config.autoHide,
            messageSpeed: config.messageSpeed,
        };
        this.state = { ...this.defaults };
        this.backgroundUrl = "";
        this.backgroundIsObjectUrl = false;
        this.connectionView = null;

        this.loadPersistedState();
        this.applyState();
        this.installSettingsTab();
        this.restoreBackground();
    }

    loadPersistedState() {
        if (!this.persistence.enabled || !this.persistence.restoreUserSettings) return;
        try {
            const saved = JSON.parse(localStorage.getItem(this.persistence.displayKey) || "{}");
            const legacyMap = {
                msgBoxOpacity: "messageBoxOpacity",
                charName: "characterName",
                userName: "userName",
                showUserText: "showUserText",
                showAIText: "showAIText",
                showMicGlow: "showMicGlow",
                showVnMenu: "showMenu",
                autoHideMsgBox: "autoHide",
                msgSpeed: "messageSpeed",
            };
            for (const [savedKey, stateKey] of Object.entries(legacyMap)) {
                if (Object.prototype.hasOwnProperty.call(saved, savedKey)) this.state[stateKey] = saved[savedKey];
            }
        } catch (error) {
            console.warn("Could not restore display settings:", error);
        }
    }

    saveState() {
        if (!this.persistence.enabled) return;
        localStorage.setItem(this.persistence.displayKey, JSON.stringify({
            msgBoxOpacity: this.state.messageBoxOpacity,
            charName: this.state.characterName,
            userName: this.state.userName,
            showUserText: this.state.showUserText,
            showAIText: this.state.showAIText,
            showMicGlow: this.state.showMicGlow,
            showVnMenu: this.state.showMenu,
            autoHideMsgBox: this.state.autoHide,
            msgSpeed: this.state.messageSpeed,
        }));
    }

    applyState() {
        const inner = document.querySelector(".message-inner");
        if (inner) inner.style.opacity = this.state.messageBoxOpacity / 100;
        const menu = document.querySelector(".vn-menu");
        if (menu) menu.style.display = this.state.showMenu ? "flex" : "none";
        const messageBox = document.getElementById("messageBox");
        messageBox.classList.toggle("auto-hidden", this.state.autoHide);
        this.ui.speakerLabelAI = this.state.characterName || "AI";
        this.ui.speakerLabelUser = this.state.userName || "User";
        if (!this.state.showMicGlow) document.getElementById("micGlow").classList.remove("active");
    }

    async restoreBackground() {
        try {
            const blob = await this.blobStore.get(this.persistence.backgroundKey);
            if (blob) this.setBackgroundUrl(URL.createObjectURL(blob), true);
        } catch (error) {
            console.warn("Could not restore background image:", error);
        }

        if (!this.persistence.legacyBackgroundKey) return;
        try {
            const oldDataUrl = localStorage.getItem(this.persistence.legacyBackgroundKey);
            if (!oldDataUrl) return;
            const blob = await fetch(oldDataUrl).then((response) => response.blob());
            await this.blobStore.put(this.persistence.backgroundKey, blob);
            localStorage.removeItem(this.persistence.legacyBackgroundKey);
        } catch (error) {
            console.warn("Could not migrate background image:", error);
        }
    }

    setBackgroundUrl(url, isObjectUrl = false) {
        if (this.backgroundIsObjectUrl && this.backgroundUrl) URL.revokeObjectURL(this.backgroundUrl);
        this.backgroundUrl = url;
        this.backgroundIsObjectUrl = isObjectUrl;
        const layer = document.getElementById("bgLayer");
        if (!url) {
            layer.replaceChildren();
            return;
        }
        let image = layer.querySelector("img");
        if (!image) {
            image = document.createElement("img");
            layer.appendChild(image);
        }
        image.src = url;
    }

    async storeBackground(blob) {
        await this.blobStore.put(this.persistence.backgroundKey, blob);
        this.setBackgroundUrl(URL.createObjectURL(blob), true);
    }

    installSettingsTab() {
        this.settingsHost.addTab("UI", (panel) => this.buildSettings(panel), { position: 2 });
        this.settingsHost.onTabReset("UI", () => this.reset());
    }

    buildSettings(panel) {
        const messageSection = document.createElement("div");
        messageSection.appendChild(sectionHeading("Message box"));

        const sliders = document.createElement("div");
        sliders.className = "settings-sliders";
        this.addSlider(sliders, {
            label: "Opacity",
            min: 0,
            max: 100,
            value: this.state.messageBoxOpacity,
            format: (value) => `${value}%`,
            onInput: (value) => {
                this.state.messageBoxOpacity = value;
                this.applyState();
                this.saveState();
            },
        });
        this.addSlider(sliders, {
            label: "Speed",
            min: 1,
            max: 100,
            value: this.state.messageSpeed,
            format: String,
            onInput: (value) => {
                this.state.messageSpeed = value;
                this.saveState();
            },
        });
        messageSection.appendChild(sliders);

        const toggles = document.createElement("div");
        toggles.style.cssText = "margin-top:8px;display:flex;flex-direction:column;gap:10px";
        const toggleDefinitions = [
            ["Show user speech", "showUserText"],
            ["Show AI speech", "showAIText"],
            ["Auto-hide", "autoHide"],
            ["Show menu buttons", "showMenu"],
        ];
        for (const [label, key] of toggleDefinitions) {
            const toggle = createToggle(label, this.state[key], (value) => {
                this.state[key] = value;
                this.applyState();
                this.saveState();
            });
            toggles.appendChild(toggle.row);
        }
        messageSection.appendChild(toggles);
        panel.appendChild(messageSection);

        panel.appendChild(this.buildSpeakerSection());
        panel.appendChild(this.buildMicrophoneSection());
        panel.appendChild(this.buildBackgroundSection());
        panel.appendChild(this.buildConnectionSection());
    }

    addSlider(container, { label, min, max, value, format, onInput }) {
        const labelElement = document.createElement("span");
        labelElement.textContent = label;
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = min;
        slider.max = max;
        slider.value = value;
        const display = document.createElement("span");
        display.textContent = format(value);
        slider.addEventListener("input", () => {
            const nextValue = Number.parseInt(slider.value, 10);
            display.textContent = format(nextValue);
            onInput(nextValue);
        });
        container.append(labelElement, slider, display);
    }

    buildSpeakerSection() {
        const section = document.createElement("div");
        section.style.cssText = "margin-top:12px";
        section.appendChild(sectionHeading("Speaker names"));
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:auto 1fr;gap:6px 8px;align-items:center";
        const inputStyle = "padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px;width:100%;box-sizing:border-box";

        for (const definition of [
            { label: "Character", key: "characterName", placeholder: "AI", uiKey: "speakerLabelAI", fallback: "AI" },
            { label: "User", key: "userName", placeholder: "User", uiKey: "speakerLabelUser", fallback: "User" },
        ]) {
            const label = document.createElement("span");
            label.style.cssText = "font-size:11px;color:#666";
            label.textContent = definition.label;
            const input = document.createElement("input");
            input.type = "text";
            input.placeholder = definition.placeholder;
            input.value = this.state[definition.key];
            input.style.cssText = inputStyle;
            input.addEventListener("input", () => {
                this.state[definition.key] = input.value.trim();
                this.ui[definition.uiKey] = this.state[definition.key] || definition.fallback;
                this.saveState();
            });
            grid.append(label, input);
        }
        section.appendChild(grid);
        return section;
    }

    buildMicrophoneSection() {
        const section = document.createElement("div");
        section.style.cssText = "margin-top:12px";
        section.appendChild(sectionHeading("Microphone"));
        const toggles = document.createElement("div");
        toggles.style.cssText = "display:flex;flex-direction:column;gap:10px";
        const indicator = createToggle("Show indicator", this.state.showMicGlow, (value) => {
            this.state.showMicGlow = value;
            this.applyState();
            this.saveState();
        });
        const mute = createToggle("Mute", this.aiavatar.isMuted, (value) => {
            if (value) this.aiavatar.mute();
            else this.aiavatar.unmute();
        });
        toggles.append(indicator.row, mute.row);
        section.appendChild(toggles);
        return section;
    }

    buildBackgroundSection() {
        const section = document.createElement("div");
        section.style.cssText = "margin-top:12px";
        section.appendChild(sectionHeading("Background image"));

        const urlRow = document.createElement("div");
        urlRow.style.cssText = "display:flex;gap:6px;margin-bottom:6px";
        const urlInput = document.createElement("input");
        urlInput.type = "text";
        urlInput.placeholder = "Image URL";
        urlInput.style.cssText = "flex:1;padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px";
        const loadButton = document.createElement("button");
        loadButton.className = "settings-button";
        loadButton.textContent = "Load";
        loadButton.addEventListener("click", async () => {
            const url = urlInput.value.trim();
            if (!url) return;
            try {
                const blob = await fetch(url).then((response) => response.blob());
                await this.storeBackground(blob);
            } catch {
                this.setBackgroundUrl(url);
            }
        });
        urlRow.append(urlInput, loadButton);

        const buttonRow = document.createElement("div");
        buttonRow.style.cssText = "display:flex;gap:6px";
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*";
        fileInput.style.display = "none";
        const fileButton = document.createElement("button");
        fileButton.className = "settings-button";
        fileButton.textContent = "Local file";
        fileButton.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", async () => {
            const file = fileInput.files[0];
            if (file) await this.storeBackground(file);
        });
        const clearButton = document.createElement("button");
        clearButton.className = "settings-button";
        clearButton.textContent = "Clear";
        clearButton.addEventListener("click", async () => {
            urlInput.value = "";
            this.setBackgroundUrl("");
            await this.blobStore.delete(this.persistence.backgroundKey);
        });
        buttonRow.append(fileButton, fileInput, clearButton);
        section.append(urlRow, buttonRow);
        return section;
    }

    buildConnectionSection() {
        const section = document.createElement("div");
        section.style.cssText = "margin-top:12px";
        const header = document.createElement("div");
        header.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:6px";
        const dot = document.createElement("span");
        dot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:50%;background:#e67e22;flex-shrink:0";
        const status = document.createElement("span");
        status.style.cssText = "font-size:11px;color:#999";
        status.textContent = "Disconnected";
        header.append(dot, status);

        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:center;font-size:11px";
        const fields = {};
        for (const key of ["session_id", "context_id"]) {
            const label = document.createElement("span");
            label.style.color = "#666";
            label.textContent = key;
            const value = document.createElement("span");
            value.style.cssText = "color:#999;word-break:break-all;-webkit-user-select:text;user-select:text";
            value.textContent = "-";
            grid.append(label, value);
            fields[key] = value;
        }

        const userIdLabel = document.createElement("span");
        userIdLabel.style.color = "#666";
        userIdLabel.textContent = "user_id";
        const userIdInput = document.createElement("input");
        userIdInput.type = "text";
        userIdInput.value = localStorage.getItem("userId") || "user01";
        userIdInput.style.cssText = "padding:2px 6px;border:1px solid #ccc;border-radius:3px;font-size:11px;width:100%;box-sizing:border-box;background:#fff;color:#333";
        userIdInput.addEventListener("change", () => {
            const userId = userIdInput.value.trim();
            if (!userId) return;
            this.ui.userId = userId;
            localStorage.setItem("userId", userId);
        });
        grid.append(userIdLabel, userIdInput);
        section.append(header, grid);
        this.connectionView = { dot, status, fields, userIdInput };
        return section;
    }

    updateConnection(connected, data = null) {
        if (!this.connectionView) return;
        const { dot, status, fields, userIdInput } = this.connectionView;
        dot.style.background = connected ? "#2ecc71" : "#e67e22";
        status.textContent = connected ? "Connected" : "Disconnected";
        for (const key of ["session_id", "context_id"]) fields[key].textContent = data?.[key] || "-";
        if (data?.user_id) userIdInput.value = data.user_id;
    }

    async reset() {
        if (this.persistence.enabled) localStorage.removeItem(this.persistence.displayKey);
        Object.assign(this.state, this.defaults);
        this.aiavatar.unmute();
        this.setBackgroundUrl("");
        await this.blobStore.delete(this.persistence.backgroundKey);
        this.applyState();
    }

    dispose() {
        if (this.backgroundIsObjectUrl && this.backgroundUrl) URL.revokeObjectURL(this.backgroundUrl);
    }
}
