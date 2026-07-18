function heading(text) {
    const element = document.createElement("div");
    element.style.cssText = "font-size:11px;color:#888;margin-bottom:6px";
    element.textContent = text;
    return element;
}

function button(text) {
    const element = document.createElement("button");
    element.type = "button";
    element.className = "settings-button";
    element.textContent = text;
    return element;
}

function toggle(labelText, checked, onChange) {
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

function addSlider(grid, { label, min, max, step = 1, value, format = String, onInput }) {
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const input = document.createElement("input");
    input.type = "range";
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;
    const output = document.createElement("span");
    output.textContent = format(value);
    input.addEventListener("input", () => {
        const next = Number.parseFloat(input.value);
        output.textContent = format(next);
        onInput(next);
    });
    grid.append(labelElement, input, output);
    return { input, output };
}

function hiddenFileInput({ accept, multiple = false, directory = false }) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.multiple = multiple;
    input.hidden = true;
    if (directory) input.webkitdirectory = true;
    return input;
}

function morphSelect(adapter, selected, { auto = true } = {}) {
    const select = document.createElement("select");
    select.className = "mmdi-input";
    select.style.width = "100%";
    if (auto) select.appendChild(new Option("Auto (name fallback)", "__auto__"));
    select.appendChild(new Option("None", ""));
    for (const name of adapter.getMorphNames()) select.appendChild(new Option(name, name));
    select.value = selected;
    if (select.selectedIndex < 0) select.value = auto ? "__auto__" : "";
    return select;
}

export function installMmdSettings(adapter) {
    const settings = adapter.settingsHost;

    settings.addTab("Load", (panel) => {
        panel.appendChild(heading("MMD model"));
        const urlRow = document.createElement("div");
        urlRow.style.cssText = "display:flex;gap:6px;margin-bottom:8px";
        const urlInput = document.createElement("input");
        urlInput.type = "url";
        urlInput.className = "mmdi-input";
        urlInput.style.flex = "1";
        urlInput.placeholder = "PMX / PMD file URL";
        const urlButton = button("Load URL");
        urlButton.addEventListener("click", async () => {
            const url = urlInput.value.trim();
            if (!url) return;
            try {
                await adapter.loadModelUrl(url);
            } catch (error) {
                console.error("Failed to load MMD model:", error);
                alert(`Failed to load MMD model from URL.\n${error.message || error}`);
            }
        });
        urlRow.append(urlInput, urlButton);
        panel.appendChild(urlRow);

        const inputs = {
            files: hiddenFileInput({ accept: ".pmx,.pmd,.bpmx", multiple: true }),
            // Do not set accept for directories: all referenced textures and
            // auxiliary files must be returned by the browser.
            folder: hiddenFileInput({ accept: "", multiple: true, directory: true }),
        };
        const loadSelectedFiles = async (files, errorMessage) => {
            const model = files.find((file) => /\.(pmx|pmd|bpmx)$/i.test(file.name));
            if (!model) return;
            try {
                await adapter.loadModelFiles(model, files, { cache: true });
            } catch (error) {
                console.error("Failed to load MMD model:", error);
                alert(`${errorMessage}\n${error.message || error}`);
            }
        };
        inputs.files.addEventListener("change", async () => {
            await loadSelectedFiles(Array.from(inputs.files.files || []), "Failed to load MMD model files.");
            inputs.files.value = "";
        });
        inputs.folder.addEventListener("change", async () => {
            await loadSelectedFiles(Array.from(inputs.folder.files || []), "Failed to load MMD model folder.");
            inputs.folder.value = "";
        });
        const modelButtons = document.createElement("div");
        modelButtons.className = "mmdi-button-row";
        const filesButton = button("Local files");
        filesButton.addEventListener("click", () => inputs.files.click());
        const folderButton = button("Folder");
        folderButton.addEventListener("click", () => inputs.folder.click());
        const unloadButton = button("Unload");
        unloadButton.addEventListener("click", () => adapter.unloadModel({ clearCache: true }));
        modelButtons.append(filesButton, folderButton, unloadButton, inputs.files, inputs.folder);
        panel.appendChild(modelButtons);

        const physicsSection = document.createElement("div");
        physicsSection.style.marginTop = "14px";
        physicsSection.appendChild(heading("Physics"));
        const physicsToggle = toggle("Enable physics", adapter.physics.enabled, async (enabled) => {
            try {
                const reloaded = await adapter.setPhysicsEnabled(enabled);
                if (adapter.currentModel && !reloaded) alert("Reload the model to apply the physics setting.");
            } catch (error) {
                console.error("Failed to reload MMD physics:", error);
                alert("Physics setting was saved. Reload the model to apply it.");
            }
        });
        physicsSection.appendChild(physicsToggle.row);
        const physicsGrid = document.createElement("div");
        physicsGrid.className = "settings-sliders";
        physicsGrid.style.marginTop = "10px";
        addSlider(physicsGrid, {
            label: "Gravity",
            min: 0,
            max: 120,
            value: adapter.physics.gravity,
            onInput: (value) => adapter.setGravity(value),
        });
        physicsSection.appendChild(physicsGrid);
        panel.appendChild(physicsSection);

        const motionSection = document.createElement("div");
        motionSection.style.marginTop = "16px";
        motionSection.appendChild(heading("VMD motion"));
        const motionInput = hiddenFileInput({ accept: ".vmd,.bvmd" });
        const motionButton = button("Load motion");
        motionButton.addEventListener("click", () => motionInput.click());
        motionInput.addEventListener("change", async () => {
            const file = motionInput.files?.[0];
            if (!file) return;
            try {
                const name = file.name.replace(/\.(vmd|bvmd)$/i, "") || "motion";
                await adapter.loadMotionBlob(name, file, { cache: true });
            } catch (error) {
                console.error("Failed to load VMD motion:", error);
                alert("Failed to load VMD motion.");
            } finally {
                motionInput.value = "";
            }
        });
        const playButton = button("Play");
        playButton.addEventListener("click", () => adapter.playMotion());
        const pauseButton = button("Pause");
        pauseButton.addEventListener("click", () => adapter.pauseMotion());
        const clearButton = button("Clear");
        clearButton.addEventListener("click", () => adapter.clearMotion());
        const motionButtons = document.createElement("div");
        motionButtons.className = "mmdi-button-row";
        motionButtons.append(motionButton, motionInput, playButton, pauseButton, clearButton);
        motionSection.appendChild(motionButtons);
        panel.appendChild(motionSection);

        const hint = document.createElement("div");
        hint.className = "mmdi-hint";
        hint.textContent = "テクスチャを含むモデルは Folder または全ファイルをまとめて選択してください。画面へのドロップにも対応します。";
        panel.appendChild(hint);
    }, { position: 0, active: true });

    settings.addTab("Light", (panel) => {
        const grid = document.createElement("div");
        grid.className = "settings-sliders";
        for (const definition of [
            { key: "ambient", label: "Ambient", max: 2, step: 0.01, format: (value) => value.toFixed(2) },
            { key: "directional", label: "Direct", max: 3, step: 0.01, format: (value) => value.toFixed(2) },
            { key: "face", label: "Face fill", max: 2, step: 0.01, format: (value) => value.toFixed(2) },
            { key: "outlineScale", label: "Edge scale", max: 1, step: 0.01, format: (value) => `${value.toFixed(2)}x` },
        ]) {
            addSlider(grid, {
                ...definition,
                min: 0,
                value: adapter.appearance[definition.key],
                onInput: (value) => adapter.setAppearance(definition.key, value),
            });
        }
        panel.appendChild(grid);
        const toon = toggle("Toon shading", adapter.appearance.toonEnabled, (enabled) => {
            adapter.setAppearance("toonEnabled", enabled);
        });
        toon.row.style.marginTop = "12px";
        panel.appendChild(toon.row);
    }, { position: 1 });

    settings.addTab("Idle", (panel) => {
        const enabled = toggle("Idle motion", adapter.idle.enabled, (value) => adapter.setIdle("enabled", value));
        panel.appendChild(enabled.row);
        const grid = document.createElement("div");
        grid.className = "settings-sliders";
        grid.style.marginTop = "12px";
        addSlider(grid, {
            label: "Arm relax",
            min: -80,
            max: 0,
            value: adapter.idle.armRelax,
            format: (value) => `${value}°`,
            onInput: (value) => adapter.setIdle("armRelax", value),
        });
        addSlider(grid, {
            label: "Breath",
            min: 0,
            max: 100,
            value: adapter.idle.breath,
            format: (value) => `${value}%`,
            onInput: (value) => adapter.setIdle("breath", value),
        });
        addSlider(grid, {
            label: "Sway",
            min: 0,
            max: 100,
            value: adapter.idle.sway,
            format: (value) => `${value}%`,
            onInput: (value) => adapter.setIdle("sway", value),
        });
        panel.appendChild(grid);
        const hint = document.createElement("div");
        hint.className = "mmdi-hint";
        hint.textContent = "VMD が設定されている間は、VMD のボーン制御を優先します。";
        panel.appendChild(hint);
    }, { position: 3 });

    settings.addTab("Expression", (panel) => {
        panel.appendChild(heading("AI face → MMD morph"));
        const mappingGrid = document.createElement("div");
        mappingGrid.style.cssText = "display:grid;grid-template-columns:70px 1fr auto;gap:6px;align-items:center";
        const selectEntries = [];

        const rebuildSelect = (entry) => {
            const mapped = Object.prototype.hasOwnProperty.call(adapter.expressionMapping, entry.key)
                ? adapter.expressionMapping[entry.key]
                : "__auto__";
            const next = morphSelect(adapter, mapped);
            entry.select.replaceWith(next);
            entry.select = next;
            next.addEventListener("change", () => {
                adapter.setExpressionMapping(entry.key, next.value === "__auto__" ? undefined : next.value);
            });
        };

        for (const definition of adapter.config.expression.keys) {
            const label = document.createElement("span");
            label.style.fontSize = "11px";
            label.textContent = definition.label;
            const mapped = Object.prototype.hasOwnProperty.call(adapter.expressionMapping, definition.key)
                ? adapter.expressionMapping[definition.key]
                : "__auto__";
            const select = morphSelect(adapter, mapped);
            select.addEventListener("change", () => {
                adapter.setExpressionMapping(
                    definition.key,
                    select.value === "__auto__" ? undefined : select.value,
                );
            });
            const test = button("Test");
            test.addEventListener("click", () => {
                adapter.applyExpression(definition.key, adapter.config.expression.testDurationSeconds);
            });
            mappingGrid.append(label, select, test);
            selectEntries.push({ key: definition.key, select });
        }
        panel.appendChild(mappingGrid);

        const motionSection = document.createElement("div");
        motionSection.style.marginTop = "16px";
        motionSection.appendChild(heading("Blink / lip sync"));
        const blinkToggle = toggle("Auto blink", adapter.faceMotion.blinkEnabled, (enabled) => {
            adapter.setFaceMotion("blinkEnabled", enabled);
        });
        motionSection.appendChild(blinkToggle.row);

        const blinkRow = document.createElement("div");
        blinkRow.style.cssText = "display:grid;grid-template-columns:70px 1fr;gap:6px;align-items:center;margin-top:8px";
        const blinkLabel = document.createElement("span");
        blinkLabel.style.fontSize = "11px";
        blinkLabel.textContent = "Blink morph";
        let blinkSelect = morphSelect(adapter, adapter.faceMotion.blinkMorph || "__auto__");
        blinkSelect.addEventListener("change", () => {
            adapter.setFaceMotion("blinkMorph", blinkSelect.value === "__auto__" ? "" : blinkSelect.value);
        });
        blinkRow.append(blinkLabel, blinkSelect);
        motionSection.appendChild(blinkRow);

        const motionGrid = document.createElement("div");
        motionGrid.className = "settings-sliders";
        motionGrid.style.marginTop = "10px";
        addSlider(motionGrid, {
            label: "Lip smooth",
            min: 0,
            max: 100,
            value: adapter.faceMotion.lipSmooth,
            onInput: (value) => adapter.setFaceMotion("lipSmooth", value),
        });
        addSlider(motionGrid, {
            label: "Lip gain",
            min: 0.5,
            max: 4,
            step: 0.1,
            value: adapter.faceMotion.lipGain,
            format: (value) => `${value.toFixed(1)}x`,
            onInput: (value) => adapter.setFaceMotion("lipGain", value),
        });
        addSlider(motionGrid, {
            label: "Blink avg",
            min: 1,
            max: 8,
            step: 0.5,
            value: adapter.faceMotion.blinkInterval,
            format: (value) => `${value.toFixed(1)}s`,
            onInput: (value) => adapter.setFaceMotion("blinkInterval", value),
        });
        motionSection.appendChild(motionGrid);
        panel.appendChild(motionSection);

        const resetButton = button("Reset mapping");
        resetButton.style.marginTop = "12px";
        resetButton.addEventListener("click", () => adapter.resetExpressionMapping());
        panel.appendChild(resetButton);

        adapter.onMorphListChanged = () => {
            for (const entry of selectEntries) rebuildSelect(entry);
            const selectedBlink = adapter.faceMotion.blinkMorph || "__auto__";
            const nextBlink = morphSelect(adapter, selectedBlink);
            blinkSelect.replaceWith(nextBlink);
            blinkSelect = nextBlink;
            blinkSelect.addEventListener("change", () => {
                adapter.setFaceMotion("blinkMorph", blinkSelect.value === "__auto__" ? "" : blinkSelect.value);
            });
        };
    }, { position: 4 });

    settings.onTabReset("Light", () => adapter.resetAppearance());
    settings.onTabReset("Idle", () => adapter.resetIdle());
    settings.onTabReset("Expression", () => adapter.resetExpressionMapping());
}
