function heading(text) {
    const element = document.createElement("div");
    element.style.cssText = "font-size:11px;color:#999;margin-bottom:6px";
    element.textContent = text;
    return element;
}

function button(text) {
    const element = document.createElement("button");
    element.className = "vrmi-sampler-btn";
    element.textContent = text;
    return element;
}

export function installVrmSettings(adapter) {
    const settings = adapter.settingsHost;
    let refreshAnimationList = () => {};

    settings.addTab("Load", (panel) => {
        panel.appendChild(heading("VRM"));

        const urlRow = document.createElement("div");
        urlRow.style.cssText = "display:flex;gap:6px;margin-bottom:8px";
        const urlInput = document.createElement("input");
        urlInput.type = "text";
        urlInput.placeholder = "VRM file URL";
        urlInput.style.cssText = "flex:1;padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px";
        const urlButton = button("Load URL");
        urlButton.addEventListener("click", async () => {
            const url = urlInput.value.trim();
            if (!url) return;
            try {
                await adapter.loadModelUrl(url, { cache: true });
            } catch (error) {
                console.error("Failed to load VRM:", error);
                alert("Failed to load VRM from URL.");
            }
        });
        urlRow.append(urlInput, urlButton);
        panel.appendChild(urlRow);

        const modelInput = document.createElement("input");
        modelInput.type = "file";
        modelInput.accept = ".vrm";
        modelInput.style.display = "none";
        const modelButton = button("Local file");
        modelButton.addEventListener("click", () => modelInput.click());
        modelInput.addEventListener("change", async () => {
            const file = modelInput.files[0];
            if (!file) return;
            try {
                await adapter.loadModelBlob(file, { cache: true });
            } catch (error) {
                console.error("Failed to load VRM:", error);
                alert("Failed to load VRM file.");
            } finally {
                modelInput.value = "";
            }
        });
        panel.append(modelButton, modelInput);

        const unloadButton = button("Unload");
        unloadButton.style.marginLeft = "6px";
        unloadButton.addEventListener("click", () => adapter.unloadModel({ clearCache: true }));
        panel.appendChild(unloadButton);

        const animationSection = document.createElement("div");
        animationSection.style.marginTop = "16px";
        animationSection.appendChild(heading("VRMA"));
        const animationInput = document.createElement("input");
        animationInput.type = "file";
        animationInput.accept = ".vrma";
        animationInput.multiple = true;
        animationInput.style.display = "none";
        const animationButton = button("Add animation");
        animationButton.addEventListener("click", () => animationInput.click());
        animationInput.addEventListener("change", async () => {
            for (const file of animationInput.files) {
                try {
                    await adapter.loadAnimationBlob(file.name.replace(/\.vrma$/i, ""), file, { cache: true });
                } catch (error) {
                    console.error("Failed to load VRMA:", error);
                    alert(`Failed to load VRMA: ${file.name}`);
                }
            }
            animationInput.value = "";
            refreshAnimationList();
        });
        animationSection.append(animationButton, animationInput);

        const animationList = document.createElement("div");
        animationList.style.marginTop = "8px";
        refreshAnimationList = () => {
            animationList.replaceChildren();
            const names = adapter.animationNames;
            if (!names.length) {
                const empty = document.createElement("div");
                empty.style.cssText = "font-size:11px;color:#999;padding:4px 0";
                empty.textContent = "No animations registered.";
                animationList.appendChild(empty);
                return;
            }

            for (const name of names) {
                const row = document.createElement("div");
                row.style.cssText = "display:flex;align-items:center;gap:6px;padding:3px 0;font-size:12px";
                const nameInput = document.createElement("input");
                nameInput.type = "text";
                nameInput.value = name;
                nameInput.style.cssText = "flex:1;font-size:12px;padding:2px 6px;border:1px solid rgba(0,0,0,0.1);border-radius:4px;background:rgba(255,255,255,0.5);outline:none;min-width:0";
                nameInput.addEventListener("blur", async () => {
                    const nextName = nameInput.value.trim().toLowerCase();
                    if (!nextName || nextName === name) {
                        nameInput.value = name;
                        return;
                    }
                    if (!await adapter.renameAnimation(name, nextName)) {
                        alert("Rename failed. Name may already exist.");
                    }
                    refreshAnimationList();
                });
                nameInput.addEventListener("keydown", (event) => {
                    if (event.key === "Enter") nameInput.blur();
                    if (event.key === "Escape") {
                        nameInput.value = name;
                        nameInput.blur();
                    }
                });
                const deleteButton = button("Delete");
                deleteButton.style.cssText = "font-size:11px;padding:2px 8px;color:#c33;flex-shrink:0";
                deleteButton.addEventListener("click", async () => {
                    await adapter.removeAnimation(name);
                    refreshAnimationList();
                });
                row.append(nameInput, deleteButton);
                animationList.appendChild(row);
            }
        };
        animationSection.appendChild(animationList);
        panel.appendChild(animationSection);
        refreshAnimationList();

        const hint = document.createElement("div");
        hint.style.cssText = "margin-top:12px;font-size:11px;color:#999";
        hint.textContent = "You can also drag & drop .vrm / .vrma files onto the screen.";
        panel.appendChild(hint);
    }, { position: 0, active: true });

    settings.addTab("Light", (panel) => {
        const grid = document.createElement("div");
        grid.className = "vrmi-sliders";
        for (const definition of adapter.lightDefinitions) {
            const label = document.createElement("span");
            label.textContent = definition.label;
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = definition.min;
            slider.max = definition.max;
            slider.step = definition.step || 1;
            slider.value = adapter.lighting[definition.key];
            const display = document.createElement("span");
            display.textContent = definition.format(adapter.lighting[definition.key]);
            slider.addEventListener("input", () => {
                const value = Number.parseFloat(slider.value);
                adapter.setLighting(definition.key, value);
                display.textContent = definition.format(value);
            });
            grid.append(label, slider, display);
        }
        panel.appendChild(grid);
    }, { position: 1 });

    settings.onTabReset("Light", () => adapter.resetLighting());
    settings.onTabReset("Load", () => adapter.clearModelCache());
    adapter.onAnimationListChanged = refreshAnimationList;
}
