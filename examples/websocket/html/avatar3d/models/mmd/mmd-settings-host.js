export function createMmdSettingsHost() {
    const element = document.createElement("div");
    element.className = "mmdi-panel";

    const tabs = document.createElement("div");
    tabs.className = "mmdi-tabs";
    const toolbar = document.createElement("div");
    toolbar.className = "mmdi-toolbar";
    const resetButton = document.createElement("button");
    resetButton.type = "button";
    resetButton.className = "mmdi-reset";
    resetButton.textContent = "Reset tab";
    toolbar.appendChild(resetButton);
    const content = document.createElement("div");
    content.className = "mmdi-content";
    element.append(tabs, toolbar, content);
    document.body.appendChild(element);

    const entries = [];
    const resetHandlers = new Map();
    let activeEntry = null;

    function render() {
        entries.sort((left, right) => left.position - right.position);
        tabs.replaceChildren(...entries.map((entry) => entry.button));
        content.replaceChildren(...entries.map((entry) => entry.pane));
    }

    function activate(entry) {
        activeEntry = entry;
        for (const item of entries) {
            item.button.classList.toggle("active", item === entry);
            item.pane.classList.toggle("active", item === entry);
        }
        resetButton.hidden = !resetHandlers.has(entry.name);
    }

    resetButton.addEventListener("click", () => {
        const handler = activeEntry && resetHandlers.get(activeEntry.name);
        handler?.();
    });

    return {
        element,
        addTab(name, build, options = {}) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "mmdi-tab";
            button.textContent = name;
            const pane = document.createElement("div");
            pane.className = "mmdi-pane";
            build(pane);
            const entry = {
                name,
                button,
                pane,
                position: options.position ?? entries.length,
            };
            entries.push(entry);
            button.addEventListener("click", () => activate(entry));
            render();
            if (options.active || !activeEntry) activate(entry);
        },
        onTabReset(name, handler) {
            resetHandlers.set(name, handler);
            if (activeEntry?.name === name) resetButton.hidden = false;
        },
        dispose() {
            element.remove();
        },
    };
}
