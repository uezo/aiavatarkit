export function installToolToasts({ durationMs = 4000 } = {}) {
    const status = document.getElementById("toolStatus");
    const container = document.getElementById("toolToastContainer");
    if (!status || !container) return { dispose() {} };

    function show(text) {
        const toast = document.createElement("div");
        toast.className = "tool-toast";
        toast.textContent = text;
        container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add("fade-out");
            toast.addEventListener("animationend", () => toast.remove(), { once: true });
        }, durationMs);
    }

    const observer = new MutationObserver(() => {
        const text = status.textContent.trim();
        if (text) show(text);
    });
    observer.observe(status, { childList: true, characterData: true, subtree: true });

    return {
        dispose() {
            observer.disconnect();
            container.replaceChildren();
        },
    };
}
