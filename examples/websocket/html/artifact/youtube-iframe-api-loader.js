const YOUTUBE_IFRAME_API_URL = "https://www.youtube.com/iframe_api";
const apiLoads = new WeakMap();

function isApiReady(windowRoot) {
    return typeof windowRoot?.YT?.Player === "function";
}

export function loadYouTubeIframeApi({
    windowRoot = globalThis.window,
    documentRoot = globalThis.document,
    timeoutMs = 15000,
} = {}) {
    if (!windowRoot || !documentRoot) {
        return Promise.reject(new Error("YouTube IFrame API requires window and document"));
    }
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
        return Promise.reject(new RangeError("YouTube IFrame API timeout must be positive"));
    }
    if (isApiReady(windowRoot)) return Promise.resolve(windowRoot.YT);

    const pending = apiLoads.get(windowRoot);
    if (pending) return pending;

    let loadPromise;
    loadPromise = new Promise((resolve, reject) => {
        let settled = false;
        let pollTimer = null;
        let timeoutTimer = null;
        let script = null;
        const previousReadyCallback = windowRoot.onYouTubeIframeAPIReady;

        const clearTimer = windowRoot.clearTimeout?.bind(windowRoot) || globalThis.clearTimeout;
        const setTimer = windowRoot.setTimeout?.bind(windowRoot) || globalThis.setTimeout;
        const clearPoll = windowRoot.clearInterval?.bind(windowRoot) || globalThis.clearInterval;
        const setPoll = windowRoot.setInterval?.bind(windowRoot) || globalThis.setInterval;

        const cleanup = () => {
            if (pollTimer !== null) clearPoll(pollTimer);
            if (timeoutTimer !== null) clearTimer(timeoutTimer);
            script?.removeEventListener?.("error", handleScriptError);
            if (windowRoot.onYouTubeIframeAPIReady === handleApiReady) {
                windowRoot.onYouTubeIframeAPIReady = previousReadyCallback;
            }
        };

        const finish = () => {
            if (settled || !isApiReady(windowRoot)) return false;
            settled = true;
            cleanup();
            resolve(windowRoot.YT);
            return true;
        };

        const fail = (error) => {
            if (settled) return;
            settled = true;
            cleanup();
            reject(error);
        };

        function handleApiReady(...args) {
            if (typeof previousReadyCallback === "function") {
                try {
                    previousReadyCallback.apply(windowRoot, args);
                } catch (error) {
                    windowRoot.console?.error?.("Existing YouTube API callback failed:", error);
                }
            }
            finish();
        }

        function handleScriptError() {
            fail(new Error("Could not load the YouTube IFrame API"));
        }

        windowRoot.onYouTubeIframeAPIReady = handleApiReady;
        script = documentRoot.querySelector?.(`script[src="${YOUTUBE_IFRAME_API_URL}"]`) || null;
        if (!script) {
            script = documentRoot.createElement?.("script");
            const parent = documentRoot.head || documentRoot.body || documentRoot.documentElement;
            if (!script || !parent?.appendChild) {
                fail(new Error("Could not create the YouTube IFrame API script"));
                return;
            }
            script.src = YOUTUBE_IFRAME_API_URL;
            script.async = true;
            parent.appendChild(script);
        }
        script.addEventListener?.("error", handleScriptError, { once: true });

        pollTimer = setPoll(finish, 50);
        timeoutTimer = setTimer(() => {
            fail(new Error("Timed out loading the YouTube IFrame API"));
        }, timeoutMs);
        finish();
    });

    apiLoads.set(windowRoot, loadPromise);
    loadPromise.catch(() => {
        if (apiLoads.get(windowRoot) === loadPromise) apiLoads.delete(windowRoot);
    });
    return loadPromise;
}
