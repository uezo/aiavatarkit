import { VIDEO_STATES, VideoDriver } from "./video-driver.js";
import { loadYouTubeIframeApi } from "./youtube-iframe-api-loader.js";

const YOUTUBE_HOSTS = new Set(["youtube.com", "www.youtube.com", "m.youtube.com"]);
const YOUTUBE_SHORT_HOSTS = new Set(["youtu.be", "www.youtu.be"]);
const VIDEO_ID_PATTERN = /^[A-Za-z0-9_-]{11}$/;
const EMBED_PATH = /^\/embed\/([A-Za-z0-9_-]{11})\/?$/;

const PLAYER_STATES = new Map([
    [-1, VIDEO_STATES.UNSTARTED],
    [0, VIDEO_STATES.ENDED],
    [1, VIDEO_STATES.PLAYING],
    [2, VIDEO_STATES.PAUSED],
    [3, VIDEO_STATES.BUFFERING],
    [5, VIDEO_STATES.CUED],
]);

const ERROR_MESSAGES = new Map([
    [2, "Invalid YouTube video ID or player parameter"],
    [5, "The YouTube video cannot be played in this HTML5 player"],
    [100, "The YouTube video was not found or is private"],
    [101, "The YouTube video owner does not allow embedding"],
    [150, "The YouTube video owner does not allow embedding"],
    [153, "The YouTube player request did not include valid client identity"],
]);

function normalizeDelaySeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) {
        throw new RangeError("YouTube autoplay delay must be a non-negative number");
    }
    return seconds;
}

function normalizeStartSeconds(value) {
    if (value === null || value === undefined || value === "") return null;
    const seconds = Number(value);
    if (!Number.isSafeInteger(seconds) || seconds < 0) {
        throw new RangeError("YouTube start time must be a non-negative integer");
    }
    return seconds;
}

function parseTime(value) {
    if (!value) return null;
    if (/^\d+$/.test(value)) return Number(value);
    const match = String(value).match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/);
    if (!match || !match[0]) return null;
    return Number(match[1] || 0) * 3600
        + Number(match[2] || 0) * 60
        + Number(match[3] || 0);
}

function extractVideoSource(url) {
    let videoId = null;
    if (YOUTUBE_SHORT_HOSTS.has(url.hostname)) {
        const path = url.pathname.replace(/^\//, "").replace(/\/$/, "");
        if (!path.includes("/")) videoId = path;
    } else if (YOUTUBE_HOSTS.has(url.hostname)) {
        if (url.pathname === "/watch") videoId = url.searchParams.get("v");
        else videoId = url.pathname.match(EMBED_PATH)?.[1] || null;
    }
    if (!VIDEO_ID_PATTERN.test(videoId || "")) {
        throw new Error("YouTube URLs must identify one embeddable video");
    }

    const start = url.searchParams.get("start");
    const startSeconds = normalizeStartSeconds(
        start === null || start === "" ? parseTime(url.searchParams.get("t")) : start,
    );
    return { videoId, startSeconds };
}

function appendIframePermission(iframe, permission) {
    const permissions = new Set(
        String(iframe.allow || "")
            .split(";")
            .map((value) => value.trim())
            .filter(Boolean),
    );
    permissions.add(permission);
    iframe.allow = Array.from(permissions).join("; ");
}

export class YouTubeDriver extends VideoDriver {
    static get provider() {
        return "youtube";
    }

    static supports(url) {
        return url.protocol === "https:"
            && (YOUTUBE_HOSTS.has(url.hostname) || YOUTUBE_SHORT_HOSTS.has(url.hostname));
    }

    static resolveUrl(source, { startSeconds = null, origin = null } = {}) {
        const url = new URL(source instanceof URL ? source.href : source);
        if (!this.supports(url)) throw new Error("Invalid YouTube host or protocol");
        if (url.username || url.password) throw new Error("YouTube URLs cannot contain credentials");

        const resolved = extractVideoSource(url);
        const start = normalizeStartSeconds(startSeconds ?? resolved.startSeconds);
        const embed = new URL(`https://www.youtube.com/embed/${resolved.videoId}`);
        embed.searchParams.set("enablejsapi", "1");
        embed.searchParams.set("playsinline", "1");
        if (start !== null && start > 0) embed.searchParams.set("start", String(start));

        if (origin) {
            const originUrl = new URL(origin);
            if (originUrl.protocol !== "https:" && originUrl.protocol !== "http:") {
                throw new Error("YouTube embed origin must use HTTP or HTTPS");
            }
            embed.searchParams.set("origin", originUrl.origin);
        }
        return embed.href;
    }

    constructor({
        iframe,
        url,
        windowRoot = globalThis.window,
        documentRoot = globalThis.document,
        apiLoader = loadYouTubeIframeApi,
        autoplay = false,
        autoplayDelaySeconds = 0,
        muted = false,
        onEvent = null,
    } = {}) {
        super({ onEvent });
        if (!iframe || typeof iframe !== "object") throw new TypeError("YouTubeDriver requires an iframe");
        if (typeof apiLoader !== "function") throw new TypeError("YouTubeDriver apiLoader must be a function");

        this.iframe = iframe;
        this.window = windowRoot;
        this.document = documentRoot;
        this.apiLoader = apiLoader;
        this.autoplay = Boolean(autoplay);
        this.autoplayDelaySeconds = normalizeDelaySeconds(autoplayDelaySeconds);
        this.muted = Boolean(muted);
        this.player = null;
        this._initializePromise = null;
        this._autoplayTimer = null;
        this._autoplayAttempted = false;
        this._hasStarted = false;
        this._autoplayDeadline = this.autoplay
            ? Date.now() + this.autoplayDelaySeconds * 1000
            : null;

        const origin = windowRoot?.location?.origin;
        this.url = this.constructor.resolveUrl(url, {
            origin: origin && origin !== "null" ? origin : null,
        });
        this.configureIframe();
    }

    get autoplayAttempted() {
        return this._autoplayAttempted;
    }

    get hasStarted() {
        return this._hasStarted;
    }

    configureIframe() {
        this.iframe.src = this.url;
        for (const permission of ["autoplay", "encrypted-media", "fullscreen", "picture-in-picture"]) {
            appendIframePermission(this.iframe, permission);
        }
        this.iframe.setAttribute?.("allowfullscreen", "");
    }

    initialize() {
        if (this.disposed) return Promise.resolve(false);
        if (this._initializePromise) return this._initializePromise;

        this._initializePromise = Promise.resolve(this.apiLoader({
            windowRoot: this.window,
            documentRoot: this.document,
        })).then((YT) => {
            if (this.disposed) return false;
            if (typeof YT?.Player !== "function") throw new Error("YouTube IFrame API did not provide YT.Player");

            return new Promise((resolve, reject) => {
                let settled = false;
                const settleReady = () => {
                    if (settled) return;
                    settled = true;
                    resolve(true);
                };
                const settleError = (error) => {
                    if (settled) return;
                    settled = true;
                    reject(error);
                };

                try {
                    this.player = new YT.Player(this.iframe, {
                        events: {
                            onReady: (event) => {
                                if (this.disposed) return;
                                this.player = event.target || this.player;
                                this._setReady();
                                this._scheduleAutoplay();
                                settleReady();
                            },
                            onStateChange: (event) => this._handleStateChange(event.data),
                            onAutoplayBlocked: () => this._handleAutoplayBlocked(),
                            onError: (event) => {
                                const error = this._handlePlayerError(event.data);
                                settleError(error);
                            },
                        },
                    });
                } catch (error) {
                    this._setState(VIDEO_STATES.ERROR);
                    this._emit("error", { error });
                    settleError(error);
                }
            });
        }).catch((error) => {
            if (!this.disposed && this.state !== VIDEO_STATES.ERROR) {
                this._setState(VIDEO_STATES.ERROR);
                this._emit("error", { error });
            }
            throw error;
        });
        return this._initializePromise;
    }

    startAutoplay() {
        if (this.disposed || !this.ready || !this.player) return false;
        if (this._autoplayAttempted || this._hasStarted) return false;
        this._autoplayAttempted = true;
        this._clearAutoplayTimer();

        try {
            if (this.muted) this.player.mute?.();
            this.player.playVideo();
            this._emit("autoplayrequested", { muted: this.muted });
            return true;
        } catch (error) {
            this._setState(VIDEO_STATES.ERROR);
            this._emit("error", { error });
            return false;
        }
    }

    _scheduleAutoplay() {
        if (!this.autoplay || this.disposed || !this.ready || this._hasStarted || this._autoplayAttempted) return;
        this._clearAutoplayTimer();
        const delayMs = Math.max(0, (this._autoplayDeadline || Date.now()) - Date.now());
        const setTimer = this.window?.setTimeout?.bind(this.window) || globalThis.setTimeout;
        this._autoplayTimer = setTimer(() => {
            this._autoplayTimer = null;
            this.startAutoplay();
        }, delayMs);
        this._emit("autoplayscheduled", { delaySeconds: delayMs / 1000 });
    }

    _handleStateChange(playerState) {
        if (this.disposed) return;
        const state = PLAYER_STATES.get(Number(playerState));
        if (!state) return;
        this._setState(state);
        if (state === VIDEO_STATES.PLAYING) {
            this._hasStarted = true;
            this._clearAutoplayTimer();
        }
    }

    _handleAutoplayBlocked() {
        if (this.disposed) return;
        this._clearAutoplayTimer();
        this._emit("autoplayblocked");
    }

    _handlePlayerError(providerCode) {
        const code = Number(providerCode);
        const error = new Error(ERROR_MESSAGES.get(code) || `YouTube player error: ${providerCode}`);
        error.code = "youtube-player-error";
        error.providerCode = providerCode;
        this._setState(VIDEO_STATES.ERROR);
        this._clearAutoplayTimer();
        this._emit("error", { error, providerCode });
        return error;
    }

    _clearAutoplayTimer() {
        if (this._autoplayTimer === null) return;
        const clearTimer = this.window?.clearTimeout?.bind(this.window) || globalThis.clearTimeout;
        clearTimer(this._autoplayTimer);
        this._autoplayTimer = null;
    }

    dispose() {
        if (this.disposed) return;
        this._clearAutoplayTimer();
        try {
            this.player?.destroy?.();
        } catch {
            // The containing artifact will remove the iframe even if the provider cleanup fails.
        }
        this.player = null;
        super.dispose();
    }
}
