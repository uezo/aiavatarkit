export const VIDEO_STATES = Object.freeze({
    IDLE: "idle",
    UNSTARTED: "unstarted",
    CUED: "cued",
    PLAYING: "playing",
    PAUSED: "paused",
    BUFFERING: "buffering",
    ENDED: "ended",
    ERROR: "error",
});

const VALID_STATES = new Set(Object.values(VIDEO_STATES));

export class VideoDriver {
    static get provider() {
        throw new Error("VideoDriver.provider must be implemented");
    }

    static supports() {
        return false;
    }

    static resolveUrl() {
        throw new Error("VideoDriver.resolveUrl() must be implemented");
    }

    constructor({ onEvent = null } = {}) {
        if (new.target === VideoDriver) throw new TypeError("VideoDriver is abstract");
        if (onEvent !== null && typeof onEvent !== "function") {
            throw new TypeError("VideoDriver onEvent must be a function");
        }
        this._ready = false;
        this._disposed = false;
        this._state = VIDEO_STATES.IDLE;
        this._onEvent = onEvent;
    }

    get ready() {
        return this._ready;
    }

    get disposed() {
        return this._disposed;
    }

    get state() {
        return this._state;
    }

    initialize() {
        throw new Error("VideoDriver.initialize() must be implemented");
    }

    startAutoplay() {
        throw new Error("VideoDriver.startAutoplay() must be implemented");
    }

    dispose() {
        this._ready = false;
        this._disposed = true;
    }

    _setReady() {
        if (this.disposed || this.ready) return false;
        this._ready = true;
        this._emit("ready");
        return true;
    }

    _setState(state) {
        if (!VALID_STATES.has(state)) throw new Error(`Invalid video state: ${state}`);
        if (this.disposed || state === this._state) return false;
        const previousState = this._state;
        this._state = state;
        this._emit("statechange", { state, previousState });
        return true;
    }

    _emit(type, detail = {}) {
        if (this.disposed || !this._onEvent) return;
        try {
            this._onEvent({
                type,
                provider: this.constructor.provider,
                ...detail,
            });
        } catch (error) {
            globalThis.console?.error?.("VideoDriver event handler failed:", error);
        }
    }
}
