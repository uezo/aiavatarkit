import { PresentationDriver } from "./presentation-driver.js";

let messageSequence = 0;

const DOCSWELL_HOSTS = new Set(["docswell.com", "www.docswell.com"]);
const EMBED_PATH = /^\/slide\/[A-Za-z0-9_-]+\/embed\/?$/;
const VIEW_PATH = /^\/s\/[^/]+\/([A-Za-z0-9]{6})(?:-[^/]+)?\/?$/;

function slideFromUrl(url) {
    return Number(new URL(url).hash.match(/^#p([1-9]\d*)$/)?.[1] || 1);
}

function documentUrl(url) {
    const value = new URL(url);
    value.hash = "";
    return value.href;
}

export class DocswellDriver extends PresentationDriver {
    static get provider() {
        return "docswell";
    }

    static supports(url) {
        return DOCSWELL_HOSTS.has(url.hostname);
    }

    static resolveUrl(source, slide = null) {
        const url = new URL(source.href);
        if (!this.supports(url)) throw new Error("Invalid Docswell host");
        for (const key of url.searchParams.keys()) {
            if (key !== "key") throw new Error(`Unsupported Docswell parameter: ${key}`);
        }
        if (url.hash && !/^#p[1-9]\d*$/.test(url.hash)) {
            throw new Error("Docswell page fragments must have the form #p7");
        }

        const viewMatch = url.pathname.match(VIEW_PATH);
        if (viewMatch) {
            url.pathname = `/slide/${viewMatch[1]}/embed`;
        } else if (!EMBED_PATH.test(url.pathname)) {
            throw new Error("Docswell URLs must be viewer or embed URLs");
        } else {
            url.pathname = url.pathname.replace(/\/$/, "");
        }
        if (slide !== null) url.hash = `p${slide}`;
        return url.href;
    }

    constructor({ iframe, url, windowRoot = globalThis.window, onSlideChange = null }) {
        super({ initialSlide: slideFromUrl(url), onSlideChange });
        this.iframe = iframe;
        this.url = url;
        this.window = windowRoot;
        this.origin = new URL(url).origin;
        this.id = `aiavatar-artifact-${++messageSequence}`;
        this.listening = false;
        this.handleWindowMessage = this.handleWindowMessage.bind(this);
    }

    initialize() {
        if (!this.listening) {
            this.window?.addEventListener?.("message", this.handleWindowMessage);
            this.listening = true;
        }
        try {
            this.iframe.contentWindow?.postMessage(
                { type: "docswell:initialize", id: this.id },
                this.origin,
            );
            return true;
        } catch {
            return false;
        }
    }

    navigate(url) {
        if (!this.ready || documentUrl(url) !== documentUrl(this.url)) return false;
        const slide = slideFromUrl(url);
        return this.postGo(slide - 1) ? slide : false;
    }

    navigateBy(offset) {
        if (!this.ready || !Number.isSafeInteger(offset) || !offset) return false;
        const upper = this.totalSlides || Number.MAX_SAFE_INTEGER;
        const slide = Math.min(upper, Math.max(1, this.currentSlide + offset));
        if (!this.postGo(slide - 1)) return false;
        this._setCurrentSlide(slide);
        return { slide };
    }

    postGo(to) {
        try {
            this.iframe.contentWindow.postMessage(
                { type: "docswell:go", to, id: this.id },
                this.origin,
            );
            return true;
        } catch {
            return false;
        }
    }

    handleWindowMessage(event) {
        if (event.source !== this.iframe.contentWindow || event.origin !== this.origin) return;
        const data = event.data;
        if (!data || data.id !== this.id) return;

        if (data.type === "docswell:initialized") {
            const total = Number(data.total);
            this._setReady(Number.isSafeInteger(total) && total > 0 ? total : null);
        } else if (data.type === "docswell:move") {
            const index = Number(data.index);
            if (Number.isSafeInteger(index) && index >= 0) this._setCurrentSlide(index + 1);
        }
    }

    dispose() {
        if (this.listening) this.window?.removeEventListener?.("message", this.handleWindowMessage);
        this.listening = false;
        super.dispose();
    }
}
