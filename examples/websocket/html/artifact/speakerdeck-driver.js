import { PresentationDriver } from "./presentation-driver.js";

const PLAYER_PATH = /^\/player\/[A-Za-z0-9_-]+\/?$/;

function slideFromUrl(url) {
    return Number(new URL(url).searchParams.get("slide") || 1);
}

function deckUrl(url) {
    const value = new URL(url);
    value.searchParams.delete("slide");
    return value.href;
}

export class SpeakerDeckDriver extends PresentationDriver {
    static get provider() {
        return "speakerdeck";
    }

    static supports(url) {
        return url.hostname === "speakerdeck.com";
    }

    static resolveUrl(source, slide = null) {
        const url = new URL(source.href);
        if (!this.supports(url)) throw new Error("Invalid Speaker Deck host");
        if (!PLAYER_PATH.test(url.pathname)) {
            throw new Error("Speaker Deck requires a player embed URL");
        }
        url.pathname = url.pathname.replace(/\/$/, "");
        for (const key of url.searchParams.keys()) {
            if (key !== "slide") throw new Error(`Unsupported Speaker Deck parameter: ${key}`);
        }
        if (slide !== null) url.searchParams.set("slide", slide);
        const resolvedSlide = url.searchParams.get("slide");
        if (resolvedSlide !== null && !/^[1-9]\d*$/.test(resolvedSlide)) {
            throw new Error("Speaker Deck slide must be a positive integer");
        }
        if (url.hash) throw new Error("Speaker Deck embed URLs cannot contain a fragment");
        return url.href;
    }

    constructor({ iframe, url }) {
        super({ initialSlide: null });
        this.iframe = iframe;
        this.url = url;
        this.requestedSlide = slideFromUrl(url);
    }

    initialize() {
        this._setReady();
        return true;
    }

    navigate(url) {
        if (!this.ready || deckUrl(url) !== deckUrl(this.url)) return false;
        const slide = slideFromUrl(url);
        this.iframe.src = url;
        this.url = url;
        this.requestedSlide = slide;
        return slide;
    }

    navigateBy(offset) {
        if (!this.ready || !Number.isSafeInteger(offset) || !offset) return false;
        const url = new URL(this.url);
        const slide = Math.max(1, this.requestedSlide + offset);
        url.searchParams.set("slide", slide);
        return this.navigate(url.href) === false ? false : { slide, url: this.url };
    }
}
