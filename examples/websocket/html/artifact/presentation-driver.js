export class PresentationDriver {
    static get provider() {
        throw new Error("PresentationDriver.provider must be implemented");
    }

    static supports() {
        return false;
    }

    static resolveUrl() {
        throw new Error("PresentationDriver.resolveUrl() must be implemented");
    }

    constructor({ initialSlide = 1, onSlideChange = null } = {}) {
        if (new.target === PresentationDriver) throw new TypeError("PresentationDriver is abstract");
        this._ready = false;
        this._currentSlide = initialSlide;
        this._totalSlides = null;
        this._onSlideChange = onSlideChange;
    }

    get ready() {
        return this._ready;
    }

    get currentSlide() {
        return this._currentSlide;
    }

    get totalSlides() {
        return this._totalSlides;
    }

    initialize() {
        throw new Error("PresentationDriver.initialize() must be implemented");
    }

    navigate() {
        throw new Error("PresentationDriver.navigate() must be implemented");
    }

    navigateBy() {
        throw new Error("PresentationDriver.navigateBy() must be implemented");
    }

    dispose() {
        this._ready = false;
    }

    _setReady(totalSlides = null) {
        this._ready = true;
        this._totalSlides = totalSlides;
    }

    _setCurrentSlide(slide) {
        this._currentSlide = slide;
        this._onSlideChange?.(slide);
    }
}
