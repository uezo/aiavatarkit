import {
    assertArtifactAttributes,
    parseArtifactDisplayAttributes,
} from "./artifact-parser.js";

const MAPS_EMBED_BASE_URL = "https://www.google.com/maps/embed/v1/";
const MAP_TYPES = new Set(["roadmap", "satellite"]);
const TRAVEL_MODES = new Set(["driving", "walking", "bicycling", "transit", "flying"]);
const DEFAULT_ZOOM = 15;
const MAX_MAP_TERM_LENGTH = 512;

function normalizeApiKey(value) {
    if (typeof value !== "string" || !value.trim()) {
        throw new TypeError("Google Maps Embed API key is required");
    }
    const apiKey = value.trim();
    if (apiKey.length > 512 || /\s/.test(apiKey)) {
        throw new TypeError("Google Maps Embed API key is invalid");
    }
    return apiKey;
}

function parseCoordinate(value, name, minimum, maximum) {
    if (value === undefined || value === null || String(value).trim() === "") {
        throw new Error(`Map ${name} is required`);
    }
    const coordinate = Number(value);
    if (!Number.isFinite(coordinate) || coordinate < minimum || coordinate > maximum) {
        throw new RangeError(`Map ${name} must be between ${minimum} and ${maximum}`);
    }
    return coordinate;
}

function parseMapTerm(value, name) {
    if (value === undefined || value === null) return null;
    const term = String(value).trim();
    if (!term) throw new Error(`Map ${name} must not be empty`);
    if (term.length > MAX_MAP_TERM_LENGTH || /[\u0000-\u001f\u007f]/.test(term)) {
        throw new Error(`Map ${name} is invalid or too long`);
    }
    return term;
}

function parseZoom(value, defaultZoom) {
    if (value === undefined || value === null || String(value).trim() === "") return defaultZoom;
    const zoom = Number(value);
    if (!Number.isInteger(zoom) || zoom < 0 || zoom > 21) {
        throw new RangeError("Map zoom must be an integer between 0 and 21");
    }
    return zoom;
}

function parseMapType(value, defaultMapType) {
    const mapType = String(value || defaultMapType).toLowerCase();
    if (!MAP_TYPES.has(mapType)) {
        throw new Error(`Unsupported map type: ${value}`);
    }
    return mapType;
}

function parseTravelMode(value) {
    if (value === undefined || value === null || String(value).trim() === "") return null;
    const travelMode = String(value).toLowerCase();
    if (!TRAVEL_MODES.has(travelMode)) {
        throw new Error(`Unsupported map travel mode: ${value}`);
    }
    return travelMode;
}

function normalizeLanguage(value) {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value !== "string" || !/^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$/i.test(value)) {
        throw new TypeError("Google Maps language is invalid");
    }
    return value;
}

function normalizeRegion(value) {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value !== "string" || !/^[a-z]{2}$/i.test(value)) {
        throw new TypeError("Google Maps region must be a two-letter code");
    }
    return value.toUpperCase();
}

export class GoogleMapArtifactPlugin {
    constructor({
        apiKey,
        defaultZoom = DEFAULT_ZOOM,
        defaultMapType = "roadmap",
        language = null,
        region = null,
    } = {}) {
        this.type = "map";
        this.aliases = [];
        this.apiKey = normalizeApiKey(apiKey);
        this.defaultZoom = parseZoom(defaultZoom, DEFAULT_ZOOM);
        this.defaultMapType = parseMapType(defaultMapType, "roadmap");
        this.language = normalizeLanguage(language);
        this.region = normalizeRegion(region);
    }

    parse(attributes) {
        assertArtifactAttributes(attributes, [
            "location",
            "latitude",
            "longitude",
            "origin",
            "destination",
            "travel-mode",
            "zoom",
            "maptype",
        ]);
        if (attributes.src !== undefined || attributes.href !== undefined) {
            throw new Error("Map artifacts use a location, coordinates, or route instead of src or href");
        }

        const location = parseMapTerm(attributes.location, "location");
        const origin = parseMapTerm(attributes.origin, "origin");
        const destination = parseMapTerm(attributes.destination, "destination");
        const travelMode = parseTravelMode(attributes["travel-mode"]);
        const hasLatitude = attributes.latitude !== undefined;
        const hasLongitude = attributes.longitude !== undefined;
        const hasCoordinates = hasLatitude || hasLongitude;
        const hasRoute = origin !== null || destination !== null;

        if (hasRoute && (!origin || !destination)) {
            throw new Error("Map origin and destination must be specified together");
        }
        const inputCount = Number(Boolean(location)) + Number(hasCoordinates) + Number(hasRoute);
        if (inputCount === 0) {
            throw new Error("Map location, coordinates, or route are required");
        }
        if (inputCount > 1) {
            throw new Error("Map artifacts use either a location, coordinates, or route");
        }
        if (hasCoordinates && hasLatitude !== hasLongitude) {
            throw new Error("Map latitude and longitude must be specified together");
        }
        if (travelMode && !hasRoute) {
            throw new Error("Map travel-mode is available only for routes");
        }

        const mode = hasRoute ? "directions" : location ? "place" : "view";
        const zoom = mode === "directions" && attributes.zoom === undefined
            ? null
            : parseZoom(attributes.zoom, this.defaultZoom);

        return {
            action: "show",
            type: this.type,
            mode,
            location,
            latitude: hasCoordinates ? parseCoordinate(attributes.latitude, "latitude", -90, 90) : null,
            longitude: hasCoordinates ? parseCoordinate(attributes.longitude, "longitude", -180, 180) : null,
            origin,
            destination,
            travelMode,
            zoom,
            mapType: parseMapType(attributes.maptype, this.defaultMapType),
            ...parseArtifactDisplayAttributes(attributes),
        };
    }

    resolveSource(command) {
        const url = new URL(command.mode, MAPS_EMBED_BASE_URL);
        url.searchParams.set("key", this.apiKey);
        if (command.mode === "place") url.searchParams.set("q", command.location);
        else if (command.mode === "directions") {
            url.searchParams.set("origin", command.origin);
            url.searchParams.set("destination", command.destination);
            if (command.travelMode) url.searchParams.set("mode", command.travelMode);
        } else {
            url.searchParams.set("center", `${command.latitude},${command.longitude}`);
        }
        if (command.zoom !== null) url.searchParams.set("zoom", String(command.zoom));
        url.searchParams.set("maptype", command.mapType);
        if (this.language) url.searchParams.set("language", this.language);
        if (this.region) url.searchParams.set("region", this.region);
        return { provider: "google-maps-embed", url: url.href };
    }

    getDefaults() {
        return { aspect: "16:9" };
    }

    mount({ documentRoot, command, source, view }) {
        const iframe = documentRoot.createElement("iframe");
        iframe.className = "artifact-media";
        iframe.src = source.url;
        const defaultTitle = command.mode === "directions"
            ? `${command.origin} → ${command.destination}`
            : command.location || "Google Map";
        iframe.title = command.title || command.alt || defaultTitle;
        iframe.loading = "eager";
        iframe.referrerPolicy = "strict-origin-when-cross-origin";
        iframe.setAttribute("allowfullscreen", "");
        iframe.addEventListener("load", () => view.loaded(), { once: true });
        iframe.addEventListener("error", () => view.error("地図を表示できませんでした"), { once: true });

        return {
            element: iframe,
            dispose: () => iframe.removeAttribute("src"),
        };
    }
}
