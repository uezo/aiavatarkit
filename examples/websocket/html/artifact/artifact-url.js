const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

export function parseArtifactHttpUrl(source) {
    if (typeof source !== "string" || !source || source.length > 4096) {
        throw new Error("Artifact URL is missing or too long");
    }

    let url;
    try {
        url = new URL(source);
    } catch {
        throw new Error("Artifact URL must be absolute");
    }
    if (url.username || url.password) throw new Error("Artifact URLs cannot contain credentials");
    if (url.protocol !== "https:" && !(url.protocol === "http:" && LOCAL_HOSTS.has(url.hostname))) {
        throw new Error("Artifact URLs must use HTTPS (HTTP is allowed only for localhost)");
    }
    return url;
}

export function normalizeArtifactSlide(slide) {
    if (slide === null || slide === undefined || slide === "") return null;
    const value = String(slide);
    if (!/^[1-9]\d*$/.test(value)) throw new Error("Artifact slide must be a positive integer");
    return value;
}
