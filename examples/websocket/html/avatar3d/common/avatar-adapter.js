const REQUIRED_METHODS = [
    "initialize",
    "importFiles",
    "handleResponse",
    "stop",
    "dispose",
];

/**
 * Runtime contract for a 3D avatar implementation.
 *
 * The common application only deals with semantic lifecycle and response
 * events. Rendering engines, file extensions, cameras, assets and animation
 * formats remain private to the adapter implementation.
 */
export function assertAvatarAdapter(adapter) {
    if (!adapter || typeof adapter !== "object") {
        throw new TypeError("An avatar adapter is required");
    }
    for (const method of REQUIRED_METHODS) {
        if (typeof adapter[method] !== "function") {
            throw new TypeError(`Avatar adapter must implement ${method}()`);
        }
    }
    return adapter;
}
