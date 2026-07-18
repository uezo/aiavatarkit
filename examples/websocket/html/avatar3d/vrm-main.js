import { startAvatarApp } from "./common/app.js";
import { createBlobStore } from "./common/blob-store.js";
import { VrmAdapter } from "./models/vrm/vrm-adapter.js";

export function startVrmPage({ model, ...commonConfig }) {
    if (!model || typeof model !== "object") throw new TypeError("model configuration is required");
    const blobStore = createBlobStore(commonConfig.persistence);
    const modelAdapter = new VrmAdapter({
        config: model,
        persistence: commonConfig.persistence,
        blobStore,
    });
    return startAvatarApp({ config: commonConfig, modelAdapter, blobStore });
}
