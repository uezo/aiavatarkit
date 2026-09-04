function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
        reader.addEventListener("error", () => reject(reader.error), { once: true });
        reader.readAsDataURL(blob);
    });
}

export async function prepareImageDataUrl(file, {
    maxLongEdge = 1536,
    jpegQuality = 0.7,
} = {}) {
    const bitmap = await createImageBitmap(file);
    try {
        const longEdge = Math.max(bitmap.width, bitmap.height);
        if (!longEdge) throw new Error("Image has invalid dimensions");

        const limit = Number.isFinite(maxLongEdge) && maxLongEdge > 0
            ? maxLongEdge
            : 1536;
        const scale = Math.min(1, limit / longEdge);
        const width = Math.max(1, Math.round(bitmap.width * scale));
        const height = Math.max(1, Math.round(bitmap.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Could not create image canvas");
        context.fillStyle = "#fff";
        context.fillRect(0, 0, width, height);
        context.drawImage(bitmap, 0, 0, width, height);

        const quality = Math.max(0, Math.min(1, jpegQuality));
        const blob = await new Promise((resolve, reject) => {
            canvas.toBlob(
                (result) => result ? resolve(result) : reject(new Error("Could not encode image")),
                "image/jpeg",
                quality,
            );
        });
        return blobToDataUrl(blob);
    } finally {
        bitmap.close?.();
    }
}

export function installRequestInput({ aiavatar, ui, imageOptions = {}, onSent = () => {} }) {
    const form = document.getElementById("requestTextForm");
    const input = document.getElementById("requestTextInput");
    const imageButton = document.getElementById("requestImageButton");
    const imageInput = document.getElementById("requestImageInput");
    const imageIcon = document.getElementById("requestImageIcon");
    const imagePreview = document.getElementById("requestImagePreview");
    const imageRemove = document.getElementById("requestImageRemove");

    if (!form || !input) return { dispose() {} };

    let imageDataUrl = null;
    let imageSelection = 0;

    const hasImageControls = imageButton
        && imageInput
        && imageIcon
        && imagePreview
        && imageRemove;

    const clearImage = () => {
        imageSelection += 1;
        imageDataUrl = null;
        if (!hasImageControls) return;
        imageInput.value = "";
        imageIcon.hidden = false;
        imagePreview.hidden = true;
        imagePreview.removeAttribute("src");
        imageRemove.hidden = true;
        imageButton.title = "Attach image";
    };

    const onImageButtonClick = () => imageInput.click();
    const onImageRemoveClick = () => clearImage();
    const onImageChange = async () => {
        const file = imageInput.files?.[0];
        if (!file) return;
        if (file.type && !file.type.toLowerCase().startsWith("image/")) {
            clearImage();
            return;
        }

        const selection = ++imageSelection;
        try {
            const dataUrl = await prepareImageDataUrl(file, imageOptions);
            if (selection !== imageSelection || !dataUrl) return;
            imageDataUrl = dataUrl;
            imageIcon.hidden = true;
            imagePreview.src = dataUrl;
            imagePreview.hidden = false;
            imageRemove.hidden = false;
            imageButton.title = file.name || "Replace image";
        } catch (error) {
            if (selection !== imageSelection) return;
            console.error("Failed to attach image:", error);
            clearImage();
        }
    };

    const onSubmit = (event) => {
        event.preventDefault();
        const text = input.value.trim();
        if (!text && !imageDataUrl) return;
        if (!aiavatar.chat(ui.sessionId, ui.userId, text, imageDataUrl)) return;

        onSent({ text, imageDataUrl });
        input.value = "";
        clearImage();
        ui.updateMessage("user", text || "📎 Image", false);
    };

    form.addEventListener("submit", onSubmit);
    if (hasImageControls) {
        imageButton.addEventListener("click", onImageButtonClick);
        imageInput.addEventListener("change", onImageChange);
        imageRemove.addEventListener("click", onImageRemoveClick);
    }
    return {
        dispose() {
            imageSelection += 1;
            form.removeEventListener("submit", onSubmit);
            if (hasImageControls) {
                imageButton.removeEventListener("click", onImageButtonClick);
                imageInput.removeEventListener("change", onImageChange);
                imageRemove.removeEventListener("click", onImageRemoveClick);
            }
        },
    };
}
