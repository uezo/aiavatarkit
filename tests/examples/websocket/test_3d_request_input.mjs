import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const controllerSource = await readFile(
    new URL("avatar3d/common/request-input-controller.js", htmlDirectory),
    "utf8",
);
const displaySource = await readFile(
    new URL("avatar3d/common/display-controller.js", htmlDirectory),
    "utf8",
);
const threeDSource = await readFile(new URL("3d.html", htmlDirectory), "utf8");
const { installRequestInput, prepareImageDataUrl } = await import(
    `data:text/javascript;base64,${Buffer.from(controllerSource).toString("base64")}`
);
const { DisplayController } = await import(
    `data:text/javascript;base64,${Buffer.from(displaySource).toString("base64")}`
);

function eventTarget(properties = {}) {
    const listeners = new Map();
    return {
        ...properties,
        addEventListener(name, listener) {
            listeners.set(name, listener);
        },
        removeEventListener(name, listener) {
            if (listeners.get(name) === listener) listeners.delete(name);
        },
        dispatch(name, event = {}) {
            return listeners.get(name)?.(event);
        },
        hasListener(name) {
            return listeners.has(name);
        },
    };
}

test("request text input is disabled by default", () => {
    assert.match(threeDSource, /id="requestTextForm"[^>]*hidden/);
    assert.match(threeDSource, /showRequestInput: false/);
    assert.match(threeDSource, /id="requestImageInput" accept="image\/\*" hidden/);
    assert.doesNotMatch(threeDSource, /id="requestImageInput"[^>]*multiple/);
});

test("request form attaches one image and clears it after sending", async () => {
    const previousDocument = globalThis.document;
    const previousFileReader = globalThis.FileReader;
    const previousCreateImageBitmap = globalThis.createImageBitmap;
    const form = eventTarget();
    const input = { value: "" };
    const imageButton = eventTarget({ title: "Attach image", clickCount: 0, click() { this.clickCount++; } });
    const imageInput = eventTarget({
        files: [{ name: "avatar.png", type: "image/png", dataUrl: "data:image/png;base64,AAAA" }],
        value: "avatar.png",
        clickCount: 0,
        click() { this.clickCount++; },
    });
    const imageIcon = { hidden: false };
    const imagePreview = {
        hidden: true,
        src: "",
        removeAttribute(name) {
            if (name === "src") this.src = "";
        },
    };
    const imageRemove = eventTarget({ hidden: true });
    const elements = {
        requestTextForm: form,
        requestTextInput: input,
        requestImageButton: imageButton,
        requestImageInput: imageInput,
        requestImageIcon: imageIcon,
        requestImagePreview: imagePreview,
        requestImageRemove: imageRemove,
    };
    const chatCalls = [];
    const messages = [];
    const canvas = {
        width: 0,
        height: 0,
        getContext() {
            return { fillRect() {}, drawImage() {} };
        },
        toBlob(callback) {
            callback({ dataUrl: "data:image/jpeg;base64,BBBB" });
        },
    };
    globalThis.document = {
        getElementById: (id) => elements[id] || null,
        createElement: (name) => name === "canvas" ? canvas : null,
    };
    globalThis.createImageBitmap = async () => ({ width: 120, height: 80, close() {} });
    globalThis.FileReader = class {
        constructor() {
            this.listeners = new Map();
        }
        addEventListener(name, listener) {
            this.listeners.set(name, listener);
        }
        readAsDataURL(blob) {
            this.result = blob.dataUrl;
            this.listeners.get("load")?.();
        }
    };

    try {
        const controller = installRequestInput({
            aiavatar: {
                chat(...args) {
                    chatCalls.push(args);
                    return true;
                },
            },
            ui: {
                sessionId: "session-1",
                userId: "user-1",
                updateMessage: (...args) => messages.push(args),
            },
        });

        imageButton.dispatch("click");
        assert.equal(imageInput.clickCount, 1);
        await imageInput.dispatch("change");
        assert.equal(imageIcon.hidden, true);
        assert.equal(imagePreview.hidden, false);
        assert.equal(imagePreview.src, "data:image/jpeg;base64,BBBB");
        assert.equal(imageRemove.hidden, false);

        form.dispatch("submit", { preventDefault() {} });
        assert.deepEqual(chatCalls, [[
            "session-1",
            "user-1",
            "",
            "data:image/jpeg;base64,BBBB",
        ]]);
        assert.deepEqual(messages, [["user", "📎 Image", false]]);
        assert.equal(imageInput.value, "");
        assert.equal(imagePreview.hidden, true);
        assert.equal(imagePreview.src, "");
        assert.equal(imageRemove.hidden, true);

        controller.dispose();
        assert.equal(imageButton.hasListener("click"), false);
        assert.equal(imageInput.hasListener("change"), false);
        assert.equal(imageRemove.hasListener("click"), false);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
        if (previousFileReader === undefined) delete globalThis.FileReader;
        else globalThis.FileReader = previousFileReader;
        if (previousCreateImageBitmap === undefined) delete globalThis.createImageBitmap;
        else globalThis.createImageBitmap = previousCreateImageBitmap;
    }
});

test("attached images are resized and JPEG-compressed before sending", async () => {
    const previousDocument = globalThis.document;
    const previousFileReader = globalThis.FileReader;
    const previousCreateImageBitmap = globalThis.createImageBitmap;
    const calls = [];
    const bitmap = { width: 3000, height: 2000, close: () => calls.push(["close"]) };
    const context = {
        fillStyle: "",
        fillRect: (...args) => calls.push(["fillRect", ...args]),
        drawImage: (...args) => calls.push(["drawImage", ...args]),
    };
    const canvas = {
        width: 0,
        height: 0,
        getContext: () => context,
        toBlob(callback, type, quality) {
            calls.push(["toBlob", type, quality]);
            callback({ dataUrl: "data:image/jpeg;base64,COMPRESSED" });
        },
    };
    globalThis.document = { createElement: () => canvas };
    globalThis.createImageBitmap = async () => bitmap;
    globalThis.FileReader = class {
        constructor() {
            this.listeners = new Map();
        }
        addEventListener(name, listener) {
            this.listeners.set(name, listener);
        }
        readAsDataURL(blob) {
            this.result = blob.dataUrl;
            this.listeners.get("load")?.();
        }
    };

    try {
        const result = await prepareImageDataUrl(
            { name: "large.jpg", type: "image/jpeg" },
            { maxLongEdge: 1536, jpegQuality: 0.7 },
        );

        assert.equal(result, "data:image/jpeg;base64,COMPRESSED");
        assert.equal(canvas.width, 1536);
        assert.equal(canvas.height, 1024);
        assert.deepEqual(calls, [
            ["fillRect", 0, 0, 1536, 1024],
            ["drawImage", bitmap, 0, 0, 1536, 1024],
            ["toBlob", "image/jpeg", 0.7],
            ["close"],
        ]);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
        if (previousFileReader === undefined) delete globalThis.FileReader;
        else globalThis.FileReader = previousFileReader;
        if (previousCreateImageBitmap === undefined) delete globalThis.createImageBitmap;
        else globalThis.createImageBitmap = previousCreateImageBitmap;
    }
});

test("request form sends trimmed text through the active conversation", () => {
    const previousDocument = globalThis.document;
    const form = eventTarget();
    const input = { value: "  Hello avatar  " };
    const chatCalls = [];
    const messages = [];
    globalThis.document = {
        getElementById(id) {
            return { requestTextForm: form, requestTextInput: input }[id] || null;
        },
    };

    try {
        const controller = installRequestInput({
            aiavatar: {
                chat(...args) {
                    chatCalls.push(args);
                    return true;
                },
            },
            ui: {
                sessionId: "session-1",
                userId: "user-1",
                updateMessage: (...args) => messages.push(args),
            },
        });
        let prevented = false;
        form.dispatch("submit", { preventDefault: () => { prevented = true; } });

        assert.equal(prevented, true);
        assert.deepEqual(chatCalls, [["session-1", "user-1", "Hello avatar", null]]);
        assert.deepEqual(messages, [["user", "Hello avatar", false]]);
        assert.equal(input.value, "");

        controller.dispose();
        assert.equal(form.hasListener("submit"), false);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test("request form keeps unsent text when the conversation is disconnected", () => {
    const previousDocument = globalThis.document;
    const form = eventTarget();
    const input = { value: "Try again" };
    let updateCount = 0;
    globalThis.document = {
        getElementById(id) {
            return { requestTextForm: form, requestTextInput: input }[id] || null;
        },
    };

    try {
        installRequestInput({
            aiavatar: { chat: () => false },
            ui: { sessionId: "session-1", userId: "user-1", updateMessage: () => { updateCount++; } },
        });
        form.dispatch("submit", { preventDefault() {} });

        assert.equal(input.value, "Try again");
        assert.equal(updateCount, 0);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test("display state toggles request input visibility", () => {
    const previousDocument = globalThis.document;
    const form = { hidden: true };
    const messageBox = {
        classList: {
            toggle() {},
            remove() {},
        },
    };
    globalThis.document = {
        querySelector(selector) {
            if (selector === ".message-inner") return { style: {} };
            if (selector === ".vn-menu") return { style: {} };
            return null;
        },
        getElementById(id) {
            return {
                messageBox,
                requestTextForm: form,
                micGlow: { classList: { remove() {} } },
            }[id] || null;
        },
    };

    try {
        const display = Object.create(DisplayController.prototype);
        display.state = {
            messageBoxOpacity: 80,
            showMenu: true,
            autoHide: false,
            showRequestInput: true,
            characterName: "",
            userName: "",
            showMicGlow: true,
        };
        display.ui = {};
        display.applyState();
        assert.equal(form.hidden, false);

        display.state.showRequestInput = false;
        display.applyState();
        assert.equal(form.hidden, true);
    } finally {
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});
