import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(
    new URL("../../../examples/websocket/html/avatar3d/common/app.js", import.meta.url),
    "utf8",
);
const testableAppSource = appSource
    .replace(/^import .*$/gm, "")
    .replace("export async function startAvatarApp", "async function startAvatarApp");
const { importDroppedFiles } = await import(
    `data:text/javascript;base64,${Buffer.from(testableAppSource).toString("base64")}`
);

function file(name, type = "") {
    return { name, type };
}

test("dropping an image stores it as the background", async () => {
    const storedBackgrounds = [];
    const importedModels = [];

    await importDroppedFiles([file("background.png", "image/png")], {
        display: { storeBackground: async (image) => storedBackgrounds.push(image) },
        modelAdapter: { importFiles: async (files) => importedModels.push(files) },
    });

    assert.deepEqual(storedBackgrounds, [file("background.png", "image/png")]);
    assert.deepEqual(importedModels, []);
});

test("dropping model and image files routes both to the correct importer", async () => {
    const files = [
        file("avatar.vrm", "application/octet-stream"),
        file("background.webp", "image/webp"),
        file("idle.vrma"),
    ];
    const storedBackgrounds = [];
    const importedModels = [];

    await importDroppedFiles(files, {
        display: { storeBackground: async (image) => storedBackgrounds.push(image) },
        modelAdapter: { importFiles: async (modelFiles) => importedModels.push(modelFiles) },
    });

    assert.deepEqual(storedBackgrounds, [files[1]]);
    assert.deepEqual(importedModels, [[files[0], files[2]]]);
});

test("image extensions are recognized when the browser omits the MIME type", async () => {
    const files = [file("first.JPG"), file("last.svg")];
    const storedBackgrounds = [];

    await importDroppedFiles(files, {
        display: { storeBackground: async (image) => storedBackgrounds.push(image) },
        modelAdapter: { importFiles: async () => assert.fail("images must not reach the model adapter") },
    });

    assert.deepEqual(storedBackgrounds, [files[1]]);
});
