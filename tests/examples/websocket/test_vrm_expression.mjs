import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const htmlDirectory = new URL("../../../examples/websocket/html/", import.meta.url);
const vrmIdleSource = await readFile(new URL("vrm-idle.js", htmlDirectory), "utf8");
const VRMIdle = new Function(`${vrmIdleSource}; return VRMIdle;`)();

function createIdle(currentFaceName = "happy") {
    const calls = [];
    const idle = Object.create(VRMIdle.prototype);
    idle.currentFaceName = currentFaceName;
    idle._exprTimeout = null;
    idle._exprUpdateId = 0;
    idle._vrm = {
        expressionManager: {
            setValue(name, value) {
                calls.push([name, value]);
            },
        },
    };
    return { idle, calls };
}

test("neutral is applied as an explicit VRM expression without a revert timer", () => {
    const { idle, calls } = createIdle("happy");

    idle.applyExpression(undefined, 2);

    assert.equal(idle.currentFaceName, "neutral");
    assert.equal(idle._exprTimeout, null);
    assert.deepEqual(calls, [
        ["happy", 0],
        ["neutral", 1],
    ]);
});

test("neutral is cleared before another VRM expression is applied", () => {
    const { idle, calls } = createIdle("neutral");

    idle.applyExpression("Joy", 0);

    assert.equal(idle.currentFaceName, "happy");
    assert.deepEqual(calls, [
        ["neutral", 0],
        ["happy", 1],
    ]);
});
