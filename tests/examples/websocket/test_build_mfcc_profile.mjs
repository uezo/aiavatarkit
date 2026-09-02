import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);
const repositoryDirectory = fileURLToPath(new URL("../../../", import.meta.url));
const toolUrl = new URL(
    "../../../examples/websocket/tools/build-mfcc-profile.mjs",
    import.meta.url,
);
const toolPath = fileURLToPath(toolUrl);
const { buildProfile } = await import(toolUrl);
const engineSource = await readFile(new URL(
    "../../../examples/websocket/html/mfcc-lipsync.js",
    import.meta.url,
), "utf8");
const MFCCLipSyncEngine = new Function(`${engineSource}\nreturn MFCCLipSyncEngine;`)();

const SAMPLE_RATE = 16000;
const VOWEL_FREQUENCIES = Object.freeze({
    A: [250, 750, 1250],
    I: [312.5, 2250, 3000],
    U: [187.5, 625, 1000],
    E: [375, 1750, 2750],
    O: [250, 500, 1000],
});

async function temporaryDirectory(t) {
    const directory = await mkdtemp(join(tmpdir(), "aiavatarkit-mfcc-profile-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    return directory;
}

function calibrationSignal(frequencies, {
    sampleRate = SAMPLE_RATE,
    durationSec = 1.5,
    activeStartSec = 0.2,
    activeEndSec = 1.3,
} = {}) {
    const sampleCount = Math.round(durationSec * sampleRate);
    const samples = new Float64Array(sampleCount);
    const activeStart = Math.round(activeStartSec * sampleRate);
    const activeEnd = Math.round(activeEndSec * sampleRate);
    for (let index = activeStart; index < activeEnd; index++) {
        const time = index / sampleRate;
        for (let frequencyIndex = 0; frequencyIndex < frequencies.length; frequencyIndex++) {
            samples[index] += 0.18 * Math.sin(
                2 * Math.PI * frequencies[frequencyIndex] * time + frequencyIndex * 0.37,
            );
        }
    }
    return samples;
}

function repeatedCalibrationSignal(frequencies, {
    sampleRate = SAMPLE_RATE,
    repetitions = 4,
    burstSec = 0.1,
    gapSec = 0.16,
    paddingSec = 0.2,
} = {}) {
    const durationSec = 2 * paddingSec
        + repetitions * burstSec
        + (repetitions - 1) * gapSec;
    const samples = new Float64Array(Math.round(durationSec * sampleRate));
    for (let repetition = 0; repetition < repetitions; repetition++) {
        const startSec = paddingSec + repetition * (burstSec + gapSec);
        const start = Math.round(startSec * sampleRate);
        const end = Math.round((startSec + burstSec) * sampleRate);
        for (let index = start; index < end; index++) {
            const time = index / sampleRate;
            for (let frequencyIndex = 0; frequencyIndex < frequencies.length; frequencyIndex++) {
                samples[index] += 0.18 * Math.sin(
                    2 * Math.PI * frequencies[frequencyIndex] * time + frequencyIndex * 0.37,
                );
            }
        }
    }
    return samples;
}

function mixedCalibrationSignal(firstFrequencies, secondFrequencies, {
    sampleRate = SAMPLE_RATE,
    durationSec = 1.5,
    activeStartSec = 0.2,
    activeEndSec = 1.3,
} = {}) {
    const samples = new Float64Array(Math.round(durationSec * sampleRate));
    const activeStart = Math.round(activeStartSec * sampleRate);
    const activeEnd = Math.round(activeEndSec * sampleRate);
    const split = Math.round((activeStart + activeEnd) / 2);
    for (let index = activeStart; index < activeEnd; index++) {
        const time = index / sampleRate;
        const frequencies = index < split ? firstFrequencies : secondFrequencies;
        for (let frequencyIndex = 0; frequencyIndex < frequencies.length; frequencyIndex++) {
            samples[index] += 0.18 * Math.sin(
                2 * Math.PI * frequencies[frequencyIndex] * time + frequencyIndex * 0.37,
            );
        }
    }
    return samples;
}

function monoWav(samples, { sampleRate = SAMPLE_RATE, format = "pcm16" } = {}) {
    const isFloat = format === "float32";
    const bytesPerSample = isFloat ? 4 : 2;
    const dataLength = samples.length * bytesPerSample;
    const buffer = Buffer.alloc(44 + dataLength);
    buffer.write("RIFF", 0, "ascii");
    buffer.writeUInt32LE(36 + dataLength, 4);
    buffer.write("WAVE", 8, "ascii");
    buffer.write("fmt ", 12, "ascii");
    buffer.writeUInt32LE(16, 16);
    buffer.writeUInt16LE(isFloat ? 3 : 1, 20);
    buffer.writeUInt16LE(1, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(sampleRate * bytesPerSample, 28);
    buffer.writeUInt16LE(bytesPerSample, 32);
    buffer.writeUInt16LE(bytesPerSample * 8, 34);
    buffer.write("data", 36, "ascii");
    buffer.writeUInt32LE(dataLength, 40);

    for (let index = 0; index < samples.length; index++) {
        const value = Math.max(-1, Math.min(1, samples[index]));
        const offset = 44 + index * bytesPerSample;
        if (isFloat) buffer.writeFloatLE(value, offset);
        else buffer.writeInt16LE(Math.round(value * 32767), offset);
    }
    return buffer;
}

async function writeCalibrationWavs(directory, {
    format = "pcm16",
    omit = null,
    signalFactory = calibrationSignal,
} = {}) {
    const signals = {};
    await Promise.all(Object.entries(VOWEL_FREQUENCIES).map(async ([phoneme, frequencies]) => {
        if (phoneme === omit) return;
        const samples = signalFactory(frequencies);
        signals[phoneme] = samples;
        await writeFile(
            join(directory, `${phoneme.toLowerCase()}.wav`),
            monoWav(samples, { format }),
        );
    }));
    return signals;
}

async function runCli(inputDirectory, outputPath = null) {
    const args = [toolPath, inputDirectory];
    if (outputPath) args.push(outputPath);
    return execFileAsync(process.execPath, args, {
        cwd: repositoryDirectory,
        encoding: "utf8",
    });
}

function assertProfileShape(profile) {
    assert.equal(profile.mfccNum, 12);
    assert.equal(profile.mfccDataCount, 16);
    assert.equal(profile.melFilterBankChannels, 26);
    assert.equal(profile.targetSampleRate, SAMPLE_RATE);
    assert.equal(profile.sampleCount, 1024);
    assert.equal(profile.useStandardization, false);
    assert.equal(profile.compareMethod, 2);
    assert.deepEqual(profile.mfccs.map(({ name }) => name), ["A", "I", "U", "E", "O"]);

    for (const entry of profile.mfccs) {
        assert.equal(entry.mfccCalibrationDataList.length, 16);
        for (const calibration of entry.mfccCalibrationDataList) {
            assert.equal(calibration.array.length, 12);
            assert.ok(calibration.array.every(Number.isFinite));
        }
    }
}

function assertSummaryShape(summary) {
    const tolerance = 1e-12;
    for (const key of ["mean", "p10", "minimum", "maximum"]) {
        assert.ok(Number.isFinite(summary[key]), `${key} must be finite`);
    }
    assert.ok(summary.minimum <= summary.mean + tolerance);
    assert.ok(summary.mean <= summary.maximum + tolerance);
    assert.ok(summary.minimum <= summary.p10 + tolerance);
    assert.ok(summary.p10 <= summary.maximum + tolerance);
}

function assertQualityShape(validation) {
    const quality = validation.quality;
    assert.equal(quality.metric, "cosine");
    assert.equal(quality.heuristic, true);
    assert.ok(Number.isFinite(quality.looAccuracy));
    assert.ok(Math.abs(
        quality.looAccuracy - quality.looCorrectFrames / validation.totalFrames
    ) < 1e-12);
    assert.ok(quality.looCorrectFrames >= 0);
    assert.ok(quality.looCorrectFrames <= validation.totalFrames);
    assert.ok(quality.looZeroScoreFrames >= 0);
    for (const value of Object.values(quality.thresholds)) assert.ok(Number.isFinite(value));
    assert.deepEqual(Object.keys(quality.phonemes), ["A", "I", "U", "E", "O"]);

    let summedLooCorrect = 0;
    let summedLooZero = 0;
    for (const [phoneme, item] of Object.entries(quality.phonemes)) {
        assert.equal(item.frameCount, 16);
        assert.ok(item.misclassifiedFrames >= 0 && item.misclassifiedFrames <= 16);
        assert.ok(item.looCorrectFrames >= 0 && item.looCorrectFrames <= 16);
        assert.ok(item.looMisclassifiedFrames >= 0 && item.looMisclassifiedFrames <= 16);
        assert.equal(item.looMisclassifiedFrames, item.frameCount - item.looCorrectFrames);
        assert.equal(item.looConfusion[phoneme], item.looCorrectFrames);
        assert.equal(
            Object.values(item.looConfusion).reduce((sum, count) => sum + count, 0),
            item.frameCount,
        );
        assert.equal(
            Object.values(item.competitorCounts).reduce((sum, count) => sum + count, 0),
            item.frameCount,
        );
        assert.equal(item.competitorCounts[phoneme], 0);
        assert.notEqual(item.nearestPhoneme, phoneme);
        assert.ok(Object.hasOwn(VOWEL_FREQUENCIES, item.nearestPhoneme));
        assert.ok(Number.isFinite(item.nearestCentroidCosine));
        assert.ok(item.nearestCentroidCosine >= -1 - 1e-12);
        assert.ok(item.nearestCentroidCosine <= 1 + 1e-12);
        assert.ok(item.nearestCentroidAngleDegrees >= 0);
        assert.ok(item.nearestCentroidAngleDegrees <= 180);
        assert.ok(Number.isFinite(item.stabilityP90AngleDegrees));
        assertSummaryShape(item.stability);
        assertSummaryShape(item.margin);
        assert.ok(item.stability.minimum >= -1e-12);
        assert.ok(item.stability.maximum <= 1 + 1e-12);
        assert.ok(item.margin.minimum >= -1 - 1e-12);
        assert.ok(item.margin.maximum <= 1 + 1e-12);
        summedLooCorrect += item.looCorrectFrames;
        summedLooZero += item.looZeroScoreFrames;
    }
    assert.equal(summedLooCorrect, quality.looCorrectFrames);
    assert.equal(summedLooZero, quality.looZeroScoreFrames);

    assert.equal(quality.closestPairs.length, 10);
    const pairNames = new Set();
    for (let index = 0; index < quality.closestPairs.length; index++) {
        const pair = quality.closestPairs[index];
        assert.equal(pair.phonemes.length, 2);
        assert.notEqual(pair.phonemes[0], pair.phonemes[1]);
        assert.ok(pair.phonemes.every((phoneme) => Object.hasOwn(VOWEL_FREQUENCIES, phoneme)));
        pairNames.add([...pair.phonemes].sort().join("/"));
        for (const key of ["cosine", "angleDegrees", "spread90Degrees", "competitorScore"]) {
            assert.ok(Number.isFinite(pair[key]), `${key} must be finite`);
        }
        assert.ok(pair.cosine >= -1 - 1e-12 && pair.cosine <= 1 + 1e-12);
        assert.ok(pair.angleDegrees >= 0 && pair.angleDegrees <= 180);
        assert.ok(pair.spread90Degrees >= 0);
        assert.ok(pair.competitorScore >= 0 && pair.competitorScore <= 1);
        const expectedAngle = Math.acos(Math.max(-1, Math.min(1, pair.cosine))) * 180 / Math.PI;
        assert.ok(Math.abs(pair.angleDegrees - expectedAngle) < 1e-12);
        assert.ok(pair.overlapRatio == null || Number.isFinite(pair.overlapRatio));
        assert.ok(pair.overlapRatio == null || pair.overlapRatio >= 0);
        if (index > 0) {
            assert.ok(quality.closestPairs[index - 1].cosine >= pair.cosine);
        }
    }
    assert.equal(pairNames.size, 10);

    for (const [phoneme, item] of Object.entries(quality.phonemes)) {
        const nearestPair = quality.closestPairs.find(({ phonemes }) => (
            phonemes.includes(phoneme)
        ));
        const nearestPhoneme = nearestPair.phonemes.find((candidate) => candidate !== phoneme);
        assert.equal(item.nearestPhoneme, nearestPhoneme);
        assert.ok(Math.abs(item.nearestCentroidCosine - nearestPair.cosine) < 1e-12);
    }

    for (const warning of quality.warnings) {
        assert.equal(typeof warning.code, "string");
        assert.ok(["strong", "warning"].includes(warning.severity));
        assert.equal(typeof warning.message, "string");
        assert.ok("value" in warning);
        assert.ok("threshold" in warning);
    }
}

test("profile builder CLI generates an MFCC profile from PCM16 vowel WAVs", async (t) => {
    const directory = await temporaryDirectory(t);
    const outputPath = join(directory, "generated", "voice-profile.json");
    const signals = await writeCalibrationWavs(directory);

    const { stdout, stderr } = await runCli(directory, outputPath);

    assert.match(stdout, /Wrote MFCC Profile:/);
    assert.equal(stdout.match(/mode=continuous/g)?.length, 5);
    assert.match(stdout, /Quality analysis \(cosine similarity; heuristic\):/);
    assert.equal(stdout.match(/^  [AIUEO]: self=/gm)?.length, 5);
    assert.match(stdout, /Closest centroid pairs/);
    assert.match(stdout, /Quality warnings \(heuristic\): none/);
    assert.equal(stderr, "");
    const profile = JSON.parse(await readFile(outputPath, "utf8"));
    assertProfileShape(profile);

    const engine = new MFCCLipSyncEngine({ profile });
    assert.deepEqual(engine.entries.map(({ name }) => name), ["A", "I", "U", "E", "O"]);
    const input = {
        pcm: signals.A,
        sampleRate: SAMPLE_RATE,
        samplePosition: Math.round(0.8 * SAMPLE_RATE),
    };
    const result = engine.processAudioData(input);
    assert.deepEqual(Object.keys(result.visemes), ["A", "I", "U", "E", "O"]);
    assert.ok(Object.values(result.visemes).every(Number.isFinite));
});

test("profile builder CLI accepts IEEE Float32 vowel WAVs", async (t) => {
    const directory = await temporaryDirectory(t);
    await writeCalibrationWavs(directory, { format: "float32" });

    const { stdout, stderr } = await runCli(directory);

    assert.match(stdout, /Wrote MFCC Profile:/);
    assert.equal(stderr, "");
    const profile = JSON.parse(await readFile(
        join(directory, "mfcc-profile.json"),
        "utf8",
    ));
    assertProfileShape(profile);
});

test("profile builder automatically combines repeated short vowel sections", async (t) => {
    const directory = await temporaryDirectory(t);
    await writeCalibrationWavs(directory, { signalFactory: repeatedCalibrationSignal });

    const result = await buildProfile({ inputDirectory: directory });

    assertProfileShape(result.profile);
    assertQualityShape(result.validation);
    const engine = new MFCCLipSyncEngine({ profile: result.profile });
    for (const analysis of result.analyses) {
        const entry = engine.entries.find(({ name }) => name === analysis.phoneme);
        const angles = analysis.selectedFrames
            .map(({ mfcc }) => {
                const cosine = engine._scoreDetail(mfcc, entry.average).metric;
                return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
            })
            .sort((left, right) => left - right);
        const expectedP90 = angles[Math.floor((angles.length - 1) * 0.9)];
        assert.ok(Math.abs(
            result.validation.quality.phonemes[analysis.phoneme]
                .stabilityP90AngleDegrees - expectedP90
        ) < 1e-12);
    }
    for (const analysis of result.analyses) {
        assert.equal(analysis.selectionMode, "combined");
        assert.deepEqual(analysis.activeRunLengths, [10, 10, 10, 10]);
        assert.equal(analysis.usedRunCount, 4);
        assert.equal(analysis.stableFrameCount, 24);
        assert.equal(analysis.selectedFrames.length, 16);

        const usedRanges = new Set();
        for (const frame of analysis.selectedFrames) {
            const timeSec = frame.samplePosition / SAMPLE_RATE;
            const rangeIndex = analysis.stableRanges.findIndex(({ startSec, endSec }) => (
                timeSec >= startSec && timeSec <= endSec
            ));
            assert.notEqual(rangeIndex, -1);
            usedRanges.add(rangeIndex);
        }
        assert.deepEqual([...usedRanges].sort((a, b) => a - b), [0, 1, 2, 3]);
    }
});

test("profile quality analysis warns about unstable and overlapping calibrations", async (t) => {
    const directory = await temporaryDirectory(t);
    await writeCalibrationWavs(directory);
    const mixed = mixedCalibrationSignal(VOWEL_FREQUENCIES.A, VOWEL_FREQUENCIES.I);
    await Promise.all(["a.wav", "i.wav"].map((fileName) => (
        writeFile(join(directory, fileName), monoWav(mixed))
    )));

    const result = await buildProfile({ inputDirectory: directory });

    assertQualityShape(result.validation);
    const { phonemes, closestPairs, thresholds } = result.validation.quality;
    assert.equal(phonemes.A.nearestPhoneme, "I");
    assert.equal(phonemes.I.nearestPhoneme, "A");
    assert.ok(phonemes.A.nearestCentroidCosine > 0.999);
    assert.ok(phonemes.I.nearestCentroidCosine > 0.999);
    assert.deepEqual(new Set(closestPairs[0].phonemes), new Set(["A", "I"]));
    for (const phoneme of ["A", "I"]) {
        assert.ok(phonemes[phoneme].stability.p10 < 0.8);
        assert.ok(Math.abs(phonemes[phoneme].margin.maximum) <= 1e-12);
        assert.ok(phonemes[phoneme].stability.p10 < thresholds.stabilityP10Cosine);
        assert.ok(phonemes[phoneme].margin.p10 < thresholds.marginP10Cosine);
    }
    const { warnings } = result.validation.quality;
    for (const phoneme of ["A", "I"]) {
        assert.ok(warnings.some((warning) => (
            warning.code === "variable_calibration" && warning.phoneme === phoneme
        )));
        assert.ok(warnings.some((warning) => (
            warning.code === "thin_margin" && warning.phoneme === phoneme
        )));
    }
    assert.ok(warnings.some(({ code, phonemes }) => (
        code === "overlapping_centroids"
        && phonemes.includes("A")
        && phonemes.includes("I")
    )));
    assert.ok(warnings.some(({ code, phoneme }) => (
        code === "misclassified_frames" && phoneme === "I"
    )));

    const { stdout, stderr } = await runCli(directory, join(directory, "warning-profile.json"));
    assert.match(stdout, /Quality warnings \(heuristic\):/);
    assert.match(stdout, /\[STRONG\]/);
    assert.match(stdout, /A\/I: centroids are close/);
    assert.equal(stderr, "");
});

test("profile builder CLI fails without every required vowel WAV", async (t) => {
    const directory = await temporaryDirectory(t);
    const outputPath = join(directory, "missing-profile.json");
    await writeCalibrationWavs(directory, { omit: "O" });

    await assert.rejects(
        runCli(directory, outputPath),
        (error) => {
            assert.equal(error.code, 1);
            assert.match(error.stderr, /Missing calibration file: .*o\.wav/);
            return true;
        },
    );
    await assert.rejects(
        access(outputPath),
        (error) => error.code === "ENOENT",
    );
});

test("profile builder CLI rejects a sample-rate mismatch instead of resampling", async (t) => {
    const directory = await temporaryDirectory(t);
    const outputPath = join(directory, "wrong-rate-profile.json");
    const sampleRate = 8000;
    const samples = calibrationSignal(VOWEL_FREQUENCIES.A, { sampleRate });
    await writeFile(join(directory, "a.wav"), monoWav(samples, { sampleRate }));

    await assert.rejects(
        runCli(directory, outputPath),
        (error) => {
            assert.equal(error.code, 1);
            assert.match(error.stderr, /a\.wav is 8000 Hz; expected 16000 Hz/);
            return true;
        },
    );
    await assert.rejects(
        access(outputPath),
        (error) => error.code === "ENOENT",
    );
});

test("profile builder CLI never overwrites a calibration WAV", async (t) => {
    const directory = await temporaryDirectory(t);
    await writeCalibrationWavs(directory);
    const inputPath = join(directory, "a.wav");
    const original = await readFile(inputPath);

    await assert.rejects(
        runCli(directory, inputPath),
        (error) => {
            assert.equal(error.code, 1);
            assert.match(error.stderr, /Output path would overwrite a calibration WAV/);
            return true;
        },
    );
    assert.deepEqual(await readFile(inputPath), original);
});
