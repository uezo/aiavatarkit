#!/usr/bin/env node

// Generates an MFCC Profile JSON compatible with MFCCLipSyncEngine.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const DEFAULT_ENGINE_PATH = resolve(SCRIPT_DIRECTORY, "../html/mfcc-lipsync.js");

export const VOWELS = Object.freeze([
    { fileName: "a.wav", phoneme: "A" },
    { fileName: "i.wav", phoneme: "I" },
    { fileName: "u.wav", phoneme: "U" },
    { fileName: "e.wav", phoneme: "E" },
    { fileName: "o.wav", phoneme: "O" },
]);

export const PROFILE_SETTINGS = Object.freeze({
    mfccNum: 12,
    mfccDataCount: 16,
    melFilterBankChannels: 26,
    targetSampleRate: 16000,
    sampleCount: 1024,
    useStandardization: false,
    compareMethod: 2,
});

const FRAME_HOP_SAMPLES = 256;
const ACTIVE_RMS_RATIO = 0.2;
const MIN_ACTIVE_RMS = 1e-4;
const EDGE_TRIM_RATIO = 0.15;
const MIN_COMBINED_RUN_FRAMES = 5;
const WAVE_EXTENSIBLE_GUID_TAIL = Buffer.from("00001000800000aa00389b71", "hex");
const QUALITY_WARNING_THRESHOLDS = Object.freeze({
    stabilityP10Cosine: 0.95,
    strongStabilityP10Cosine: 0.9,
    strongMinimumStabilityCosine: 0.87,
    marginP10Cosine: 0.02,
    overlapCentroidCosine: 0.9,
    overlapRatio: 1,
    strongOverlapRatio: 0.75,
});

function fail(message) {
    throw new Error(message);
}

function readPcmSample(buffer, offset, format, bitsPerSample) {
    if (format === 3) {
        if (bitsPerSample === 32) return buffer.readFloatLE(offset);
        if (bitsPerSample === 64) return buffer.readDoubleLE(offset);
        fail(`Unsupported IEEE float WAV bit depth: ${bitsPerSample}`);
    }

    if (format !== 1) fail(`Unsupported WAV audio format: ${format}`);
    if (bitsPerSample === 8) return (buffer.readUInt8(offset) - 128) / 128;
    if (bitsPerSample === 16) return buffer.readInt16LE(offset) / 32768;
    if (bitsPerSample === 24) {
        let value = buffer.readUIntLE(offset, 3);
        if (value & 0x800000) value -= 0x1000000;
        return value / 8388608;
    }
    if (bitsPerSample === 32) return buffer.readInt32LE(offset) / 2147483648;
    fail(`Unsupported PCM WAV bit depth: ${bitsPerSample}`);
}

export function decodeWav(input, sourceName = "WAV input") {
    const buffer = Buffer.isBuffer(input) ? input : Buffer.from(input);
    if (buffer.length < 12
        || buffer.toString("ascii", 0, 4) !== "RIFF"
        || buffer.toString("ascii", 8, 12) !== "WAVE") {
        fail(`${sourceName} is not a RIFF/WAVE file`);
    }

    let format = null;
    let dataOffset = null;
    let dataLength = null;
    for (let offset = 12; offset + 8 <= buffer.length;) {
        const id = buffer.toString("ascii", offset, offset + 4);
        const length = buffer.readUInt32LE(offset + 4);
        const start = offset + 8;
        const end = start + length;
        if (end > buffer.length) fail(`${sourceName} contains a truncated ${id} chunk`);

        if (id === "fmt ") {
            if (length < 16) fail(`${sourceName} contains an invalid fmt chunk`);
            const containerFormat = buffer.readUInt16LE(start);
            format = {
                audioFormat: containerFormat,
                channels: buffer.readUInt16LE(start + 2),
                sampleRate: buffer.readUInt32LE(start + 4),
                blockAlign: buffer.readUInt16LE(start + 12),
                bitsPerSample: buffer.readUInt16LE(start + 14),
            };
            if (containerFormat === 0xfffe) {
                const extensionLength = length >= 18 ? buffer.readUInt16LE(start + 16) : 0;
                const subFormat = length >= 40 ? buffer.readUInt32LE(start + 24) : 0;
                const hasStandardGuid = length >= 40 && buffer.subarray(
                    start + 28,
                    start + 40,
                ).equals(WAVE_EXTENSIBLE_GUID_TAIL);
                if (extensionLength < 22
                    || length < 18 + extensionLength
                    || !hasStandardGuid
                    || (subFormat !== 1 && subFormat !== 3)) {
                    fail(`${sourceName} contains an unsupported extensible WAV format`);
                }
                format.audioFormat = subFormat;
            }
        } else if (id === "data" && dataOffset == null) {
            dataOffset = start;
            dataLength = length;
        }
        offset = end + (length & 1);
    }

    if (!format) fail(`${sourceName} does not contain a fmt chunk`);
    if (dataOffset == null) fail(`${sourceName} does not contain a data chunk`);
    if (format.channels < 1) fail(`${sourceName} has no audio channels`);
    if (format.sampleRate < 1) fail(`${sourceName} has an invalid sample rate`);
    const bytesPerSample = format.bitsPerSample / 8;
    if (!Number.isInteger(bytesPerSample) || bytesPerSample < 1) {
        fail(`${sourceName} has an invalid bit depth: ${format.bitsPerSample}`);
    }
    if (format.blockAlign !== bytesPerSample * format.channels) {
        fail(`${sourceName} has an invalid block alignment`);
    }
    if (dataLength % format.blockAlign !== 0) {
        fail(`${sourceName} contains an incomplete audio frame`);
    }

    const sampleCount = Math.floor(dataLength / format.blockAlign);
    const pcm = new Float64Array(sampleCount);
    for (let index = 0; index < sampleCount; index++) {
        const offset = dataOffset + index * format.blockAlign;
        const value = readPcmSample(
            buffer,
            offset,
            format.audioFormat,
            format.bitsPerSample,
        );
        pcm[index] = Number.isFinite(value)
            ? Math.max(-1, Math.min(1, value))
            : 0;
    }

    return {
        pcm,
        sampleRate: format.sampleRate,
        channels: format.channels,
        bitsPerSample: format.bitsPerSample,
        durationSec: sampleCount / format.sampleRate,
    };
}

async function loadMfccLipSyncEngine(enginePath = DEFAULT_ENGINE_PATH) {
    const source = await readFile(enginePath, "utf8");
    return new Function(`${source}\nreturn MFCCLipSyncEngine;`)();
}

function seedProfile() {
    return {
        ...PROFILE_SETTINGS,
        mfccs: [{
            name: "seed",
            mfccCalibrationDataList: [{
                array: Array(PROFILE_SETTINGS.mfccNum).fill(0),
            }],
        }],
    };
}

function activeRuns(frames, threshold) {
    const runs = [];
    let currentStart = -1;
    for (let index = 0; index <= frames.length; index++) {
        const active = index < frames.length && frames[index].rawVolume >= threshold;
        if (active && currentStart < 0) currentStart = index;
        if (!active && currentStart >= 0) {
            runs.push(frames.slice(currentStart, index));
            currentStart = -1;
        }
    }
    return runs;
}

function combinedStableFrames(runs) {
    const usableRuns = [];
    const stableRuns = [];
    const frames = [];
    for (const run of runs) {
        if (run.length < MIN_COMBINED_RUN_FRAMES) continue;
        const edgeTrim = Math.min(
            Math.floor((run.length - 1) / 2),
            Math.max(2, Math.floor(run.length * EDGE_TRIM_RATIO)),
        );
        const stable = run.slice(edgeTrim, run.length - edgeTrim);
        if (stable.length === 0) continue;
        usableRuns.push(run);
        stableRuns.push(stable);
        frames.push(...stable);
    }
    return { frames, usableRuns, stableRuns };
}

function percentile(values, ratio) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.min(
        sorted.length - 1,
        Math.max(0, Math.floor((sorted.length - 1) * ratio)),
    );
    return sorted[index];
}

function averageMfcc(frames) {
    const average = new Float64Array(PROFILE_SETTINGS.mfccNum);
    for (const frame of frames) {
        for (let index = 0; index < average.length; index++) {
            average[index] += frame.mfcc[index];
        }
    }
    for (let index = 0; index < average.length; index++) average[index] /= frames.length;
    return average;
}

function cosineSimilarity(a, b) {
    let product = 0;
    let aNorm = 0;
    let bNorm = 0;
    for (let index = 0; index < a.length; index++) {
        product += a[index] * b[index];
        aNorm += a[index] * a[index];
        bNorm += b[index] * b[index];
    }
    const denominator = Math.sqrt(aNorm * bNorm);
    return denominator > 0 ? product / denominator : -1;
}

function cosineAngleDegrees(similarity) {
    const clamped = Math.max(-1, Math.min(1, similarity));
    return Math.acos(clamped) * 180 / Math.PI;
}

function summarizeNumbers(values) {
    return {
        mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        p10: percentile(values, 0.1),
        minimum: Math.min(...values),
        maximum: Math.max(...values),
    };
}

function selectDistributedFrames(frames, count) {
    const centroid = averageMfcc(frames);
    const selected = [];
    for (let bin = 0; bin < count; bin++) {
        const start = Math.floor(bin * frames.length / count);
        const end = Math.max(start + 1, Math.floor((bin + 1) * frames.length / count));
        let best = frames[start];
        let bestSimilarity = cosineSimilarity(best.mfcc, centroid);
        for (let index = start + 1; index < end; index++) {
            const similarity = cosineSimilarity(frames[index].mfcc, centroid);
            if (similarity > bestSimilarity) {
                best = frames[index];
                bestSimilarity = similarity;
            }
        }
        selected.push(best);
    }
    return selected;
}

export function analyzeVowel(engine, audio, phoneme, sourceName) {
    if (audio.sampleRate !== PROFILE_SETTINGS.targetSampleRate) {
        fail(
            `${sourceName} is ${audio.sampleRate} Hz; `
            + `expected ${PROFILE_SETTINGS.targetSampleRate} Hz`,
        );
    }
    if (audio.pcm.length < PROFILE_SETTINGS.sampleCount) {
        fail(`${sourceName} is too short for a 64 ms MFCC window`);
    }

    const positions = [];
    for (
        let position = PROFILE_SETTINGS.sampleCount;
        position <= audio.pcm.length;
        position += FRAME_HOP_SAMPLES
    ) {
        positions.push(position);
    }
    if (positions.at(-1) !== audio.pcm.length) positions.push(audio.pcm.length);

    const frames = positions.map((samplePosition) => {
        const features = engine.extractMfcc({
            pcm: audio.pcm,
            sampleRate: audio.sampleRate,
            samplePosition,
        });
        return {
            samplePosition,
            rawVolume: features.rawVolume,
            mfcc: features.mfcc,
        };
    });
    const volumes = frames.map(({ rawVolume }) => rawVolume);
    let maxVolume = 0;
    for (const volume of volumes) maxVolume = Math.max(maxVolume, volume);
    if (!(maxVolume > MIN_ACTIVE_RMS)) {
        fail(`${sourceName} does not contain a detectable ${phoneme} vowel`);
    }

    // A transient click can be the loudest sample in a TTS file. Basing the
    // threshold on p90 keeps that click from hiding the sustained vowel.
    const referenceVolume = percentile(volumes, 0.9);
    const activeThreshold = Math.max(MIN_ACTIVE_RMS, referenceVolume * ACTIVE_RMS_RATIO);
    const runs = activeRuns(frames, activeThreshold);
    let activeRun = runs[0] || [];
    for (const run of runs) {
        if (run.length > activeRun.length) activeRun = run;
    }

    let selectionMode = "continuous";
    let usedRunCount = 1;
    let stableFrames;
    let stableRuns;
    if (activeRun.length >= PROFILE_SETTINGS.mfccDataCount) {
        // Preserve the original sustained-vowel path exactly when one run is
        // already long enough.
        const edgeTrim = Math.floor(activeRun.length * EDGE_TRIM_RATIO);
        stableFrames = activeRun.slice(edgeTrim, activeRun.length - edgeTrim);
        if (stableFrames.length < PROFILE_SETTINGS.mfccDataCount) stableFrames = activeRun;
        stableRuns = [stableFrames];
    } else {
        const combined = combinedStableFrames(runs);
        stableFrames = combined.frames;
        stableRuns = combined.stableRuns;
        usedRunCount = combined.usableRuns.length;
        selectionMode = "combined";
        if (stableFrames.length < PROFILE_SETTINGS.mfccDataCount) {
            fail(
                `${sourceName} has no long ${phoneme} section `
                + `(longest=${activeRun.length} frames) and only `
                + `${stableFrames.length} stable frames across ${usedRunCount} repetitions; `
                + `provide more or longer repetitions`,
            );
        }
    }
    const selectedFrames = selectDistributedFrames(
        stableFrames,
        PROFILE_SETTINGS.mfccDataCount,
    );
    const toRange = (run) => ({
        startSec: run[0].samplePosition / audio.sampleRate,
        endSec: run.at(-1).samplePosition / audio.sampleRate,
    });
    const stableRanges = stableRuns.map(toRange);

    return {
        phoneme,
        sourceName,
        durationSec: audio.durationSec,
        channels: audio.channels,
        bitsPerSample: audio.bitsPerSample,
        maxVolume,
        activeThreshold,
        selectionMode,
        activeRunCount: runs.length,
        activeRunLengths: runs.map((run) => run.length),
        usedRunCount,
        stableFrameCount: stableFrames.length,
        longestRunStartSec: activeRun[0].samplePosition / audio.sampleRate,
        longestRunEndSec: activeRun.at(-1).samplePosition / audio.sampleRate,
        stableStartSec: selectionMode === "continuous" ? stableRanges[0].startSec : null,
        stableEndSec: selectionMode === "continuous" ? stableRanges[0].endSec : null,
        stableRanges,
        selectedFrames,
    };
}

function validateProfile(Engine, profile, analyses) {
    const engine = new Engine({ profile });
    const entryIndexByName = new Map(
        engine.entries.map((entry, index) => [entry.name, index]),
    );
    const profileEntryByName = new Map(profile.mfccs.map((entry) => [entry.name, entry]));
    const confusionRow = () => ({
        ...Object.fromEntries(VOWELS.map(({ phoneme }) => [phoneme, 0])),
        unclassified: 0,
    });
    const confusion = Object.fromEntries(
        VOWELS.map(({ phoneme }) => [phoneme, confusionRow()]),
    );
    const qualitySamples = Object.fromEntries(
        VOWELS.map(({ phoneme }) => [
            phoneme,
            {
                stability: [],
                margin: [],
                competitors: Object.fromEntries(
                    VOWELS.map(({ phoneme: candidate }) => [candidate, 0]),
                ),
                looCorrect: 0,
                looZeroScore: 0,
                looConfusion: confusionRow(),
            },
        ]),
    );
    const classifyDetails = (details) => {
        const scoreSum = details.reduce((sum, { score }) => sum + score, 0);
        if (!(scoreSum > 0)) return { winnerIndex: null, scoreSum };
        let winnerIndex = 0;
        for (let index = 1; index < details.length; index++) {
            if (details[index].score > details[winnerIndex].score) winnerIndex = index;
        }
        return { winnerIndex, scoreSum };
    };
    let correct = 0;
    let zeroScoreFrames = 0;
    let looCorrectFrames = 0;
    let looZeroScoreFrames = 0;
    let total = 0;
    for (const analysis of analyses) {
        const actualIndex = entryIndexByName.get(analysis.phoneme);
        const quality = qualitySamples[analysis.phoneme];
        const storedVectors = profileEntryByName
            .get(analysis.phoneme)
            .mfccCalibrationDataList
            .map(({ array }) => array);
        for (let frameIndex = 0; frameIndex < analysis.selectedFrames.length; frameIndex++) {
            const frame = analysis.selectedFrames[frameIndex];
            const details = engine.entries.map((entry) => (
                engine._scoreDetail(frame.mfcc, entry.average)
            ));
            let competitorIndex = actualIndex === 0 ? 1 : 0;
            for (let index = 0; index < details.length; index++) {
                if (index !== actualIndex
                    && details[index].metric > details[competitorIndex].metric) {
                    competitorIndex = index;
                }
            }
            quality.stability.push(details[actualIndex].metric);
            quality.margin.push(
                details[actualIndex].metric - details[competitorIndex].metric,
            );
            quality.competitors[engine.entries[competitorIndex].name]++;
            total++;
            const classification = classifyDetails(details);
            if (classification.winnerIndex == null) {
                zeroScoreFrames++;
                confusion[analysis.phoneme].unclassified++;
            } else {
                const winner = engine.entries[classification.winnerIndex].name;
                confusion[analysis.phoneme][winner]++;
                if (winner === analysis.phoneme) correct++;
            }

            const storedVector = storedVectors[frameIndex];
            const looAverage = new Float64Array(profile.mfccNum);
            for (let index = 0; index < looAverage.length; index++) {
                looAverage[index] = (
                    engine.entries[actualIndex].average[index] * storedVectors.length
                    - storedVector[index]
                ) / (storedVectors.length - 1);
            }
            const looDetails = engine.entries.map((entry, index) => (
                engine._scoreDetail(
                    frame.mfcc,
                    index === actualIndex ? looAverage : entry.average,
                )
            ));
            const looClassification = classifyDetails(looDetails);
            if (looClassification.winnerIndex == null) {
                quality.looZeroScore++;
                quality.looConfusion.unclassified++;
                looZeroScoreFrames++;
            } else {
                const looWinner = engine.entries[looClassification.winnerIndex].name;
                quality.looConfusion[looWinner]++;
                if (looWinner === analysis.phoneme) {
                    quality.looCorrect++;
                    looCorrectFrames++;
                }
            }
        }
    }

    const phonemes = {};
    const warnings = [];
    for (const analysis of analyses) {
        const phoneme = analysis.phoneme;
        const entryIndex = entryIndexByName.get(phoneme);
        const entry = engine.entries[entryIndex];
        let nearestIndex = entryIndex === 0 ? 1 : 0;
        let nearestCosine = cosineSimilarity(
            entry.average,
            engine.entries[nearestIndex].average,
        );
        for (let index = 0; index < engine.entries.length; index++) {
            if (index === entryIndex) continue;
            const similarity = cosineSimilarity(entry.average, engine.entries[index].average);
            if (similarity > nearestCosine) {
                nearestIndex = index;
                nearestCosine = similarity;
            }
        }
        const samples = qualitySamples[phoneme];
        const stability = summarizeNumbers(samples.stability);
        const stabilityAngles = samples.stability.map(cosineAngleDegrees);
        const margin = summarizeNumbers(samples.margin);
        const frameCount = analysis.selectedFrames.length;
        const misclassifiedFrames = frameCount - confusion[phoneme][phoneme];
        const looMisclassifiedFrames = frameCount - samples.looCorrect;
        phonemes[phoneme] = {
            frameCount,
            misclassifiedFrames,
            looCorrectFrames: samples.looCorrect,
            looMisclassifiedFrames,
            looZeroScoreFrames: samples.looZeroScore,
            looConfusion: samples.looConfusion,
            stability,
            stabilityP90AngleDegrees: percentile(stabilityAngles, 0.9),
            nearestPhoneme: engine.entries[nearestIndex].name,
            nearestCentroidCosine: nearestCosine,
            nearestCentroidAngleDegrees: cosineAngleDegrees(nearestCosine),
            margin,
            competitorCounts: samples.competitors,
        };

        if (misclassifiedFrames > 0) {
            const destinations = [
                ...VOWELS.map(({ phoneme: candidate }) => candidate),
                "unclassified",
            ]
                .filter((candidate) => candidate !== phoneme && confusion[phoneme][candidate] > 0)
                .map((candidate) => `${candidate}:${confusion[phoneme][candidate]}`)
                .join(", ");
            warnings.push({
                code: "misclassified_frames",
                severity: "strong",
                phoneme,
                value: misclassifiedFrames,
                threshold: 0,
                message: `${phoneme}: ${misclassifiedFrames}/${frameCount} calibration frames `
                    + `were not classified as ${phoneme} (${destinations})`,
            });
        }
        if (looMisclassifiedFrames > 0) {
            warnings.push({
                code: "loo_misclassified_frames",
                severity: looMisclassifiedFrames >= 2 ? "strong" : "warning",
                phoneme,
                value: looMisclassifiedFrames,
                threshold: 0,
                message: `${phoneme}: leave-one-out classified `
                    + `${samples.looCorrect}/${frameCount} frames correctly `
                    + `(zero-score=${samples.looZeroScore})`,
            });
        }
        if (margin.p10 < QUALITY_WARNING_THRESHOLDS.marginP10Cosine) {
            warnings.push({
                code: "thin_margin",
                severity: margin.minimum <= 0 ? "strong" : "warning",
                phoneme,
                value: margin.p10,
                threshold: QUALITY_WARNING_THRESHOLDS.marginP10Cosine,
                message: `${phoneme}: classification boundary is thin (cosine margin `
                    + `p10=${margin.p10.toFixed(3)}, minimum=${margin.minimum.toFixed(3)})`,
            });
        }
        if (stability.p10 < QUALITY_WARNING_THRESHOLDS.stabilityP10Cosine
            || stability.minimum < QUALITY_WARNING_THRESHOLDS.strongMinimumStabilityCosine) {
            const strong = stability.p10
                < QUALITY_WARNING_THRESHOLDS.strongStabilityP10Cosine
                || stability.minimum
                    < QUALITY_WARNING_THRESHOLDS.strongMinimumStabilityCosine;
            const minimumOnly = stability.p10
                >= QUALITY_WARNING_THRESHOLDS.stabilityP10Cosine;
            warnings.push({
                code: "variable_calibration",
                severity: strong ? "strong" : "warning",
                phoneme,
                trigger: minimumOnly ? "minimum" : "p10",
                value: minimumOnly ? stability.minimum : stability.p10,
                threshold: minimumOnly
                    ? QUALITY_WARNING_THRESHOLDS.strongMinimumStabilityCosine
                    : QUALITY_WARNING_THRESHOLDS.stabilityP10Cosine,
                stabilityP10: stability.p10,
                stabilityMinimum: stability.minimum,
                message: `${phoneme}: calibration frames vary widely `
                    + `(stability mean=${stability.mean.toFixed(3)}, `
                    + `p10=${stability.p10.toFixed(3)}, `
                    + `minimum=${stability.minimum.toFixed(3)})`,
            });
        }
    }

    const closestPairs = [];
    for (let left = 0; left < engine.entries.length; left++) {
        for (let right = left + 1; right < engine.entries.length; right++) {
            const similarity = cosineSimilarity(
                engine.entries[left].average,
                engine.entries[right].average,
            );
            const angleDegrees = cosineAngleDegrees(similarity);
            const leftQuality = phonemes[engine.entries[left].name];
            const rightQuality = phonemes[engine.entries[right].name];
            const spread90Degrees = leftQuality.stabilityP90AngleDegrees
                + rightQuality.stabilityP90AngleDegrees;
            closestPairs.push({
                phonemes: [engine.entries[left].name, engine.entries[right].name],
                cosine: similarity,
                angleDegrees,
                spread90Degrees,
                overlapRatio: spread90Degrees > 1e-12
                    ? angleDegrees / spread90Degrees
                    : null,
                competitorScore: engine._scoreDetail(
                    engine.entries[left].average,
                    engine.entries[right].average,
                ).score,
            });
        }
    }
    closestPairs.sort((a, b) => b.cosine - a.cosine);
    for (const pair of closestPairs) {
        const overlaps = pair.overlapRatio != null
            && pair.overlapRatio < QUALITY_WARNING_THRESHOLDS.overlapRatio;
        if (pair.cosine <= QUALITY_WARNING_THRESHOLDS.overlapCentroidCosine
            || (!overlaps && pair.cosine <= 0.97)) {
            continue;
        }
        const strong = pair.cosine > 0.97
            || (pair.overlapRatio != null
                && pair.overlapRatio < QUALITY_WARNING_THRESHOLDS.strongOverlapRatio);
        const cosineOnly = !overlaps && pair.cosine > 0.97;
        warnings.push({
            code: "overlapping_centroids",
            severity: strong ? "strong" : "warning",
            phonemes: pair.phonemes,
            trigger: cosineOnly ? "centroidCosine" : "overlapRatio",
            value: cosineOnly ? pair.cosine : pair.overlapRatio,
            threshold: cosineOnly ? 0.97 : QUALITY_WARNING_THRESHOLDS.overlapRatio,
            centroidCosine: pair.cosine,
            overlapRatio: pair.overlapRatio,
            message: `${pair.phonemes.join("/")}: centroids are close `
                + `(cosine=${pair.cosine.toFixed(3)}, `
                + `angle=${pair.angleDegrees.toFixed(1)}deg, `
                + `spread90=${pair.spread90Degrees.toFixed(1)}deg, `
                + `ratio=${pair.overlapRatio?.toFixed(2) ?? "n/a"})`,
        });
    }

    return {
        totalFrames: total,
        correctFrames: correct,
        accuracy: total > 0 ? correct / total : 0,
        zeroScoreFrames,
        confusion,
        quality: {
            metric: "cosine",
            heuristic: true,
            thresholds: { ...QUALITY_WARNING_THRESHOLDS },
            phonemes,
            closestPairs,
            warnings,
            looCorrectFrames,
            looAccuracy: total > 0 ? looCorrectFrames / total : 0,
            looZeroScoreFrames,
        },
    };
}

export async function buildProfile({
    inputDirectory,
    outputPath = null,
    enginePath = DEFAULT_ENGINE_PATH,
} = {}) {
    if (!inputDirectory) fail("inputDirectory is required");
    const absoluteInputDirectory = resolve(inputDirectory);
    const absoluteOutputPath = resolve(
        outputPath ?? join(absoluteInputDirectory, "mfcc-profile.json"),
    );
    const inputPaths = VOWELS.map(({ fileName }) => join(absoluteInputDirectory, fileName));
    if (inputPaths.includes(absoluteOutputPath)) {
        fail(`Output path would overwrite a calibration WAV: ${absoluteOutputPath}`);
    }
    const Engine = await loadMfccLipSyncEngine(enginePath);
    const analyzer = new Engine({ profile: seedProfile() });
    const analyses = [];
    for (let index = 0; index < VOWELS.length; index++) {
        const { phoneme } = VOWELS[index];
        const sourcePath = inputPaths[index];
        let wav;
        try {
            wav = await readFile(sourcePath);
        } catch (error) {
            if (error?.code === "ENOENT") fail(`Missing calibration file: ${sourcePath}`);
            throw error;
        }
        const audio = decodeWav(wav, sourcePath);
        analyses.push(analyzeVowel(analyzer, audio, phoneme, sourcePath));
    }

    const profile = {
        ...PROFILE_SETTINGS,
        mfccs: analyses.map(({ phoneme, selectedFrames }) => ({
            name: phoneme,
            mfccCalibrationDataList: selectedFrames.map(({ mfcc }) => ({
                array: Array.from(mfcc, (value) => Math.fround(value)),
            })),
        })),
    };
    const validation = validateProfile(Engine, profile, analyses);
    await mkdir(dirname(absoluteOutputPath), { recursive: true });
    await writeFile(absoluteOutputPath, `${JSON.stringify(profile, null, 2)}\n`, "utf8");
    return {
        inputDirectory: absoluteInputDirectory,
        outputPath: absoluteOutputPath,
        profile,
        analyses,
        validation,
    };
}

function formatSeconds(value) {
    return `${value.toFixed(3)}s`;
}

function formatQualityNumber(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function printResult(result) {
    console.log(`Wrote MFCC Profile: ${result.outputPath}`);
    for (const analysis of result.analyses) {
        const selection = analysis.selectionMode === "combined"
            ? `mode=combined, runs=${analysis.usedRunCount}, `
                + `stableFrames=${analysis.stableFrameCount}`
            : `mode=continuous, stable=${formatSeconds(analysis.stableStartSec)}`
                + `-${formatSeconds(analysis.stableEndSec)}`;
        console.log(
            `${analysis.phoneme}: ${formatSeconds(analysis.durationSec)}, `
            + `${selection}, `
            + `selected=${analysis.selectedFrames.length}`,
        );
    }
    const { validation } = result;
    console.log(
        `Self-check: ${validation.correctFrames}/${validation.totalFrames} `
        + `(${(validation.accuracy * 100).toFixed(1)}%), `
        + `zero-score=${validation.zeroScoreFrames}`,
    );
    for (const { phoneme } of VOWELS) {
        const counts = VOWELS
            .map(({ phoneme: candidate }) => `${candidate}:${validation.confusion[phoneme][candidate]}`)
            .join(" ");
        console.log(
            `  ${phoneme} -> ${counts} ?:${validation.confusion[phoneme].unclassified}`,
        );
    }

    const { quality } = validation;
    console.log("Quality analysis (cosine similarity; heuristic):");
    console.log(
        "  stability=own centroid mean/p10/min; "
        + "margin=p10/min vs each frame's strongest competitor",
    );
    for (const { phoneme } of VOWELS) {
        const item = quality.phonemes[phoneme];
        console.log(
            `  ${phoneme}: self=${item.frameCount - item.misclassifiedFrames}`
            + `/${item.frameCount}, LOO=${item.looCorrectFrames}/${item.frameCount}`
            + (item.looZeroScoreFrames > 0 ? `(zero=${item.looZeroScoreFrames})` : "")
            + ", "
            + `stability=${formatQualityNumber(item.stability.mean)}`
            + `/${formatQualityNumber(item.stability.p10)}`
            + `/${formatQualityNumber(item.stability.minimum)}, `
            + `margin=${formatQualityNumber(item.margin.p10)}`
            + `/${formatQualityNumber(item.margin.minimum)}, `
            + `nearest=${item.nearestPhoneme}`
            + `(${formatQualityNumber(item.nearestCentroidCosine)})`,
        );
    }

    console.log("Closest centroid pairs (lower overlap ratio means less separation):");
    for (const pair of quality.closestPairs.slice(0, 3)) {
        console.log(
            `  ${pair.phonemes.join("/")}: cosine=${formatQualityNumber(pair.cosine)}, `
            + `angle=${formatQualityNumber(pair.angleDegrees, 1)}deg, `
            + `spread90=${formatQualityNumber(pair.spread90Degrees, 1)}deg, `
            + `overlapRatio=${formatQualityNumber(pair.overlapRatio, 2)}`,
        );
    }

    if (quality.warnings.length === 0) {
        console.log("Quality warnings (heuristic): none");
    } else {
        console.log("Quality warnings (heuristic):");
        const severityOrder = { strong: 0, warning: 1 };
        const orderedWarnings = [...quality.warnings].sort((left, right) => (
            (severityOrder[left.severity] ?? 2) - (severityOrder[right.severity] ?? 2)
        ));
        for (const warning of orderedWarnings) {
            console.log(`  [${warning.severity.toUpperCase()}] ${warning.message}`);
        }
    }
}

function printUsage() {
    console.log(`Usage:
  node examples/websocket/tools/build-mfcc-profile.mjs <wav-directory> [output.json]

The directory must contain uncompressed 16 kHz PCM or IEEE-float WAV files named:
  a.wav  i.wav  u.wav  e.wav  o.wav

Each file may contain one sustained vowel or several repetitions separated by silence.

When output.json is omitted, mfcc-profile.json is written in the WAV directory.`);
}

async function main(args) {
    if (args.includes("--help") || args.includes("-h")) {
        printUsage();
        return;
    }
    if (args.length < 1 || args.length > 2) {
        printUsage();
        process.exitCode = 1;
        return;
    }
    const inputDirectory = resolve(args[0]);
    const outputPath = args[1]
        ? resolve(args[1])
        : join(inputDirectory, "mfcc-profile.json");
    const result = await buildProfile({ inputDirectory, outputPath });
    printResult(result);
}

const entryPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (entryPath === import.meta.url) {
    main(process.argv.slice(2)).catch((error) => {
        console.error(`Profile generation failed: ${error.message}`);
        process.exitCode = 1;
    });
}
