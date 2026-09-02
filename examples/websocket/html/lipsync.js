class LipSyncEngine {
    static VISEMES = Object.freeze(["A", "I", "U", "E", "O"]);

    static MOUTH_VISEMES = Object.freeze({
        closed: Object.freeze({}),
        half: Object.freeze({ A: 0.4 }),
        open: Object.freeze({ A: 1.0 }),
        u: Object.freeze({ U: 0.8 }),
        e: Object.freeze({ E: 0.7 }),
    });

    static clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    static input(input) {
        if (ArrayBuffer.isView(input) || Array.isArray(input)) {
            return {
                pcm: input,
                sampleRate: 16000,
                samplePosition: input.length,
                tSec: 0,
                gain: 1,
            };
        }
        const pcm = input?.pcm;
        if (!pcm || typeof pcm.length !== "number") {
            throw new TypeError("Lip sync input must include a pcm array");
        }
        const sampleRate = Number(input.sampleRate);
        if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
            throw new RangeError("Lip sync input sampleRate must be greater than zero");
        }
        return {
            pcm,
            sampleRate,
            samplePosition: Number.isFinite(input.samplePosition)
                ? input.samplePosition
                : pcm.length,
            tSec: Number.isFinite(input.tSec) ? input.tSec : 0,
            gain: Number.isFinite(input.gain) ? Math.max(0, input.gain) : 1,
        };
    }

    static frame(input, {
        sampleCount = 256,
        targetSampleRate = null,
        timeOffsetSec = 0,
    } = {}) {
        const audio = LipSyncEngine.input(input);
        const count = Math.max(1, Math.trunc(sampleCount));
        const targetRate = Number.isFinite(targetSampleRate) && targetSampleRate > 0
            ? targetSampleRate
            : audio.sampleRate;
        const sourceCount = Math.max(1, Math.ceil(count * audio.sampleRate / targetRate));
        const end = LipSyncEngine.clamp(
            Math.floor(audio.samplePosition + timeOffsetSec * audio.sampleRate),
            0,
            audio.pcm.length,
        );
        const start = end - sourceCount;
        const frame = new Float64Array(sourceCount);
        const sourceStart = Math.max(0, start);
        const targetStart = sourceStart - start;
        for (let i = sourceStart; i < end; i++) {
            const value = Number(audio.pcm[i]);
            frame[targetStart + i - sourceStart] = Number.isFinite(value) ? value : 0;
        }
        return { ...audio, pcm: frame };
    }

    static rms(pcm) {
        if (!pcm.length) return 0;
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
        return Math.sqrt(sum / pcm.length);
    }

    static analyze(input, { sampleCount = 256 } = {}) {
        const audio = LipSyncEngine.frame(input, { sampleCount });
        const rms = LipSyncEngine.rms(audio.pcm) * audio.gain;
        const fftSize = LipSyncEngine.nextPowerOfTwo(audio.pcm.length);
        const windowed = new Float64Array(fftSize);
        const denominator = Math.max(1, audio.pcm.length - 1);
        for (let i = 0; i < audio.pcm.length; i++) {
            const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / denominator);
            windowed[i] = audio.pcm[i] * window;
        }
        const spectrum = LipSyncEngine.fftMagnitude(windowed);
        const nyquistBin = fftSize / 2;
        let weighted = 0;
        let total = 0;
        for (let i = 0; i < nyquistBin; i++) {
            const magnitude = spectrum[i];
            weighted += magnitude * i;
            total += magnitude;
        }
        const centroid01 = total > 0
            ? LipSyncEngine.clamp((weighted / total) / nyquistBin, 0, 1)
            : 0;
        return { rms, centroid01 };
    }

    static nextPowerOfTwo(value) {
        let result = 1;
        while (result < value) result *= 2;
        return result;
    }

    static fftMagnitude(data) {
        const n = data.length;
        if (n === 0 || (n & (n - 1)) !== 0) {
            throw new RangeError("FFT input length must be a non-zero power of two");
        }
        const real = Float64Array.from(data);
        const imag = new Float64Array(n);

        for (let i = 1, j = 0; i < n; i++) {
            let bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                [real[i], real[j]] = [real[j], real[i]];
            }
        }

        for (let length = 2; length <= n; length *= 2) {
            const angle = -2 * Math.PI / length;
            const stepReal = Math.cos(angle);
            const stepImag = Math.sin(angle);
            for (let offset = 0; offset < n; offset += length) {
                let twiddleReal = 1;
                let twiddleImag = 0;
                const half = length / 2;
                for (let i = 0; i < half; i++) {
                    const even = offset + i;
                    const odd = even + half;
                    const oddReal = real[odd] * twiddleReal - imag[odd] * twiddleImag;
                    const oddImag = real[odd] * twiddleImag + imag[odd] * twiddleReal;
                    real[odd] = real[even] - oddReal;
                    imag[odd] = imag[even] - oddImag;
                    real[even] += oddReal;
                    imag[even] += oddImag;
                    const nextReal = twiddleReal * stepReal - twiddleImag * stepImag;
                    twiddleImag = twiddleReal * stepImag + twiddleImag * stepReal;
                    twiddleReal = nextReal;
                }
            }
        }

        const magnitude = new Float64Array(n);
        for (let i = 0; i < n; i++) magnitude[i] = Math.hypot(real[i], imag[i]);
        return magnitude;
    }

    constructor({
        audioHz = 30,
        cutoffHz = 8.0,
        minVowelInterval = 0.12,
        peakMargin = 0.02,
        historySeconds = 10,
        thresholds = {},
        levels = [],
        vowelBands = [],
        analysisSampleCount = 256,
    } = {}) {
        const cfg = { audioHz, cutoffHz, minVowelInterval, peakMargin, historySeconds };
        this.cfg = Object.freeze(cfg);
        this.analysisSampleCount = analysisSampleCount;

        // Default mouth opening levels and vowel bands
        const defaultLevels = [
            { thresh: 0.0, shape: "closed" },
            { thresh: 0.30, shape: "half" },
            { thresh: 0.52, shape: "open" },
        ];
        const defaultVowelBands = [
            { upper: 0.16, shape: "u" },
            { upper: 0.20, shape: "open" },
            { upper: 1.0, shape: "e" },
        ];

        this.levels = this.sortLevels(levels.length ? levels : defaultLevels);
        this.vowelBands = this.sortVowelBands(vowelBands.length ? vowelBands : defaultVowelBands);

        // 1-pole LPF coefficient (beta)
        this.beta = 1.0 - Math.exp(-2.0 * Math.PI * cutoffHz / audioHz);

        // Online normalization state
        this.normalization = {
            noise: 1e-4,
            peak: 1e-3,
            peakDecay: 0.995,
        };

        // Short-term smoothing
        this.smoothing = {
            rmsQueue: [],
            rmsQueueMax: 3,
            envLP: 0,
        };

        // History buffers
        this.histories = {
            env: [],
            centroid: [],
            max: Math.floor(audioHz * historySeconds),
        };

        // thresholds (auto-tuned)
        this.thresholds = {
            talk: thresholds.talk ?? 0.06,
            half: thresholds.half ?? this.findLevelThreshold("half") ?? 0.30,
            open: thresholds.open ?? this.findLevelThreshold("open") ?? 0.52,
            u: thresholds.u ?? this.findVowelUpper("u") ?? 0.16,
            e: thresholds.e ?? this.findVowelUpper("e") ?? 0.20,
        };
        this.syncThresholdsToLevels();

        // vowel logic
        this.currentOpenShape = "open";
        this.lastVowelChangeT = -999;

        // peak detection helpers
        this.ePrev2 = 0;
        this.ePrev1 = 0;

        this.mouthShape = "closed";
        this.env = 0;
        this.centroid = 0;
        this.processedFrameCount = 0;
    }

    initialize() {
        return this;
    }

    processAudioData(audio) {
        const input = LipSyncEngine.input(audio);
        const { rms, centroid01 } = LipSyncEngine.analyze(input, {
            sampleCount: this.analysisSampleCount,
        });
        const mouthShape = this._update({ rms, centroid01, tSec: input.tSec });
        const visemes = this.visemesForShape(mouthShape);
        const [mainViseme, mainVisemeWeight] = this.mainViseme(visemes);
        return {
            visemes,
            mainViseme,
            mainVisemeWeight,
        };
    }

    _update(inputOrRmsRaw, centroid01, tSec) {
        const input = (typeof inputOrRmsRaw === "object" && inputOrRmsRaw !== null)
            ? inputOrRmsRaw
            : { rmsRaw: inputOrRmsRaw, centroid01, tSec };

        const rmsRaw = Number.isFinite(input.rmsRaw ?? input.rms) ? (input.rmsRaw ?? input.rms) : 0;
        const centroidNorm = Number.isFinite(input.centroid01 ?? input.centroid)
            ? (input.centroid01 ?? input.centroid)
            : 0;
        const timeSec = Number.isFinite(input.tSec ?? input.timeSec ?? input.time)
            ? (input.tSec ?? input.timeSec ?? input.time)
            : 0;

        // Online normalization
        const { normalization, smoothing, histories, thresholds } = this;
        if (rmsRaw < normalization.noise + 0.0005) normalization.noise = 0.99 * normalization.noise + 0.01 * rmsRaw;
        else normalization.noise = 0.999 * normalization.noise + 0.001 * rmsRaw;

        normalization.peak = Math.max(rmsRaw, normalization.peak * normalization.peakDecay);
        const denom = Math.max(normalization.peak - normalization.noise, 1e-6);
        const rmsNorm = Math.pow(this.clamp((rmsRaw - normalization.noise) / denom, 0, 1), 0.5);

        // Short-term smoothing
        smoothing.rmsQueue.push(rmsNorm);
        if (smoothing.rmsQueue.length > smoothing.rmsQueueMax) smoothing.rmsQueue.shift();
        const rmsSm = smoothing.rmsQueue.reduce((a, b) => a + b, 0) / smoothing.rmsQueue.length;

        // Envelope low-pass
        smoothing.envLP = smoothing.envLP + this.beta * (rmsSm - smoothing.envLP);
        const env = this.clamp(0.75 * smoothing.envLP + 0.25 * rmsSm, 0, 1);

        this.env = env;
        this.centroid = this.clamp(centroidNorm, 0, 1);

        // History
        histories.env.push(env);
        histories.centroid.push(this.centroid);
        if (histories.env.length > histories.max) histories.env.shift();
        if (histories.centroid.length > histories.max) histories.centroid.shift();
        this.processedFrameCount++;

        // Threshold auto-update roughly every second
        if (histories.env.length > this.cfg.audioHz * 3
            && this.processedFrameCount % this.cfg.audioHz === 0) {
            this.autoUpdateThresholds();
        }

        // Mouth level: map arbitrary levels defined in the array
        const levelShape = this.pickLevelShape(env);
        let mouthShape = levelShape;

        // Vowel update only when env exceeds the open gate
        const openGate = this.thresholds.open;
        if (env >= openGate) {
            const isPeak =
                (this.ePrev2 < this.ePrev1) &&
                (this.ePrev1 >= env) &&
                (this.ePrev1 > openGate + this.cfg.peakMargin);

            if (isPeak && (timeSec - this.lastVowelChangeT) >= this.cfg.minVowelInterval) {
                const cm = this.meanLast(histories.centroid, 5, this.centroid);
                this.currentOpenShape = this.pickVowelShape(cm);
                this.lastVowelChangeT = timeSec;
            }
            mouthShape = this.currentOpenShape;
        }

        this.mouthShape = mouthShape;

        this.ePrev2 = this.ePrev1;
        this.ePrev1 = env;

        return this.mouthShape;
    }

    visemesForShape(mouthShape) {
        const weights = Object.fromEntries(LipSyncEngine.VISEMES.map((name) => [name, 0]));
        Object.assign(weights, LipSyncEngine.MOUTH_VISEMES[mouthShape] || {});
        return weights;
    }

    mainViseme(visemes) {
        let mainViseme = null;
        let mainVisemeWeight = 0;
        for (const viseme of LipSyncEngine.VISEMES) {
            const weight = Number(visemes[viseme]) || 0;
            if (weight > mainVisemeWeight) {
                mainViseme = viseme;
                mainVisemeWeight = weight;
            }
        }
        return [mainViseme, mainVisemeWeight];
    }

    autoUpdateThresholds() {
        const { thresholds, histories } = this;
        const vals = Float32Array.from(histories.env);
        const sorted = Array.from(vals).sort((a, b) => a - b);
        const k = Math.max(1, Math.floor(0.2 * sorted.length));
        const noiseFloorEnv = this.median(sorted.slice(0, k));
        thresholds.talk = this.clamp(noiseFloorEnv + 0.05, 0.03, 0.18);

        const talkVals = Array.from(vals).filter(v => v > thresholds.talk);
        if (talkVals.length > 20) {
            const half = this.percentile(talkVals, 25);
            const open = this.percentile(talkVals, 58);
            thresholds.half = Math.max(half, thresholds.talk + 0.02);
            thresholds.open = Math.max(open, thresholds.half + 0.05);

            const cents = histories.centroid;
            const openMask = histories.env.map(v => v >= thresholds.open);
            let centOpen = [];
            for (let i = 0; i < openMask.length; i++) if (openMask[i]) centOpen.push(cents[i]);
            if (centOpen.length <= 20) centOpen = cents.filter((_, i) => histories.env[i] > thresholds.talk);

            if (centOpen.length > 20) {
                thresholds.u = this.percentile(centOpen, 20);
                thresholds.e = this.percentile(centOpen, 80);
            }
        }

        this.syncThresholdsToLevels();
    }

    meanLast(arr, n, fallback) {
        const m = Math.min(n, arr.length);
        if (m <= 0) return fallback;
        let s = 0;
        for (let i = arr.length - m; i < arr.length; i++) s += arr[i];
        return s / m;
    }
    percentile(arr, p) {
        const a = Array.from(arr).sort((x, y) => x - y);
        const idx = (p / 100) * (a.length - 1);
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        if (lo === hi) return a[lo];
        const t = idx - lo;
        return a[lo] * (1 - t) + a[hi] * t;
    }
    median(a) {
        if (a.length === 0) return 0;
        const mid = Math.floor(a.length / 2);
        return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
    }

    pickLevelShape(env) {
        if (!this.levels.length) return "closed";
        let shape = this.levels[0].shape;
        for (let i = 0; i < this.levels.length; i++) {
            const level = this.levels[i];
            if (env >= level.thresh) shape = level.shape;
            else break;
        }
        return shape;
    }

    pickVowelShape(centroid) {
        if (!this.vowelBands.length) return "open";
        for (let i = 0; i < this.vowelBands.length; i++) {
            const band = this.vowelBands[i];
            if (centroid <= band.upper) return band.shape;
        }
        return this.vowelBands[this.vowelBands.length - 1].shape;
    }

    sortLevels(levels) {
        return [...levels].sort((a, b) => a.thresh - b.thresh);
    }

    sortVowelBands(bands) {
        return [...bands].sort((a, b) => a.upper - b.upper);
    }

    findLevelThreshold(shapeName) {
        const hit = this.levels.find(l => l.shape === shapeName);
        return hit ? hit.thresh : undefined;
    }

    findVowelUpper(shapeName) {
        const hit = this.vowelBands.find(b => b.shape === shapeName);
        return hit ? hit.upper : undefined;
    }

    syncThresholdsToLevels() {
        // Sync thresholds into level definitions
        const halfIdx = this.levels.findIndex(l => l.shape === "half");
        if (halfIdx >= 0) this.levels[halfIdx].thresh = this.thresholds.half;
        const openIdx = this.levels.findIndex(l => l.shape === "open");
        if (openIdx >= 0) this.levels[openIdx].thresh = this.thresholds.open;
        this.levels = this.sortLevels(this.levels);

        // Sync thresholds into vowel bands (only matching shapes)
        const uIdx = this.vowelBands.findIndex(b => b.shape === "u");
        if (uIdx >= 0) this.vowelBands[uIdx].upper = this.thresholds.u;
        const eIdx = this.vowelBands.findIndex(b => b.shape === "e");
        if (eIdx >= 0) this.vowelBands[eIdx].upper = this.thresholds.e;
        this.vowelBands = this.sortVowelBands(this.vowelBands);
    }

    clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
}
