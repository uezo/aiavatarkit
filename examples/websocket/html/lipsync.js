class LipSyncEngine {
    constructor({
        audioHz = 60,
        cutoffHz = 8.0,
        minVowelInterval = 0.12,
        peakMargin = 0.02,
        historySeconds = 10,
        thresholds = {},
        levels = [],
        vowelBands = [],
        applyTarget = null,
        mouthPathTemplate = "images/mouth_{mouth}.png",
        hideOnClosed = true,
    } = {}) {
        const cfg = { audioHz, cutoffHz, minVowelInterval, peakMargin, historySeconds };
        this.cfg = Object.freeze(cfg);
        this.applyTarget = applyTarget;
        this.mouthPathTemplate = mouthPathTemplate;
        this.hideOnClosed = hideOnClosed;
        this.mouthCache = new Map(); // mouth -> object URL
        this.mouthPreloaded = false;

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

        // fire-and-forget preload for default shapes
        this.preloadDefaultMouths();
    }

    // Update mouth shape based on analysis results
    update(inputOrRmsRaw, centroid01, tSec) {
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

        // Threshold auto-update roughly every second
        if (histories.env.length > this.cfg.audioHz * 3 && (histories.env.length % this.cfg.audioHz === 0)) {
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

    // Convenience: update + apply to image element using a path template
    apply({ rms, centroid01, tSec }) {
        const mouth = this.update({ rms, centroid01, tSec });
        const target = this.applyTarget;
        if (!target) return mouth;

        if (mouth === "closed" && this.hideOnClosed) {
            target.src = "";
            target.style.display = "none";
            return mouth;
        }

        if (!this.mouthPreloaded) return mouth; // skip until preload finishes
        const cached = this.mouthCache.get(mouth);
        if (!cached) return mouth; // skip if missing

        target.src = cached;
        target.style.display = "block";
        return mouth;
    }

    reset() {
        if (!this.applyTarget) return;
        this.applyTarget.src = "";
        this.applyTarget.style.display = "none";
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

    async preloadDefaultMouths() {
        const mouths = new Set();
        this.levels.forEach(l => mouths.add(l.shape));
        this.vowelBands.forEach(v => mouths.add(v.shape));
        mouths.delete("closed"); // skip closed
        const tasks = [...mouths].map(mouth => this.fetchMouth(mouth).catch((err) => {
            console.warn(`LipSyncEngine: failed to preload mouth "${mouth}"`, err);
        }));
        try {
            await Promise.all(tasks);
        } finally {
            this.mouthPreloaded = true;
        }
    }

    async fetchMouth(mouth) {
        if (this.mouthCache.has(mouth)) return this.mouthCache.get(mouth);
        const path = this.buildMouthPath(mouth);
        const res = await fetch(path);
        if (!res.ok) throw new Error(`Failed to load mouth: ${mouth}`);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        this.mouthCache.set(mouth, url);
        return url;
    }

    buildMouthPath(mouth) {
        return this.mouthPathTemplate.replace("{mouth}", mouth);
    }

    clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
}
