/**
 * MFCC-based lip sync engine with uLipSync v3 Profile JSON compatibility.
 *
 * It uses the uLipSync Profile JSON format so profiles calibrated with
 * uLipSync can be reused. Its MFCC processing implementation was also
 * developed with reference to the uLipSync v3 processing pipeline.
 *
 * uLipSync: https://github.com/hecomi/uLipSync
 * Copyright (c) 2021 hecomi, licensed under the MIT License.
 * See ../README.md#acknowledgements for attribution and license terms.
 */
class MFCCLipSyncEngine {
    static DEFAULT_PHONEME_MAP = Object.freeze({
        A: "A",
        I: "I",
        U: "U",
        E: "E",
        O: "O",
    });

    static VISEMES = Object.freeze(["A", "I", "U", "E", "O"]);

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
        const audio = MFCCLipSyncEngine.input(input);
        const count = Math.max(1, Math.trunc(sampleCount));
        const targetRate = Number.isFinite(targetSampleRate) && targetSampleRate > 0
            ? targetSampleRate
            : audio.sampleRate;
        const sourceCount = Math.max(1, Math.ceil(count * audio.sampleRate / targetRate));
        const end = MFCCLipSyncEngine.clamp(
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
        return {
            ...audio,
            pcm: frame,
        };
    }

    static rms(pcm) {
        if (!pcm.length) return 0;
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
        return Math.sqrt(sum / pcm.length);
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
        profile = null,
        profileUrl = null,
        minVolume = -2.5,
        maxVolume = -1.5,
        volumeGain = 1,
        timeOffsetSec = 0,
        phonemeMap = MFCCLipSyncEngine.DEFAULT_PHONEME_MAP,
    } = {}) {
        this.profileUrl = profileUrl;
        this.minVolume = minVolume;
        this.maxVolume = maxVolume;
        this.volumeGain = volumeGain;
        this.timeOffsetSec = timeOffsetSec;
        this.phonemeMap = { ...phonemeMap };
        this.profile = null;
        this.entries = [];
        this.means = new Float64Array(0);
        this.standardDeviations = new Float64Array(0);
        if (profile) this.setProfile(profile);
    }

    async initialize() {
        if (!this.profile && this.profileUrl) await this.loadProfile(this.profileUrl);
        if (!this.profile) {
            throw new Error("MFCCLipSyncEngine requires profile or profileUrl");
        }
        return this;
    }

    async loadProfile(source) {
        let profile = source;
        if (typeof source === "string") {
            const response = await fetch(source);
            if (!response.ok) {
                throw new Error(`Failed to load MFCC profile: HTTP ${response.status}`);
            }
            profile = await response.json();
        } else if (source && typeof source.text === "function") {
            profile = JSON.parse(await source.text());
        }
        return this.setProfile(profile);
    }

    setProfile(profileOrJson) {
        const profile = typeof profileOrJson === "string"
            ? JSON.parse(profileOrJson)
            : profileOrJson;
        if (!profile || typeof profile !== "object" || Array.isArray(profile)) {
            throw new TypeError("MFCC profile must be a JSON object");
        }

        const mfccNum = this._positiveInteger(profile.mfccNum ?? 12, "mfccNum");
        const melFilterBankChannels = this._positiveInteger(
            profile.melFilterBankChannels ?? 30,
            "melFilterBankChannels",
        );
        const targetSampleRate = this._positiveInteger(
            profile.targetSampleRate ?? 16000,
            "targetSampleRate",
        );
        const sampleCount = this._positiveInteger(profile.sampleCount ?? 1024, "sampleCount");
        if ((sampleCount & (sampleCount - 1)) !== 0) {
            throw new RangeError("MFCC profile sampleCount must be a power of two");
        }
        if (mfccNum >= melFilterBankChannels) {
            throw new RangeError("MFCC profile mfccNum must be smaller than melFilterBankChannels");
        }
        if (!Array.isArray(profile.mfccs) || profile.mfccs.length === 0) {
            throw new TypeError("MFCC profile must include at least one mfcc entry");
        }

        const mfccDataCount = this._positiveInteger(
            profile.mfccDataCount ?? 16,
            "mfccDataCount",
        );
        const calibrationVectors = [];
        const entries = profile.mfccs.map((entry, index) => {
            const name = String(entry?.name ?? "");
            if (!name) throw new TypeError(`MFCC profile mfccs[${index}].name is required`);
            const calibrationList = Array.isArray(entry.mfccCalibrationDataList)
                ? entry.mfccCalibrationDataList
                : [];
            const vectors = calibrationList.map((item, calibrationIndex) => {
                const values = item?.array;
                if (!Array.isArray(values) && !ArrayBuffer.isView(values)) {
                    throw new TypeError(
                        `MFCC profile ${name} calibration ${calibrationIndex} must include an array`,
                    );
                }
                if (values.length < mfccNum) {
                    throw new RangeError(
                        `MFCC profile ${name} calibration ${calibrationIndex} has too few MFCC values`,
                    );
                }
                const vector = new Float64Array(mfccNum);
                for (let i = 0; i < mfccNum; i++) {
                    const value = Number(values[i]);
                    if (!Number.isFinite(value)) {
                        throw new TypeError(
                            `MFCC profile ${name} calibration ${calibrationIndex} contains a non-finite value`,
                        );
                    }
                    vector[i] = value;
                }
                calibrationVectors.push(vector);
                return vector;
            });
            const averageVectors = vectors.slice(-mfccDataCount);
            const average = new Float64Array(mfccNum);
            for (const vector of averageVectors) {
                for (let i = 0; i < mfccNum; i++) average[i] += vector[i];
            }
            if (averageVectors.length > 0) {
                for (let i = 0; i < mfccNum; i++) average[i] /= averageVectors.length;
            }
            return { name, average };
        });

        const useStandardization = Boolean(profile.useStandardization);
        const means = new Float64Array(mfccNum);
        const standardDeviations = new Float64Array(mfccNum);
        standardDeviations.fill(1);
        if (useStandardization && calibrationVectors.length > 0) {
            for (const vector of calibrationVectors) {
                for (let i = 0; i < mfccNum; i++) means[i] += vector[i];
            }
            for (let i = 0; i < mfccNum; i++) means[i] /= calibrationVectors.length;
            standardDeviations.fill(0);
            for (const vector of calibrationVectors) {
                for (let i = 0; i < mfccNum; i++) {
                    const delta = vector[i] - means[i];
                    standardDeviations[i] += delta * delta;
                }
            }
            for (let i = 0; i < mfccNum; i++) {
                const deviation = Math.sqrt(standardDeviations[i] / calibrationVectors.length);
                standardDeviations[i] = deviation > 1e-12 ? deviation : 1;
            }
        }

        this.profile = {
            ...profile,
            mfccNum,
            mfccDataCount,
            melFilterBankChannels,
            targetSampleRate,
            sampleCount,
            useStandardization,
            compareMethod: this._compareMethod(profile.compareMethod ?? 1),
        };
        this.entries = entries;
        this.means = means;
        this.standardDeviations = standardDeviations;
        return this;
    }

    processAudioData(input) {
        this._requireProfile();
        const sourceAudio = MFCCLipSyncEngine.input(input);
        const features = this.extractMfcc(sourceAudio);
        const volume = this._normalizeVolume(
            features.rawVolume * features.audio.gain * this.volumeGain,
        );
        if (features.rawVolume <= 1e-12) {
            return this._emptyResult();
        }

        const scores = this.entries.map((entry) => (
            this._score(features.mfcc, entry.average)
        ));
        let scoreSum = 0;
        for (let i = 0; i < scores.length; i++) {
            const score = Number.isFinite(scores[i]) ? scores[i] : 0;
            scores[i] = score;
            scoreSum += score;
        }

        const phonemeRatios = {};
        for (let i = 0; i < scores.length; i++) {
            const name = this.entries[i].name;
            phonemeRatios[name] = (phonemeRatios[name] || 0)
                + (scoreSum > 0 ? scores[i] / scoreSum : 0);
        }
        let mainPhoneme = "";
        let mainRatio = -1;
        for (const [phoneme, ratio] of Object.entries(phonemeRatios)) {
            if (ratio > mainRatio) {
                mainPhoneme = phoneme;
                mainRatio = ratio;
            }
        }
        const mainViseme = this._visemeForPhoneme(mainPhoneme);
        return {
            visemes: this._visemes(phonemeRatios, volume),
            mainViseme,
            mainVisemeWeight: mainViseme ? volume : 0,
        };
    }

    extractMfcc(input) {
        this._requireProfile();
        const audio = MFCCLipSyncEngine.frame(input, {
            sampleCount: this.profile.sampleCount,
            targetSampleRate: this.profile.targetSampleRate,
            timeOffsetSec: this.timeOffsetSec,
        });
        const rawVolume = MFCCLipSyncEngine.rms(audio.pcm);
        if (rawVolume <= 1e-12) {
            const mfcc = new Float64Array(this.profile.mfccNum);
            return { mfcc, rawVolume, audio };
        }

        const data = this._resample(
            audio.pcm,
            audio.sampleRate,
            this.profile.targetSampleRate,
            this.profile.sampleCount,
        );
        const original = Float64Array.from(data);
        for (let i = 1; i < data.length; i++) data[i] = original[i] - 0.97 * original[i - 1];
        const denominator = Math.max(1, data.length - 1);
        let max = 0;
        for (let i = 0; i < data.length; i++) {
            data[i] *= 0.54 - 0.46 * Math.cos(2 * Math.PI * i / denominator);
            max = Math.max(max, Math.abs(data[i]));
        }
        if (max > Number.EPSILON) {
            for (let i = 0; i < data.length; i++) data[i] /= max;
        }

        const spectrum = MFCCLipSyncEngine.fftMagnitude(data);
        const melSpectrum = this._melFilterBank(
            spectrum,
            this.profile.targetSampleRate,
            this.profile.melFilterBankChannels,
        );
        for (let i = 0; i < melSpectrum.length; i++) {
            melSpectrum[i] = 10 * Math.log10(Math.max(melSpectrum[i], 1e-30));
        }

        const mfcc = new Float64Array(this.profile.mfccNum);
        const dctScale = Math.PI / melSpectrum.length;
        for (let coefficient = 1; coefficient <= mfcc.length; coefficient++) {
            let sum = 0;
            for (let band = 0; band < melSpectrum.length; band++) {
                sum += melSpectrum[band]
                    * Math.cos((band + 0.5) * coefficient * dctScale);
            }
            mfcc[coefficient - 1] = sum;
        }
        return { mfcc, rawVolume, audio };
    }

    _resample(input, sourceRate, targetRate, targetCount) {
        const data = Float64Array.from(input);
        const cutoff = (targetRate / 2 - 500) / sourceRate;
        const range = 500 / sourceRate;
        let filterLength = Math.round(3.1 / range);
        if ((filterLength + 1) % 2 === 0) filterLength += 1;
        const coefficients = new Float64Array(filterLength);
        for (let i = 0; i < filterLength; i++) {
            const x = i - (filterLength - 1) / 2;
            const angle = 2 * Math.PI * cutoff * x;
            coefficients[i] = 2 * cutoff * Math.sin(angle) / angle;
        }
        for (let tap = 0; tap < filterLength; tap++) {
            const coefficient = coefficients[tap];
            for (let i = tap; i < data.length; i++) {
                data[i] += coefficient * input[i - tap];
            }
        }

        if (sourceRate <= targetRate) {
            if (data.length !== targetCount) {
                throw new RangeError("MFCC input frame length does not match profile sampleCount");
            }
            return data;
        }

        const ratio = sourceRate / targetRate;
        const output = new Float64Array(targetCount);
        if (sourceRate % targetRate === 0) {
            const skip = sourceRate / targetRate;
            for (let i = 0; i < targetCount; i++) output[i] = data[i * skip];
            return output;
        }
        for (let i = 0; i < targetCount; i++) {
            output[i] = data[Math.min(data.length - 1, Math.floor(i * ratio))];
        }
        return output;
    }

    _melFilterBank(spectrum, sampleRate, channels) {
        const result = new Float64Array(channels);
        const maxFrequency = sampleRate / 2;
        const maxMel = 1127 * Math.log(maxFrequency / 700 + 1);
        const nyquistBin = spectrum.length / 2;
        const frequencyStep = maxFrequency / nyquistBin;
        const melStep = maxMel / (channels + 1);
        const toFrequency = (mel) => 700 * (Math.exp(mel / 1127) - 1);

        for (let channel = 0; channel < channels; channel++) {
            const begin = toFrequency(melStep * channel);
            const center = toFrequency(melStep * (channel + 1));
            const end = toFrequency(melStep * (channel + 2));
            const beginIndex = Math.ceil(begin / frequencyStep);
            const centerIndex = Math.round(center / frequencyStep);
            const endIndex = Math.min(nyquistBin - 1, Math.floor(end / frequencyStep));
            let sum = 0;
            for (let index = beginIndex + 1; index <= endIndex; index++) {
                const frequency = frequencyStep * index;
                let weight = index < centerIndex
                    ? (frequency - begin) / Math.max(center - begin, Number.EPSILON)
                    : (end - frequency) / Math.max(end - center, Number.EPSILON);
                weight /= Math.max((end - begin) * 0.5, Number.EPSILON);
                sum += Math.max(0, weight) * spectrum[index];
            }
            result[channel] = sum;
        }
        return result;
    }

    _score(mfcc, phoneme) {
        return this._scoreDetail(mfcc, phoneme).score;
    }

    _scoreDetail(mfcc, phoneme) {
        const method = this.profile.compareMethod;
        if (method === "CosineSimilarity") {
            let product = 0;
            let mfccNorm = 0;
            let phonemeNorm = 0;
            for (let i = 0; i < mfcc.length; i++) {
                const x = (mfcc[i] - this.means[i]) / this.standardDeviations[i];
                const y = (phoneme[i] - this.means[i]) / this.standardDeviations[i];
                product += x * y;
                mfccNorm += x * x;
                phonemeNorm += y * y;
            }
            const denominator = Math.sqrt(mfccNorm) * Math.sqrt(phonemeNorm);
            const similarity = denominator > 0 ? Math.max(0, product / denominator) : 0;
            return {
                metric: similarity,
                // uLipSync calculates this score as float32, including underflow to zero.
                score: Math.fround(Math.pow(similarity, 100)),
            };
        }

        let distance = 0;
        for (let i = 0; i < mfcc.length; i++) {
            const x = (mfcc[i] - this.means[i]) / this.standardDeviations[i];
            const y = (phoneme[i] - this.means[i]) / this.standardDeviations[i];
            const delta = x - y;
            distance += method === "L1Norm" ? Math.abs(delta) : delta * delta;
        }
        distance = method === "L1Norm"
            ? distance / mfcc.length
            : Math.sqrt(distance / mfcc.length);
        return {
            metric: distance,
            score: Math.pow(10, -distance),
        };
    }

    _normalizeVolume(rawVolume) {
        if (rawVolume <= 0) return 0;
        const range = Math.max(1e-4, this.maxVolume - this.minVolume);
        return MFCCLipSyncEngine.clamp((Math.log10(rawVolume) - this.minVolume) / range, 0, 1);
    }

    _visemes(phonemeRatios, volume) {
        const visemes = Object.fromEntries(MFCCLipSyncEngine.VISEMES.map((name) => [name, 0]));
        for (const [phoneme, ratio] of Object.entries(phonemeRatios)) {
            const viseme = this._visemeForPhoneme(phoneme);
            if (viseme in visemes) visemes[viseme] += ratio * volume;
        }
        return visemes;
    }

    _visemeForPhoneme(phoneme) {
        const viseme = this.phonemeMap[phoneme]
            || this.phonemeMap[phoneme.toUpperCase()]
            || this.phonemeMap[phoneme.toLowerCase()];
        return MFCCLipSyncEngine.VISEMES.includes(viseme) ? viseme : null;
    }

    _emptyResult() {
        return {
            visemes: Object.fromEntries(MFCCLipSyncEngine.VISEMES.map((name) => [name, 0])),
            mainViseme: null,
            mainVisemeWeight: 0,
        };
    }

    _positiveInteger(value, name) {
        const number = Number(value);
        if (!Number.isInteger(number) || number <= 0) {
            throw new RangeError(`MFCC profile ${name} must be a positive integer`);
        }
        return number;
    }

    _compareMethod(value) {
        const methods = ["L1Norm", "L2Norm", "CosineSimilarity"];
        if (Number.isInteger(value) && methods[value]) return methods[value];
        const normalized = String(value).toLowerCase().replace(/[^a-z0-9]/g, "");
        const method = methods.find((candidate) => (
            candidate.toLowerCase() === normalized
            || candidate.toLowerCase().replace("norm", "") === normalized
        ));
        if (!method) throw new RangeError(`Unsupported MFCC compareMethod: ${value}`);
        return method;
    }

    _requireProfile() {
        if (!this.profile) throw new Error("MFCC profile has not been loaded");
    }
}
