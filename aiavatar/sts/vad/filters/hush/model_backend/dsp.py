"""Streaming DSP for the Hush 16 kHz model.

Adapted to Python from DeepFilterNet, as vendored by pulp-vision/Hush
at commit 9f6414e91461a8f4bdf9840c0cdcdcb7da986339 (src/lib.rs and src/tract.rs).
Original DeepFilterNet copyright notice:
    Copyright 2021 Hendrik Schröter

Upstream: https://github.com/pulp-vision/Hush/tree/9f6414e91461a8f4bdf9840c0cdcdcb7da986339/native/weya_nc_build/vendor/deep_filter
Upstream is dual licensed MIT/Apache-2.0; this adaptation uses Apache-2.0.
SPDX-License-Identifier: Apache-2.0

One DspState belongs to one audio session. Calls alternate analyze/synthesize;
returned feature and audio arrays reuse that session's buffers.
"""

import numpy as np


SAMPLE_RATE = 16000
HOP_SIZE = 160
FFT_SIZE = 320
FREQ_SIZE = FFT_SIZE // 2 + 1
NB_ERB = 32
NB_DF = 64
DF_ORDER = 5
MIN_ERB_BINS = 2
NORM_TAU = 1.0
# native calc_norm_alpha: round(exp(-hop/sr/tau), 3) for this configuration.
ALPHA = np.float32(0.990)
ONE_MINUS_ALPHA = np.float32(1.0) - ALPHA
WNORM = np.float32(1.0) / np.float32(FFT_SIZE * FFT_SIZE / (2 * HOP_SIZE))


def _erb_widths():
    """Port native erb_fb, retaining f32 arithmetic and positive .round()."""
    f32 = np.float32
    erb_scale = f32(9.265)
    hz_scale = f32(24.7) * erb_scale
    high = erb_scale * np.log1p(f32(SAMPLE_RATE // 2) / hz_scale)
    step = high / f32(NB_ERB)
    freq_width = f32(SAMPLE_RATE) / f32(FFT_SIZE)
    widths = []
    previous = overflow = 0
    for band in range(1, NB_ERB + 1):
        frequency = hz_scale * (np.exp((f32(band) * step) / erb_scale) - f32(1))
        # Rust round() rounds positive halfway values upward, unlike np.rint.
        boundary = int(np.floor(float(frequency / freq_width) + 0.5))
        width = boundary - previous - overflow
        if width < MIN_ERB_BINS:
            overflow = MIN_ERB_BINS - width
            width = MIN_ERB_BINS
        else:
            overflow = 0
        widths.append(width)
        previous = boundary
    widths[-1] += 1
    widths[-1] -= max(0, sum(widths) - FREQ_SIZE)
    if sum(widths) != FREQ_SIZE or min(widths) <= 0:
        raise ValueError("Invalid ERB filter bank")
    return np.asarray(widths, dtype=np.intp)


def _norm_initial(start, end, size):
    # Native calculates step and each multiply/add in f32, unlike f64 linspace.
    start, end = np.float32(start), np.float32(end)
    step = (end - start) / np.float32(size - 1)
    return start + np.arange(size, dtype=np.float32) * step


# Native constructs the Vorbis window in f64 and casts each element to f32.
_phase = 0.5 * np.pi * (np.arange(FFT_SIZE, dtype=np.float64) + 0.5) / (FFT_SIZE // 2)
WINDOW = np.sin(0.5 * np.pi * np.sin(_phase) ** 2).astype(np.float32)
ERB_WIDTHS = _erb_widths()
ERB_STARTS = np.r_[0, np.cumsum(ERB_WIDTHS)[:-1]].astype(np.intp)
ERB_BIN_INDEX = np.repeat(np.arange(NB_ERB, dtype=np.intp), ERB_WIDTHS)
ERB_BIN_WEIGHT = (np.float32(1.0) / ERB_WIDTHS.astype(np.float32))[ERB_BIN_INDEX]
MEAN_NORM_INITIAL = _norm_initial(-60.0, -90.0, NB_ERB)
UNIT_NORM_INITIAL = _norm_initial(0.001, 0.0001, NB_DF)
for _constant in (
    WINDOW, ERB_WIDTHS, ERB_STARTS, ERB_BIN_INDEX, ERB_BIN_WEIGHT,
    MEAN_NORM_INITIAL, UNIT_NORM_INITIAL,
):
    _constant.flags.writeable = False
del _phase, _constant


class DspState:
    """Independent mutable DSP state for one mono audio stream.

    ``analyze`` accepts exactly 160 float32 samples and returns model inputs.
    ``synthesize`` accepts a 32-value mask and 640 real coefficient values
    (or None to skip either stage). Native coefficient memory layout is
    [frequency=64, tap=5, real_imag=2], oldest to newest tap. Consequently the
    model output [1, 1, 64, 10] can be passed directly. The coefficient selecting
    the current spectrum is coefs.reshape(64, 5, 2)[:, 4, 0] = 1.
    """

    def __init__(self, attenuation_limit=0.0):
        self.attenuation_limit = np.float32(attenuation_limit)
        self.analysis_mem = np.zeros(HOP_SIZE, dtype=np.float32)
        self.synthesis_mem = np.zeros(HOP_SIZE, dtype=np.float32)
        self.mean_norm_state = MEAN_NORM_INITIAL.copy()
        self.unit_norm_state = UNIT_NORM_INITIAL.copy()
        self.spectra = np.zeros((DF_ORDER, FREQ_SIZE), dtype=np.complex64)
        self._analysis = np.empty(FFT_SIZE, dtype=np.float32)
        self._spectrum = np.empty(FREQ_SIZE, dtype=np.complex64)
        self._power = np.empty(FREQ_SIZE, dtype=np.float32)
        self._imag_power = np.empty(FREQ_SIZE, dtype=np.float32)
        self._erb_power = np.empty(FREQ_SIZE, dtype=np.float32)
        self._erb = np.empty(NB_ERB, dtype=np.float32)
        self._mean_update = np.empty(NB_ERB, dtype=np.float32)
        self._unit_update = np.empty(NB_DF, dtype=np.float32)
        self._unit_denominator = np.empty(NB_DF, dtype=np.float32)
        self._feat_spec = np.empty((1, 2, 1, NB_DF), dtype=np.float32)
        self._features = {
            "feat_erb": self._erb.reshape(1, 1, 1, NB_ERB),
            "feat_spec": self._feat_spec,
        }
        self._gain_bins = np.empty(FREQ_SIZE, dtype=np.float32)
        self._enhanced = np.empty(FREQ_SIZE, dtype=np.complex64)
        self._df_products = np.empty((DF_ORDER, NB_DF), dtype=np.complex64)
        self._inverse = np.empty(FFT_SIZE, dtype=np.float32)
        self._output = np.empty(HOP_SIZE, dtype=np.float32)
        self._pending = False

    def analyze(self, frame):
        """Window/FFT, update history and feature normalization; no inference."""
        if self._pending:
            raise RuntimeError("synthesize must follow each analyze")
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (HOP_SIZE,):
            raise ValueError(f"frame must have shape ({HOP_SIZE},), got {frame.shape}")

        np.multiply(self.analysis_mem, WINDOW[:HOP_SIZE], out=self._analysis[:HOP_SIZE])
        np.multiply(frame, WINDOW[HOP_SIZE:], out=self._analysis[HOP_SIZE:])
        self.analysis_mem[:] = frame
        np.fft.rfft(self._analysis, out=self._spectrum)
        self._spectrum *= WNORM
        self.spectra[:-1] = self.spectra[1:]
        self.spectra[-1] = self._spectrum

        # Match native: average re^2+im^2 within each disjoint ERB band.
        np.square(self._spectrum.real, out=self._power)
        np.square(self._spectrum.imag, out=self._imag_power)
        np.add(self._power, self._imag_power, out=self._power)
        np.multiply(self._power, ERB_BIN_WEIGHT, out=self._erb_power)
        np.add.reduceat(self._erb_power, ERB_STARTS, out=self._erb)
        self._erb += np.float32(1e-10)
        np.log10(self._erb, out=self._erb)
        self._erb *= np.float32(10.0)
        np.multiply(self._erb, ONE_MINUS_ALPHA, out=self._mean_update)
        self.mean_norm_state *= ALPHA
        self.mean_norm_state += self._mean_update
        self._erb -= self.mean_norm_state
        self._erb /= np.float32(40.0)

        # Native unit state tracks magnitude, then divides by sqrt(state).
        np.sqrt(self._power[:NB_DF], out=self._unit_update)
        self._unit_update *= ONE_MINUS_ALPHA
        self.unit_norm_state *= ALPHA
        self.unit_norm_state += self._unit_update
        np.sqrt(self.unit_norm_state, out=self._unit_denominator)
        np.divide(self._spectrum.real[:NB_DF], self._unit_denominator,
                  out=self._feat_spec[0, 0, 0])
        np.divide(self._spectrum.imag[:NB_DF], self._unit_denominator,
                  out=self._feat_spec[0, 1, 0])
        self._pending = True
        return self._features

    def synthesize(self, mask, coefs):
        """Apply ERB gains and complex deep filter, then inverse FFT/OLA."""
        if not self._pending:
            raise RuntimeError("analyze must precede synthesize")
        self._enhanced[:] = self._spectrum
        if mask is not None:
            mask = np.asarray(mask, dtype=np.float32).reshape(NB_ERB)
            np.take(mask, ERB_BIN_INDEX, out=self._gain_bins)
            self._enhanced *= self._gain_bins
        if coefs is not None:
            coefs = np.asarray(coefs, dtype=np.float32).reshape(NB_DF, DF_ORDER, 2)
            # ONNX output is contiguous; normalize an arbitrary view if needed.
            complex_coefs = np.ascontiguousarray(coefs).view(np.complex64).reshape(NB_DF, DF_ORDER)
            np.multiply(self.spectra[:, :NB_DF], complex_coefs.T, out=self._df_products)
            np.add.reduce(self._df_products, axis=0, out=self._enhanced[:NB_DF])

        if self.attenuation_limit:
            self._enhanced *= np.float32(1.0) - self.attenuation_limit
            self._enhanced += self._spectrum * self.attenuation_limit

        # Rust realfft's inverse is unnormalized. NumPy's norm="forward"
        # gives the same inverse convention; 1/N was applied in analysis.
        np.fft.irfft(self._enhanced, n=FFT_SIZE, norm="forward", out=self._inverse)
        self._inverse *= WINDOW
        np.add(self._inverse[:HOP_SIZE], self.synthesis_mem, out=self._output)
        self.synthesis_mem[:] = self._inverse[HOP_SIZE:]
        self._pending = False
        return self._output
