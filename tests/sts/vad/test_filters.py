import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from aiavatar.sts.vad.filters import AudioFilter, NearFieldAudioGate, HighShelfFilter, AGCFilter, SessionAudioRecorder

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE  # 0.032 sec


def make_chunk(amplitude: int, samples: int = CHUNK_SAMPLES) -> bytes:
    """Generate a sine wave chunk with the given peak amplitude"""
    frames = []
    for i in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE))
        frames.append(struct.pack("<h", value))
    return b"".join(frames)


def max_amplitude(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    return int(np.abs(np.frombuffer(pcm, dtype=np.int16)).max())


def make_gate(**kwargs) -> NearFieldAudioGate:
    params = dict(
        sample_rate=SAMPLE_RATE,
        closed_gain=0.05,
        lookahead_duration=0.12,     # about 4 chunks
        min_rms_db=-42.0,
        open_snr_db_threshold=12.0,
        close_snr_db_threshold=6.0,
        open_min_duration=0.06,      # about 2 chunks
        close_min_duration=0.4,
        ambient_window_duration=2.5,
        initial_ambient_db=-65.0,
    )
    params.update(kwargs)
    return NearFieldAudioGate(**params)


def feed_chunks(gate: NearFieldAudioGate, chunks, session_id: str = "session1") -> bytes:
    output = bytearray()
    for chunk in chunks:
        output.extend(gate.process(chunk, session_id))
    return bytes(output)


def build_ambient(gate: NearFieldAudioGate, amplitude: int = 30, chunks: int = 60, session_id: str = "session1") -> bytes:
    """Feed quiet chunks to settle the ambient noise floor"""
    return feed_chunks(gate, [make_chunk(amplitude)] * chunks, session_id)


def test_audio_is_conserved_through_lookahead():
    gate = make_gate()
    quiet = make_chunk(30)

    emitted = 0
    for _ in range(20):
        emitted += len(gate.process(quiet, "session1"))

    pending = gate.get_diagnostics("session1")["pending_duration"]
    assert emitted + int(pending * SAMPLE_RATE) * 2 == len(quiet) * 20
    # Delay line holds at least lookahead_duration
    assert pending >= gate.lookahead_duration

    emitted += len(gate.flush("session1"))
    assert emitted == len(quiet) * 20


def test_far_speech_is_attenuated():
    gate = make_gate()
    build_ambient(gate)

    far = make_chunk(300)  # about -43 dBFS: below min_rms_db
    output = feed_chunks(gate, [far] * 20)
    output += gate.flush("session1")

    assert gate.get_diagnostics("session1")["is_open"] is False
    # Trailing part of output is the far speech, attenuated by closed_gain
    far_part = output[-len(far) * 10:]
    assert max_amplitude(far_part) <= 300 * 0.05 + 2


def test_near_speech_passes_with_onset_preserved():
    gate = make_gate()
    build_ambient(gate)

    loud = make_chunk(8000)
    output = feed_chunks(gate, [loud] * 10)
    output += gate.flush("session1")

    assert gate.get_diagnostics("session1")["is_open"] is True
    # The loud region is the last 10 chunks of the output
    loud_part = np.abs(np.frombuffer(output[-len(loud) * 10:], dtype=np.int16))
    assert loud_part.max() >= 8000 * 0.9
    # Onset preservation: even the FIRST loud chunk was released (mostly) open
    # thanks to the lookahead buffer. Allow for the crossfade ramp.
    first_chunk = loud_part[:CHUNK_SAMPLES]
    assert first_chunk.max() >= 8000 * 0.5


def test_gate_closes_after_silence_and_rejects_far_speech_again():
    gate = make_gate()
    build_ambient(gate)

    loud = make_chunk(8000)
    quiet = make_chunk(30)
    far = make_chunk(300)

    feed_chunks(gate, [loud] * 10)
    assert gate.get_diagnostics("session1")["is_open"] is True

    # Silence longer than close_min_duration (0.4 sec = about 13 chunks)
    feed_chunks(gate, [quiet] * 20)
    assert gate.get_diagnostics("session1")["is_open"] is False

    # Far speech afterwards stays attenuated
    output = feed_chunks(gate, [far] * 20)
    output += gate.flush("session1")
    assert gate.get_diagnostics("session1")["is_open"] is False
    far_part = output[-len(far) * 10:]
    assert max_amplitude(far_part) <= 300 * 0.05 + 2


def test_gate_survives_short_pauses_within_turn():
    gate = make_gate()
    build_ambient(gate)

    loud = make_chunk(8000)
    quiet = make_chunk(30)

    feed_chunks(gate, [loud] * 10)
    # Short pause (about 0.19 sec) < close_min_duration (0.4 sec)
    feed_chunks(gate, [quiet] * 6)
    assert gate.get_diagnostics("session1")["is_open"] is True
    # Speech resumes without needing to re-open
    feed_chunks(gate, [loud] * 3)
    assert gate.get_diagnostics("session1")["is_open"] is True


def test_low_snr_speech_is_rejected():
    gate = make_gate(
        min_rms_db=None,  # SNR only
        ambient_max_rise_db_per_update=None,  # Let ambient follow the noisy environment immediately
        calibration_duration=2.0,  # Learn the noisy environment before judging (required when noise is present from the start)
    )
    # Noisy environment from the start: ambient around -32 dBFS
    build_ambient(gate, amplitude=1200, chunks=80)

    # Speech only about 6 dB above ambient
    low_snr = make_chunk(2400)
    feed_chunks(gate, [low_snr] * 20)
    assert gate.get_diagnostics("session1")["is_open"] is False
    assert gate.get_diagnostics("session1")["gate_reason"] == "below_snr"

    # Speech well above ambient opens the gate
    loud = make_chunk(16000)
    feed_chunks(gate, [loud] * 5)
    assert gate.get_diagnostics("session1")["is_open"] is True


def test_disabled_gate_passes_audio_through_without_delay():
    gate = make_gate(enabled=False)
    chunk = make_chunk(300)
    assert gate.process(chunk, "session1") == chunk


def test_sessions_are_isolated_and_resettable():
    gate = make_gate()
    build_ambient(gate, session_id="session1")

    loud = make_chunk(8000)
    feed_chunks(gate, [loud] * 10, session_id="session1")
    assert gate.get_diagnostics("session1")["is_open"] is True

    # Another session starts closed with its own ambient state
    feed_chunks(gate, [loud] * 1, session_id="session2")
    assert gate.get_diagnostics("session2")["is_open"] is False

    gate.reset_session("session1")
    assert gate.get_diagnostics("session1") is None
    assert gate.get_diagnostics("session2") is not None


def test_set_config():
    gate = make_gate()
    updated = gate.set_config({"open_snr_db_threshold": 6.0, "unknown_key": 1})
    assert updated == {"open_snr_db_threshold": 6.0}
    assert gate.open_snr_db_threshold == 6.0


def read_wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as f:
        return f.getnframes()


def test_recorder_captures_raw_and_gated_pair(tmp_path: Path):
    recorder = SessionAudioRecorder(str(tmp_path), sample_rate=SAMPLE_RATE)
    gate = make_gate()
    chain = [recorder.tap("raw"), gate, recorder.tap("gated")]

    quiet = make_chunk(30)
    loud = make_chunk(8000)
    fed = 0
    for chunk in [quiet] * 60 + [loud] * 10:
        samples = chunk
        for f in chain:
            samples = f.process(samples, "session1")
        fed += 1

    recorder.finalize_session("session1")

    raw_files = list((tmp_path / "session1").glob("raw_*.wav"))
    gated_files = list((tmp_path / "session1").glob("gated_*.wav"))
    assert len(raw_files) == 1
    assert len(gated_files) == 1

    # Raw file holds everything; gated file lacks the audio still pending
    # in the gate's lookahead buffer
    raw_frames = read_wav_frames(raw_files[0])
    gated_frames = read_wav_frames(gated_files[0])
    pending_frames = int(gate.get_diagnostics("session1")["pending_duration"] * SAMPLE_RATE)
    assert raw_frames == CHUNK_SAMPLES * fed
    assert gated_frames == raw_frames - pending_frames


def test_recorder_records_across_utterance_resets(tmp_path: Path):
    """Recording continues for the whole session, not per utterance"""
    recorder = SessionAudioRecorder(str(tmp_path), sample_rate=SAMPLE_RATE)
    tap = recorder.tap("raw")

    # Two "utterances" with silence between: still one continuous file
    for chunk in [make_chunk(8000)] * 5 + [make_chunk(0)] * 30 + [make_chunk(8000)] * 5:
        tap.process(chunk, "session1")
    recorder.finalize_session("session1")

    raw_files = list((tmp_path / "session1").glob("raw_*.wav"))
    assert len(raw_files) == 1
    assert read_wav_frames(raw_files[0]) == CHUNK_SAMPLES * 40


def test_recorder_target_session_filtering(tmp_path: Path):
    recorder = SessionAudioRecorder(str(tmp_path), sample_rate=SAMPLE_RATE, target_session_ids=["target"])
    tap = recorder.tap("raw")
    chunk = make_chunk(8000)

    tap.process(chunk, "other")
    tap.process(chunk, "target")

    # Sessions can be added at runtime
    recorder.target_session_ids.add("other")
    tap.process(chunk, "other")

    recorder.close()
    assert not (tmp_path / "other").exists() or read_wav_frames(next((tmp_path / "other").glob("raw_*.wav"))) == CHUNK_SAMPLES
    assert read_wav_frames(next((tmp_path / "target").glob("raw_*.wav"))) == CHUNK_SAMPLES


def test_recorder_rotates_files_by_duration(tmp_path: Path):
    recorder = SessionAudioRecorder(
        str(tmp_path),
        sample_rate=SAMPLE_RATE,
        max_file_duration=CHUNK_DURATION * 3  # Rotate every 3 chunks
    )
    tap = recorder.tap("raw")
    for _ in range(7):
        tap.process(make_chunk(8000), "session1")
    recorder.close()

    raw_files = sorted((tmp_path / "session1").glob("raw_*.wav"))
    assert len(raw_files) == 3  # 3 + 3 + 1 chunks
    assert sum(read_wav_frames(p) for p in raw_files) == CHUNK_SAMPLES * 7


def test_recorder_finalizes_via_tap_reset_session(tmp_path: Path):
    """Detector's delete_session propagates to taps and closes the files"""
    recorder = SessionAudioRecorder(str(tmp_path), sample_rate=SAMPLE_RATE)
    tap = recorder.tap("raw")
    tap.process(make_chunk(8000), "session1")

    tap.reset_session("session1")

    assert "session1" not in recorder._sessions
    # File header is finalized and readable
    assert read_wav_frames(next((tmp_path / "session1").glob("raw_*.wav"))) == CHUNK_SAMPLES


def test_recorder_ttl_sweep_finalizes_stale_sessions(tmp_path: Path):
    import time as time_module
    recorder = SessionAudioRecorder(str(tmp_path), sample_rate=SAMPLE_RATE, session_ttl=10.0)
    tap = recorder.tap("raw")

    tap.process(make_chunk(8000), "stale")
    # Simulate a session that ended without cleanup long ago
    recorder._last_activity["stale"] = time_module.time() - 100
    recorder._last_sweep = time_module.time() - 100

    tap.process(make_chunk(8000), "active")

    assert "stale" not in recorder._sessions
    assert "active" in recorder._sessions
    recorder.close()


def make_sine(freq: float, amplitude: int, samples: int) -> bytes:
    frames = []
    for i in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        frames.append(struct.pack("<h", value))
    return b"".join(frames)


def rms_db(pcm: bytes) -> float:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return 20.0 * math.log10(max(rms, 1.0) / 32768.0)


def test_high_shelf_boosts_high_frequencies_only():
    eq = HighShelfFilter(gain_db=6.0, cutoff_hz=2000.0, sample_rate=SAMPLE_RATE)

    # 3000 Hz (above cutoff): boosted by about +6 dB
    high = make_sine(3000, 4000, SAMPLE_RATE)  # 1 sec
    out_high = eq.process(high, "s_high")
    assert 4.5 <= rms_db(out_high[3200:]) - rms_db(high[3200:]) <= 7.5  # Skip filter transient

    # 300 Hz (below cutoff): nearly unchanged
    low = make_sine(300, 4000, SAMPLE_RATE)
    out_low = eq.process(low, "s_low")
    assert abs(rms_db(out_low[3200:]) - rms_db(low[3200:])) <= 1.0


def test_high_shelf_is_seamless_across_chunks():
    """Chunked processing gives identical output to whole-buffer processing"""
    signal = make_sine(2500, 8000, CHUNK_SAMPLES * 10)

    eq1 = HighShelfFilter(sample_rate=SAMPLE_RATE)
    whole = eq1.process(signal, "s")

    eq2 = HighShelfFilter(sample_rate=SAMPLE_RATE)
    chunked = bytearray()
    for i in range(0, len(signal), CHUNK_SAMPLES * 2):
        chunked.extend(eq2.process(signal[i:i + CHUNK_SAMPLES * 2], "s"))

    assert bytes(chunked) == whole


def test_agc_raises_quiet_audio_towards_target():
    agc = AGCFilter(target_rms_db=-20.0, max_gain_db=18.0, up_db_per_sec=10.0, sample_rate=SAMPLE_RATE)
    quiet = make_sine(440, 1000, CHUNK_SAMPLES)  # about -33 dBFS rms

    first = agc.process(quiet, "s")
    # Gain ramps up slowly: the first chunk is nearly unchanged
    assert rms_db(first) < -30.0

    for _ in range(int(3.0 / CHUNK_DURATION)):  # 3 seconds
        out = agc.process(quiet, "s")
    assert -22.0 <= rms_db(out) <= -18.0  # Settled near target


def test_agc_does_not_clip_loud_audio():
    agc = AGCFilter(target_rms_db=-3.0, max_gain_db=18.0, up_db_per_sec=1000.0, sample_rate=SAMPLE_RATE)
    loud = make_sine(440, 30000, CHUNK_SAMPLES)

    for _ in range(30):
        out = agc.process(loud, "s")
    audio = np.frombuffer(out, dtype=np.int16)
    # Peak limiter keeps samples below full scale (1 dB headroom)
    assert int(np.abs(audio).max()) <= 30000 + 1


def test_agc_holds_gain_during_silence():
    agc = AGCFilter(target_rms_db=-20.0, silence_rms_db=-55.0, sample_rate=SAMPLE_RATE)
    quiet = make_sine(440, 1000, CHUNK_SAMPLES)
    silence = b"\x00\x00" * CHUNK_SAMPLES

    for _ in range(30):
        agc.process(quiet, "s")
    gain_after_speech = agc.get_diagnostics("s")["gain_db"]

    for _ in range(100):
        agc.process(silence, "s")
    # Silence must not pump the gain towards max
    assert abs(agc.get_diagnostics("s")["gain_db"] - gain_after_speech) <= 1.0


def test_filter_interfaces(tmp_path: Path):
    recorder = SessionAudioRecorder(str(tmp_path))
    assert isinstance(recorder.tap("raw"), AudioFilter)
    assert issubclass(NearFieldAudioGate, AudioFilter)
    assert issubclass(HighShelfFilter, AudioFilter)
    assert issubclass(AGCFilter, AudioFilter)
