import io
import math
import struct
import wave

import pytest

from aiavatar.adapter.asterisk.audio import audio_to_slin16, chunk_slin16


def _wav(sample_rate, *, channels=1, duration=0.05):
    frame_count = int(sample_rate * duration)
    mono = [
        int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(frame_count)
    ]
    samples = mono if channels == 1 else [value for sample in mono for value in (sample, sample)]
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


@pytest.mark.parametrize("sample_rate", [16_000, 24_000, 44_100])
def test_wav_sample_rates_are_normalized_to_slin16(sample_rate):
    result = audio_to_slin16(
        _wav(sample_rate),
        raw_sample_rate=24_000,
    )

    assert len(result) % 2 == 0
    assert abs(len(result) - 1600) <= 4


def test_stereo_wav_is_mixed_to_mono():
    result = audio_to_slin16(
        _wav(16_000, channels=2),
        raw_sample_rate=24_000,
    )

    assert len(result) == 1600


def test_chunks_preserve_audio_and_only_final_fragment_may_be_short():
    samples = b"\x01\x00" * 100_000
    chunks = list(chunk_slin16(
        samples,
        optimal_frame_size=640,
        target_duration_ms=100,
    ))

    assert chunks
    assert all(len(chunk) % 640 == 0 for chunk in chunks[:-1])
    assert all(len(chunk) <= 65_500 for chunk in chunks)
    assert b"".join(chunks) == samples


def test_frame_larger_than_websocket_limit_is_rejected():
    with pytest.raises(ValueError):
        list(chunk_slin16(b"\x00\x00", optimal_frame_size=65_501))
