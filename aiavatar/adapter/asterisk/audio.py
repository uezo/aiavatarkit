import audioop
import io
import wave
from typing import Iterator

from .protocol import MAX_WEBSOCKET_MESSAGE_SIZE


def audio_to_slin16(
    audio_data: bytes,
    *,
    raw_sample_rate: int,
    target_sample_rate: int = 16_000,
) -> bytes:
    """Convert a PCM WAV or raw mono linear16 buffer to mono 16-bit PCM."""

    if not isinstance(audio_data, bytes):
        raise TypeError("audio_data must be bytes")
    if raw_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")

    sample_rate = raw_sample_rate
    sample_width = 2
    channels = 1
    samples = audio_data

    if len(audio_data) >= 12 and audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                if wav_file.getcomptype() != "NONE":
                    raise ValueError("Only uncompressed PCM WAV audio is supported")
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                samples = wav_file.readframes(wav_file.getnframes())
        except (EOFError, wave.Error) as ex:
            raise ValueError("Invalid PCM WAV audio") from ex

    if channels not in (1, 2):
        raise ValueError("Only mono or stereo audio is supported")
    if sample_width not in (1, 2, 3, 4):
        raise ValueError("PCM sample width must be 8, 16, 24, or 32 bits")
    if len(samples) % (sample_width * channels):
        raise ValueError("PCM audio is not aligned to complete samples")

    # WAV 8-bit PCM is unsigned while audioop expects signed samples.
    if sample_width == 1:
        samples = audioop.bias(samples, 1, -128)

    if channels == 2:
        samples = audioop.tomono(samples, sample_width, 0.5, 0.5)

    if sample_width != 2:
        samples = audioop.lin2lin(samples, sample_width, 2)

    if sample_rate != target_sample_rate and samples:
        samples, _ = audioop.ratecv(
            samples,
            2,
            1,
            sample_rate,
            target_sample_rate,
            None,
        )

    return samples


def chunk_slin16(
    samples: bytes,
    *,
    optimal_frame_size: int,
    sample_rate: int = 16_000,
    target_duration_ms: int = 100,
    max_message_size: int = MAX_WEBSOCKET_MESSAGE_SIZE,
) -> Iterator[bytes]:
    """Yield bounded messages; only the final buffering fragment may be short."""

    if optimal_frame_size <= 0:
        raise ValueError("optimal_frame_size must be positive")
    if optimal_frame_size > max_message_size:
        raise ValueError("optimal_frame_size exceeds the WebSocket message limit")
    if sample_rate <= 0 or target_duration_ms <= 0:
        raise ValueError("sample_rate and target_duration_ms must be positive")
    if len(samples) % 2:
        raise ValueError("slin16 audio must contain complete 16-bit samples")
    if not samples:
        return

    max_aligned_size = (max_message_size // optimal_frame_size) * optimal_frame_size
    target_size = sample_rate * 2 * target_duration_ms // 1000
    target_size = max(optimal_frame_size, target_size)
    message_size = min(target_size, max_aligned_size)
    message_size = max(
        optimal_frame_size,
        (message_size // optimal_frame_size) * optimal_frame_size,
    )

    for offset in range(0, len(samples), message_size):
        yield samples[offset:offset + message_size]
