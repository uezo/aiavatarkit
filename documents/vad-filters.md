# Audio filters

Audio filters run on the incoming stream before voice activity detection. Use them to level
out volume, tame room noise, reject far-field speech, or capture raw audio while debugging.

`SileroSpeechDetector` and `SileroStreamSpeechDetector` can run audio through `audio_filters` before VAD, recording, and speech recognition. Filters are applied in order, and downstream processing sees the filtered audio.

This is useful for acoustic preprocessing such as near-field gating, EQ, gain normalization, and debug recording.

```python
from aiavatar.sts.vad.filters import (
    AGCFilter,
    HighShelfFilter,
    NearFieldAudioGate,
    SessionAudioRecorder,
)
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector

audio_recorder = SessionAudioRecorder("debug_audio")

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    audio_filters=[
        audio_recorder.tap("raw"),
        NearFieldAudioGate(
            min_rms_db=-42.0,
            open_snr_db_threshold=12.0,
            close_snr_db_threshold=6.0,
        ),
        HighShelfFilter(gain_db=6.0, cutoff_hz=2000.0),
        AGCFilter(target_rms_db=-20.0),
        audio_recorder.tap("processed"),
    ],
)
```

Built-in filters:

- `NearFieldAudioGate`: attenuates far-field or low-SNR audio before it reaches VAD. It uses a short lookahead buffer so speech onsets are not clipped.
- `HighShelfFilter`: boosts or cuts high frequencies above a cutoff. This can help intelligibility on band-limited telephony audio.
- `AGCFilter`: automatic gain control that raises quiet speech toward a target RMS level while avoiding clipping.
- `SessionAudioRecorder`: debug tap that writes audio at selected points in the filter chain to WAV files.

Filter order matters. Put `NearFieldAudioGate` before `AGCFilter`; otherwise AGC may amplify far-field audio and make the gate less useful. `SessionAudioRecorder.tap()` can be placed before and after filters to compare raw and processed audio.

You can implement a custom filter by subclassing `AudioFilter`:

```python
from aiavatar.sts.vad.filters import AudioFilter

class MyAudioFilter(AudioFilter):
    def process(self, samples: bytes, session_id: str) -> bytes:
        # samples are 16-bit linear PCM bytes
        return samples

    def reset_session(self, session_id: str):
        # Optional: release per-session state
        pass
```

Filters may keep short internal buffers and return `b""` while warming up. The detector treats this as "no output yet" and keeps the current recording state unchanged.

## See also

- [Speech detector](vad.md) — attaching filters to a detector
- [Semantic turn end](vad-turn-end.md) — deciding when a turn is complete

---

[← Documentation index](../README.md#-documentation)
