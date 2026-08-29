# Speech detector (VAD)

A `SpeechDetector` decides when the user starts and stops speaking. It is the first stage of
the pipeline and the one that most directly determines how responsive the avatar feels.

| Detector | Class | Module | Extra install |
| --- | --- | --- | --- |
| Silero VAD | `SileroSpeechDetector` | `aiavatar.sts.vad.silero` | — |
| Silero VAD, streaming | `SileroStreamSpeechDetector` | `aiavatar.sts.vad.stream` | — |
| Azure Speech | `AzureStreamSpeechDetector` | `aiavatar.sts.vad.azure_stream` | `azure-cognitiveservices-speech` |
| Amazon Transcribe (AWS) | `AmazonTranscribeStreamSpeechDetector` | `aiavatar.sts.vad.amazon_transcribe_stream` | `amazon-transcribe` |
| Parapper ASR | `ParapperStreamSpeechDetector` | `aiavatar.sts.vad.parapper_stream` | — |
| Volume threshold (legacy) | `StandardSpeechDetector` | `aiavatar.sts.vad.standard` | — |

**Which one to use.** `SileroStreamSpeechDetector` is the default choice: it recognises
speech in segments while the user is still talking, so by the time the turn ends the text is
usually already there. That overlap is where most of the pipeline's latency advantage comes
from. Use plain `SileroSpeechDetector` when you want one clean recognition of the whole
utterance. The Azure, Amazon, and Parapper detectors delegate both detection and
recognition to a cloud stream, and therefore do not take a separate `SpeechRecognizer`.

AIAvatarKit includes Voice Activity Detection (VAD) components to automatically detect when speech starts and ends in audio streams. This enables seamless conversation flow without manual input controls.

## Silero Speech Detector

The default Speech Detector is `SileroSpeechDetector`, which employs AI-based voice activity detection using the Silero VAD model:

```python
from aiavatar.sts.vad.silero import SileroSpeechDetector

vad = SileroSpeechDetector(
    speech_probability_threshold=0.5,    # AI model confidence threshold (0.0-1.0)
    silence_duration_threshold=0.5,      # Seconds of silence to end recording
    volume_db_threshold=None,            # Optional: filter by volume in dB (e.g., -30.0)
    max_duration=10.0,                   # Maximum recording duration
    min_duration=0.2,                    # Minimum recording duration
    sample_rate=16000,                   # Audio sample rate
    channels=1,                          # Audio channels
    chunk_size=512,                      # Audio processing chunk size
    model_pool_size=1,                   # Number of parallel AI models
    debug=True
)

aiavatar_app = AIAvatarWebSocketServer(vad=vad, openai_api_key=OPENAI_API_KEY)
```

For high-concurrency applications:

```python
vad = SileroSpeechDetector(
    speech_probability_threshold=0.6,    # Stricter threshold for noisy environments
    model_pool_size=4,                   # 4 parallel AI models for load balancing
    debug=False
)
```

To use a local Silero VAD hub repository/cache instead of downloading from the network, set `hub_cache_path`.
The path must point to a Silero VAD repository/cache directory that contains `hubconf.py`:

```python
vad = SileroSpeechDetector(
    hub_cache_path="/models/silero-vad"
)
```

When `hub_cache_path` is set, the detector loads the Silero model and utility functions from that local path only.
If the path does not exist, initialization fails instead of downloading from the network.

For custom JIT model files, `model_path` is still supported. In that case, the detector first loads the Silero utilities from `hub_cache_path` or the default Torch Hub source, then replaces the model with the local JIT file:

```python
vad = SileroSpeechDetector(
    hub_cache_path="/models/silero-vad",
    model_path="path/to/silero_vad.jit"
)
```

Turn detection is only half the decision. Silence long enough to trip
`silence_duration_threshold` still means "the user stopped talking", not "the user is
finished" — someone pausing after a conjunction or mid-recall trips it too. Attach
[turn-end gates](vad-turn-end.md) with `turn_end_gates=[...]` and they run after the
threshold is reached to confirm the utterance is actually complete.

What the detector receives is also worth deciding deliberately. Room noise, an over-hot
microphone, and a second person talking nearby all reach it as speech unless something
removes them first — see [audio filters](vad-filters.md).

## Silero Stream Speech Detector

`SileroStreamSpeechDetector` extends `SileroSpeechDetector` with segment-based speech recognition. It performs partial transcription during recording, allowing you to receive intermediate results before the final transcription.

```python
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.stt.google import GoogleSpeechRecognizer

vad = SileroStreamSpeechDetector(
    speech_recognizer=GoogleSpeechRecognizer(...),
    segment_silence_threshold=0.2,       # Silence duration to trigger segment recognition
    silence_duration_threshold=0.5,      # Silence duration to finalize recording
    # Inherits all SileroSpeechDetector parameters
)
```

Streaming detection pairs particularly well with turn-end gates: recognition of the segments
so far has usually finished while a gate is still deciding, so holding the turn open costs
almost nothing. See [Semantic turn end](vad-turn-end.md), and
[Audio filters](vad-filters.md) for what runs before any of this.

### Segment Recognition Callback

The `on_speech_detecting` callback is triggered when a speech segment is recognized:

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, session):
    print(f"Partial text: {text}")

    # For WebSocket apps, send partial text to client via info message
    # resp = STSResponse(
    #     type="info",
    #     session_id=session.session_id,
    #     metadata={"partial_request_text": text}
    # )
    # await ws_app.handle_response(resp)
```

### Text Validation

Use `validate_recognized_text` to filter out invalid recognition results:

```python
@vad.validate_recognized_text
def validate(text):
    if len(text) < 2:
        return "Text too short"  # Return error message to reject
    return None  # Return None to accept
```

## Azure Stream Speech Detector

`AzureStreamSpeechDetector` uses Azure's streaming speech recognition service for both speech detection and transcription. Audio is continuously streamed to Azure, and speech boundaries are determined by Azure's recognition events.

```sh
pip install azure-cognitiveservices-speech
```

```python
from aiavatar.sts.vad.azure_stream import AzureStreamSpeechDetector

vad = AzureStreamSpeechDetector(
    azure_subscription_key=AZURE_API_KEY,
    azure_region=AZURE_REGION
)
```

This detector also supports the `on_speech_detecting` callback for partial transcription results:

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, session):
    print(f"Partial text: {text}")

    # For WebSocket apps, send partial text to client via info message
    # resp = STSResponse(
    #     type="info",
    #     session_id=session.session_id,
    #     metadata={"partial_request_text": text}
    # )
    # await ws_app.handle_response(resp)
```

## AWS Stream Speech Detector

`AmazonTranscribeStreamSpeechDetector` uses Amazon Transcribe's streaming speech recognition service for both speech detection and transcription. Audio is continuously streamed to Amazon Transcribe, and speech boundaries are determined by the recognition results combined with a configurable silence duration threshold.

```sh
pip install amazon-transcribe
```

```python
from aiavatar.sts.vad.amazon_transcribe_stream import AmazonTranscribeStreamSpeechDetector

vad = AmazonTranscribeStreamSpeechDetector(
    aws_region="ap-northeast-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,         # Optional: uses default credential chain if omitted
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  # Optional: uses default credential chain if omitted
    aws_language="ja-JP",
    silence_duration_threshold=0.5,  # Seconds of silence after last recognition to finalize
    max_duration=20.0,               # Maximum recording duration in seconds
)
```

When `silence_duration_threshold > 0`, multiple recognition results from Amazon Transcribe are accumulated into a single speech detection event. A silence timer starts after each final result, and if new speech arrives before the timer expires, the timer is cancelled and transcription continues. This allows natural pauses within a sentence without splitting the utterance.

> **Note:** The `silence_duration_threshold` timer starts from when Amazon Transcribe returns a final recognition result, not from when the user actually stops speaking. Since Amazon Transcribe takes some time to process audio and return a final result, the actual delay from the user's perspective is: **Transcribe processing delay + `silence_duration_threshold`**. For example, if Transcribe takes ~0.5s to return a final result and `silence_duration_threshold=0.5`, the total delay from the end of speech to firing `on_speech_detected` will be approximately 1.0s.

When `max_duration` is reached during recording, if there are accumulated recognition results, speech detection is triggered immediately with the combined text.

This detector also supports the `on_speech_detecting` callback for partial transcription results. When texts have been accumulated from previous final results, they are prepended to the current partial text:

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, session):
    print(f"Partial text: {text}")

    # For WebSocket apps, send partial text to client via info message
    # resp = STSResponse(
    #     type="info",
    #     session_id=session.session_id,
    #     metadata={"partial_request_text": text}
    # )
    # await ws_app.handle_response(resp)
```

Use `validate_recognized_text` to filter out invalid recognition results:

```python
@vad.validate_recognized_text
def validate(text):
    if len(text) < 2:
        return "Text too short"  # Return error message to reject
    return None  # Return None to accept
```

## Customization

### on_recording_started Callback

The `on_recording_started` callback is triggered when recording has been active long enough to be considered meaningful speech. This is useful for stopping AI speech when the user starts talking.

```python
# Option 1: Pass callback in constructor
async def my_recording_started_handler(session_id: str):
    print(f"Recording started for session: {session_id}")
    await stop_ai_speech()

vad = SileroSpeechDetector(
    on_recording_started=my_recording_started_handler,
    on_recording_started_min_duration=1.5,    # Trigger after 1.5 sec of speech (default)
    # other parameters...
)

# Option 2: Use decorator
@vad.on_recording_started
async def on_recording_started(session_id):
    await stop_ai_speech()
```

For stream-based detectors (`SileroStreamSpeechDetector`, `AzureStreamSpeechDetector`), the callback can also be triggered by recognized text length:

```python
vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    on_recording_started_min_duration=1.5,    # Trigger after 1.5 sec of speech
    on_recording_started_min_text_length=2,   # OR trigger when text >= 2 chars
)
```

### Custom Trigger Condition

You can customize when `on_recording_started` fires using the `should_trigger_recording_started` decorator:

```python
@vad.should_trigger_recording_started
def custom_trigger(text, session):
    # text: Recognized text (None for non-stream detectors)
    # session: Recording session object
    # Return True to trigger the callback
    return text and len(text) >= 5
```

## Standard Speech Detector (Legacy)

`StandardSpeechDetector` uses simple volume-based detection. Consider using `SileroSpeechDetector` for better accuracy. This detector is suitable for environments with limited computing resources:

```python
from aiavatar.sts.vad.standard import StandardSpeechDetector

vad = StandardSpeechDetector(
    volume_db_threshold=-30.0,           # Voice detection threshold in dB
    silence_duration_threshold=0.5,      # Seconds of silence to end recording
    max_duration=10.0,                   # Maximum recording duration
    min_duration=0.2,                    # Minimum recording duration
    sample_rate=16000,                   # Audio sample rate
    channels=1,                          # Audio channels
    preroll_buffer_count=5,              # Pre-recording buffer size
    debug=True
)
```

## Muting

Every detector exposes a `should_mute` callable. While it returns `True`, incoming audio is
discarded and no turn is started. The default never mutes.

```python
vad.should_mute = lambda: avatar_is_speaking
```

This is the mechanism behind barge-in handling. Adapters that accept `mute_on_barge_in=True`
register an `on_recording_started` callback that stops the current response the moment the
user starts talking, so the avatar yields the floor instead of talking over them:

```python
adapter = AIAvatarWebSocketServer(
    openai_api_key=OPENAI_API_KEY,
    mute_on_barge_in=True,
)
```

Use `should_mute` directly when the decision depends on something the adapter does not know
about — a push-to-talk button, a physical switch, or a state machine in your application.

## Parapper Stream Speech Detector

`ParapperStreamSpeechDetector` connects to a Parapper ASR WebSocket endpoint and uses the
server's own voice activity detection and turn-end decision. The server's `turn.final`
signal is trusted as the only turn-end signal: this detector applies no client-side silence
timeout and does not re-segment turns.

```python
from aiavatar.sts.vad.parapper_stream import ParapperStreamSpeechDetector

vad = ParapperStreamSpeechDetector(
    url="ws://127.0.0.1:8080/ws/recognition",
    api_key=PARAPPER_API_KEY,     # Optional
    sample_rate=16000,
    channels=1,
    preroll_buffer_sec=2.0,       # Audio retained before speech starts
    connect_timeout=10.0,
    drain_timeout=10.0,
)
```

Pass `to_linear16` when your transport delivers something other than 16-bit linear PCM — for
example a telephony channel sending µ-law — and the callable will be applied to each chunk
before it is sent upstream.

Because detection and recognition happen in one place, do not also configure a
`SpeechRecognizer` for this detector.

## See also

- [Semantic turn end](vad-turn-end.md) — gates that hold a turn through a pause
- [Audio filters](vad-filters.md) — AGC, EQ, and near-field gating before detection
- [Speech-to-Text](stt.md) — the recognizers batch and streaming detectors call
- [Pipeline](pipeline.md) — what happens once a turn ends

---

[← Documentation index](../README.md#-documentation)
