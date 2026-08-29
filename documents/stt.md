# Speech-to-Text

A `SpeechRecognizer` turns recorded or streaming audio into text. AIAvatarKit ships four
providers, all of which call their vendor's REST API directly — no vendor SDK is required.

| Provider | Class |
| --- | --- |
| [Azure Speech](#azure-speech) | `AzureSpeechRecognizer` |
| [Google Cloud Speech-to-Text](#google-cloud-speech-to-text) | `GoogleSpeechRecognizer` |
| [OpenAI](#openai) | `OpenAISpeechRecognizer` |
| [AmiVoice](#amivoice) | `AmiVoiceSpeechRecognizer` |

All four share `sample_rate` (default `16000`), the HTTP pool arguments
(`max_connections`, `max_keepalive_connections`, `timeout`), and `debug`.

## Azure Speech

The fastest of the four in practice, and the one to reach for when latency matters.

```python
from aiavatar.sts.stt.azure import AzureSpeechRecognizer

stt = AzureSpeechRecognizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION,
    language="ja-JP",
    alternative_languages=["en-US", "zh-CN"],
)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `azure_region` | — | Required, e.g. `japaneast` |
| `language` | `"ja-JP"` | |
| `alternative_languages` | `None` | Candidates for automatic language detection |
| `use_classic` | `False` | Use the classic recognition endpoint |
| `cid` | `None` | Custom model id for a domain-trained deployment |
| `max_retries` | `2` | Retries on transient failures |

`cid` is how a domain-specific model reaches the pipeline — train it in Speech Studio and
pass its id here. Combined with a [per-session switch](#per-session-stt-switching) it lets
you use a specialised model only for the turns that need it.

## Google Cloud Speech-to-Text

```python
from aiavatar.sts.stt.google import GoogleSpeechRecognizer

stt = GoogleSpeechRecognizer(
    google_api_key=GOOGLE_API_KEY,
    language="ja-JP",
    alternative_languages=["en-US"],
)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `language` | `"ja-JP"` | |
| `alternative_languages` | `None` | Candidates for automatic language detection |
| `timeout` | `10.0` | |

## OpenAI

```python
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

stt = OpenAISpeechRecognizer(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-transcribe",
    language="ja",
)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `model` | `"gpt-transcribe"` | Also accepts Whisper and the other transcribe models |
| `language` | `"ja"` | ISO code, not a locale — `ja`, not `ja-JP` |
| `base_url` | `"https://api.openai.com/v1"` | Any OpenAI-compatible transcription endpoint |
| `min_data_length` | `4096` | Audio shorter than this is not sent |

`gpt-transcribe` is OpenAI's high-accuracy speech-to-text model for file and streamed
transcription. See the [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-transcribe).

`base_url` makes this the general-purpose option: anything that implements OpenAI's
transcription API works through this class.

`min_data_length` guards against sending near-empty clips. Raise it if you see the
recognizer returning noise from very short bursts.

## AmiVoice

Strong on Japanese, and notably better than the others at proper nouns — place names,
company names, product names. That makes it a good candidate for a
[per-session switch](#per-session-stt-switching) on turns where a name is expected.

```python
from aiavatar.sts.stt.amivoice import AmiVoiceSpeechRecognizer

stt = AmiVoiceSpeechRecognizer(
    amivoice_api_key=AMIVOICE_API_KEY,
    engine="-a2-ja-general",
)
```

| Argument | Default | Notes |
| --- | --- | --- |
| `engine` | `"-a2-ja-general"` | AmiVoice engine name; pick the one matching your domain |
| `target_sample_rate` | `0` | Resample before sending; `0` sends the audio as captured |
| `timeout` | `30.0` | Higher than the others by default |

If you want to configure in detail, create instance of `SpeechRecognizer` with custom parameters and set it to `AIAvatar`. We support Azure, Google and OpenAI Speech-to-Text services.

NOTE: **`AzureSpeechRecognizer` is much faster** than Google and OpenAI(default).

```python
# Create AzureSpeechRecognizer
from aiavatar.sts.stt.azure import AzureSpeechRecognizer
stt = AzureSpeechRecognizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION
)

# Create the adapter with AzureSpeechRecognizer
aiavatar_app = AIAvatarWebSocketServer(
    stt=stt,
    openai_api_key=OPENAI_API_KEY   # API Key for LLM
)
```

You can also make custom STT components by implementing `SpeechRecognizer` interface.

## Preprocessing and Postprocessing

You can add custom preprocessing and postprocessing to any `SpeechRecognizer` implementation. This is useful for tasks like speaker verification, audio filtering, or text normalization.

```python
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

# Create recognizer
recognizer = OpenAISpeechRecognizer(openai_api_key="your-api-key")

# Add preprocessing - e.g., speaker verification
@recognizer.preprocess
async def verify_speaker(session_id: str, audio_data: bytes):
    # Perform speaker verification
    is_valid_speaker = await check_speaker_identity(audio_data)
    
    if not is_valid_speaker:
        # Return empty bytes to skip transcription
        return b"", {"rejected": True, "reason": "speaker_mismatch"}
    
    # Return processed audio and metadata
    filtered_audio = apply_noise_filter(audio_data)
    return filtered_audio, {"speaker_verified": True, "session_id": session_id}

# Add postprocessing - e.g., text formatting
@recognizer.postprocess
async def format_text(session_id: str, text: str, audio_data: bytes, preprocess_metadata: dict):
    # Format transcribed text
    formatted_text = text.strip().capitalize()
    
    # Add punctuation if missing
    if formatted_text and formatted_text[-1] not in '.!?':
        formatted_text += '.'
    
    # Return formatted text and metadata
    return formatted_text, {
        "original_text": text,
        "formatting_applied": True,
        "preprocess_info": preprocess_metadata
    }

# Use the recognizer with preprocessing and postprocessing
result = await recognizer.recognize(
    session_id="user-123",
    data=audio_bytes
)

print(f"Text: {result.text}")
print(f"Preprocess metadata: {result.preprocess_metadata}")
print(f"Postprocess metadata: {result.postprocess_metadata}")
```

The preprocessing and postprocessing functions can return either:
- Just the processed data (bytes for preprocess, string for postprocess)
- A tuple of (processed_data, metadata_dict) for additional information

If preprocessing returns empty bytes, the transcription is skipped and the result will have `text=None`.


## Speaker Diarization

AIAvatarKit provides speaker diarization functionality to suppress responses to voices other than the main speaker. This prevents interruptions from surrounding conversations or venue announcements at events.

The `MainSpeakerGate` provides the following features:

- Calculates voice embeddings from request audio
- Registers a voice as the main speaker when similarity exceeds threshold for 2 consecutive requests (per session)
- Returns `accepted=True` when request audio similarity exceeds threshold after main speaker registration
- Returns `accepted=True` when no main speaker is registered yet

**NOTE:** While mechanically ignoring non-main speaker voices (Example 1) is simplest, it risks stopping conversation due to misidentification and cannot handle speaker changes. Consider context-aware handling (Example 2) as well.

```python
from aiavatar.sts.stt.speaker_gate import MainSpeakerGate
speaker_gate = MainSpeakerGate()

# Example 1: Drop request when the voice is not from main speaker
@aiavatar_app.sts.stt.preprocess
async def stt_preprocess(session_id: str, audio_bytes: bytes):
    # Compare with main speaker's voice embedding
    gate_response = await speaker_gate.evaluate(session_id, audio_bytes, aiavatar_app.sts.vad.sample_rate)
    # Branch processing based on similarity with main speaker's voice
    if not gate_response.accepted:
        logger.info(f"Ignore other speaker's voice: confidence={gate_response.confidence}")
        return None, gate_response.to_dict()
    else:
        return audio_bytes, gate_response.to_dict()

# Example 2: Add annotation for LLM that the voice is not from main speaker
@aiavatar_app.sts.stt.postprocess
async def stt_postprocess(session_id: str, text: str, audio_bytes: bytes, preprocess_metadata: dict):
    # Compare with main speaker's voice embedding
    gate_response = await speaker_gate.evaluate(session_id, audio_bytes, aiavatar_app.sts.vad.sample_rate)
    # Branch processing based on similarity with main speaker's voice
    if not gate_response.accepted:
        logger.info(f"Adding note that this may be from a different speaker: confidence={gate_response.confidence}")
        return f"$The following request may not be from the main speaker (similarity: {gate_response.confidence}). Determine from the content whether to respond. If you should not respond, output just[wait:user] as the answer:\n\n{text}", gate_response.to_dict()
    else:
        return text, gate_response.to_dict()
```


## Per-session STT Switching

You can override the STT engine for a specific session without changing the default engine used by other sessions. Batch VAD and stream VAD perform recognition in different places, so use the API belonging to the component that performs STT:

- For batch VAD, use `STSPipeline.set/get/clear_speech_recognizer()`.
- For `SileroStreamSpeechDetector`, use the detector's `set/get/clear_speech_recognizer()` methods.

The examples below ask the LLM to prefix questions whose answers may contain proper nouns with `<require_noun />`. The tag is detected before TTS, so the specialized STT is already active if the user barges in while the question is being spoken.

```python
from aiavatar.sts.llm import LLMResponse
from aiavatar.sts.stt.amivoice import AmiVoiceSpeechRecognizer

REQUIRE_NOUN_TAG = "<require_noun />"

# Create this once and share it across sessions. Do not mutate the engine on a
# shared recognizer because concurrent sessions may require different engines.
proper_noun_stt = AmiVoiceSpeechRecognizer(
    amivoice_api_key=AMIVOICE_API_KEY,
    engine="YOUR_PROPER_NOUN_ENGINE",
)

# Prompt convention for the LLM:
# When the next user response may contain a place or other proper noun, prefix
# the response with <require_noun />.
# Example: <require_noun />Where are you traveling for summer vacation?
```

### Batch VAD

Batch recognition is performed by `STSPipeline`, which owns the per-session override:

```python
# aiavatar_app.sts.get_speech_recognizer(session_id) returns the override,
# or the pipeline's default STT when no override is set.

@aiavatar_app.sts.process_llm_chunk
async def switch_stt_for_travel_question(
    llm_chunk: LLMResponse,
    session_id: str,
    user_id: str,
) -> dict:
    if REQUIRE_NOUN_TAG in (llm_chunk.text or ""):
        aiavatar_app.sts.set_speech_recognizer(
            session_id,
            proper_noun_stt,
        )
    return {}


@aiavatar_app.sts.on_finish
async def restore_default_stt(request, response):
    # Keep the specialized STT after the tagged travel question. The response
    # generated after the user's answer normally has no tag, so it restores the
    # default STT for the following turn.
    if REQUIRE_NOUN_TAG not in (response.text or ""):
        aiavatar_app.sts.clear_speech_recognizer(request.session_id)
```

### Stream VAD

Stream recognition is performed inside the VAD, so the override goes on the detector rather than on the pipeline. Call the `SileroStreamSpeechDetector` instance that was passed to the app:

```python
# stream_vad is the SileroStreamSpeechDetector instance passed to AIAvatar.
# stream_vad.get_speech_recognizer(session_id) returns the override,
# or the stream detector's default STT when no override is set.

@aiavatar_app.sts.process_llm_chunk
async def switch_stream_stt_for_travel_question(
    llm_chunk: LLMResponse,
    session_id: str,
    user_id: str,
) -> dict:
    if REQUIRE_NOUN_TAG in (llm_chunk.text or ""):
        stream_vad.set_speech_recognizer(
            session_id,
            proper_noun_stt,
        )
    return {}


@aiavatar_app.sts.on_finish
async def restore_default_stream_stt(request, response):
    if REQUIRE_NOUN_TAG not in (response.text or ""):
        stream_vad.clear_speech_recognizer(request.session_id)
```

Switching back is intentionally controlled by user code. Batch overrides remain in the pipeline until `aiavatar_app.sts.clear_speech_recognizer()` is called, so also clear them from a `finally` block or an adapter disconnect callback if a session can end before `on_finish` runs. Stream overrides are stored in the VAD recording session and are naturally discarded when that session is deleted, but explicit cleanup is still recommended for consistent application behavior.

## Where speaker data is stored

`SpeakerRegistry` keeps embeddings in a `BaseSpeakerStore`. Use the default `InMemoryStore`,
which persists to a file through its `data_path` argument.

For PostgreSQL-backed storage, use `PGVectorStore` from
`aiavatar.sts.stt.speaker_registry.postgres`. It implements the asynchronous
`BaseSpeakerStore` interface and performs Top-K search in PostgreSQL through `pgvector`.
Enable PostgreSQL's `vector` extension (provided by `pgvector`) before use and, for
application-managed connection lifecycle, pass a shared `asyncpg` pool provider through
`get_pool`.

## See also

- [Speech detector](vad.md) — streaming detectors that drive recognition
- [Text-to-Speech](tts.md) — the other end of the pipeline
- [Database](database.md) — persisting the speaker registry

---

[← Documentation index](../README.md#-documentation)
