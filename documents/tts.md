# Text-to-Speech

A `SpeechSynthesizer` turns each speakable chunk of the LLM response into audio. The
pipeline calls it as soon as the first chunk is ready, not after the response completes, so
synthesis overlaps generation.

| Provider | Class | Module |
| --- | --- | --- |
| VOICEVOX, AivisSpeech | `VoicevoxSpeechSynthesizer` | `aiavatar.sts.tts.voicevox` |
| Azure | `AzureSpeechSynthesizer` | `aiavatar.sts.tts.azure` |
| Google | `GoogleSpeechSynthesizer` | `aiavatar.sts.tts.google` |
| OpenAI | `OpenAISpeechSynthesizer` | `aiavatar.sts.tts.openai` |
| VOISONA | `VoisonaSpeechSynthesizer` | `aiavatar.sts.tts.voisona` |
| SpeechGateway | `SpeechGatewaySpeechSynthesizer` | `aiavatar.sts.tts.speech_gateway` |
| Any HTTP endpoint | `create_instant_synthesizer()` | `aiavatar.sts.tts` |
| Several at once | `SpeechSynthesizerRouter` | `aiavatar.sts.tts` |

Every synthesizer accepts the same base arguments: `style_mapper`, `preprocessors`,
`postprocessors`, `sample_rate`, `cache_dir`, `cache_ext`, `timeout`, `max_connections`,
`max_keepalive_connections`, and `debug`.

You can use a variety of text-to-speech (TTS) services as `SpeechSynthesizer` components.

## VOICEVOX

[VOICEVOX](https://voicevox.hiroshiba.jp) is a free text-to-speech engine popular for its wide variety of Japanese character voices. AIAvatarKit uses `VoicevoxSpeechSynthesizer` by default, and you can configure the speaker ID and VOICEVOX server URL directly when initializing an adapter.

```python
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key="YOUR_OPENAI_API_KEY",
    # 46 is Sayo. See http://127.0.0.1:50021/speakers to get all ids for characters
    voicevox_speaker=46,
    voicevox_url="http://127.0.0.1:50021",
)
```

For more control, create a `VoicevoxSpeechSynthesizer` instance with custom parameters.

```python
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer

tts = VoicevoxSpeechSynthesizer(
    base_url="http://127.0.0.1:50021",
    speaker=46,  # Sayo; see /speakers on your VOICEVOX server for other IDs
    timeout=10.0,
    cache_dir="./tts_cache/voicevox",
    debug=True,
)
```

[AivisSpeech](https://aivis-project.com) can also be used with `VoicevoxSpeechSynthesizer` because it provides a VOICEVOX-compatible API.

```python
tts = VoicevoxSpeechSynthesizer(
    base_url="http://127.0.0.1:10101",  # Your AivisSpeech API server
    speaker="888753761",  # Anneli
)
```

## Azure

```python
from aiavatar.sts.tts.azure import AzureSpeechSynthesizer

tts = AzureSpeechSynthesizer(
    azure_api_key="YOUR_AZURE_API_KEY",
    azure_region="YOUR_AZURE_REGION",
    speaker="ja-JP-NanamiNeural",
)
```

## Google

```python
from aiavatar.sts.tts.google import GoogleSpeechSynthesizer

tts = GoogleSpeechSynthesizer(
    google_api_key="YOUR_GOOGLE_API_KEY",
    speaker="ja-JP-Neural2-B",
)
```

## OpenAI

```python
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer

tts = OpenAISpeechSynthesizer(
    openai_api_key="YOUR_OPENAI_API_KEY",
    speaker="sage",
    audio_format="wav",
    sample_rate=16000,
)
```

`sample_rate` selects the final sample rate for synthesized audio. The built-in postprocessor resamples PCM WAV output in-process without ffmpeg. For other formats, add a compatible `TTSPostprocessor`.

Irodori-TTS can also be used through `OpenAISpeechSynthesizer` with [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server/), an OpenAI Text-to-Speech API-compatible server.

## SpeechGateway

[SpeechGateway](https://github.com/uezo/speech-gateway) provides a unified API for speech synthesis, along with features such as response caching and language-based routing across TTS services.

```python
from aiavatar.sts.tts.speech_gateway import SpeechGatewaySpeechSynthesizer

tts = SpeechGatewaySpeechSynthesizer(
    service_name="sbv2",
    speaker="0-0",
    tts_url="http://127.0.0.1:8000/tts",
)
```

## TTS Routing

Use `SpeechSynthesizerRouter` to select one of several TTS synthesizers for each request. Register a synchronous routing function with `@tts.route`; it receives the text, style information, and language, and returns a key from the `synthesizers` mapping.

```python
import os

from aiavatar.sts.tts import SpeechSynthesizerRouter
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer

japanese_tts = VoicevoxSpeechSynthesizer(
    sample_rate=16000,
    cache_dir="./tts_cache/voicevox",
)
multilingual_tts = OpenAISpeechSynthesizer(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    sample_rate=16000,
    cache_dir="./tts_cache/openai",
)

tts = SpeechSynthesizerRouter(
    synthesizers={
        "ja": japanese_tts,
        "multi": multilingual_tts,
    },
    default="ja",
)

@tts.route
def select_tts(text, style_info, language):
    if not language:
        return None  # Use the default route
    if language.lower().split("-", 1)[0] == "ja":
        return "ja"
    return "multi"
```

The router only selects and delegates. Preprocessors, postprocessors, sample-rate conversion, and caching belong to each registered synthesizer and run after routing. This allows language-specific processing, such as applying `AlphabetToKanaPreprocessor` only to the Japanese synthesizer. Cached audio is also stored and read by the selected synthesizer rather than by the router.

## TTS Caching

All TTS synthesizers support optional response caching. When `cache_dir` is set, synthesized audio is saved to disk and reused for identical requests, avoiding redundant API calls.

```python
tts = AzureSpeechSynthesizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION,
    speaker="ja-JP-MayuNeural",
    cache_dir="./tts_cache/azure",  # Enable caching
    cache_ext="wav",                # File extension (default: "wav")
)
```

- Cache files are stored as `{sha256_hash}.{cache_ext}` in the specified directory
- The hash is computed from all request parameters (URL, headers, body, etc.)
- Set `cache_dir=None` (default) to disable caching
- Works with all built-in TTS classes, including SpeechGateway and InstantSynthesizer

## Adjusting Speech Speed

With `SpeechGatewaySpeechSynthesizer`, you can change the speech speed per session by setting the speed either on the entire instance or in `style_info`.

Here is an example of storing the speech speed as `tts_speed` in session data when using WebSocketAdapter.

```python
# Apply speech speed per session
from aiavatar.sts.llm import LLMResponse
@aiavatar_app.sts.process_llm_chunk
async def process_llm_chunk(llm_stream_chunk: LLMResponse, session_id: str, user_id: str) -> dict:
    if session_data := aiavatar_app.sessions.get(session_id):
        if speed := session_data.data.get("tts_speed"):
            return {"speed": float(speed)}
```

NOTE: To configure `tts_speed`, you can either set up a REST API endpoint to update it directly, or use control tags included in responses to update it.

## VOISONA

`VoisonaSpeechSynthesizer` drives a local [VOISONA](https://voisona.com) Talk API server. It
submits a synthesis request, polls until the audio is ready, and then reads the produced
file.

```python
from aiavatar.sts.tts.voisona import VoisonaSpeechSynthesizer

tts = VoisonaSpeechSynthesizer(
    base_url="http://127.0.0.1:32766/api/talk/v1",
    username=VOISONA_USERNAME,
    password=VOISONA_PASSWORD,
    speaker="YOUR_VOICE_LIBRARY",
    voice_version=None,          # None uses the newest installed version
    default_language="ja_JP",
    global_parameters={},        # Engine-wide synthesis parameters
    poll_interval=0.05,          # How often to check whether audio is ready
    delete_request=True,         # Remove the server-side request when finished
)
```

Because synthesis is file-based, the synthesizer needs a writable directory. It picks a
platform-appropriate default; set `output_dir` to override it. Set `delete_request=False`
when you want to inspect what the engine produced.

## Style control

Every synthesizer accepts a `style_mapper`, which maps a style key coming from the LLM to
whatever the engine expects — a speaker ID, a style name, an emotion parameter.

```python
tts = VoicevoxSpeechSynthesizer(
    base_url="http://127.0.0.1:50021",
    speaker=46,
    style_mapper={
        "neutral": "46",
        "joy": "47",
        "angry": "48",
    },
)
```

Extract the style from the response in `@process_llm_chunk` and pass it to synthesis as
`style_info`. A common pattern is to have the LLM emit `<face name="angry" />`, use that tag to drive
the avatar's expression, and reuse the same key to switch voice style.

## Audio format conversion

Channels disagree about audio format. A browser is happy with WAV at 24 kHz; a telephony
channel wants 8 kHz µ-law. Two mechanisms handle that:

- **`sample_rate`** — pass it to any synthesizer and the built-in `WavSampleRatePostprocessor`
  resamples WAV output to that rate. It is always installed, after any postprocessors you
  supply. Conversion requires uncompressed PCM WAV.
- **`AudioConverter`** — for providers that return raw PCM or a compressed format, convert
  the HTTP response explicitly.

| Method | What it does | Requires |
| --- | --- | --- |
| `convert()` | Transcodes between formats and sample rates | **`ffmpeg` on `PATH`** |
| `pcm_to_wave()` | Wraps raw PCM in a WAV header | Nothing — pure Python |

`convert()` spawns `ffmpeg` as a subprocess and pipes the response through it. `ffmpeg` is
not a Python package, so `pip install` will not bring it: install it with your system package
manager (`brew install ffmpeg`, `apt install ffmpeg`) and make sure the process running
AIAvatarKit can find it. A missing binary surfaces at synthesis time, not at startup, so it
is worth checking before you deploy.

Reach for `pcm_to_wave()` when the provider can already give you raw PCM at the right sample
rate — it avoids the dependency entirely. Better still, ask the provider for WAV directly
where that is an option.

```python
from aiavatar.sts.tts import AudioConverter
```

Postprocessors run in order after synthesis and before the audio reaches the adapter, so
this is also the place to add loudness normalisation or any other transform of your own by
implementing `TTSPostprocessor`.

## See also

- [Instant TTS](tts-instant.md) — ElevenLabs, Aivis Cloud, CoeFont, Polly, and others
- [TTS preprocessing](tts-preprocessing.md) — pronunciation dictionaries
- [Avatar control](avatar.md) — driving expressions from the same response
- [Pipeline](pipeline.md) — where synthesis sits in a turn

---

[← Documentation index](../README.md#-documentation)
