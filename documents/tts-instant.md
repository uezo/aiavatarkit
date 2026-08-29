# Instant TTS

`create_instant_synthesizer()` wraps any HTTP text-to-speech endpoint in a
`SpeechSynthesizer` without writing a class. You supply the method, URL, headers, and body,
and get back something the pipeline can use like any built-in synthesizer.

```python
from aiavatar.sts.tts import create_instant_synthesizer
```

The `{text}` and `{language}` placeholders in `params`, `headers`, and `json` are replaced
with the processed text and language during synthesis.

Services whose API needs more than one substituted string — a signature, a session token, a
different HTTP verb per request — supply a `request_maker` instead. Services that return
something other than a WAV body supply a `response_parser`. Both are used below.

Each recipe below shows the shape a working configuration takes for that service. Vendor
APIs change, so check the linked reference before relying on one — the parameter names, and
whether a value belongs in `params` or `json`, are the details that move.

## Style-Bert-VITS2

A local Style-Bert-VITS2 server needs nothing but the endpoint and its parameters. Note that
`POST /voice` declares `text`, `model_id`, and `speaker_id` as **query parameters**, not a
JSON body — so they go in `params`. Sending them as `json` leaves `text` missing from the
query string and the server answers `422`.

```python
sbv2_tts = create_instant_synthesizer(
    method="POST",
    url="http://127.0.0.1:5000/voice",
    params={
        "model_id": 0,
        "speaker_id": 0,
        "text": "{text}"  # Placeholder for processed text
    }
)
```

→ [`server_fastapi.py`](https://github.com/litagin02/Style-Bert-VITS2/blob/master/server_fastapi.py)

Style-Bert-VITS2 is also reachable through SpeechGateway — see
[Text-to-Speech](tts.md#speechgateway).

## ElevenLabs

The voice is part of the URL, so build it with an f-string. The API key goes in a header.

```python
voice_id = "YOUR_VOICE_ID"

elevenlabs_tts = create_instant_synthesizer(
    method="POST",
    url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
    headers={
        "xi-api-key": ELEVENLABS_API_KEY
    },
    params={
        "output_format": "wav_16000",   # Query parameter, not part of the body
    },
    json={
        "text": "{text}",
        "model_id": "eleven_v3",
    }
)
```

Two things to get right here. `output_format` is a **query parameter** on this endpoint, so
it goes in `params`, not in `json` — putting it in the body leaves the default
`mp3_44100_128` in effect.

And pick a containerized format. The `pcm_*` values return raw, headerless PCM, which a
browser cannot play: you would have to wrap it yourself with
`AudioConverter(input_sample_rate=16000).pcm_to_wave`. `wav_16000` avoids that entirely, and
matches the pipeline's default sample rate.

→ [ElevenLabs text-to-speech reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)

## Aivis Cloud API

Aivis Cloud returns audio that needs converting, so pass `AudioConverter.convert` as the
response parser.

**`AudioConverter.convert` shells out to `ffmpeg`**, which must be installed and on `PATH`.
It is not a Python dependency, so `pip install` will not provide it — see
[Text-to-Speech](tts.md#audio-format-conversion).

```python
from aiavatar.sts.tts import AudioConverter

aivis_tts = create_instant_synthesizer(
    method="POST",
    url="https://api.aivis-project.com/v1/tts/synthesize",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIVIS_API_KEY}"
    },
    json={
        "model_uuid": "22e8ed77-94fe-4ef2-871f-a86f94e9a579",   # Kohaku
        "text": "{text}"
    },
    response_parser=AudioConverter(debug=True).convert
)
```

Aivis Cloud can also be configured as the built-in application's Japanese TTS without any
Python at all — see [Getting started](getting-started.md#built-in-tts-routing).

## Kotodama

Kotodama returns base64-encoded audio inside JSON, so it needs a small response parser.
Implement `make_request` instead if you want to apply a style or a language per request.

```python
import base64

async def base64_to_bytes(http_response) -> bytes:
    response_json = http_response.json()
    b64audio = response_json["audios"][0]
    return base64.b64decode(b64audio)

kotodama_tts = create_instant_synthesizer(
    method="POST",
    url=f"https://tts3.spiral-ai-app.com/api/tts_generate",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": KOTODAMA_API_KEY
    },
    json={
        "text": "{text}",
        "speaker_id": "Marlo",
        "decoration_id": "neutral",
        "audio_format": "wav"
    },
    response_parser=base64_to_bytes
)
```

## CoeFont

CoeFont signs each request with an HMAC over the timestamp and body, which a static
configuration cannot express. Build the request yourself and pass it as `request_maker`.
CoeFont responds with a redirect, so set `follow_redirects=True`.

```python
import hashlib
import hmac
import json
from datetime import datetime, timezone

import httpx

def make_coefont_request(text: str, style_info: dict, language: str):
    date = str(int(datetime.now(tz=timezone.utc).timestamp()))

    data = json.dumps({
        "coefont": "33e0a2ff-5050-434c-9506-defe97e52f15",  # Yuko Goto
        "text": text
    })

    signature = hmac.new(
        key=bytes(COEFONT_ACCESS_SECRET, "utf-8"),
        msg=(date+data).encode("utf-8"),
        digestmod=hashlib.sha256
    ).hexdigest()

    return httpx.Request(
        method="post",
        url="https://api.coefont.cloud/v2/text2speech",
        headers={
            "Content-Type": "application/json",
            "Authorization": COEFONT_ACCESS_KEY,
            "X-Coefont-Date": date,
            "X-Coefont-Content": signature
        },
        data=data
    )

tts = create_instant_synthesizer(
    request_maker=make_coefont_request,
    follow_redirects=True
)
```

## Amazon Polly

Polly needs AWS SigV4 signing, so it also goes through `request_maker`. It returns raw PCM,
which `AudioConverter.pcm_to_wave` wraps in a WAV container. Unlike `convert()`, this one is
pure Python and needs no `ffmpeg`.

```python
import json

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from aiavatar.sts.tts import AudioConverter

region = "ap-northeast-1"
voice_id = "Mizuki"

session = boto3.Session()
# Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as environment variables
credentials = session.get_credentials().get_frozen_credentials()

convert_pcm_to_wave = AudioConverter(input_sample_rate=16000).pcm_to_wave

def aws_polly_request_maker(text, style_info=None, language=None):
    url = f"https://polly.{region}.amazonaws.com/v1/speech"
    body = json.dumps({
        "OutputFormat": "pcm",
        "SampleRate": "16000",
        "Text": text,
        "VoiceId": voice_id,
    })
    aws_request = AWSRequest(method="POST", url=url, data=body, headers={"Content-Type": "application/json"})
    SigV4Auth(credentials, "polly", region).add_auth(aws_request)
    return httpx.Request(method="POST", url=url, headers=dict(aws_request.headers), content=body)

tts = create_instant_synthesizer(
    request_maker=aws_polly_request_maker,
    response_parser=convert_pcm_to_wave,
)
```

## COEIROINK

A local COEIROINK server takes its synthesis parameters in the body. This example also sets
`cache_dir`, so repeated lines are synthesized once and replayed from disk.

```python
tts = create_instant_synthesizer(
    method="POST",
    url="http://127.0.0.1:50032/v1/synthesis",
    headers={"Content-Type": "application/json"},
    json={
        "speakerUuid": "3c37646f-3881-5374-2a83-149267990abc",  # Tsukuyomi-chan
        "styleId": 0,
        "text": "{text}",
        "volumeScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "prePhonemeLength": 0.0,
        "postPhonemeLength": 0.0,
        "outputSamplingRate": 16000,
        "speedScale": 1.0,
    },
    cache_dir="ttscache/coeiroink/tsukuyomi-chan",
)
```

## Writing a synthesizer class instead

When a service needs more than a request and a response — connection pooling of its own,
multi-step synthesis, a websocket — implement the `SpeechSynthesizer` interface directly.

The base `synthesize()` handles empty text, preprocessing, caching, and postprocessing, so a
minimal implementation only needs to provide `generate()`. The `text` passed to both methods
has already been preprocessed.

The default cache key is built from the synthesizer class, the processed text, the style
information, and the language. Override `make_synthesis_cache_key()` when synthesis also
depends on provider-specific settings such as the model, speaker, or speed — otherwise two
different voices will collide on one cache entry.

## See also

- [Text-to-Speech](tts.md) — built-in synthesizers, routing, and caching
- [TTS preprocessing](tts-preprocessing.md) — fixing pronunciation before synthesis
- [Getting started](getting-started.md) — configuring instant TTS from the CLI

---

[← Documentation index](../README.md#-documentation)
