# AIAvatarKit WebSocket Example

AIAvatarKit supports low-latency, real-time conversations not only from standalone programs but also from various client applications such as web browsers over WebSocket connections.

In addition to dialogue, you can drive facial expressions and motion by following the control data included in WebSocket responses.


## Quickstart (Web Browser)

💡 Prerequisite: Install [VOICEVOX](https://voicevox.hiroshiba.jp) in advance and keep it running on localhost port 50021.

Get the code from GitHub. [Downloading the ZIP](https://github.com/uezo/aiavatarkit/archive/refs/heads/main.zip) also works.

```sh
git clone https://github.com/uezo/aiavatarkit
```

Move into the WebSocket example directory.

```sh
cd aiavatarkit/examples/websocket
```

Install the required libraries.

```sh
pip install aiavatar
```

Open `server.py` and set your OpenAI API key to `OPENAI_API_KEY`.

```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

Start the server.

```sh
uvicorn server:app
```

Set `AVATAR_MODE` in `html/index.html` to `"image"` or `"mpt"`. Then visit http://localhost:8000/static/index.html, click `Start`, and try talking to the avatar.


## Lip sync engines

The Image, VRM, and MMD viewers can use either the legacy `LipSyncEngine` or the MFCC-based `MFCCLipSyncEngine`. Both receive decoded playback PCM from `AIAvatarClient` and expose the same interface:

```javascript
await engine.initialize();
const result = engine.processAudioData(audio);
// result.visemes: { A, I, U, E, O } (volume-scaled blend weights)
// result.mainViseme: "A" | "I" | "U" | "E" | "O" | null
// result.mainVisemeWeight: 0.0 ... 1.0
```

Set the engine independently in the VRM or MMD model options:

```javascript
lipsync: {
    usePhonemeBlend: false,
    maxVisemeWeight: 0.5,
    engine: new MFCCLipSyncEngine({
        profileUrl: "profiles/default-female.json",
        minVolume: -2.5,
        maxVolume: -0.8,
        volumeGain: 1,
    }),
},
```

Use `engine: new LipSyncEngine({...})` for the legacy implementation. The adapters call the injected object without branching on its class. `profile` can be used instead of `profileUrl` to pass an already parsed Profile object. For VRM and MMD, `maxVisemeWeight` scales the final viseme weights proportionally: `0.5` makes an engine weight of `0.7` apply as `0.35`. It defaults to `1.0` when omitted.

The Image viewer injects an `MFCCLipSyncEngine` into `ImageAvatar` by default. Omitting `lipsyncEngine` keeps the legacy engine as a fallback. Image mouths are selected statelessly from each result: silence closes the mouth, a low-volume `A` or `O` and an ambiguous viseme distribution use `half`, stronger `A` or `O` use `open`, `U` uses `u`, and `I` or `E` uses `e`. MotionPNGTuber continues to use its own dedicated lip sync implementation.

`MFCCLipSyncEngine` reads MFCC Profile JSON compatible with the [uLipSync](https://github.com/hecomi/uLipSync) v3 format. The bundled `default-female.json` was independently calibrated for the example's default female TTS voice and contains no uLipSync Sample Profile data. The engine always returns both volume-scaled blend weights and the highest-scoring viseme with its full normalized opening weight. The 3D adapter's `lipsync.usePhonemeBlend` setting decides which representation to apply: `false` applies only `mainViseme` at `mainVisemeWeight`, while `true` applies `visemes`. Profiles are voice-dependent, so replace the default with one calibrated for the target voice when better accuracy is required. The VRM adapter maps the common `A/I/U/E/O` output to the three-vrm expression presets `aa/ih/ou/ee/oh`; three-vrm exposes these unified names for both VRM 0.x and 1.0 models.

### Generate an MFCC Profile for a TTS voice

Create five files containing Japanese vowels (for example, `あー` rather than the spoken letter name). A sustained vowel about 1.5 seconds long is ideal. If the TTS cannot prolong a sound, several short repetitions such as `あ、あ、あ、あ、あ、あ` are also supported; separate them with short audible pauses. Include a little silence before and after the speech:

```text
calibration/
  a.wav
  i.wav
  u.wav
  e.wav
  o.wav
```

The WAV files must be uncompressed 16 kHz PCM or IEEE-float audio. Mono is recommended; for a multichannel file, only the first channel is analyzed. The tool deliberately does not resample calibration audio, so an accidental sample-rate mismatch is reported instead of being hidden.

From `examples/websocket`, generate a Profile with:

```sh
node tools/build-mfcc-profile.mjs calibration html/profiles/custom-voice.json
```

When the output argument is omitted, the tool writes `calibration/mfcc-profile.json`. It first looks for one sufficiently long voiced section. If none exists, it automatically combines the stable centers of repeated short sections; no option is required. It then selects 16 distributed MFCC frames and prints a five-vowel self-check plus a quality analysis. The analysis reports leave-one-out (LOO) classification, within-vowel stability, the margin from the nearest competing vowel, the closest vowel pairs, and heuristic warnings for variable or overlapping calibration data. The self-check is a basic training-data sanity check; a perfect score can still accompany a thin classification margin, so inspect the LOO result and warnings as well. These metrics are calibration hints rather than a guarantee for arbitrary sentences. They are printed to the terminal and are not added to the compatible Profile JSON.

The generated Profile contains `A/I/U/E/O`; silence still closes the mouth through the engine's volume gate, so a `-.wav` file is not required. Select the generated file with `profileUrl` in the engine options shown above. A low self-check or LOO score usually means that a clip contains the wrong vowel, has too few usable repetitions, or changes voice quality between repetitions.

### Acknowledgements

The MFCC processing in `html/mfcc-lipsync.js` was developed with reference to the [uLipSync v3 processing pipeline](https://github.com/hecomi/uLipSync). Its Profile JSON reader remains compatible with uLipSync v3 so existing calibrated profiles can be reused. No uLipSync runtime, library, or official Sample Profile data is bundled with this example.

uLipSync is Copyright (c) 2021 hecomi and is distributed under the MIT License.

<details>
<summary>uLipSync MIT License</summary>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

</details>

## Artifacts in the web viewers

`html/index.html` and `html/3d.html` can display an image, chart, presentation, YouTube video, sandboxed web app, or Google map when an AI response contains a self-closing `artifact` tag. The adapter parses and resolves registered tags into `AIAvatarResponse.control_tags`, which is the viewer's only artifact command source. The viewer does not parse tags from response `text`. The surrounding speech continues to use `voice_text`.

```html
<artifact type="image" src="https://example.com/image.png" alt="Generated image" />
<artifact type="chart" src="https://example.com/chart.svg" aspect="4:3" />
<artifact type="presentation" src="https://speakerdeck.com/player/DECK_ID" slide="7" />
<artifact type="presentation" src="https://www.docswell.com/s/USER/SLIDE_ID-2026-01-23-123456" slide="7" />
<artifact type="presentation" slide="12" />
<artifact type="presentation" offset="+1" />
<artifact type="presentation" offset="+2" />
<artifact type="video" src="https://www.youtube.com/watch?v=VIDEO_ID" autoplay-delay="3" />
<artifact type="webapp" src="https://example.com/app" />
<artifact type="map" location="Tokyo Station" zoom="16" />
<artifact type="map" origin="Tokyo Station" destination="Tokyo Tower" travel-mode="walking" />
<artifact action="clear" />
```

`webapp` loads an HTTPS page in a sandboxed iframe; the page must permit embedding. It can request a new chat turn by posting `{ type: "aiavatar.webapp.invoke", version: 1, text, imageDataUrl }` to its parent window; `imageDataUrl` is optional. Payloads, source windows, sizes, and invocation frequency are validated by the viewer.

`map` uses the Google Maps Embed API. Specify either `location`, a `latitude`/`longitude` pair, or both `origin` and `destination` for directions. `travel-mode` accepts `driving`, `walking`, `bicycling`, `transit`, or `flying`; `zoom` accepts an integer from `0` to `21`. Enable the Maps Embed API, restrict its browser key, and replace `YOUR_GOOGLE_MAPS_EMBED_API_KEY` in the viewer HTML before use.

`href` is accepted as an alias of `src` for URL-based artifacts. `slide` is a positive absolute page number; when `src` is omitted, it moves the currently displayed presentation. A signed `offset` such as `+1`, `+2`, or `-1` moves relative to the current page. The viewer converts navigation to the provider-specific Speaker Deck query or Docswell message. `offset` operates only within the currently displayed presentation. Docswell applies it to the player's actual position, including manual navigation. Speaker Deck applies it to the last page requested through an artifact tag, so manual navigation is intentionally ignored. `autoplay-delay` is a number from `0` to `3600` seconds from video display until the first YouTube playback attempt; it defaults to `0`. YouTube `t` or `start` URL parameters select the position inside the video and are independent of this delay. Browsers can block autoplay with sound, in which case the embedded player remains available for manual playback. `size` accepts `small`, `medium`, `large`, or `full`; `aspect` accepts `auto`, `16:9`, `4:3`, `3:2`, `1:1`, or `9:16`. Images and charts default to `auto`; presentations, videos, web apps, and maps default to `16:9`. Speaker Deck requires a `/player/...` embed URL. Docswell accepts either a `/slide/.../embed` URL or its normal `/s/...` viewing URL. Other provider URL formats are rejected.

While an artifact is visible, the VRM moves into a compact overlay and uses a separate camera state. Its first position automatically frames the full model; later drag, rotation, and zoom adjustments are stored separately in local storage. Closing the artifact restores the normal camera without changing it.

Adapters register `face`, `animation`, `vision`, and `artifact` by default. Applications can keep long URLs out of the LLM response by replacing the built-in artifact catalog:

```python
ARTIFACTS = {
    "about_company": {
        "type": "presentation",
        "src": "https://speakerdeck.com/player/DECK_ID",
        "slide": 1,
        "aspect": "16:9",
        "title": "About the company",
    },
}

aiavatar_app.set_artifacts(ARTIFACTS)
```

The LLM can then emit `<artifact id="about_company" />`. Display and navigation attributes in the LLM tag override catalog values, so `<artifact id="about_company" slide="5" size="full" />` opens page 5 at full size. When an `id` is present, the configured `type` and `src` are protected and cannot be replaced by tag attributes. A tag without an `id` continues to accept a direct `src` or `href`, which is useful for images returned by search or generation tools.

- `set_artifacts(configs)` replaces the complete shared catalog.
- `update_artifacts(configs)` adds or replaces multiple entries while retaining other IDs.
- `add_artifact(id, config)` adds or replaces one entry.

These methods update the adapter-wide catalog shared by all sessions; they do not create session-private artifacts.

Direct artifact URLs cause the browser to request the resolved location. The server application is responsible for allowing only URLs that are safe for its users and environment. An `on_response` handler receives `AIAvatarResponse.control_tags` after ID resolution, and can validate, rewrite, or remove artifacts before they are sent:

```python
from urllib.parse import urlsplit

TRUSTED_ARTIFACT_HOSTS = {"cdn.example.com"}

def is_trusted_artifact_url(source):
    try:
        url = urlsplit(source)
        return (
            url.scheme == "https"
            and url.hostname in TRUSTED_ARTIFACT_HOSTS
            and url.username is None
            and url.password is None
        )
    except ValueError:
        return False

@aiavatar_app.on_response
async def validate_artifact_urls(response, _):
    if not response.control_tags:
        return

    validated = []
    for tag in response.control_tags:
        if tag.name != "artifact":
            validated.append(tag)
            continue

        source = tag.attributes.get("src") or tag.attributes.get("href")
        if source and not is_trusted_artifact_url(source):
            continue
        validated.append(tag)

    response.control_tags = validated
```


## Deep Dive

The project README describes how to configure and customize the speech-to-speech pipeline or its components (VAD / STT / LLM / TTS).

https://github.com/uezo/aiavatarkit?tab=readme-ov-file#-contents
