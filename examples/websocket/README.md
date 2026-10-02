# AIAvatarKit WebSocket Example

AIAvatarKit supports low-latency, real-time conversations not only from standalone programs but also from various client applications such as web browsers over WebSocket connections.

In addition to dialogue, you can drive facial expressions and motion by following the control data included in WebSocket responses.

To add Nod to your STT → LLM → TTS pipeline, follow the
[Nod setup guide](../nod/README.md#add-nod-to-a-speech-pipeline).
It uses the stream VAD, automatically registered event hooks, in-memory nod
history, cached audio, and the existing 3D viewer. The browser preserves nods
when receiving a normal `stop` message; no adapter setting is required.

## Nod playback

Enable BARGE-IN in `3d.html` (VRM/MMD) to continue microphone input during nods.
Use complete WAV delivery (`response_audio_chunk_size=0`) for the shared browser.
The page uses `AIAvatarClient` in `aiavatar.js`; no additional client script is
needed.

The client preserves queued, decoding, and audible nods across speech resumption.
Main response acceptance/start/audio discards pending nods while audible ones
finish. Main-response generation proceeds immediately; its playback waits behind
an audible nod in the serial browser queue.

On a `stop` message, the browser stops main-response audio and removes queued main
audio while retaining nods identified by `metadata.nod=true` on audio messages.
Stop messages need no additional metadata. Without nod metadata, ordinary response
playback and barge-in behavior remain unchanged.
The Stop button and WebSocket disconnection stop all playback. Disconnection also
releases microphone capture and the audio context, including a microphone acquired
after disconnection while permission was pending.

## Playback context for turn-taking

### Recognition preview

`AvatarUI` defaults to `separatePartialTranscript: false`: partial speech
recognition appears directly in the main message box, preserving the original
behavior. Set it to `true` to show recognition below the message box while keeping
the main AI message and 3D typewriter running. Custom pages can specify it in the
constructor:

```javascript
const ui = new AvatarUI({ aiavatar, camera, separatePartialTranscript: true });
```

In `3d.html`, the **UI → Message box → Live transcript below** switch controls
this option. The 3D settings save and restore it; resetting restores the configured
default (off by default).
The separate preview used by `index.html` and `3d.html` is a small, left-aligned
line without a separate panel or visible label. Long text follows the latest
words on that single line. A visually hidden label identifies it as
`User · Transcribing` (using the configured user name) for assistive technology.
In either mode, the existing `start.metadata.recognized_text` event clears any
separate preview and shows the accepted user utterance in the main message box.

When enabled, the separate preview disappears three seconds after its last
update, so discarded backchannels do not linger. This is a display timeout, not a turn-taking
decision. Stopping, disconnecting, starting a new session, or receiving a
`canceled`/`error` event also clears it. AI `chunk`/`final` events keep the preview
visible until that timeout or the accepted input's `start`. The 3D viewer's
"Show user speech" setting also applies to new recognition previews. No server
notification or protocol change is required.

### Playback notifications

`index.html` and `3d.html` install the shared `playback-context.js` helper. After
decoding a main-response audio chunk, immediately before starting its audio
source, the browser sends a separate WebSocket message:

```json
{
  "type": "playback",
  "session_id": "session-1",
  "metadata": {
    "event": "start",
    "playback_id": "browser-generated-unique-id",
    "transaction_id": "response-transaction-id",
    "text": "This is the text being read aloud.",
    "duration_seconds": 3.0
  }
}
```

`text` uses the chunk's `voice_text`. Duration comes from the decoded audio buffer
in seconds. The browser generates the playback ID for each chunk and copies the
response's top-level `transaction_id` into this notification. The transaction ID
groups chunks of one AI response; a missing or null ID is omitted for older
servers. Nod, backlog audio, and messages without a
nonempty `voice_text` or session ID do not send start events. Queued or
still-decoding audio has not started yet and does not send a start event.

Natural completion sends an end event for that same ID:

```json
{
  "type": "playback",
  "session_id": "session-1",
  "metadata": {
    "event": "end",
    "playback_id": "browser-generated-unique-id",
    "completed": true
  }
}
```

Interruption, including stopping playback, uses `completed: false`. These events
are independent of microphone `data` messages. Use the shared browser's complete
WAV delivery mode (`response_audio_chunk_size=0`).

When the normal response `final` arrives, the helper identifies its last audio
chunk using the matching session/transaction and playback queue. Once that chunk
has started, it sends one additional notification:

```json
{
  "type": "playback",
  "session_id": "session-1",
  "metadata": {
    "event": "final",
    "playback_id": "last-audio-chunk-id",
    "transaction_id": "response-transaction-id"
  }
}
```

This identifies the final audio chunk; it does not mean playback has ended.
If the response `final` arrives after that chunk starts or naturally finishes,
the notification is sent then. Interrupted/error responses, missing transaction
IDs, nods, and backlog audio do not establish a final chunk. An empty queue alone
does not establish one either.

For Jev integration, import `JevTurnTakingGate` from
`aiavatar.sts.vad.turn_taking_gates.jev` and pass it to the VAD constructor as
`turn_taking_gate=...`. Other classifiers can implement the
[`TurnTakingGate` base class](../../documents/vad-turn-taking.md#custom-gates)
and reuse the same playback and session handling. The gate keeps one
`TurnTakingSession` per connection's unique session ID. A gate can be shared
across connections and VADs with different session IDs. With the standard
WebSocket adapter and Silero or SileroStream VAD, the `start` request initializes
the helper through the VAD's session setup, and disconnect finalizes it through
the pipeline. No dedicated start/disconnect hooks or helper reference in
connection data are needed. For standalone gate use,
first register the helper with `gate.get_session(session_id, create=True)`.
In the existing server `on_request` hook, verify that the connection is current,
then forward the playback metadata:

```python
if request.type == "playback" and vad.turn_taking_gate is not None:
    vad.turn_taking_gate.handle_playback_event(
        request.session_id, **(request.metadata or {}),
    )
```

`handle_playback_event()` is available on both a single gate and
`TurnTakingGateManager`. It routes `start`, `end`, and `final` to the existing
session's corresponding methods and returns their boolean result. It ignores
extra metadata fields and treats `completed` as true only when it is the literal
boolean `True`. Missing or closed sessions and unknown events return `False`;
the method never creates a helper. `session_id` can also be passed by keyword.

The VAD calls `gate.evaluate()` before notifying `on_speech_detected`. The gate
handles session lookup, classification, error fallback, cancellation, logging,
and the check that the helper is still current after classification. Evaluation
does not recreate a missing session. Only allowed speech reaches the callbacks
and the pipeline that creates an `STSRequest`.
There is no application speech-filter callback or registration-order requirement;
callback return values do not suppress later callbacks. The VAD passes the
optional recording ID for diagnostics and its `vad_performance.speech_end_at`
timestamp to correct for processing delay. Neither a missing recording ID nor
missing recognized text rejects an input; text-free VAD output passes through
without classification.

The server accumulates started chunks only for the current transaction ID,
retaining each chunk's text, duration, and server playback timestamps. A new ID
replaces that response's state. A missing, empty, or invalid ID uses independent
single-chunk tracking, so unrelated responses are never grouped by a missing ID.

With a valid timezone-aware `speech_end_at`, the server maps the user's speech
end onto its monotonic playback timeline. It concatenates the text already
spoken by that time, estimating a partial chunk from its own elapsed time and
audio duration. Later chunks contribute no future words. If playback has moved
to the next chunk of the same response, the estimate can still use the preceding
chunk; a gap between known chunks uses the preceding spoken text. A sentence
boundary alone does not mean the assistant has yielded the turn.

For Jev evaluation only, if a completed chunk has a known following chunk of
the same response with zero estimated characters, the server appends that next
chunk's first two characters plus `…` (or the whole text if shorter plus `…`).
This applies to known gaps too, but not to a target inside the preceding chunk
or a chunk that was cut short. Playback progress stays unchanged. Session debug
logs separate `assistant_spoken_text`, supplemented `evaluation_text`, and
`continuation_hint` so this preview can be checked against the decision.

Missing, invalid, or timezone-naive timestamps use decision time; future
timestamps contribute zero delay. Before the retained response starts or after
its last known playback ends, input passes without calling Jev. Playback that
ended after `speech_end_at` can still be judged at that earlier time. The server
does not predict unreceived chunks or retain previous transactions. Without a
transaction ID, the original single-chunk behavior remains: ended playback or
an empty estimated prefix allows input. Network delay and text-to-audio
alignment still affect this approximation.

To allow a quick reply at the end of the whole response, configure
`JevTurnTakingGate(response_end_grace_seconds=0.3, ...)`. When the corrected user
speech end falls within the last 300 ms of the confirmed final audio chunk, the
session allows the input without calling Jev, with reason
`playback_response_end_grace`. This intentionally allows acknowledgments there
as well. Earlier chunks, gaps, interrupted audio, and unconfirmed final chunks
keep their existing behavior. The default is `0.0` (disabled); this is a duration
in seconds, not a percentage of each chunk. The grace also applies when the
audio has already naturally finished by decision time but the corrected speech
end falls in that final window.

The estimate is frozen before awaiting Jev, so later controls or recordings
cannot change it. Matching end events stop only the identified current playback.
Response state is cleared on replacement or connection cleanup.

Each session keeps only its latest finalized-input decision eligible. A new
`should_take_turn()` call cancels the preceding pending call without waiting for its cleanup,
even when the new input bypasses Jev. Superseded results and exceptions propagate
as `asyncio.CancelledError` instead of reaching the pipeline; decisions already
returned are unaffected. Deleting the VAD session removes and closes its helper
through the gate, cancelling pending calls; `await vad.finalize_session(session_id)` also awaits their cleanup,
including superseded calls still unwinding. A connection hook can additionally
await its captured helper's `aclose()`; cleanup is idempotent. The default `request_timeout=1.0`
allows the current input on timeout (`jev_timeout`); other request
failures allow it with `jev_error`.

For classification-only barge-in, use `mute_on_barge_in=False` so playback
continues until the decision. To stop after a configurable long utterance instead,
use `mute_on_barge_in=True`, set `vad.on_recording_started_min_duration` (for
example, `3.0` seconds). With a `turn_taking_gate` configured, the VAD's default
recording-start condition uses duration only and ignores the text-length
alternative, so a short recognized acknowledgment does not stop audio. No custom
callback is needed. Without a gate, the existing duration-or-text behavior remains.
An explicit `vad.should_trigger_recording_started` callback overrides the default.
With the gate's default `skip_condition=None`, the VAD supplies the matching
final-input condition: `recorded_duration >= vad.on_recording_started_min_duration`.
Matching inputs pass without Jev with reason `turn_take_skipped`, while still
cancelling an older pending decision. Each input uses its own duration, so no
persistent skip flag or reset is needed. An explicit synchronous
`skip_condition(text, recorded_duration)` takes precedence; use
`lambda text, recorded_duration: False` to disable the duration bypass. The
default is supplied per session, so sharing a gate across VADs does not mix their
duration policies. Maximum-duration flushes use the VAD's reported total duration,
which includes trailing silence.
Changing the gate's skip condition does not change the recording-start condition.
The turn-end gate and existing playback stop behavior are unchanged; no new
suppression of later generated audio is added. See the
[duration threshold example](../../documents/vad-turn-taking.md#interrupting-longer-utterances).

Keep `debug=True` to trace helper creation (`event=session_started`), initial
closure (`event=session_closed`), final-input entry, playback estimates, Jev
decisions, and bypasses. Repeated cleanup does not duplicate the closure log.
`event=superseded` records when newer finalized input replaces a pending decision.
The lifecycle logs describe the helper. `handle_playback_event()` also logs
ignored controls for missing/closed sessions and unknown events; controls
rejected before reaching the gate remain transport-level diagnostics.
Stop/Finish controls must end current playback, and disconnect/replacement must
finalize the old VAD session. Use a new session ID when reconnecting. The gate
obtains the helper when evaluation starts and checks its ownership after
classification. Accepted inputs follow the normal VAD callback dispatch without
further turn-taking session checks between handlers.
The playback bridge must ignore controls from disconnected/replaced connections
before calling `handle_playback_event`. Playback controls never recreate state.
A response transaction ID mismatch is not a reason to drop the user's speech.
See [Jev turn-taking](../../documents/vad-turn-taking.md#quick-start-with-jev)
for setup, classification policy, and cleanup details. The injected HTTP client
remains application-owned and must be closed separately at shutdown.

For custom pages, call `installPlaybackContext(aiavatar)` after binding the avatar
or initializing the 3D model adapter. It composes existing `onPlaybackStart` and
`onPlaybackEnd` callbacks while preserving audio/lip-sync and response callbacks.
Call the returned helper's `handleResponse(response)` at the start of the page's
`onResponseReceived` callback, as the maintained 2D and 3D pages do, to identify
final chunks and clear response tracking on connection/stop events.
Its returned `dispose()` sends an interrupted end for the current context and
restores the callbacks it owns; it does not stop the audio itself.
`onPlaybackStart` receives `{message, playbackId, durationSeconds}` immediately
before `source.start()`. `onPlaybackEnd` receives the same information plus `completed`,
which is `true` for natural completion and `false` for interruption. Controls
from an old or disconnected socket are not sent into a replacement connection.

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

Use `engine: new LipSyncEngine({...})` for the legacy implementation. The adapters call the injected object without branching on its class. `profile` can be used instead of `profileUrl` to pass an already parsed Profile object. `phonemeScoreMultipliers`, such as `{ U: 1.4 }`, can bias individual Profile phoneme scores before normalization when a calibrated mouth shape is systematically under-selected. Keys are case-insensitive, omitted multipliers default to `1.0`, and adjusted scores are capped at `1.0`. This changes which viseme is selected, not how far the mouth opens. For VRM and MMD, `maxVisemeWeight` scales the final viseme weights proportionally: `0.5` makes an engine weight of `0.7` apply as `0.35`. It defaults to `1.0` when omitted.

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

For a higher-quality context-aware Profile, use the unsuffixed files for
consonant-vowel calibration recordings and optionally add a complete clean-vowel
reference set:

```text
calibration/
  a.wav             a_reference.wav
  i.wav             i_reference.wav
  u.wav             u_reference.wav
  e.wav             e_reference.wav
  o.wav             o_reference.wav
```

When all five reference files are present, the Profile data still comes only
from `a.wav` through `o.wav`. The clean reference set is used to rank stable
candidate frames by similarity to the intended vowel and separation from the
other four vowels. Candidate selection is balanced across the voiced sections
before the best 16 frames are stored. A partial reference set is rejected so
that every vowel is selected under the same conditions. Without reference
files, generation behaves exactly as described below.

The WAV files must be uncompressed 16 kHz PCM or IEEE-float audio. Mono is recommended; for a multichannel file, only the first channel is analyzed. The tool deliberately does not resample calibration audio, so an accidental sample-rate mismatch is reported instead of being hidden.

From `examples/websocket`, generate a Profile with:

```sh
node tools/build-mfcc-profile.mjs calibration html/profiles/custom-voice.json
```

When the output argument is omitted, the tool writes `calibration/mfcc-profile.json`. Without a reference set, it first looks for one sufficiently long voiced section. If none exists, it automatically combines the stable centers of repeated short sections; no option is required. It then selects 16 distributed MFCC frames and prints a five-vowel self-check plus a quality analysis. The analysis reports leave-one-out (LOO) classification, within-vowel stability, the margin from the nearest competing vowel, the closest vowel pairs, and heuristic warnings for variable or overlapping calibration data. The self-check is a basic training-data sanity check; a perfect score can still accompany a thin classification margin, so inspect the LOO result and warnings as well. These metrics are calibration hints rather than a guarantee for arbitrary sentences. They are printed to the terminal and are not added to the compatible Profile JSON.

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
