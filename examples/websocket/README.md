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


## Artifacts in the web viewers

`html/index.html` and `html/3d.html` can display an image, chart, Speaker Deck presentation, Docswell presentation, or YouTube video when an AI response contains a self-closing `artifact` tag. The adapter exposes registered tags through `AIAvatarResponse.control_tags`; the viewer falls back to parsing `text` when connected to an older server. The surrounding speech continues to use `voice_text`.

```html
<artifact type="image" src="https://example.com/image.png" alt="Generated image" />
<artifact type="chart" src="https://example.com/chart.svg" aspect="4:3" />
<artifact type="presentation" src="https://speakerdeck.com/player/DECK_ID" slide="7" />
<artifact type="presentation" src="https://www.docswell.com/s/USER/SLIDE_ID-2026-01-23-123456" slide="7" />
<artifact type="presentation" slide="12" />
<artifact type="presentation" offset="+1" />
<artifact type="presentation" offset="+2" />
<artifact type="video" src="https://www.youtube.com/watch?v=VIDEO_ID" autoplay-delay="3" />
<artifact action="clear" />
```

`href` is accepted as an alias of `src`. `slide` is a positive absolute page number; when `src` is omitted, it moves the currently displayed presentation. A signed `offset` such as `+1`, `+2`, or `-1` moves relative to the current page. The viewer converts navigation to the provider-specific Speaker Deck query or Docswell message. `offset` operates only within the currently displayed presentation. Docswell applies it to the player's actual position, including manual navigation. Speaker Deck applies it to the last page requested through an artifact tag, so manual navigation is intentionally ignored. `autoplay-delay` is a non-negative number of seconds from video display until the first YouTube playback attempt; it defaults to `0`. YouTube `t` or `start` URL parameters select the position inside the video and are independent of this delay. Browsers can block autoplay with sound, in which case the embedded player remains available for manual playback. `size` accepts `small`, `medium`, `large`, or `full`; `aspect` accepts `auto`, `16:9`, `4:3`, `3:2`, `1:1`, or `9:16`. Images and charts default to `auto`; presentations and videos default to `16:9`. Speaker Deck requires a `/player/...` embed URL. Docswell accepts either a `/slide/.../embed` URL or its normal `/s/...` viewing URL. Other provider URL formats are rejected.

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

The LLM can then emit `<artifact id="about_company" />`. Attributes in the LLM tag override catalog values, so `<artifact id="about_company" slide="5" size="full" />` opens page 5 at full size. A tag without an `id` continues to accept a direct `src` or `href`, which is useful for images returned by search or generation tools.

- `set_artifacts(configs)` replaces the complete shared catalog.
- `update_artifacts(configs)` adds or replaces multiple entries while retaining other IDs.
- `add_artifact(id, config)` adds or replaces one entry.

These methods update the adapter-wide catalog shared by all sessions; they do not create session-private artifacts.


## Deep Dive

The project README describes how to configure and customize the speech-to-speech pipeline or its components (VAD / STT / LLM / TTS).

https://github.com/uezo/aiavatarkit?tab=readme-ov-file#-contents
