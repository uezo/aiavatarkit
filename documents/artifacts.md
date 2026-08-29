# Artifacts

An artifact is something the avatar puts on screen while it talks: an image, a chart, a
slide deck, a video, a map, or a whole sandboxed web app. The model asks for it with an
`<artifact />` tag in its response and the client renders it.

This shares a transport with [avatar expressions](avatar.md) — both are control tags — but it
is doing a different job. Expressions make the avatar look right; artifacts change what the
user can actually see and do. An artifact can be a running web app, so treat what reaches
this channel with the same care you would treat anything else you embed in a page.

Artifacts arrive on `AIAvatarResponse.control_tags` as a list of `ControlTag`, each with a
`name` and an `attributes` dict, in the order the model wrote them. They are excluded from
`voice_text`, so they are never read aloud. The bundled viewers consume only these
structured tags and never parse tags out of the response `text`.

Both the standard and 3D bundled viewers support images, charts, presentations, YouTube
videos, sandboxed web apps, Google maps, and directions.

## Registering a catalog

Register application-owned artifacts on the Adapter so the LLM only needs to select a short, stable ID:

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

The LLM can display the configured artifact with `<artifact id="about_company" />`. Display and navigation attributes supplied by the LLM override configured defaults, so `<artifact id="about_company" slide="5" />` starts at page 5. When an `id` is present, configured `type` and `src` values are protected and cannot be replaced by tag attributes. When the LLM knows a browser-accessible HTTPS URL, it can also display an image with `<artifact type="image" src="https://example.com/image.png" />` or a YouTube video with `<artifact type="video" src="https://www.youtube.com/watch?v=VIDEO_ID" autoplay-delay="3" />`. Use `<artifact action="clear" />` to hide the currently displayed artifact.

## Media, maps, and web apps

YouTube watch, shortened `youtu.be`, and embed URLs are supported. `autoplay-delay` specifies a fixed delay of 0 to 3600 seconds from displaying the video until its first autoplay attempt and defaults to `0`. The `t` and `start` URL parameters select the initial position within the video; they do not affect the autoplay delay. Browsers may block autoplay with sound, in which case the embedded controls remain available for manual playback.

Web apps use `<artifact type="webapp" src="https://example.com/app" />` and run in a sandboxed iframe. Maps use `<artifact type="map" location="Tokyo Station" />`; directions use `<artifact type="map" origin="Tokyo Station" destination="Tokyo Tower" travel-mode="walking" />`. Maps require a Maps Embed API key in the viewer HTML; restrict it to approved websites and the Maps Embed API. See the [WebSocket example documentation](../examples/websocket/README.md#artifacts-in-the-web-viewers) for supported attributes and web-app invocation messages.

## Changing the catalog at runtime

The catalog can be changed at runtime:

- `set_artifacts(configs)` replaces the complete catalog.
- `update_artifacts(configs)` adds or replaces multiple entries while retaining other IDs.
- `add_artifact(id, config)` adds or replaces one entry.

The catalog belongs to the Adapter and is shared by all sessions. Use an application-level or session-level store instead when generated artifacts must remain private to one user.

## Validating what reaches the browser

Direct artifact URLs cause the user's browser to request the resolved location. The server application is responsible for allowing only URLs that are safe for its users and network environment. `on_response` runs after artifact ID resolution, so it can validate, rewrite, or remove the resolved `control_tags` before delivery:

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

## Prompting the model

A compact system-prompt section is usually sufficient:

```markdown
## Artifacts
To display an image, chart, slide, video, map, or route, insert an `<artifact />` tag immediately before the relevant sentence in the response body. Do not read the tag aloud or explain it.

- Registered artifact: `<artifact id="{ARTIFACT_ID}" />`
- HTTPS URL: `<artifact type="{TYPE}" src="{HTTPS_URL}" />`
  - Use `image` for an image, `chart` for a chart, `presentation` for slides, or `video` for a YouTube video.
  - A Docswell viewing URL (`https://www.docswell.com/s/...`) can be used directly as a presentation `src`.
  - Speaker Deck requires an embed URL (`https://speakerdeck.com/player/...`).
  - YouTube videos accept `autoplay-delay` from `0` to `3600` as the number of seconds before the first autoplay attempt: `<artifact type="video" src="https://www.youtube.com/watch?v=VIDEO_ID" autoplay-delay="3" />`
- Map: `<artifact type="map" location="{PLACE_NAME_OR_ADDRESS}" />`
  - Use this when the user asks to display a map for a place name or address.
  - Optionally set `zoom` to an integer from `0` to `21`.
- Directions: `<artifact type="map" origin="{ORIGIN}" destination="{DESTINATION}" travel-mode="{TRAVEL_MODE}" />`
  - Use `driving` for driving, `walking` for walking, `bicycling` for bicycling, or `transit` for public transit. Omit `travel-mode` when the user does not specify a mode of travel.
- Presentation controls are available only with `type="presentation"`, not with images, charts, or videos.
  - Move the displayed presentation to a numbered page: `<artifact type="presentation" slide="3" />`
  - To set the starting page of a new presentation, specify `id` or `src` together with `slide` in the same tag.
  - Move relative to the current page: `<artifact type="presentation" offset="+1" />`, `offset="-1"`, `offset="+2"`, and so on.
  - For relative navigation, use a signed `offset` instead of `slide` and specify it in a single tag.
  - Never use numbered page navigation when the request is relative to the current page.
- Never invent unknown IDs or URLs.
- Only the most recently specified artifact is displayed. A new artifact replaces the previous one.
- To hide the current artifact, output `<artifact action="clear" />`.

### Available Artifacts
- `about_company`: Company overview presentation
```

## See also

- [Avatar control](avatar.md) — expressions and gestures on the same channel
- [Tools](tools.md) — structured content as the other way to reach the client
- [Adapters](adapters.md) — registering and configuring control tags
- [Vision](vision.md) — images arriving from the user

---

[← Documentation index](../README.md#-documentation)
