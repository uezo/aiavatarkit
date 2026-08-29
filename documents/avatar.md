# Avatar control

A spoken answer is only part of what the avatar produces. The same LLM response also carries
its facial expression and its gestures, as control tags the model writes inline.

The chain is short and worth holding in your head:

1. The model writes `<face name="joy" />` or `<animation name="wave_hands" />` in its response.
2. The adapter parses them into `AIAvatarResponse.control_tags`; WebSocket responses also
   fill `avatar_control_request` for existing avatar clients.
3. The tags remain in `text` but are stripped from `voice_text`, so the avatar never reads
   them aloud.
4. Your client reads either representation and moves the avatar.

Nothing in the pipeline knows what a face *is*. It transports the intent; the client decides
what "joy" looks like.

## What arrives at the client

Face and animation currently travel in two forms. They appear in the generic `control_tags`
list, while WebSocket responses also populate the typed `avatar_control_request` field for
existing avatar clients.

New integrations can use `control_tags`. The typed field remains available for clients built
around `AvatarControlRequest`, which has four members:

| Field | Type | Meaning |
| --- | --- | --- |
| `face_name` | `str` | The expression key the model asked for |
| `face_duration` | `float` | How long to hold it, in seconds |
| `animation_name` | `str` | The gesture key the model asked for |
| `animation_duration` | `float` | How long to play it, in seconds |

Both durations default to `4.0` when the adapter fills the request from a tag. Fields the
model did not ask for stay `None`.

A WebSocket chunk can therefore look like this:

```json
{
    "type": "chunk",
    "text": "<face name=\"joy\" />Hey, you can see the ocean!",
    "voice_text": "Hey, you can see the ocean!",
    "control_tags": [
        {
            "name": "face",
            "attributes": {"name": "joy"}
        }
    ],
    "avatar_control_request": {
        "face_name": "joy",
        "face_duration": 4.0,
        "animation_name": null,
        "animation_duration": null
    },
    "audio_data": "..."
}
```

Note that `text` still contains the inline tag, while `voice_text` does not. The text the
avatar speaks and the instruction for how it should look therefore travel separately, so
the synthesizer never has a chance to pronounce `<face name="joy" />`.

## Telling the model what to emit

Expression keys are yours to define. The model only needs to know which ones exist and how
to write them, so a short system-prompt section is enough.

```python
aiavatar_app.sts.llm.system_prompt = """# Face Expression

* You have the following expressions:

- joy
- angry
- sorrow
- fun

* If you want to express a particular emotion, please insert it at the beginning of the sentence like <face name="joy" />.

Example
<face name="joy" />Hey, you can see the ocean! <face name="fun" />Let's go swimming.
"""
```

Animations work the same way. List the gestures your avatar can actually perform and show
the tag format:

```markdown
# Animation

* You can play the following animations:

- wave_hands
- nod
- think
- point

* Insert the animation at the beginning of the sentence like <animation name="wave_hands" />.

Example
<animation name="wave_hands" />Nice to meet you! <animation name="think" />Let me see...
```

Keep the key list short and the names obvious. A model asked to choose between five clearly
named emotions is reliable; one asked to choose between twenty subtle ones is not.

## Handling it in a browser client

The bundled browser client reads the field off each message as it plays back audio. This is
the whole of the face handling in [`aiavatar.js`](../examples/websocket/html/aiavatar.js):

```javascript
if (msg.avatar_control_request && msg.avatar_control_request.face_name) {
    this.updateFace(
        msg.avatar_control_request.face_name,
        msg.avatar_control_request.face_duration
    );
}
```

`updateFace` is yours to implement. In the 2D example it swaps an image; in the 3D VRM
viewer it drives a blend shape and returns to neutral after `face_duration`.

Animations follow the same shape. From the VRM viewer:

```javascript
const animationRequest = response.avatar_control_request;
if (animationRequest?.animation_name) {
    this.idle.playAnimation(
        animationRequest.animation_name,
        animationRequest.animation_duration
    );
}
```

Because the payload is plain JSON, this works the same from Unity, from a native app, or
from anything else that can read a WebSocket message. The mapping from key to asset lives in
the client, next to the assets.

## Handling it from Python

`AIAvatarWebSocketClient` — the local client in `aiavatar.adapter.websocket.client` — can
hand the work to a controller object instead of making you react to each response. This is
mainly how metaverse integrations are built, where the avatar lives in someone else's
runtime and is driven over a side channel.

```python
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient(
    url="ws://localhost:8000/ws",
    face_controller=face_controller,
    animation_controller=animation_controller,
)
```

A controller maps the model's keys to whatever that runtime understands. `FaceController`
holds a `faces` dict and `AnimationController` holds an `animations` dict:

```python
from aiavatar.face import FaceControllerBase
from aiavatar.animation import AnimationControllerBase

face_controller = FaceControllerBase()
face_controller.faces = {
    "neutral": "🙂",
    "joy": "😀",
    "angry": "😠",
    "sorrow": "😞",
    "fun": "🥳",
}

animation_controller = AnimationControllerBase(
    animations={"idling": 0, "waving_arm": 3, "nodding_once": 4},
    idling_key="idling",
)
```

The base implementations just log the mapped value, which is enough to confirm the model is
emitting what you expect. Both reset themselves once the duration expires — back to
`neutral` for faces, back to `idling_key` for animations.

For a real platform, subclass `FaceControllerBase` or `AnimationControllerBase` and send
whatever the runtime needs in `set_face` and `animate`. `VRChatFaceController` in
`aiavatar.face.vrchat` and `VRChatAnimationController` in `aiavatar.animation.vrchat` do
this over OSC and are worth reading before you write your own — see
[Platforms and devices](platforms.md).

If your client is a browser or an app, you do not need a controller at all. Read
`avatar_control_request` off the response and skip this section.

## See also

- [Artifacts](artifacts.md) — putting images, slides, and maps on screen
- [Vision](vision.md) — images travelling the other way
- [Text-to-Speech](tts.md) — matching voice style to the expression
- [Platforms and devices](platforms.md) — VRChat controllers over OSC
- [Adapters](adapters.md) — registering and configuring control tags per channel

---

[← Documentation index](../README.md#-documentation)
