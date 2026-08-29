# Platforms and devices

Where the avatar actually lives: a metaverse world, a single-board computer, or a machine
with a microphone plugged into it.

AIAvatarKit is capable of operating on any platform that allows applications to hook into audio input and output. The platforms that have been tested include:

- VRChat
- cluster
- Vket Cloud

In addition to running on PCs to operate AI avatars on these platforms, you can also create a communication robot by connecting speakers, a microphone, and, if possible, a display to a Raspberry Pi.

## VRChat

* __2 Virtual audio devices (e.g. VB-CABLE) are required.__
* __Multiple VRChat accounts are required to chat with your AIAvatar.__


### Get started

First, run the commands below in python interpreter to check the audio devices.

```sh
$ python

>>> from aiavatar.device import AudioDevice
>>> AudioDevice().list_audio_devices()
0: Headset Microphone (Oculus Virt
    :
6: CABLE-B Output (VB-Audio Cable
7: Microsoft サウンド マッパー - Output
8: SONY TV (NVIDIA High Definition
    :
13: CABLE-A Input (VB-Audio Cable A
    :
```

In this example,

- To use `VB-Cable-A` as the VRChat microphone, `output_device_index` is `13` (CABLE-A Input).
- To use `VB-Cable-B` as the VRChat speaker, `input_device_index` is `6` (CABLE-B Output). Don't forget to set `VB-Cable-B Input` as the default output device of Windows OS.

A VRChat setup has two halves. The pipeline runs as an ordinary WebSocket server, and a
local client on the same machine as VRChat owns the audio devices and the avatar.

The server is the usual one — see [Getting started](getting-started.md):

```python
# server.py
import os
from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

app = FastAPI()
app.include_router(aiavatar_app.get_websocket_router())
```

```sh
python -m uvicorn server:app
```

The client is `AIAvatarWebSocketClient`, which captures from the microphone, plays to the
speaker, and drives the avatar. Point its device indexes at the virtual cables:

```python
# client.py
import asyncio
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient(
    url="ws://localhost:8000/ws",
    input_device_index=6,     # CABLE-B Output: listen to sound from VRChat
    output_device_index=13,   # CABLE-A Input: speak into the VRChat microphone
)

asyncio.run(client.start_listening())
```

```sh
pip install "aiavatar[local-audio]"
python client.py
```

Launch VRChat as desktop mode on the machine that runs `run.py` and log in with the account for AIAvatar. Then set `VB-Cable-A` to microphone in VRChat setting window.

That's all! Let's chat with the AIAvatar. Log in to VRChat on another machine (or Quest) and go to the world the AIAvatar is in.


### Face Expression

AIAvatarKit controls the face expression by [Avatar OSC](https://docs.vrchat.com/docs/osc-avatar-parameters).

LLM(ChatGPT/Claude/Gemini)
↓ *response with face tag* `<face name="joy" />Hello!`
AIAvatarKit(VRCFaceExpressionController)  
↓ *osc* `FaceOSC=1`  
VRChat(FX AnimatorController)  
↓  
😆

So at first, setup your avatar the following steps:

1. Add avatar parameter `FaceOSC` (type: int, default value: 0, saved: false, synced: true).
1. Add `FaceOSC` parameter to the FX animator controller.
1. Add layer and put states and transitions for face expression to the FX animator controller.
1. (option) If you use the avatar that is already used in VRChat, add input parameter configuration to avatar json.


Next, use `VRChatFaceController`.

```python
from aiavatar.face.vrchat import VRChatFaceController

# Setup VRChatFaceContorller
vrc_face_controller = VRChatFaceController(
    faces={
        "neutral": 0,   # always set `neutral: 0`

        # key = the name that LLM can understand the expression
        # value = FaceOSC value that is set to the transition on the FX animator controller
        "joy": 1,
        "angry": 2,
        "sorrow": 3,
        "fun": 4
    }
)
```

Pass it to the client, which is what receives `avatar_control_request` and drives the
avatar:

```python
client = AIAvatarWebSocketClient(
    url="ws://localhost:8000/ws",
    input_device_index=6,
    output_device_index=13,
    face_controller=vrc_face_controller,
)
```

The system prompt belongs on the server, because that is where the LLM lives:

```python
# server.py
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    system_prompt="""
# Face Expression

* You have following expressions:

- joy
- angry
- sorrow
- fun

* If you want to express a particular emotion, insert it at the beginning of the sentence like <face name="joy" />.

Example
<face name="joy" />Hey, you can see the ocean! <face name="fun" />Let's go swimming.
""",
)
```

Animations work the same way: build a `VRChatAnimationController` from
`aiavatar.animation.vrchat`, pass it to the client as `animation_controller`, and list the
gesture names in the prompt. See [Avatar control](avatar.md).

You can test it not only through the voice conversation but also via the [HTTP adapter](adapters-http.md).

## Raspberry Pi

Now writing... ✍️

## Audio device

You can specify the audio devices to be used in components by device index.

First, check the device indexes you want to use.

```sh
$ python

>>> from aiavatar.device import AudioDevice
>>> AudioDevice().list_audio_devices()
{'index': 0, 'name': '外部マイク', 'max_input_channels': 1, 'max_output_channels': 0, 'default_sample_rate': 44100.0}
{'index': 1, 'name': '外部ヘッドフォン', 'max_input_channels': 0, 'max_output_channels': 2, 'default_sample_rate': 44100.0}
{'index': 2, 'name': 'MacBook Airのマイク', 'max_input_channels': 3, 'max_output_channels': 0, 'default_sample_rate': 44100.0}
{'index': 3, 'name': 'MacBook Airのスピーカー', 'max_input_channels': 0, 'max_output_channels': 2, 'default_sample_rate': 44100.0}
```

Pass the indexes to the local client. Audio devices belong to the client, not to the
pipeline — the server never touches hardware.

```python
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient(
    url="ws://localhost:8000/ws",
    input_device_index=2,     # MacBook Airのマイク
    output_device_index=3,    # MacBook Airのスピーカー
)
```

Leave an index at its default of `-1` to use the system default device. `cancel_echo`
defaults to `True` and stops the client from hearing its own output; turn it off when the
speaker and microphone are acoustically isolated, as they are with virtual cables.

## See also

- [Avatar control](avatar.md) — the control tags platform controllers consume
- [Speech detector](vad.md) — tuning detection for a noisy room

---

[← Documentation index](../README.md#-documentation)
