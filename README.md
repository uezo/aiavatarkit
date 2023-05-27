# AIAvatarKit

🥰 Building AI-based conversational avatars lightning fast ⚡️💬

![AIAvatarKit Architecture Overview](documents/images/aiavatarkit_overview.png) 

# ✨ Features

* Live anywhere: VRChat, cluster and any other metaverse platforms, and even devices in real world.
* Extensible: Unlimited capabilities that depends on you.
* Easy to start: Ready to start conversation right out of the box.


# 🍩 Requirements

- VOICEVOX API in your computer or network reachable machine (Text-to-Speech)
- API key for Google Speech Services (Speech-to-Text)
- API key for OpenAI API (ChatGPT)
- Python 3.10 (Runtime)

# 🚀 Quick start

Install AIAvatarKit.

```bash
$ pip install aiavatar
```

Make the script as  `run.py`.

```python
import asyncio
import logging
from aiavatar import AIAvatar

GOOGLE_API_KEY = "YOUR API KEY"
OPENAI_API_KEY = "YOUR API KEY"
VV_URL = "http://127.0.0.1:50021"
VV_SPEAKER = 46

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)

# Prompt
system_message_content = """あなたは「joy」「angry」「sorrow」「fun」の4つの表情を持っています。
特に表情を表現したい場合は、文章の先頭に[face:joy]のように挿入してください。

例
[face:joy]ねえ、海が見えるよ！[face:fun]早く泳ごうよ。
"""

# Create AIAvatar
app = AIAvatar(
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    VV_URL,
    VV_SPEAKER,
    system_message_content=system_message_content,
)

# Start AIAvatar
asyncio.run(app.start())
```

Start AIAvatar.

```bash
$ python run.py
```

When you say the wake word "こんにちは" the AIAvatar will respond with "どうしたの？".
Feel free to enjoy the conversation afterwards!

# 🐈 Use in VRChat

* __2 Virtual audio devices (e.g. VB-CABLE) are required.__
* __Multiple VRChat accounts are required to chat with your AIAvatar.__

First, run the commands below in python interpreter to check the audio devices.

```bash
$ % python

>>> from aiavatar import AudioDevice
>>> AudioDevice.list_audio_devices()
Available audio devices:
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

- To use `VB-Cable-A` for microphone for VRChat, index for `output_device` is `13` (CABLE-A Input).
- To use `VB-Cable-B` for speaker for VRChat, index for `input_device` is `6` (CABLE-B Output). Don't forget to set `VB-Cable-B Input` as the default output device of Windows OS.

Then edit `run.py` like below.

```python
# Create AIAvatar
app = AIAvatar(
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    VV_URL,
    VV_SPEAKER,
    system_message_content=system_message_content,
    input_device=6      # Listen sound from VRChat
    output_device=13,   # Speak to VRChat microphone
)
```

Run it.

```bash
$ run.py
```

Launch VRChat as desktop mode on the machine that runs `run.py` and log in with the account for AIAvatar. Then set `VB-Cable-A` to microphone in VRChat setting window.

That's all! Let's chat with the AIAvatar. Log in to VRChat on another machine (or Quest) and go to the world the AIAvatar is in.
