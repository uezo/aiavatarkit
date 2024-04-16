# AIAvatarKit

🥰 Building AI-based conversational avatars lightning fast ⚡️💬

![AIAvatarKit Architecture Overview](documents/images/aiavatarkit_overview.png) 

# ✨ Features

* Live anywhere: VRChat, cluster and any other metaverse platforms, and even devices in real world.
* Extensible: Unlimited capabilities that depends on you.
* Easy to start: Ready to start conversation right out of the box.


# 🍩 Requirements

- VOICEVOX API in your computer or network reachable machine (Text-to-Speech)
- API key for Speech Services of Google or Azure (Speech-to-Text)
- API key for OpenAI API (ChatGPT)
- Python 3.10 (Runtime)

# 🚀 Quick start

Install AIAvatarKit.

```bash
$ pip install aiavatar
```

Make the script as  `run.py`.

```python
import logging
from aiavatar import AIAvatar, WakewordListener

GOOGLE_API_KEY = "YOUR_API_KEY"
OPENAI_API_KEY = "YOUR_API_KEY"
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
    google_api_key=GOOGLE_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    voicevox_url=VV_URL,
    voicevox_speaker_id=VV_SPEAKER,
    # volume_threshold=2000,    # <- Set to adjust microphone sensitivity
    model="gpt-3.5-turbo",
    system_message_content=system_message_content,
)

# Create WakewordListener
wakewords = ["こんにちは"]

async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    await app.start_chat()

wakeword_listener = WakewordListener(
    api_key=GOOGLE_API_KEY,
    volume_threshold=app.volume_threshold,
    wakewords=wakewords,
    on_wakeword=on_wakeword,
    device_index=app.input_device
)

# Start listening
ww_thread = wakeword_listener.start()
ww_thread.join()

# Tips: To terminate with Ctrl+C on Windows, use `while` below instead of `ww_thread.join()`
# while True:
#     time.sleep(1)
```

Start AIAvatar.

```bash
$ python run.py
```

When you say the wake word "こんにちは" the AIAvatar will respond with "どうしたの？".
Feel free to enjoy the conversation afterwards!

If you want to set face expression and animation, configure as follows:

```python
# Add face expresions
app.avatar_controller.face_controller.faces["on_wake"] = 10
app.avatar_controller.face_controller.faces["on_listening"] = 11
app.avatar_controller.face_controller.faces["on_thinking"] = 12

# Set face when the character is listening the users voice
async def set_listening_face():
    await app.avatar_controller.face_controller.set_face("on_listening", 3.0)
app.request_listener.on_start_listening = set_listening_face

# Set face when the character is processing the request
async def set_thinking_face():
    await app.avatar_controller.face_controller.set_face("on_thinking", 3.0)
app.chat_processor.on_start_processing = set_thinking_face

# Add animations (also add "walk" to the prompt)
app.avatar_controller.animation_controller.animations["walk"] = 9

async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    # Set face when wakeword detected
    await app.avatar_controller.face_controller.set_face("on_wake", 2.0)
    await app.start_chat(request_on_start=text, skip_start_voice=True)
```


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
    model="gpt-3.5-turbo",
    system_message_content=system_message_content,
    input_device=6      # Listen sound from VRChat
    output_device=13,   # Speak to VRChat microphone
)
```

You can also set the name of audio devices instead of index (partial match, ignore case).

```python
    input_device="CABLE-B Out"      # Listen sound from VRChat
    output_device="cable-a input",   # Speak to VRChat microphone
```


Run it.

```bash
$ run.py
```

Launch VRChat as desktop mode on the machine that runs `run.py` and log in with the account for AIAvatar. Then set `VB-Cable-A` to microphone in VRChat setting window.

That's all! Let's chat with the AIAvatar. Log in to VRChat on another machine (or Quest) and go to the world the AIAvatar is in.

# 🟦 Use Azure Listeners

We strongly recommend using AzureWakewordListener and AzureRequestListner that are more stable than the default listners. Check [examples/run_azure.py](https://github.com/uezo/aiavatarkit/blob/main/examples/run_azure.py) that works out-of-the-box.

Install Azure SpeechSDK.

```sh
$ pip install azure-cognitiveservices-speech
```

Change script to use AzureRequestListener and AzureWakewordListener.

```python
YOUR_SUBSCRIPTION_KEY = "YOUR_SUBSCRIPTION_KEY"
YOUR_REGION_NAME = "japanwest"

# Create AzureRequestListener
from aiavatar.listeners.azurevoicerequest import AzureVoiceRequestListener
request_listener = AzureVoiceRequestListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
)

# Create AIAVater with AzureRequestListener
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    system_message_content=system_message_content,
    request_listener=request_listener,
    voicevox_url=VV_URL,
    voicevox_speaker_id=VV_SPEAKER,
)

# Create AzureWakewordListner
async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    await app.start_chat()

from aiavatar.listeners.azurewakeword import AzureWakewordListener
wakeword_listener = AzureWakewordListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
    on_wakeword=on_wakeword,
    wakewords=["こんにちは"]
)
```

To specify the microphone device by setting `device_name` argument.
See Microsoft Learn to know how to check the device UID on each platform.
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-select-audio-input-devices

We provide [a script for MacOS](https://github.com/uezo/aiavatarkit/blob/main/examples/audio_device_checker/main.m). Just run it on Xcode.

```
Device UID: BuiltInMicrophoneDevice, Name: MacBook Proのマイク
Device UID: com.vbaudio.vbcableA:XXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX, Name: VB-Cable A
Device UID: com.vbaudio.vbcableB:XXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX, Name: VB-Cable B
```

For example, the UID for the built-in microphone on MacOS is `BuiltInMicrophoneDevice`.

Then, set it as the value of `device_name`.

```python
request_listener = AzureVoiceRequestListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
    device_name="BuiltInMicrophoneDevice"
)

wakeword_listener = AzureWakewordListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
    on_wakeword=on_wakeword,
    wakewords=["こんにちは"],
    device_name="BuiltInMicrophoneDevice"
)
```

# 🗣️ Use alternative Text-to-Speech services

Set speech controller after instantiate AIAvatar.

## Azure

```python
from aiavatar.speech.azurespeech import AzureSpeechController

app.avatar_controller.speech_controller = AzureSpeechController(
    AZURE_SUBSCRIPTION_KEY, AZURE_REGION,
    device_index=app.output_device,
    # # Set params if you want to customize
    # speaker_name="en-US-AvaNeural",
    # speaker_gender="Female",
    # lang="en-US"
)
```

The default speaker is `en-US-JennyMultilingualNeural` that support multi languages.

https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/language-support?tabs=tts


## VOICEVOX (subprocess version for noise reduction)

```python
from aiavatar.speech.voicevox import VoicevoxSpeechControllerSubProcess

app.avatar_controller.speech_controller = VoicevoxSpeechControllerSubProcess(
    base_url="http://127.0.0.1:50021",
    speaker_id=46,
    device_index=app.output_device
)
```


# ⚡️ Function Calling

Use `chat_processor.add_function` to use ChatGPT function calling. In this example, `get_weather` will be called autonomously.

```python
# Add function
async def get_weather(location: str):
    await asyncio.sleep(1.0)
    return {"weather": "sunny partly cloudy", "temperature": 23.4}

app.chat_processor.add_function(
    name="get_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string"
            }
        }
    },
    func=get_weather
)
```

And, after `get_weather` called, message to get voice response will be sent to ChatGPT internally.

```json
{
    "role": "function",
    "content": "{\"weather\": \"sunny partly cloudy\", \"temperature\": 23.4}",
    "name": "get_weather"
}
```


# 🎤 Testing audio I/O

Using the script below to test the audio I/O before configuring AIAvatar.

- Step-by-Step audio device configuration.
- Speak immediately after start if the output device is correctly configured.
- All recognized text will be shown in console if the input device is correctly configured.
- Just echo on wakeword recognized.

```python
import asyncio
import logging
from aiavatar import (
    AudioDevice,
    VoicevoxSpeechController,
    WakewordListener
)

GOOGLE_API_KEY = "YOUR_API_KEY"
VV_URL = "http://127.0.0.1:50021"
VV_SPEAKER = 46
VOLUME_THRESHOLD = 3000
INPUT_DEVICE = -1
OUTPUT_DEVICE = -1

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)

# Select input device
if INPUT_DEVICE < 0:
    input_device_info = AudioDevice.get_input_device_with_prompt()
else:
    input_device_info = AudioDevice.get_device_info(INPUT_DEVICE)
input_device = input_device_info["index"]

# Select output device
if OUTPUT_DEVICE < 0:
    output_device_info = AudioDevice.get_output_device_with_prompt()
else:
    output_device_info = AudioDevice.get_device_info(OUTPUT_DEVICE)
output_device = output_device_info["index"]

logger.info(f"Input device: [{input_device}] {input_device_info['name']}")
logger.info(f"Output device: [{output_device}] {output_device_info['name']}")

# Create speaker
speaker = VoicevoxSpeechController(
    VV_URL,
    VV_SPEAKER,
    device_index=output_device
)

asyncio.run(speaker.speak("オーディオデバイスのテスターを起動しました。私の声が聞こえていますか？"))

# Create WakewordListener
wakewords = ["こんにちは"]
async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    await speaker.speak(f"{text}")

wakeword_listener = WakewordListener(
    api_key=GOOGLE_API_KEY,
    volume_threshold=VOLUME_THRESHOLD,
    wakewords=["こんにちは"],
    on_wakeword=on_wakeword,
    verbose=True,
    device_index=input_device
)

# Start listening
ww_thread = wakeword_listener.start()
ww_thread.join()
```


# ⚡️ Use custom listener

It's very easy to add your original listeners. Just make it run on other thread and invoke `app.start_chat()` when the listener handles the event.

Here the example of `FileSystemListener` that invokes chat when `test.txt` is found on the file system.

```python
import asyncio
import os
from threading import Thread
from time import sleep

class FileSystemListener:
    def __init__(self, on_file_found):
        self.on_file_found = on_file_found

    def start_listening(self):
        while True:
            # Check file every 3 seconds
            if os.path.isfile("test.txt"):
                asyncio.run(self.on_file_found())
            sleep(3)

    def start(self):
        th = Thread(target=self.start_listening, daemon=True)
        th.start()
        return th
```

Use this listener in `run.py` like below.

```python
# Event handler
def on_file_found():
    asyncio.run(app.chat())

# Instantiate
fs_listener = FileSystemListener(on_file_found)
fs_thread = fs_listener.start()
    :
# Wait for finish
fs_thread.join()
```
