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

Make the script as `run.py`.

```python
from aiavatar import AIAvatar

app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    google_api_key="YOUR_GOOGLE_API_KEY"
)
app.start_listening_wakeword()

# # Tips: To terminate with Ctrl+C on Windows, use `while` to wait instead of `app.start_listening_wakeword()`
# app.start_listening_wakeword(False)
# while True:
#     time.sleep(1)
```

Start AIAvatar. Also, don't forget to launch VOICEVOX beforehand.

```bash
$ python run.py
```

Conversation will start when you say the wake word "こんにちは" (or "Hello" when language is not `ja-JP`).

Feel free to enjoy the conversation afterwards!


# 🔖 Contents

- [📕 Configuration Guide](#-configuration-guide)
  - [🎓 Generative AI](#-generative-ai)
    - [ChatGPT](#chatgpt)
    - [Claude](#claude)
    - [Gemini](#gemini)
    - [Other LLMs](#other-llms)
  - [🗣️ Voice](#️-voice)
  - [🐓 Wakeword Listener](#-wakeword-listener)
  - [🙏 Request Listener](#-request-listener)
  - [✨ Using Azure Listeners](#-using-azure-listeners)
  - [🔈 Audio Device](#-audio-device)
  - [🥰 Face Expression](#-face-expression)
  - [💃 Animation](#-animation)
  - [🎭 Custom Behavior](#-custom-behavior)
- [🌎 Platform Guide](#-platform-guide)
  - [🐈 VRChat](#-vrchat)
  - [🍓 Raspberry Pi](#-raspberry-pi)
- [🧩 RESTful APIs](#-restful-apis)
- [🤿 Deep Dive](#-deep-dive)
  - [⚡️ Function Calling](#️-function-calling)
  - [👀 Vision](#️-vision)
- [🔍 Other Tips](#-other-tips)
  - [🎤 Testing Audio I/O](#-testing-audio-io)
  - [🎚️ Noise Filter](#-noise-filter)
  - [🧪 LM Studio API](#-lm-studio-api)
  - [⚡️ Use Custom Listener](#️-use-custom-listener)


# 📕 Configuration Guide

Here are the configuration for each component.


## 🎓 Generative AI

You can set model and system message content when instantiate `AIAvatar`.

```python
app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    google_api_key="YOUR_GOOGLE_API_KEY",
    model="gpt-4-turbo",
    system_message_content="You are my cat."
)
```

### ChatGPT

If you want to configure in detail, create instance of `ChatGPTProcessor` with custom parameters and set it to `AIAvatar`.

```python
from aiavatar.processors.chatgpt import ChatGPTProcessor

chat_processor = ChatGPTProcessor(
    api_key=OPENAI_API_KEY,
    model="gpt-4-turbo",
    temperature=0.0,
    max_tokens=200,
    system_message_content="You are my cat.",
    history_count=20,       # Count of messages included in request to ChatGPT as context
    history_timeout=120.0   # Duration in seconds to expire histories
)

app.chat_processor = chat_processor
```

### Claude

Create instance of `ClaudeProcessor` with custom parameters and set it to `AIAvatar`. The default model is `claude-3-sonnet-20240229`.

```python
from aiavatar.processors.claude import ClaudeProcessor

claude_processor = ClaudeProcessor(
    api_key="ANTHROPIC_API_KEY"
)

app = AIAvatar(
    google_api_key=GOOGLE_API_KEY,
    chat_processor=claude_processor
)
```

NOTE: We support Claude 3 on Anthropic API, not Amazon Bedrock for now.


### Gemini

Create instance of `GeminiProcessor` with custom parameters and set it to `AIAvatar`. The default model is `gemini-pro`.

```python
from aiavatar.processors.gemini import GeminiProcessor

gemini_processor = GeminiProcessor(
    api_key="YOUR_GOOGLE_API_KEY"
)

app = AIAvatar(
    google_api_key=GOOGLE_API_KEY,
    chat_processor=gemini_processor
)
```

NOTE: We support Gemini on Google AI Studio, not Vertex AI for now.


### Other LLMs

You can make your custom processor that uses other generative AIs such as Llama3 by implementing `ChatProcessor` interface. We provide the example later.🙏


## 🗣️　Voice

You can set speaker id and the base url for VOICEVOX server when instantiate `AIAvatar`.

```python
app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    google_api_key="YOUR_GOOGLE_API_KEY",
    # 46 is Sayo. See http://127.0.0.1:50021/speakers to get all ids for characters
    voicevox_speaker_id=46
)
```

If you want to configure in detail, create instance of `VoicevoxSpeechController` with custom parameters and set it to `AIAvatar`.

```python
from aiavatar.speech.voicevox import VoicevoxSpeechController

speech_controller = VoicevoxSpeechController(
    base_url="https",
    speaker_id=46,
    device_index=app.audio_devices.output_device
)

app.avatar_controller.speech_controller = speech_controller
```


Speech is handled in a separate subprocess to improve audio quality and reduce noises such as popping, caused by thread blocking during parallel processing of AI responses and speech output. For systems with limited resources, setting `use_subprocess=False` allows speech processing within the main process, potentially reintroducing some noise.

```python
app.avatar_controller.speech_controller = VoicevoxSpeechController(
    base_url="http://127.0.0.1:50021",
    speaker_id=46,
    device_index=app.audio_devices.output_device,
    use_subprocess=False  # Set to False to handle speech in the main process
)
```


You can also set speech controller that uses alternative Text-to-Speech services. We provide `AzureSpeechController` for now.

```python
from aiavatar.speech.azurespeech import AzureSpeechController

AzureSpeechController(
    AZURE_SUBSCRIPTION_KEY, AZURE_REGION,
    device_index=app.audio_devices.output_device,
    # # Set params if you want to customize
    # speaker_name="en-US-AvaNeural",
    # speaker_gender="Female",
    # lang="en-US"
)
```

The default speaker is `en-US-JennyMultilingualNeural` that support multi languages.

https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/language-support?tabs=tts


You can make custom speech controller by impelemting `SpeechController` interface or extending `SpeechControllerBase`.


## 🐓 Wakeword listener

Set wakewords when instantiate `AIAvatar`. Conversation will start when AIAvatar recognizes the one of the words in this list.

```python
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    google_api_key=GOOGLE_API_KEY,
    wakewords=["Hello", "こんにちは"],
)
```

If you want to configure in detail, create instance of `WakewordListener` with custom parameters and set it to `AIAvatar`.

```python
from aiavatar.listeners.wakeword import WakewordListener

wakeword_listener = WakewordListener(
    api_key=GOOGLE_API_KEY,
    wakewords=["Hello", "こんにちは"],
    device_index=app.audio_devices.input_device,
    timeout=0.2,        # Duration in seconds to wait for silence before ending speech recognition
    max_duration=1.5    # Maximum duration in seconds to recognize speech before stopping
)

app.wakeword_listener = wakeword_listener
```


## 🙏 Request listener

If you want to configure in detail, create instance of `VoiceRequestListener` with custom parameters and set it to `AIAvatar`.

```python
from aiavatar.listeners.voicerequest import VoiceRequestListener

request_listener = VoiceRequestListener(
    api_key=GOOGLE_API_KEY,
    device_index=app.audio_devices.input_device,,
    detection_timeout=15.0, # Timeout in seconds to end the process if speech does not start within this duration
    timeout=0.5,            # Duration in seconds to wait for silence before ending speech recognition
    max_duration=20.0,      # Maximum duration in seconds to recognize speech before stopping
    min_duration=0.2,       # Minimum duration in seconds for speech to be recognized; shorter sounds are ignored
)

app.request_listener = request_listener
```


## ✨ Using Azure Listeners

We **strongly recommend using AzureWakewordListener and AzureRequestListner** that are more stable than the default listners. Check [examples/run_azure.py](https://github.com/uezo/aiavatarkit/blob/main/examples/run_azure.py) that works out-of-the-box.

Install Azure SpeechSDK.

```sh
$ pip install azure-cognitiveservices-speech
```

Change script to use AzureRequestListener and AzureWakewordListener.

```python
from aiavatar.listeners.azurevoicerequest import AzureVoiceRequestListener
from aiavatar.listeners.azurewakeword import AzureWakewordListener

YOUR_SUBSCRIPTION_KEY = "YOUR_SUBSCRIPTION_KEY"
YOUR_REGION_NAME = "YOUR_REGION_NAME"

# Create AzureRequestListener
azure_request_listener = AzureVoiceRequestListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME
)

# Create AzureWakewordListner
async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    await app.start_chat()

azrue_wakeword_listener = AzureWakewordListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
    on_wakeword=on_wakeword,
    wakewords=["こんにちは"]
)

# Create AIAVater with AzureRequestListener and Azure WakewordListener
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    request_listener=azure_request_listener,
    wakeword_listener=azrue_wakeword_listener
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
azure_request_listener = AzureVoiceRequestListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
    device_name="BuiltInMicrophoneDevice"
)

azure_wakeword_listener = AzureWakewordListener(
    YOUR_SUBSCRIPTION_KEY,
    YOUR_REGION_NAME,
    on_wakeword=on_wakeword,
    wakewords=["Hello", "こんにちは"],
    device_name="BuiltInMicrophoneDevice"
)
```


## 🍥 Using OpenAI's audio APIs

OpenAI's Speech-to-Text and Text-to-Speech capabilities provide dynamic speech recognition and voice output across multiple languages, without the need for fixed language settings.

```python
from aiavatar import AIAvatar
from aiavatar.device import AudioDevice
from aiavatar.listeners.openailisteners import (
    OpenAIWakewordListener,
    OpenAIVoiceRequestListener
)
from aiavatar.speech.openaispeech import OpenAISpeechController

# Get default audio devices
devices = AudioDevice()

# Speech
speech_controller = OpenAISpeechController(
    api_key=OPENAI_API_KEY,
    device_index=devices.output_device
)

# Wakeword
async def on_wakeword(text):
    await app.start_chat(request_on_start=text, skip_start_voice=True)

wakeword_listener = OpenAIWakewordListener(
    api_key=OPENAI_API_KEY,
    device_index=devices.input_device,
    wakewords=["こんにちは"],
    on_wakeword=on_wakeword
)

# Request
request_listener = OpenAIVoiceRequestListener(
    api_key=OPENAI_API_KEY,
    device_index=devices.input_device
)

# Create AIAvatar with OpenAI Components
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    wakeword_listener=wakeword_listener,
    request_listener=request_listener,
    speech_controller=speech_controller,
    noise_margin=10.0,
    verbose=True
)
app.start_listening_wakeword()
```


## 🔈 Audio device

You can specify the audio devices to be used in components by name or index.

```python
from aiavatar.device import AudioDevice

# Get devices by name or index
audio_device = AudioDevice(
    input_device="マイク",
    output_device="スピーカー"
)
```

Set device to components.

```python
# Set output device to SpeechController
speech_controller = VoicevoxSpeechControllerSubProcess(
    device_index=audio_device.output_device,
    base_url="http://127.0.0.1:50021",
    speaker_id=46,
)

# Set input device to Listeners
request_listener = VoiceRequestListener(
    device_index=audio_device.input_device
)

wakeword_listener = WakewordListener(
    device_index=audio_device.input_device,
    wakewords=["Hello", "こんにちは"]
)

# Set components to AIAvatar
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    speech_controller=speech_controller,
    request_listener=request_listener,
    wakeword_listener=wakeword_listener
)
```


## 🥰 Face expression

To control facial expressions within conversations, set the facial expression names and values in `FaceController.faces` as shown below, and then include these expression keys in the response message by adding instructions to the prompt.

```python
app.avatar_controller.face_controller.faces = {
    "neutral": "🙂",
    "joy": "😀",
    "angry": "😠",
    "sorrow": "😞",
    "fun": "🥳"
}

app.chat_processor.system_message_content = """# Face Expression

* You have the following expressions:

- joy
- angry
- sorrow
- fun

* If you want to express a particular emotion, please insert it at the beginning of the sentence like [face:joy].

Example
[face:joy]Hey, you can see the ocean! [face:fun]Let's go swimming.
"""
```

This allows emojis like 🥳 to be autonomously displayed in the terminal during conversations. To actually control the avatar's facial expressions in a metaverse platform, instead of displaying emojis like 🥳, you will need to use custom implementations tailored to the integration mechanisms of each platform. Please refer to our `VRChatFaceController` as an example.


## 💃 Animation

Now writing... ✍️


##　🎭 Custom Behavior

You can invoke custom implementations when listening to requests from user, processing those requests, or when recognized a wake word to start conversation.

In the following example, changing face expressions at each timing aims to enhance the interaction experience with the AI avatar.

```python
# Set face when the character is listening the users voice
async def set_listening_face():
    await app.avatar_controller.face_controller.set_face("listening", 3.0)
app.request_listener.on_start_listening = set_listening_face

# Set face when the character is processing the request
async def set_thinking_face():
    await app.avatar_controller.face_controller.set_face("thinking", 3.0)
app.chat_processor.on_start_processing = set_thinking_face

async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    # Set face when wakeword detected
    await app.avatar_controller.face_controller.set_face("smile", 2.0)
    await app.start_chat(request_on_start=text, skip_start_voice=True)
```


# 🌎 Platform Guide

AIAvatarKit is capable of operating on any platform that allows applications to hook into audio input and output. The platforms that have been tested include:

- VRChat
- cluster
- Vket Cloud

In addition to running on PCs to operate AI avatars on these platforms, you can also create a communication robot by connecting speakers, a microphone, and, if possible, a display to a Raspberry Pi.

## 🐈 VRChat

* __2 Virtual audio devices (e.g. VB-CABLE) are required.__
* __Multiple VRChat accounts are required to chat with your AIAvatar.__


### Get started

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


### Face Expression

AIAvatarKit controls the face expression by [Avatar OSC](https://docs.vrchat.com/docs/osc-avatar-parameters).

LLM(ChatGPT/Claude/Gemini)  
↓ *response with face tag* `[face:joy]Hello!`  
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

Lastly, add face expression section to the system prompt.

```python
# Make system prompt
system_message_content = """
# Face Expression

* You have following expressions:

- joy
- angry
- sorrow
- fun

* If you want to express a particular emotion, please insert it at the beginning of the sentence like [face:joy].

Example
[face:joy]Hey, you can see the ocean! [face:fun]Let's go swimming.
"""

# Set them to AIAvatar
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    google_api_key=GOOGLE_API_KEY,
    face_controller=vrc_face_controller,
    system_message_content=system_message_content
)
```

You can test it not only through the voice conversation but also via the [REST API](#-restful-apis).


## 🍓 Raspberry Pi

Now writing... ✍️


# 🧩 RESTful APIs

You can control AIAvatar via RESTful APIs. The provided functions are:

- WakewordLister
    - start: Start WakewordListener
    - stop: Stop WakewordListener
    - status: Show status of WakewordListener

- Avatar
    - speech: Speak text with face expression and animation
    - face: Set face expression
    - animation: Set animation

- System
    - log: Show recent logs

To use REST APIs, create API app and set router instead of calling `app.start_listening_wakeword()`.

```python
from fastapi import FastAPI
from aiavatar import AIAvatar
from aiavatar.api.router import get_router

app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    google_api_key=GOOGLE_API_KEY
)

# app.start_listening_wakeword()

# Create API app and set router
api = FastAPI()
api_router = get_router(app, "aiavatar.log")
api.include_router(api_router)
```

Start API with uvicorn.

```bash
$ uvicorn run:api
```

Call `/wakeword/start` to start wakeword listener.

```bash
$ curl -X 'POST' \
  'http://127.0.0.1:8000/wakeword/start' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "wakewords": []
}'
```

See API spec and try it on http://127.0.0.1:8000/docs .

**NOTE**: AzureWakewordListeners stops immediately but the default WakewordListener stops after it recognizes wakeword.


# 🤿 Deep dive

Advanced usases.

## ⚡️ Function Calling

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


## 👀 Vision

We provide the experimental support for vision input to ChatGPT. A new class, ChatGPTProcessorWithVisionBase, has been added to handle image inputs, inheriting from ChatGPTProcessor.

An example implementation, ChatGPTProcessorWithVisionScreenShot, demonstrates how to capture screenshots using pyautogui. This gives "eyes" to your AIAvatar in metaverse platforms like VRChat.

```python
import io
import pyautogui

class ChatGPTProcessorWithVisionScreenShot(ChatGPTProcessorWithVisionBase):
    async def get_image(self) -> bytes:
        buffered = io.BytesIO()
        image = pyautogui.screenshot(region=(0, 0, 1280, 720))
        image.save(buffered, format="PNG")
        image.save("image_to_chatgpt.png")
        return buffered.getvalue()
```

To use this new feature, you can instantiate ChatGPTProcessorWithVisionScreenShot instead of ChatGPTProcessor and set it in the AIAvatar.

```python
chat_processor = ChatGPTProcessorWithVisionScreenShot(
    api_key=OPENAI_API_KEY,
    system_message_content=PROMPT
)

app = AIAvatar(
    google_api_key=GOOGLE_API_KEY,
    chat_processor=chat_processor
)
```

**NOTE**

* Only the latest image will be sent to ChatGPT to avoid performance issues.
* The system uses function calling to determine if image retrieval is necessary, which adds approximately 500 milliseconds to 1 second to the processing time.


# 🔍 Other Tips

Useful information for developping and debugging.

## 🎤 Testing audio I/O

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
    wakewords=["こんにちは"],
    on_wakeword=on_wakeword,
    verbose=True,
    device_index=input_device
)

# Start listening
ww_thread = wakeword_listener.start()
ww_thread.join()
```

## 🎚️ Noise Filter

AIAvatarKit automatically adjusts the noise filter for listeners when you instantiate an AIAvatar object. To manually set the noise filter level for voice detection, set `auto_noise_filter_threshold` to `False` and specify the `volume_threshold_db` in decibels (dB).

```python
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    google_api_key=GOOGLE_API_KEY,
    auto_noise_filter_threshold=False,
    volume_threshold_db=-40   # Set the voice detection threshold to -40 dB
)
```


## 🧪 LM Studio API

Use ChatGPTProcessor with some arguments.

- base_url: URL for LM Studio local server
- model: Name of model
- parse_function_call_in_response: Always set `False`

```python
from aiavatar import AIAvatar
from aiavatar.processors.chatgpt import ChatGPTProcessor

chat_processor = ChatGPTProcessor(
    api_key=OPENAI_API_KEY,
    base_url="http://127.0.0.1:1234/v1",
    model="mmnga/DataPilot-ArrowPro-7B-KUJIRA-gguf",
    parse_function_call_in_response=False
)

app = AIAvatar(
    google_api_key=GOOGLE_API_KEY,
    chat_processor=chat_processor
)
app.start_listening_wakeword()
```


## ⚡️ Use custom listener

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
