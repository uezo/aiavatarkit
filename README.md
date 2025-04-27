# AIAvatarKit

🥰 Building AI-based conversational avatars lightning fast ⚡️💬

![AIAvatarKit Architecture Overview](documents/images/aiavatarkit_overview.png) 

## ✨ Features

- Live anywhere: VRChat, cluster and any other metaverse platforms, and even devices in the real world.
- Extensible: Unlimited capabilities that depends on you.
- Easy to start: Ready to start conversation right out of the box.


## 🍩 Requirements

- VOICEVOX API in your computer or network reachable machine (Text-to-Speech)
- API key for OpenAI API (ChatGPT and Speech-to-Text)
- Python 3.10 (Runtime)


## 🚀 Quick start

Install AIAvatarKit.

```sh
pip install git+https://github.com/uezo/aiavatarkit.git@v0.6.4
```

**NOTE:** Since technical blogs assume [v0.5.8](https://github.com/uezo/aiavatarkit/tree/v0.5.8), the PyPI version will remain based on v0.5.8 during the transition period. We plan to update to the v0.6 series around May 2025.

Make the script as `run.py`.

```python
import asyncio
from aiavatar import AIAvatar

aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    debug=True
)
asyncio.run(aiavatar_app.start_listening())
```

Start AIAvatar. Also, don't forget to launch VOICEVOX beforehand.

```bash
$ python run.py
```

Conversation will start when you say the wake word "こんにちは" (or "Hello" when language is not `ja-JP`).

Feel free to enjoy the conversation afterwards!


## 🔖 Contents

- [🎓 Generative AI](#-generative-ai)
    - [ChatGPT](#chatgpt)
    - [Claude](#claude)
    - [Gemini](#gemini)
    - [Dify](#dify)
    - [Other LLMs](#other-llms)
- [🗣️ Voice](#️voice)
- [👂 Speech Listener](#-speech-listener)
- [🥰 Face Expression](#-face-expression)
- [💃 Animation](#-animation)

- [🌎 Platform Guide](#-platform-guide)
    - [🐈 VRChat](#-vrchat)
    - [🍓 Raspberry Pi](#-raspberry-pi)

- [🤿 Deep Dive](#-deep-dive)
    - [⚡️ Function Calling](#️-function-calling)
    - [👀 Vision](#-vision)
    - [💾 Long-term Memory](#-long-term-memory)
    - [🐓 Wakeword](#-wakeword-listener)
    - [🔈 Audio Device](#-audio-device)
    - [💫 Streaming API](#-streaming-api)
    - [🔌 WebSocket](#-websocket)
    - [🎭 Custom Behavior](#-custom-behavior)
    - [🧩 RESTful APIs](#-restful-apis)
    - [🎚️ Noise Filter](#-noise-filter)


## 🎓 Generative AI

You can set model and system prompt when instantiate `AIAvatar`.

```python
aiavatar_app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    model="gpt-4o",
    system_prompt="You are my cat."
)
```

### ChatGPT

If you want to configure in detail, create instance of `ChatGPTService` with custom parameters and set it to `AIAvatar`.

```python
# Create ChatGPTService
from litests.llm.chatgpt import ChatGPTService
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create AIAvatar with ChatGPTService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

### Claude

Create instance of `ClaudeService` with custom parameters and set it to `AIAvatar`. The default model is `claude-3-5-sonnet-latest`.

```python
# Create ClaudeService
from litests.llm.claude import ClaudeService
llm = ClaudeService(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-3-7-sonnet-20250219",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create AIAvatar with ClaudeService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

NOTE: We support Claude on Anthropic API, not Amazon Bedrock for now. Use LiteLLM or other API Proxies.


### Gemini

Create instance of `GeminiService` with custom parameters and set it to `AIAvatar`. The default model is `gemini-2.0-flash-exp`.

```python
# Create GeminiService
# pip install google-generativeai
from litests.llm.gemini import GeminiService
llm = GeminiService(
    gemini_api_key=GEMINI_API_KEY,
    model="gemini-2.0-pro-latest",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create AIAvatar with GeminiService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

NOTE: We support Gemini on Google AI Studio, not Vertex AI for now. Use LiteLLM or other API Proxies.


### Dify

You can use the Dify API instead of a specific LLM's API. This eliminates the need to manage code for tools or RAG locally.

```python
# Create DifyService
from litests.llm.dify import DifyService
llm = DifyService(
    api_key=DIFY_API_KEY,
    base_url=DIFY_URL,
    user="aiavatarkit_user",
    is_agent_mode=True
)

# Create AIAvatar with DifyService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```


### Other LLMs

You can use other LLMs by using `LiteLLMService` or implementing `LLMService` interface.

See the details of LiteLLM here: https://github.com/BerriAI/litellm


## 🗣️　Voice

You can set speaker id and the base url for VOICEVOX server when instantiate `AIAvatar`.

```python
aiavatar_app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    # 46 is Sayo. See http://127.0.0.1:50021/speakers to get all ids for characters
    voicevox_speaker=46
)
```

If you want to configure in detail, create instance of `VoicevoxSpeechSynthesizer` with custom parameters and set it to `AIAvatar`.
Here is the example for [AivisSpeech](https://aivis-project.com).

```python
# Create VoicevoxSpeechSynthesizer with AivisSpeech configurations
from litests.tts.voicevox import VoicevoxSpeechSynthesizer
tts = VoicevoxSpeechSynthesizer(
    base_url="http://127.0.0.1:10101",  # Your AivisSpeech API server
    speaker="888753761"   # Anneli
)

# Create AIAvatar with VoicevoxSpeechSynthesizer
aiavatar_app = AIAvatar(
    tts=tts,
    openai_api_key=OPENAI_API_KEY   # API Key for LLM and STT
)
```

You can also set speech controller that uses alternative Text-to-Speech services. We support Azure, Google, OpenAI and any other TTS services supported by [SpeechGateway](https://github.com/uezo/speech-gateway) such as Style-Bert-VITS2 and NijiVoice.

```python
from litests.tts.azure import AzureSpeechSynthesizer
from litests.tts.google import GoogleSpeechSynthesizer
from litests.tts.openai import OpenAISpeechSynthesizer
from litests.tts.speech_gateway import SpeechGatewaySpeechSynthesizer
```

You can also make custom tts components by impelemting `SpeechSynthesizer` interface.


## 👂 Speech listener

If you want to configure in detail, create instance of `SpeechRecognizer` with custom parameters and set it to `AIAvatar`. We support Azure, Google and OpenAI Speech-to-Text services.

NOTE: **`AzureSpeechRecognizer` is much faster** than Google and OpenAI(default).

```python
# Create AzureSpeechRecognizer
from litests.stt.azure import AzureSpeechRecognizer
stt = AzureSpeechRecognizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION
)

# Create AIAvatar with AzureSpeechRecognizer
aiavatar_app = AIAvatar(
    stt=stt,
    openai_api_key=OPENAI_API_KEY   # API Key for LLM
)
```

## 🥰 Face expression

To control facial expressions within conversations, set the facial expression names and values in `FaceController.faces` as shown below, and then include these expression keys in the response message by adding instructions to the prompt.

```python
aiavatar_app.adapter.face_controller.faces = {
    "neutral": "🙂",
    "joy": "😀",
    "angry": "😠",
    "sorrow": "😞",
    "fun": "🥳"
}

aiavatar_app.sts.llm.system_prompt = """# Face Expression

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


## 🌎 Platform Guide

AIAvatarKit is capable of operating on any platform that allows applications to hook into audio input and output. The platforms that have been tested include:

- VRChat
- cluster
- Vket Cloud

In addition to running on PCs to operate AI avatars on these platforms, you can also create a communication robot by connecting speakers, a microphone, and, if possible, a display to a Raspberry Pi.

### 🐈 VRChat

* __2 Virtual audio devices (e.g. VB-CABLE) are required.__
* __Multiple VRChat accounts are required to chat with your AIAvatar.__


#### Get started

First, run the commands below in python interpreter to check the audio devices.

```sh
$ python

>>> from aiavatar import AudioDevice
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

- To use `VB-Cable-A` for microphone for VRChat, index for `output_device` is `13` (CABLE-A Input).
- To use `VB-Cable-B` for speaker for VRChat, index for `input_device` is `6` (CABLE-B Output). Don't forget to set `VB-Cable-B Input` as the default output device of Windows OS.

Then edit `run.py` like below.

```python
# Create AIAvatar
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    input_device=6,     # Listen sound from VRChat
    output_device=13,   # Speak to VRChat microphone
)
```

Run it.

```bash
$ run.py
```

Launch VRChat as desktop mode on the machine that runs `run.py` and log in with the account for AIAvatar. Then set `VB-Cable-A` to microphone in VRChat setting window.

That's all! Let's chat with the AIAvatar. Log in to VRChat on another machine (or Quest) and go to the world the AIAvatar is in.


#### Face Expression

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
system_prompt = """
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
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    face_controller=vrc_face_controller,
    system_prompt=system_prompt
)
```

You can test it not only through the voice conversation but also via the [REST API](#-restful-apis).


### 🍓 Raspberry Pi

Now writing... ✍️


## 🤿 Deep dive

Advanced usases.

### ⚡️ Function Calling

Register tool with spec by `@aiavatar_app.sts.llm.tool`. The spec should be in the format for each LLM.

```python
# Spec (for ChatGPT)
weather_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
        },
    }
}

# Implement tool and register it with spec
@aiavatar_app.sts.llm.tool(weather_tool_spec)    # NOTE: Gemini doesn't take spec as argument
async def get_weather(location: str = None):
    weather = await weather_api(location=location)  # Call weather API
    return weather  # {"weather": "clear", "temperature": 23.4}
```


### 👀 Vision

AIAvatarKit captures and sends image to AI dynamically when the AI determine that vision is required to process the request. This gives "eyes" to your AIAvatar in metaverse platforms like VRChat.


```python
# Instruct vision tag in the system message
SYSTEM_PROMPR = """
## Using Vision

If you need an image to process a user's request, you can obtain it using the following methods:

- screenshot
- camera

If an image is needed to process the request, add an instruction like [vision:screenshot] to your response to request an image from the user.

By adding this instruction, the user will provide an image in their next utterance. No comments about the image itself are necessary.

Example:

user: Look! This is the sushi I had today.
assistant: [vision:screenshot] Let me take a look.
"""

# Create AIAvatar with the system prompt
aiavatar_app = AIAvatar(
    system_prompt=SYSTEM_PROMPT,
    openai_api_key=OPENAI_API_KEY
)

# Implement get_image_url
import base64
import io
import pyautogui    # pip install pyautogui
from aiavatar.device.video import VideoDevice   # pip install opencv-python
default_camera = VideoDevice(device_index=0, width=960, height=540)

@aiavatar_app.adapter.get_image_url
async def get_image_url(source: str) -> str:
    image_bytes = None

    if source == "camera":
        # Capture photo by camera
        image_bytes = await default_camera.capture_image("camera.jpg")
    elif source == "screenshot":
        # Capture screenshot
        buffered = io.BytesIO()
        image = pyautogui.screenshot(region=(0, 0, 1280, 720))
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

    if image_bytes:
        # Upload and get url, or, make base64 encoded url
        b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        b64_url = f"data:image/jpeg;base64,{b64_encoded}"
        return b64_url
```


### 💾 Long-term Memory

To recall information from past conversations across different contexts, a long-term memory service is used.

To store conversation history, define a function decorated with `@aiavatar_app.sts.on_finish`. To retrieve memories from the conversation history, call the search function of the long-term memory service as a tool.

Below is an example using [ChatMemory](https://github.com/uezo/chatmemory).

```python
# Create client for ChatMemory
from examples.misc.chatmemory import ChatMemoryClient
chat_memory_client = ChatMemoryClient(
    base_url="http://your_chatmemory_host",
    debug=True
)

# Add messages to ChatMemory service
@aiavatar_app.sts.on_finish
async def on_finish(request, response):
    try:
        await chat_memory_client.enqueue_messages(request, response)
    except Exception as ex:
        print(ex)

# Retrieve memory by calling tool
search_memory_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
        },
    }
}
@aiavatar_app.sts.llm.tool(search_memory_tool_spec)
async def search_memory(query: str, metadata: dict = None):
    """Search long-term memory

    Args:
        query: Query to search memory.
    """
    result = await chat_memory_client.search(metadata["user_id"], query)
    return result.__dict__
```


### 🐓 Wakeword

Set `wakewords` when instantiating `AIAvatar`. Conversation will start when the AIAvatar recognizes one of the words in this list. You can also set `wakeword_timeout`, after which the AIAvatar will return to listening for the wakeword again.

```python
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    wakewords=["Hello", "こんにちは"],
    wakeword_timeout=60,
)
```


### 📋 System Prompt Parameters

You can embed parameters into your system prompt dynamically.

First, define your `AIAvatar` instance with a system prompt containing placeholders:

```python
aiavatar_app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    model="gpt-4o",
    system_prompt="User's name is {name}."
)
```

When invoking, pass the parameters as a dictionary using `system_prompt_params`:

```python
aiavatar_app.sts.invoke(STSRequest(
    # (other fields omitted)
    system_prompt_params={"name": "Nekochan"}
))
```

Placeholders in the system prompt, such as `{name}`, will be replaced with the corresponding values at runtime.


### 🔈 Audio device

You can specify the audio devices to be used in components by device index.

First, check the device indexes you want to use.

```sh
$ python

>>> from aiavatar import AudioDevice
>>> AudioDevice().list_audio_devices()
{'index': 0, 'name': '外部マイク', 'max_input_channels': 1, 'max_output_channels': 0, 'default_sample_rate': 44100.0}
{'index': 1, 'name': '外部ヘッドフォン', 'max_input_channels': 0, 'max_output_channels': 2, 'default_sample_rate': 44100.0}
{'index': 2, 'name': 'MacBook Airのマイク', 'max_input_channels': 3, 'max_output_channels': 0, 'default_sample_rate': 44100.0}
{'index': 3, 'name': 'MacBook Airのスピーカー', 'max_input_channels': 0, 'max_output_channels': 2, 'default_sample_rate': 44100.0}
```

Set indexes to AIAvatar.

```python
aiavatar_app = AIAvatar(
    input_device=2,     # MacBook Airのマイク
    output_device=3,    # MacBook Airのスピーカー
    openai_api_key=OPENAI_API_KEY
)
```


### 💫 Streaming API

You can host AIAvatarKit on a server to enable multiple clients to have independent context-aware conversations via RESTful API with streaming responses (Server-Sent Events).

Below is the simplest example of a server program:

```python
from fastapi import FastAPI
from aiavatar.adapter.http.server import AIAvatarHttpServer

# AIAvatar
aiavatar_app = AIAvatarHttpServer(
    openai_api_key=OPENAI_API_KEY,
    debug=True
)

# Setup FastAPI app with AIAvatar components 
app = FastAPI()
router = aiavatar_app.get_api_router()
app.include_router(router)
```

Save the above code as `server.py` and run it using:

```sh
uvicorn server:app
```


Next is the simplest example of a client program:

```python
import asyncio
from aiavatar.adapter.http.client import AIAvatarHttpClient

aiavatar_app = AIAvatarHttpClient(
    debug=True
)
asyncio.run(aiavatar_app.start_listening(session_id="http_session", user_id="http_user"))
```

Save the above code as `client.py` and run it using:

```sh
python client.py
```

You can now perform voice interactions just like when running locally.


When using the streaming API via HTTP, clients communicate with the server using JSON-formatted requests.

Below is the format for initiating a session:

```json
{
    "type": "start",          // Always `start`
    "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d",
    "user_id": "user_id",
    "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7",   // Set null or provided id in `start` response
    "text": "こんにちは",       // If set, audio_data will be ignored         
    "audio_data": "XXXX",     // Base64 encoded audio data
    "files": [
        {
            "type": "image",        // Only `image` is supported for now
            "url": "https://xxx",
        }
    ],
    "metadata": {}
}
```

The server returns responses as a stream of JSON objects in the following structure.

The communication flow typically consists of:

```json
{
    "type": "chunk",    // start -> chunk -> final
    "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d",
    "user_id": "user01",
    "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7",
    "text": "[face:joy]こんにちは！",   // Response text with info
    "voice_text": "こんにちは！",       // Response text for voice synthesis
    "avatar_control_request": {
        "animation_name": null,       // Parsed animation name
        "animation_duration": null,   // Parsed duration for animation
        "face_name": "joy",           // Parsed facial expression name
        "face_duration": 4.0          // Parsed duration for the facial expression
    },
    "audio_data": "XXXX",   // Base64 encoded. Playback this as the character's voice.
    "metadata": {
        "is_first_chunk": true
    }
}
```


You can test the streaming API using a simple curl command:

```sh
curl -N -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
        "type": "start",
        "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d",
        "user_id": "user01",
        "text": "こんにちは"
    }'

```

Sample response (streamed from the server):

```sh
data: {"type": "start", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": null, "voice_text": null, "avatar_control_request": null, "audio_data": "XXXX", "metadata": {"request_text": "こんにちは"}}

data: {"type": "chunk", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "[face:joy]こんにちは！", "voice_text": "こんにちは！", "avatar_control_request": {"animation_name": null, "animation_duration": null, "face_name": "joy", "face_duration": 4.0}, "audio_data": "XXXX", "metadata": {"is_first_chunk": true}}

data: {"type": "chunk", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "今日はどんなことをお手伝いしましょうか？", "voice_text": "今日はどんなことをお手伝いしましょうか？", "avatar_control_request": {"animation_name": null, "animation_duration": null, "face_name": null, "face_duration": null}, "audio_data": "XXXX", "metadata": {"is_first_chunk": false}}

data: {"type": "final", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "[face:joy]こんにちは！今日はどんなことをお手伝いしましょうか？", "voice_text": "こんにちは！今日はどんなことをお手伝いしましょうか？", "avatar_control_request": null, "audio_data": "XXXX", "metadata": {}}
```

To continue the conversation, include the `context_id` provided in the `start` response in your next request.

**NOTE:** When using the RESTful API, voice activity detection (VAD) must be performed client-side.

**NOTE:** To protect API with API Key, set `api_key=API_KEY_YOU_MAKE` to AIAvatarHttpServer and send `Authorization: Bearer {API_KEY_YOU_MAKE}` as HTTP header from client.


### 🔌 WebSocket

You can host AIAvatarKit on a server to enable multiple clients to have independent context-based conversations via WebSocket.

Below is the simplest example of a server program:

```python
from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

# Create AIAvatar
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=OPENAI_API_KEY,
    volume_db_threshold=-30,  # <- Adjust for your audio env
    debug=True
)

# Set router to FastAPI app
app = FastAPI()
router = aiavatar_app.get_websocket_router()
app.include_router(router)
```

Save the above code as `server.py` and run it using:

```sh
uvicorn server:app
```


Next is the simplest example of a client program:

```python
import asyncio
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient()
asyncio.run(client.start_listening(session_id="ws_session", user_id="ws_user"))
```

Save the above code as `client.py` and run it using:

```sh
python client.py
```

You can now perform voice interactions just like when running locally.

**NOTE:** When using the WebSocket API, voice activity detection (VAD) is performed on the server side, so clients can simply stream microphone input directly to the server.


### 🎭 Custom Behavior

You can invoke custom implementations `on_response(response_type)`. In the following example, show "thinking" face expression while processing request to enhance the interaction experience with the AI avatar.

```python
# Set face when the character is thinking the answer
@aiavatar_app.on_response("start")
async def on_start_response(response):
    await aiavatar_app.adapter.face_controller.set_face("thinking", 3.0)

# Reset face before answering
@aiavatar_app.on_response("chunk")
async def on_chunk_response(response):
    if response.metadata.get("is_first_chunk"):
        aiavatar_app.adapter.face_controller.reset()
```


### 🧩 RESTful APIs

**NOTE:** Not ready for v0.6.x

You can control AIAvatar via RESTful APIs. The provided functions are:

- Lister
    - start: Start Listener
    - stop: Stop Listener
    - status: Show status of Listener

- Avatar
    - face: Set face expression
    - animation: Set animation

- System
    - log: Show recent logs

To use REST APIs, create API app and set router instead of calling `aiavatar_app.start_listening()`.

```python
from fastapi import FastAPI
from aiavatar import AIAvatar
from aiavatar.api.router import get_router

aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY
)

# aiavatar_app.start_listening()

# Create API app and set router
api = FastAPI()
api_router = get_router(aiavatar_app, "aiavatar.log")
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


### 🎚️ Noise Filter

AIAvatarKit automatically adjusts the noise filter for listeners when you instantiate an AIAvatar object. To manually set the noise filter level for voice detection, set `auto_noise_filter_threshold` to `False` and specify the `volume_threshold_db` in decibels (dB).

```python
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    auto_noise_filter_threshold=False,
    volume_threshold_db=-40   # Set the voice detection threshold to -40 dB
)
```
