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
pip install aiavatar silero-vad fastapi uvicorn websockets
```

Open `server.py` and set your OpenAI API key to `OPENAI_API_KEY`.

```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

Start the server.

```sh
uvicorn server:app
```

Visit http://localhost:8000/static/index.html and click `Start`, then try talking to the avatar.


## Deep Dive

The project README describes how to configure and customize the speech-to-speech pipeline or its components (VAD / STT / LLM / TTS).

https://github.com/uezo/aiavatarkit?tab=readme-ov-file#-contents
