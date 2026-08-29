# Vision

The avatar can look at things. Instruct the model to emit a vision tag when it needs to see,
implement `get_image_url` to supply the image, and the captured frame is attached to the
request.

AIAvatarKit captures and sends image to AI dynamically when the AI determine that vision is required to process the request. This gives "eyes" to your AIAvatar in metaverse platforms like VRChat.

```python
# Instruct vision tag in the system message
SYSTEM_PROMPT = """
## Using Vision

If you need an image to process a user's request, you can obtain it using the following methods:

- screenshot
- camera

If an image is needed to process the request, add an instruction like <vision source="screenshot" /> to your response to request an image from the user.

By adding this instruction, the user will provide an image in their next utterance. No comments about the image itself are necessary.

Example:

user: Look! This is the sushi I had today.
assistant: <vision source="screenshot" /> Let me take a look.
"""

# The server only needs the prompt; capturing the image is the client's job
aiavatar_app = AIAvatarWebSocketServer(
    system_prompt=SYSTEM_PROMPT,
    openai_api_key=OPENAI_API_KEY
)
```

## Supplying the image from the client

When the adapter sees a vision tag in the final response, it emits a response of
`type="vision"` carrying `metadata["source"]`. The client captures whatever that names and
sends it back as a new request with a `files` entry. Nothing about capture happens on the
server.

`get_image_url` is a **client-side** decorator, defined on `AIAvatarClientBase` — so it goes
on `AIAvatarWebSocketClient` or `AIAvatarHttpClient`, not on the server:

```python
import base64
import io
import pyautogui    # pip install pyautogui
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient
from aiavatar.device.video import VideoDevice   # pip install opencv-python

client = AIAvatarWebSocketClient(url="ws://localhost:8000/ws")
default_camera = VideoDevice(device_index=0, width=960, height=540)

@client.get_image_url
async def get_image_url(source: str) -> str:
    image_bytes = None

    if source == "camera":
        # Capture photo by camera
        image_bytes = await default_camera.capture_image("camera.jpg")
    elif source == "screenshot":
        # Capture screenshot
        buffered = io.BytesIO()
        image = pyautogui.screenshot(region=(0, 0, 1280, 720))
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

    if image_bytes:
        # Upload and get url, or, make base64 encoded url
        b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        b64_url = f"data:image/jpeg;base64,{b64_encoded}"
        return b64_url
```

> **Note:** XML-style tag is also supported: `<vision source="screenshot" />`

## In a browser client

The same flow works without any Python client. The bundled viewer watches for the vision
response type and captures from the page:

```javascript
if (response.type === "vision" && response.metadata !== null) {
    if (this.onVisionRequested) {
        this.onVisionRequested(response.metadata.source);
    } else if (response.metadata.source === "camera" && this.cameraEnabled) {
        this.camera.capture();
    }
}
```

Whatever your client is, the contract is the same: read `metadata.source`, capture, and send
a follow-up request whose `files` carries the image URL or a `data:` URL.

## See also

- [Avatar control](avatar.md) — what the avatar sends back to the screen
- [LLM](llm.md) — models that accept image input

---

[← Documentation index](../README.md#-documentation)
