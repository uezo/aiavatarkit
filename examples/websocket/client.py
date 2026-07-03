import asyncio
import os
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

# This Python client uses local audio devices.
# Install optional dependencies with: pip install "aiavatar[local-audio]"
client = AIAvatarWebSocketClient(
    api_key=os.environ.get("AIAVATAR_API_KEY")
)
asyncio.run(
    client.start_listening(
        session_id="ws_session", user_id="ws_user"
    )
)
