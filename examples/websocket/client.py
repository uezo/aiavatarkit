import asyncio
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient()
asyncio.run(
    client.start_listening(
        session_id="ws_session", user_id="ws_user"
    )
)
