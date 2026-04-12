from datetime import datetime, timezone
import logging
import secrets
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from ....sts.llm import Tool
from ...channel_context_bridge import ChannelContextBridge

logger = logging.getLogger(__name__)


class LinebotChannelLinkTool(Tool):
    """ChannelLinkTool variant for LINE channels.

    Instead of a numeric state, generates a LINE Login URL.
    The user opens the URL, authenticates with LINE, and the callback
    verifies the state parameter and exchanges the code for a LINE user ID.
    """

    LINE_AUTHORIZE_URL = "https://access.line.me/oauth2/v2.1/authorize"
    LINE_TOKEN_URL = "https://api.line.me/oauth2/v2.1/token"
    LINE_PROFILE_URL = "https://api.line.me/v2/profile"

    DEFAULT_SUCCESS_HTML = "<h1>LINE account linked successfully!</h1><p>You can close this window.</p>"
    DEFAULT_ERROR_HTML = "<h1>Login failed</h1><p>Invalid or expired link.</p>"

    def __init__(
        self,
        *,
        channel_id: str,
        client_secret: str,
        base_url: str,
        channel_context_bridge: ChannelContextBridge = None,
        success_html: str = None,
        error_html: str = None,
        state_timeout: float = 300,
        name=None,
        spec=None,
        instruction=None,
        is_dynamic=False,
    ):
        self.state_timeout = state_timeout
        self._states: Dict[str, Tuple[str, datetime]] = {}

        self.channel_id = channel_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.callback_path = "/login-callback"
        self.channel_context_bridge = channel_context_bridge
        self.success_html = success_html or self.DEFAULT_SUCCESS_HTML
        self.error_html = error_html or self.DEFAULT_ERROR_HTML

        tool_name = name or "get_login_url"
        super().__init__(
            name=tool_name,
            spec=spec or {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Generate a LINE Login URL to link a LINE account to this conversation context.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            func=self.get_login_url,
            instruction=instruction,
            is_dynamic=is_dynamic,
        )

    async def get_login_url(self, metadata: dict = None):
        try:
            user_id = metadata["user_id"]
            state = secrets.token_urlsafe(32)
            self._states[user_id] = (state, datetime.now(timezone.utc))
            login_url = self.LINE_AUTHORIZE_URL + "?" + urlencode({
                "response_type": "code",
                "client_id": self.channel_id,
                "redirect_uri": self.base_url + self.callback_path,
                "state": state,
                "scope": "profile openid",
            })
            return {
                "login_url": login_url,
                "instruction": "Tell the user this LINE Login URL.",
            }

        except Exception:
            logger.exception("Error at get_login_url")
            return {"error": "Failed to generate LINE Login URL."}

    async def verify_state(self, state: str, code: str) -> Optional[Tuple[str, str]]:
        """Verify state, exchange code for token, and return (user_id, line_user_id) or None."""
        self.cleanup_expired()

        # Find user_id by state
        found_user_id = None
        for user_id, (stored_state, _) in list(self._states.items()):
            if stored_state == state:
                found_user_id = user_id
                del self._states[user_id]
                break

        if found_user_id is None:
            return None

        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                self.LINE_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.base_url + self.callback_path,
                    "client_id": self.channel_id,
                    "client_secret": self.client_secret,
                },
            )
            if token_resp.status_code != 200:
                logger.error(f"LINE token exchange failed: {token_resp.status_code} {token_resp.text}")
                return None
            access_token = token_resp.json()["access_token"]

            # Get LINE profile
            profile_resp = await client.get(
                self.LINE_PROFILE_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if profile_resp.status_code != 200:
                logger.error(f"LINE profile fetch failed: {profile_resp.status_code} {profile_resp.text}")
                return None
            line_user_id = profile_resp.json()["userId"]

        return (found_user_id, line_user_id)

    def cleanup_expired(self):
        """Remove all expired states."""
        now = datetime.now(timezone.utc)
        self._states = {
            user_id: (state, created_at)
            for user_id, (state, created_at) in self._states.items()
            if (now - created_at).total_seconds() <= self.state_timeout
        }

    def get_callback_router(self) -> APIRouter:
        router = APIRouter()

        @router.get(self.callback_path, response_class=HTMLResponse)
        async def line_login_callback(request: Request):
            state = request.query_params.get("state")
            code = request.query_params.get("code")

            if not state or not code:
                return HTMLResponse(self.error_html, status_code=400)

            result = await self.verify_state(state=state, code=code)
            if result is None:
                return HTMLResponse(self.error_html, status_code=400)

            user_id, line_user_id = result
            await self.channel_context_bridge.link_channel_user(
                channel_id="linebot",
                channel_user_id=line_user_id,
                user_id=user_id,
            )

            return HTMLResponse(self.success_html)

        return router
