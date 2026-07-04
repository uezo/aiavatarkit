import os
from uuid import uuid4

import httpx
import pytest


CHAT_COMPLETIONS_SERVER_URL = os.getenv("CHAT_COMPLETIONS_SERVER_URL")

pytestmark = pytest.mark.skipif(
    not CHAT_COMPLETIONS_SERVER_URL,
    reason="CHAT_COMPLETIONS_SERVER_URL is not set",
)


def post_chat_completions(
    *,
    token: str,
    content: str,
    stream: bool = False,
) -> httpx.Response:
    return httpx.post(
        CHAT_COMPLETIONS_SERVER_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openclaw",
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        },
        timeout=120.0,
    )


def test_chat_completions():
    token = f"test_chatcompletions_{uuid4()}"

    response = post_chat_completions(
        token=token,
        content="こんにちは。10文字以内で返事して。",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "openclaw"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_stream():
    token = f"test_chatcompletions_stream_{uuid4()}"

    response = post_chat_completions(
        token=token,
        content="こんにちは。10文字以内で返事して。",
        stream=True,
    )

    assert response.status_code == 200
    assert "chat.completion.chunk" in response.text
    assert '"role":"assistant"' in response.text
    assert "[DONE]" in response.text


def test_chat_completions_requires_bearer_token():
    response = httpx.post(
        CHAT_COMPLETIONS_SERVER_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": "openclaw",
            "messages": [
                {
                    "role": "user",
                    "content": "こんにちは",
                }
            ],
        },
        timeout=30.0,
    )

    assert response.status_code == 401
