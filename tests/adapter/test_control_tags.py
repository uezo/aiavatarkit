import json
from unittest.mock import AsyncMock

import pytest

from aiavatar.adapter import Adapter, AIAvatarResponse, ControlTagConfigResolver
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.sts.models import STSResponse


class StubPipeline:
    def add_response_handler(self, handler):
        self.response_handler = handler


class StubAdapter(Adapter):
    async def handle_response(self, response):
        pass

    async def stop_response(self, session_id: str, context_id: str):
        pass


def create_adapter():
    return StubAdapter(StubPipeline())


def test_parse_registered_control_tags_in_source_order():
    adapter = create_adapter()

    tags = adapter.parse_control_tags(
        'x<1 [face:joy] '
        '<artifact type="presentation" src="https://example.com/player?id=1" /> '
        '<artifact type="presentation" offset="-1" /> '
        '<unknown value="ignored" /> '
        "<vision source='screen_shot' />"
    )

    assert [tag.model_dump() for tag in tags] == [
        {"name": "face", "attributes": {"name": "joy"}},
        {
            "name": "artifact",
            "attributes": {
                "type": "presentation",
                "src": "https://example.com/player?id=1",
            },
        },
        {
            "name": "artifact",
            "attributes": {
                "type": "presentation",
                "offset": "-1",
            },
        },
        {"name": "vision", "attributes": {"source": "screen_shot"}},
    ]


def test_register_control_tag_with_attribute_normalizer():
    adapter = create_adapter()

    def parse_navigation(attributes):
        if "page" not in attributes:
            raise ValueError("page is required")
        return {"page": int(attributes["page"])}

    adapter.register_control_tag("navigation", parser=parse_navigation)

    tags = adapter.parse_control_tags(
        '<navigation page="3" /><navigation target="ignored" />'
    )

    assert [tag.model_dump() for tag in tags] == [
        {"name": "navigation", "attributes": {"page": 3}},
    ]


def test_config_resolver_merges_llm_attributes_over_config():
    resolver = ControlTagConfigResolver({
        "about_company": {
            "type": "presentation",
            "src": "https://speakerdeck.com/player/deck_1",
            "slide": 1,
            "aspect": "16:9",
        },
    })

    assert resolver({"id": "about_company", "slide": "5", "size": "full"}) == {
        "type": "presentation",
        "src": "https://speakerdeck.com/player/deck_1",
        "slide": "5",
        "aspect": "16:9",
        "size": "full",
    }
    assert resolver({"type": "image", "src": "https://example.com/generated.png"}) == {
        "type": "image",
        "src": "https://example.com/generated.png",
    }


def test_set_artifacts_replaces_the_catalog():
    adapter = create_adapter()
    adapter.set_artifacts({
        "first": {
            "type": "image",
            "src": "https://example.com/first.png",
            "aspect": "1:1",
        },
    })
    assert adapter.parse_control_tags('<artifact id="first" />')[0].attributes["src"].endswith("first.png")

    adapter.set_artifacts({
        "second": {"type": "image", "src": "https://example.com/second.png"},
    })
    assert adapter.parse_control_tags('<artifact id="first" />') == []
    assert adapter.parse_control_tags('<artifact id="second" />')[0].attributes["src"].endswith("second.png")


def test_update_and_add_artifacts_preserve_other_entries():
    adapter = create_adapter()
    adapter.set_artifacts({
        "first": {
            "type": "image",
            "src": "https://example.com/first.png",
            "aspect": "1:1",
        },
    })
    adapter.update_artifacts({
        "second": {"type": "image", "src": "https://example.com/second.png"},
    })
    adapter.add_artifact(
        "third",
        {"type": "image", "src": "https://example.com/third.png"},
    )

    for artifact_id in ("first", "second", "third"):
        tags = adapter.parse_control_tags(f'<artifact id="{artifact_id}" />')
        assert tags[0].attributes["src"].endswith(f"{artifact_id}.png")

    adapter.add_artifact(
        "first",
        {"type": "image", "src": "https://example.com/replaced.png"},
    )
    first = adapter.parse_control_tags('<artifact id="first" />')[0]
    assert first.attributes == {
        "type": "image",
        "src": "https://example.com/replaced.png",
    }


def test_malformed_registered_tag_is_ignored():
    adapter = create_adapter()

    assert adapter.parse_control_tags(
        '<artifact type="image" type="chart" src="https://example.com/a.png" />'
    ) == []


def test_control_tags_are_serialized_on_aiavatar_response():
    adapter = create_adapter()
    response = AIAvatarResponse(
        type="chunk",
        control_tags=adapter.parse_control_tags("[animation:wave]"),
    )

    payload = json.loads(response.model_dump_json())
    assert payload["control_tags"] == [
        {"name": "animation", "attributes": {"name": "wave"}},
    ]


@pytest.mark.asyncio
async def test_websocket_attaches_control_tags_to_chunks_only():
    server = object.__new__(AIAvatarWebSocketServer)
    Adapter.__init__(server, StubPipeline())
    server.sessions = {"session": WebSocketSessionData()}
    server.response_audio_chunk_size = 0
    server.debug = False
    server.send_response = AsyncMock()
    server.set_artifacts({
        "about_company": {
            "type": "presentation",
            "src": "https://speakerdeck.com/player/deck_1",
            "slide": 1,
            "aspect": "16:9",
        },
    })
    text = '<artifact id="about_company" slide="3" />'

    await server.handle_response(STSResponse(
        type="chunk",
        session_id="session",
        text=text,
        voice_text="",
        metadata={},
    ))
    await server.handle_response(STSResponse(
        type="final",
        session_id="session",
        text=text,
        voice_text="",
        metadata={},
    ))

    chunk_response = server.send_response.await_args_list[0].args[0]
    final_response = server.send_response.await_args_list[1].args[0]
    assert [tag.model_dump() for tag in chunk_response.control_tags] == [{
        "name": "artifact",
        "attributes": {
            "type": "presentation",
            "src": "https://speakerdeck.com/player/deck_1",
            "slide": "3",
            "aspect": "16:9",
        },
    }]
    assert final_response.control_tags is None
