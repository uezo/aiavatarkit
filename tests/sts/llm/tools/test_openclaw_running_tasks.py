import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiavatar.sts.llm.tools.openclaw_tool import OpenClawTool, OpenClawConfig
from aiavatar.sts.llm.base import LLMServiceDummy


def make_openclaw_tool(**kwargs):
    return OpenClawTool(
        openclaw_api_key="test-key",
        openclaw_base_url="http://localhost:9999",
        **kwargs,
    )


# --- Running tasks store ---

def test_add_and_get_running_task():
    tool = make_openclaw_tool()
    task_id = tool.add_running_task("Search weather", {"context_id": "ctx1", "user_id": "user1"}, "Searching...")

    tasks = tool.get_running_tasks(context_id="ctx1")
    assert len(tasks) == 1
    assert tasks[0] == {"request": "Search weather", "progress": "Searching..."}
    assert task_id in tool._running_tasks


def test_add_progress():
    tool = make_openclaw_tool()
    task_id = tool.add_running_task("Search weather", {"context_id": "ctx1", "user_id": "user1"}, "Starting...")
    tool.add_progress(task_id, "Almost done...")

    tasks = tool.get_running_tasks(context_id="ctx1")
    assert tasks[0]["progress"] == "Starting...Almost done..."


def test_add_progress_nonexistent_task_is_noop():
    tool = make_openclaw_tool()
    tool.add_progress("nonexistent", "progress")
    assert len(tool._running_tasks) == 0


def test_remove_running_task():
    tool = make_openclaw_tool()
    task_id = tool.add_running_task("Search weather", {"context_id": "ctx1", "user_id": "user1"})
    tool.remove_running_task(task_id)

    assert len(tool._running_tasks) == 0
    assert tool.get_running_tasks(context_id="ctx1") == []


def test_remove_nonexistent_task_is_noop():
    tool = make_openclaw_tool()
    tool.remove_running_task("nonexistent")  # should not raise


def test_get_running_tasks_by_context_id():
    tool = make_openclaw_tool()
    tool.add_running_task("Task A", {"context_id": "ctx1", "user_id": "user1"}, "Running")
    tool.add_running_task("Task B", {"context_id": "ctx2", "user_id": "user1"}, "Running")
    tool.add_running_task("Task C", {"context_id": "ctx1", "user_id": "user2"}, "Running")

    tasks = tool.get_running_tasks(context_id="ctx1")
    assert len(tasks) == 2
    requests = {t["request"] for t in tasks}
    assert requests == {"Task A", "Task C"}


def test_get_running_tasks_by_user_id():
    tool = make_openclaw_tool()
    tool.add_running_task("Task A", {"context_id": "ctx1", "user_id": "user1"}, "Running")
    tool.add_running_task("Task B", {"context_id": "ctx2", "user_id": "user2"}, "Running")
    tool.add_running_task("Task C", {"context_id": "ctx3", "user_id": "user1"}, "Running")

    tasks = tool.get_running_tasks(user_id="user1")
    assert len(tasks) == 2
    requests = {t["request"] for t in tasks}
    assert requests == {"Task A", "Task C"}


def test_get_running_tasks_empty():
    tool = make_openclaw_tool()
    assert tool.get_running_tasks(context_id="ctx1") == []


def test_multiple_tasks_remove_one():
    tool = make_openclaw_tool()
    id1 = tool.add_running_task("Task A", {"context_id": "ctx1", "user_id": "user1"}, "Running")
    id2 = tool.add_running_task("Task B", {"context_id": "ctx1", "user_id": "user1"}, "Running")

    tool.remove_running_task(id1)

    tasks = tool.get_running_tasks(context_id="ctx1")
    assert len(tasks) == 1
    assert tasks[0]["request"] == "Task B"


# --- invoke_openclaw integration ---

@pytest.mark.asyncio
async def test_invoke_openclaw_registers_and_removes_task():
    tool = make_openclaw_tool()

    with patch.object(tool, "_call_openclaw_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "sunny"

        result = await tool.invoke_openclaw("weather?", {"context_id": "ctx1", "user_id": "user1"})

    assert result == {"answer": "sunny"}
    # Task should be removed after completion
    assert tool.get_running_tasks(context_id="ctx1") == []


@pytest.mark.asyncio
async def test_invoke_openclaw_task_visible_during_execution():
    tool = make_openclaw_tool()
    captured_tasks = []

    async def mock_api(query, context_id, user_id, task_id):
        # Capture running tasks during execution
        captured_tasks.extend(tool.get_running_tasks(context_id="ctx1"))
        return "result"

    with patch.object(tool, "_call_openclaw_api", side_effect=mock_api):
        await tool.invoke_openclaw("do something", {"context_id": "ctx1", "user_id": "user1"})

    # Task was visible during execution
    assert len(captured_tasks) == 1
    assert captured_tasks[0]["request"] == "do something"
    assert captured_tasks[0]["progress"] == "Start processing...\n"

    # Task removed after execution
    assert tool.get_running_tasks(context_id="ctx1") == []


@pytest.mark.asyncio
async def test_invoke_openclaw_removes_task_on_error():
    tool = make_openclaw_tool()

    with patch.object(tool, "_call_openclaw_api", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = RuntimeError("API error")

        result = await tool.invoke_openclaw("fail", {"context_id": "ctx1", "user_id": "user1"})

    assert result == {"answer": "Error: API error"}
    # Task should be removed even on error
    assert tool.get_running_tasks(context_id="ctx1") == []


# --- create_check_tool ---

@pytest.mark.asyncio
async def test_create_check_tool_returns_running_tasks():
    tool = make_openclaw_tool()
    check_tool = tool.create_check_tool()

    tool.add_running_task("Search weather", {"context_id": "ctx1", "user_id": "user1"}, "Searching...")

    result = await check_tool.func(metadata={"context_id": "ctx1", "user_id": "user1"})
    assert result == {"running_tasks": [{"request": "Search weather", "progress": "Searching..."}]}


@pytest.mark.asyncio
async def test_create_check_tool_returns_empty():
    tool = make_openclaw_tool()
    check_tool = tool.create_check_tool()

    result = await check_tool.func(metadata={"context_id": "ctx1", "user_id": "user1"})
    assert result == {"running_tasks": [], "message": "No running tasks."}


def test_create_check_tool_spec():
    tool = make_openclaw_tool()
    check_tool = tool.create_check_tool()

    assert check_tool.name == "check_running_openclaw_tasks"
    assert check_tool.spec["type"] == "function"
    assert check_tool.spec["function"]["name"] == "check_running_openclaw_tasks"


def test_create_check_tool_custom_name():
    tool = make_openclaw_tool()
    check_tool = tool.create_check_tool(name="my_check", description="Custom desc")

    assert check_tool.name == "my_check"
    assert check_tool.spec["function"]["description"] == "Custom desc"


# --- Per-user OpenClaw config ---

def test_get_openclaw_config_defaults_when_no_user_config():
    tool = make_openclaw_tool(
        openclaw_session_key="default-session",
        openclaw_session_key_key="x-openclaw-session-key",
        openclaw_model="openclaw",
    )
    config = tool.get_openclaw_config("unknown_user")

    assert config.openclaw_api_key == "test-key"
    assert config.openclaw_base_url == "http://localhost:9999"
    assert config.openclaw_session_key == "default-session"
    assert config.openclaw_session_key_key == "x-openclaw-session-key"
    assert config.openclaw_model == "openclaw"


def test_get_openclaw_config_partial_override():
    tool = make_openclaw_tool(
        openclaw_session_key="default-session",
        openclaw_session_key_key="x-openclaw-session-key",
        openclaw_model="openclaw",
        openclaw_configs={
            "user1": OpenClawConfig(
                openclaw_api_key="user1-key",
                openclaw_base_url="http://user1-server:8000",
            ),
        },
    )
    config = tool.get_openclaw_config("user1")

    # Overridden by user config
    assert config.openclaw_api_key == "user1-key"
    assert config.openclaw_base_url == "http://user1-server:8000"
    # Falls back to tool defaults
    assert config.openclaw_session_key == "default-session"
    assert config.openclaw_session_key_key == "x-openclaw-session-key"
    assert config.openclaw_model == "openclaw"


def test_get_openclaw_config_full_override():
    tool = make_openclaw_tool(
        openclaw_session_key="default-session",
        openclaw_session_key_key="x-openclaw-session-key",
        openclaw_model="openclaw",
        openclaw_configs={
            "user1": OpenClawConfig(
                openclaw_api_key="user1-key",
                openclaw_base_url="http://user1-hermes:8000",
                openclaw_session_key="user1-session",
                openclaw_session_key_key="X-Hermes-Session-Id",
                openclaw_model="hermes-agent",
            ),
        },
    )
    config = tool.get_openclaw_config("user1")

    assert config.openclaw_api_key == "user1-key"
    assert config.openclaw_base_url == "http://user1-hermes:8000"
    assert config.openclaw_session_key == "user1-session"
    assert config.openclaw_session_key_key == "X-Hermes-Session-Id"
    assert config.openclaw_model == "hermes-agent"


def test_get_openclaw_config_different_users_get_different_configs():
    tool = make_openclaw_tool(
        openclaw_session_key="default-session",
        openclaw_session_key_key="x-openclaw-session-key",
        openclaw_model="openclaw",
        openclaw_configs={
            "user1": OpenClawConfig(openclaw_api_key="key-1", openclaw_base_url="http://server-1"),
            "user2": OpenClawConfig(openclaw_api_key="key-2", openclaw_model="hermes-agent"),
        },
    )

    c1 = tool.get_openclaw_config("user1")
    c2 = tool.get_openclaw_config("user2")

    assert c1.openclaw_api_key == "key-1"
    assert c1.openclaw_base_url == "http://server-1"
    assert c1.openclaw_model == "openclaw"  # fallback

    assert c2.openclaw_api_key == "key-2"
    assert c2.openclaw_base_url == "http://localhost:9999"  # fallback
    assert c2.openclaw_model == "hermes-agent"


def test_update_openclaw_config():
    tool = make_openclaw_tool()
    tool.update_openclaw_config("user1", OpenClawConfig(openclaw_api_key="new-key"))

    config = tool.get_openclaw_config("user1")
    assert config.openclaw_api_key == "new-key"


def test_update_openclaw_config_replaces_existing():
    tool = make_openclaw_tool(
        openclaw_configs={
            "user1": OpenClawConfig(openclaw_api_key="old-key"),
        },
    )
    tool.update_openclaw_config("user1", OpenClawConfig(openclaw_api_key="new-key"))

    config = tool.get_openclaw_config("user1")
    assert config.openclaw_api_key == "new-key"


def test_delete_openclaw_config():
    tool = make_openclaw_tool(
        openclaw_configs={
            "user1": OpenClawConfig(openclaw_api_key="user1-key"),
        },
    )
    tool.delete_openclaw_config("user1")

    # Should fall back to tool defaults
    config = tool.get_openclaw_config("user1")
    assert config.openclaw_api_key == "test-key"


def test_delete_openclaw_config_nonexistent_is_noop():
    tool = make_openclaw_tool()
    tool.delete_openclaw_config("nonexistent")  # should not raise


# --- _call_openclaw_api uses per-user config ---

@pytest.mark.asyncio
async def test_call_openclaw_api_uses_user_config():
    """Verify that _call_openclaw_api creates a client with per-user config values."""
    tool = make_openclaw_tool(
        openclaw_session_key="default-session",
        openclaw_session_key_key="x-openclaw-session-key",
        openclaw_model="openclaw",
        openclaw_configs={
            "user1": OpenClawConfig(
                openclaw_api_key="user1-key",
                openclaw_base_url="http://user1-server:8000",
                openclaw_session_key_key="X-Hermes-Session-Id",
                openclaw_model="hermes-agent",
            ),
        },
    )

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "user1 response"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
    mock_client.close = AsyncMock()

    with patch("aiavatar.sts.llm.tools.openclaw_tool.openai.AsyncClient", return_value=mock_client) as mock_ctor:
        answer = await tool._call_openclaw_api("hello", "ctx1", "user1", "task1")

    # Client created with user1's config
    mock_ctor.assert_called_once_with(
        api_key="user1-key",
        base_url="http://user1-server:8000",
        timeout=30000,
    )
    # API called with user1's model and header
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "hermes-agent"
    assert call_kwargs.kwargs["extra_headers"]["X-Hermes-Session-Id"] == "ctx1"
    assert answer == "user1 response"
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_openclaw_api_falls_back_to_defaults():
    """Verify that _call_openclaw_api uses tool defaults for unknown users."""
    tool = make_openclaw_tool(
        openclaw_session_key="default-session",
        openclaw_session_key_key="x-openclaw-session-key",
        openclaw_model="openclaw",
    )

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "default response"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
    mock_client.close = AsyncMock()

    with patch("aiavatar.sts.llm.tools.openclaw_tool.openai.AsyncClient", return_value=mock_client) as mock_ctor:
        answer = await tool._call_openclaw_api("hello", "ctx1", "unknown_user", "task1")

    # Client created with tool defaults
    mock_ctor.assert_called_once_with(
        api_key="test-key",
        base_url="http://localhost:9999",
        timeout=30000,
    )
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "openclaw"
    assert call_kwargs.kwargs["extra_headers"]["x-openclaw-session-key"] == "ctx1"
    assert answer == "default response"


@pytest.mark.asyncio
async def test_call_openclaw_api_closes_client_on_error():
    """Verify that client.close() is called even when the API call fails."""
    tool = make_openclaw_tool()

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API error"))
    mock_client.close = AsyncMock()

    with patch("aiavatar.sts.llm.tools.openclaw_tool.openai.AsyncClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="API error"):
            await tool._call_openclaw_api("hello", "ctx1", "user1", "task1")

    mock_client.close.assert_awaited_once()


# --- base_url guard in invoke_openclaw ---

@pytest.mark.asyncio
async def test_invoke_openclaw_rejects_unconfigured_user():
    """When no base_url is resolved for the user, return an error message without calling the API."""
    tool = OpenClawTool(
        openclaw_configs={
            "configured_user": OpenClawConfig(
                openclaw_api_key="key",
                openclaw_base_url="http://server:8000",
                openclaw_session_key_key="x-openclaw-session-key",
                openclaw_model="openclaw",
            ),
        },
        stream=False,
    )

    result = await tool.invoke_openclaw("hello", {"context_id": "ctx1", "user_id": "unknown_user"})

    assert "not configured" in result["answer"].lower()
    # No running task should remain
    assert tool.get_running_tasks(user_id="unknown_user") == []


@pytest.mark.asyncio
async def test_invoke_openclaw_allows_configured_user():
    """Configured user should pass the base_url check and call the API."""
    tool = OpenClawTool(
        openclaw_configs={
            "user1": OpenClawConfig(
                openclaw_api_key="key",
                openclaw_base_url="http://server:8000",
                openclaw_session_key_key="x-openclaw-session-key",
                openclaw_model="openclaw",
            ),
        },
        stream=False,
    )

    with patch.object(tool, "_call_openclaw_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "success"
        result = await tool.invoke_openclaw("hello", {"context_id": "ctx1", "user_id": "user1"})

    assert result == {"answer": "success"}
    mock_api.assert_awaited_once()
