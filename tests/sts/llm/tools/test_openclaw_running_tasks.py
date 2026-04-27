import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiavatar.sts.llm.tools.openclaw_tool import OpenClawTool
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

    async def mock_api(query, context_id, task_id=None):
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

    assert result == {"answer": "Error"}
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
