import asyncio
from datetime import datetime, timezone
import json
from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from aiavatar.sts.llm.base import LLMServiceDummy, ToolCall
from aiavatar.sts.llm.tools.grokbot_tool import GrokBotTool


@pytest_asyncio.fixture
async def make_tool(monkeypatch):
    original_client = httpx.AsyncClient
    tools, clients = [], []

    def make(handler, **kwargs):
        def client_factory(**client_kwargs):
            client = original_client(
                transport=httpx.MockTransport(handler), **client_kwargs
            )
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", client_factory)
        tool = GrokBotTool(
            webhook_url="https://grokbot.example/webhook",
            webhook_key="test-webhook-key",
            **kwargs,
        )
        tools.append(tool)
        return tool

    yield make

    for tool in tools:
        await tool.aclose()
    assert all(client.is_closed for client in clients)


@pytest_asyncio.fixture
async def advance_time(monkeypatch):
    loop = asyncio.get_running_loop()
    now = loop.time()
    monkeypatch.setattr(loop, "time", lambda: now)

    async def advance(seconds):
        nonlocal now
        now += seconds
        # Let due timers fire and the tasks they wake finish without real sleeps.
        for _ in range(3):
            await asyncio.sleep(0)

    return advance


def make_service(tool, tmp_path):
    service = LLMServiceDummy(db_connection_str=str(tmp_path / "context.db"))
    service.add_tool(tool, use_original=True)
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {"task_id": "request-123"}])
async def test_post_and_callback_before_http_response(make_tool, metadata):
    sent = []

    def handle(request):
        assert request.method == "POST"
        assert str(request.url) == "https://grokbot.example/webhook"
        assert request.headers["Authorization"] == "Bearer test-webhook-key"
        payload = json.loads(request.content)
        sent.append(payload)
        assert set(payload) == {"text", "request_id", "requested_at"}
        assert payload["text"] == "今日の天気を教えて"
        assert datetime.fromisoformat(payload["requested_at"]).tzinfo == timezone.utc
        request_id = payload["request_id"]
        if metadata:
            assert request_id == metadata["task_id"]
        else:
            assert str(UUID(request_id)) == request_id
        assert tool.receive_response(request_id, "晴れです") is True
        assert tool.receive_response(request_id, "duplicate") is False
        return httpx.Response(202)

    tool = make_tool(handle)
    result = await tool.invoke_grokbot("今日の天気を教えて", metadata)
    assert result == {
        "answer": "晴れです",
        "request_id": sent[0]["request_id"],
        "requested_at": sent[0]["requested_at"],
        "request": "今日の天気を教えて",
    }
    assert not tool._running_tasks
    assert tool.receive_response("unknown", "ignored") is False


@pytest.mark.asyncio
async def test_concurrent_callbacks_match_requests_out_of_order(make_tool):
    sent = asyncio.Queue()

    def handle(request):
        sent.put_nowait(json.loads(request.content))
        return httpx.Response(202)

    tool = make_tool(handle)
    first = asyncio.create_task(tool.invoke_grokbot("first", {"task_id": "first"}))
    second = asyncio.create_task(tool.invoke_grokbot("second", {"task_id": "second"}))
    payloads = [await asyncio.wait_for(sent.get(), 1) for _ in range(2)]
    requested_at = {payload["request_id"]: payload["requested_at"] for payload in payloads}
    assert set(requested_at) == {"first", "second"}
    assert tool.receive_response("first", "still working", status="progress") is True
    assert tool.receive_response("second", "second answer", status="done") is True
    assert await second == {
        "answer": "second answer",
        "request_id": "second",
        "requested_at": requested_at["second"],
        "request": "second",
    }
    assert not first.done()
    assert tool._running_tasks["first"]["progress"].endswith("still working\n")
    assert tool.receive_response("first", "first answer", status="done") is True
    assert await first == {
        "answer": "first answer",
        "request_id": "first",
        "requested_at": requested_at["first"],
        "request": "first",
    }
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_progress_accumulates_without_completing_request(make_tool):
    sent = asyncio.Event()

    def handle(request):
        sent.set()
        return httpx.Response(202)

    tool = make_tool(handle)
    request = asyncio.create_task(tool.invoke_grokbot("long task", {"task_id": "progress"}))
    await asyncio.wait_for(sent.wait(), 1)
    initial_progress = tool._running_tasks["progress"]["progress"]

    for progress in ("first step finished", "second step in progress"):
        assert tool.receive_response("progress", progress, status="progress") is True
        assert not tool._running_tasks["progress"]["future"].done()
        assert not request.done()

    assert tool._running_tasks["progress"]["progress"] == (
        initial_progress + "first step finished\nsecond step in progress\n"
    )
    assert tool.receive_response("progress", "ignored", status="unknown") is False
    assert tool.receive_response("unknown", "ignored", status="progress") is False
    assert not tool._running_tasks["progress"]["future"].done()
    assert tool.receive_response("progress", "final answer", status="done") is True
    assert tool.receive_response("progress", "late progress", status="progress") is False
    assert tool.receive_response("progress", "duplicate answer", status="done") is False
    assert (await request)["answer"] == "final answer"
    assert not tool._running_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("submitted_at", [0, 1729200000.125])
async def test_requested_at_uses_original_submission_timestamp(make_tool, submitted_at):
    expected = datetime.fromtimestamp(submitted_at, timezone.utc)
    sent = []

    def handle(request):
        payload = json.loads(request.content)
        sent.append(payload)
        assert datetime.fromisoformat(payload["requested_at"]) == expected
        assert tool.receive_response(payload["request_id"], "answer", status="done") is True
        return httpx.Response(202)

    tool = make_tool(handle)
    metadata = {"task_id": "original", "submitted_at": submitted_at}
    result = await tool.invoke_grokbot("original request", metadata)
    assert result == {
        "answer": "answer",
        "request_id": "original",
        "requested_at": sent[0]["requested_at"],
        "request": "original request",
    }
    assert metadata == {"task_id": "original", "submitted_at": submitted_at}


@pytest.mark.asyncio
async def test_running_tasks_and_check_tool_are_scoped_and_cleaned_up(make_tool):
    sent = asyncio.Queue()

    def handle(request):
        sent.put_nowait(json.loads(request.content))
        return httpx.Response(202)

    tool = make_tool(handle)
    requests = [
        asyncio.create_task(tool.invoke_grokbot(query, {
            "task_id": task_id, "context_id": context_id, "user_id": user_id,
        }))
        for task_id, query, context_id, user_id in [
            ("a", "request A", "ctx1", "user1"),
            ("b", "request B", "ctx2", "user1"),
            ("c", "request C", "ctx1", "user2"),
        ]
    ]
    payloads = [await asyncio.wait_for(sent.get(), 1) for _ in requests]
    requested_at = {payload["request_id"]: payload["requested_at"] for payload in payloads}
    assert tool.receive_response("a", "checking sources", status="progress") is True
    assert tool.get_running_tasks() == []
    assert tool.get_running_tasks(context_id="missing") == []
    assert {task["request_id"] for task in tool.get_running_tasks(context_id="ctx1")} == {"a", "c"}
    assert {task["request_id"] for task in tool.get_running_tasks(user_id="user1")} == {"a", "b"}
    scoped = tool.get_running_tasks(context_id="ctx1", user_id="user1")
    assert len(scoped) == 1
    assert scoped[0] == {
        "request_id": "a",
        "request": "request A",
        "requested_at": requested_at["a"],
        "progress": tool._running_tasks["a"]["progress"],
    }
    assert scoped[0]["progress"].endswith("checking sources\n")

    check_tool = tool.create_check_tool()
    assert check_tool.name == "check_running_grokbot_tasks"
    assert check_tool.spec["function"]["name"] == check_tool.name
    assert await check_tool.func(metadata={"context_id": "ctx1", "user_id": "user1"}) == {
        "running_tasks": scoped,
    }
    assert (await check_tool.func(metadata={"user_id": "user1"}))["running_tasks"] == tool.get_running_tasks(user_id="user1")
    for metadata in (None, {}, {"context_id": "missing", "user_id": "user1"}):
        result = await check_tool.func(metadata=metadata)
        assert result["running_tasks"] == []
        assert result["message"]

    assert tool.receive_response("a", "answer A", status="done") is True
    await requests[0]
    assert tool.get_running_tasks(context_id="ctx1", user_id="user1") == []
    assert {task["request_id"] for task in tool.get_running_tasks(user_id="user1")} == {"b"}
    for request in requests[1:]:
        request.cancel()
    await asyncio.gather(*requests[1:], return_exceptions=True)
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_completion_is_deferred_and_preserves_conversation_metadata(make_tool, tmp_path):
    sent = asyncio.Queue()
    completed = []
    ready = asyncio.Event()

    def handle(request):
        request_id = json.loads(request.content)["request_id"]
        tool.receive_response(request_id, "result")
        sent.put_nowait(request_id)
        return httpx.Response(202)

    tool = make_tool(handle)

    @tool.on_completed
    async def on_completed(result, metadata):
        completed.append((result, metadata))
        ready.set()

    service = make_service(tool, tmp_path)
    metadata = {"context_id": "ctx", "user_id": "user", "session_id": "session", "channel": "websocket"}
    results = [result async for result in service.execute_tool(tool.name, {"query": "question"}, metadata)]
    assert len(results) == 1
    accepted = results[0]
    assert accepted.data == {"message": tool.immediate_message, "task_id": accepted.task_id}
    assert await asyncio.wait_for(sent.get(), 1) == accepted.task_id
    assert completed == []
    service._start_deferred_callbacks([ToolCall(name=tool.name, result=accepted)])
    await asyncio.wait_for(ready.wait(), 1)
    result, completion_metadata = completed[0]
    assert result == {
        "answer": "result",
        "request_id": accepted.task_id,
        "requested_at": datetime.fromtimestamp(completion_metadata["submitted_at"], timezone.utc).isoformat(),
        "request": "question",
    }
    assert all(completion_metadata[key] == value for key, value in metadata.items())
    assert completion_metadata["task_id"] == accepted.task_id
    assert completion_metadata["arguments"] == {"query": "question"}
    assert "submitted_at" in completion_metadata
    assert "task_id" not in metadata


@pytest.mark.asyncio
async def test_callback_timeout_clears_pending_request(make_tool, advance_time):
    tool = make_tool(lambda request: httpx.Response(202))
    request = asyncio.create_task(tool.invoke_grokbot(
        "question", {"task_id": "timeout", "submitted_at": 0},
    ))
    await advance_time(0)
    await advance_time(299)
    assert not request.done()
    await advance_time(2)
    assert request.done()
    result = await request
    assert "timed out" in result["answer"].lower()
    assert result["request_id"] == "timeout"
    assert result["request"] == "question"
    assert datetime.fromisoformat(result["requested_at"]) == datetime.fromtimestamp(0, timezone.utc)
    assert not tool._running_tasks
    assert tool.receive_response("timeout", "too late") is False


@pytest.mark.asyncio
async def test_repeated_progress_keeps_request_alive_until_done(make_tool, advance_time):
    tool = make_tool(lambda request: httpx.Response(202))
    request = asyncio.create_task(tool.invoke_grokbot("long task", {"task_id": "active"}))
    await advance_time(0)

    for step in range(3):
        await advance_time(240)
        assert not request.done()
        assert tool.receive_response("active", f"step {step}", status="progress") is True

    await advance_time(240)
    assert not request.done()
    assert tool.receive_response("active", "finished", status="done") is True
    assert (await request)["answer"] == "finished"
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_timeout_counts_from_last_progress(make_tool, advance_time):
    tool = make_tool(lambda request: httpx.Response(202))
    request = asyncio.create_task(tool.invoke_grokbot("stalled task", {"task_id": "stalled"}))
    await advance_time(0)
    for progress in ("first step", "second step"):
        await advance_time(200)
        assert tool.receive_response("stalled", progress, status="progress") is True

    await advance_time(299)
    assert not request.done()
    await advance_time(2)
    assert request.done()
    assert "timed out" in (await request)["answer"].lower()
    assert not tool._running_tasks
    assert tool.receive_response("stalled", "too late", status="progress") is False
    assert tool.receive_response("stalled", "too late", status="done") is False


@pytest.mark.asyncio
async def test_progress_does_not_extend_another_requests_timeout(make_tool, advance_time):
    tool = make_tool(lambda request: httpx.Response(202))
    active = asyncio.create_task(tool.invoke_grokbot("active task", {"task_id": "active"}))
    stalled = asyncio.create_task(tool.invoke_grokbot("stalled task", {"task_id": "stalled"}))
    await advance_time(0)
    await advance_time(240)
    assert tool.receive_response("active", "still working", status="progress") is True

    await advance_time(61)
    assert stalled.done()
    assert "timed out" in (await stalled)["answer"].lower()
    assert not active.done()
    assert set(tool._running_tasks) == {"active"}
    assert tool.receive_response("active", "finished", status="done") is True
    assert (await active)["answer"] == "finished"
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_progress_before_post_returns_preserves_full_response_window(make_tool, advance_time):
    release_post = asyncio.Event()

    async def handle(request):
        await release_post.wait()
        return httpx.Response(202)

    tool = make_tool(handle)
    request = asyncio.create_task(tool.invoke_grokbot("question", {"task_id": "posting"}))
    await advance_time(0)
    await advance_time(10)
    assert tool.receive_response("posting", "already working", status="progress") is True
    await advance_time(10)
    release_post.set()
    await advance_time(0)

    await advance_time(290)
    assert not request.done()
    assert tool.receive_response("posting", "finished", status="done") is True
    assert (await request)["answer"] == "finished"
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_http_failure_clears_pending_request_without_retry(make_tool):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(503)

    tool = make_tool(handle)
    result = await tool.invoke_grokbot("question", {"task_id": "failed", "submitted_at": 0})
    assert "failed" in result["answer"].lower()
    assert result["request_id"] == "failed"
    assert result["request"] == "question"
    assert datetime.fromisoformat(result["requested_at"]) == datetime.fromtimestamp(0, timezone.utc)
    assert len(requests) == 1
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_cancel_clears_pending_request(make_tool):
    sent = asyncio.Event()

    def handle(request):
        sent.set()
        return httpx.Response(202)

    tool = make_tool(handle)
    request = asyncio.create_task(tool.invoke_grokbot("question", {"task_id": "cancelled"}))
    await asyncio.wait_for(sent.wait(), 1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert not tool._running_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("start_callback_first", [False, True])
async def test_canceled_callback_discards_progress_without_completion(
    make_tool, tmp_path, caplog, start_callback_first, advance_time,
):
    sent = asyncio.Event()
    completed = []

    def handle(request):
        sent.set()
        return httpx.Response(202)

    tool = make_tool(handle)

    @tool.on_completed
    async def on_completed(result, metadata):
        completed.append(result)

    service = make_service(tool, tmp_path)
    results = [result async for result in service.execute_tool(
        tool.name, {"query": "obsolete request"}, {"context_id": "ctx"},
    )]
    accepted = results[0]
    await asyncio.wait_for(sent.wait(), 1)
    request = tool._running_tasks[accepted.task_id]["task"]
    pending = tool._running_tasks[accepted.task_id]["future"]
    await advance_time(240)
    assert tool.receive_response(accepted.task_id, "halfway done", status="progress") is True
    assert tool.get_running_tasks(context_id="ctx")[0]["progress"].endswith("halfway done\n")
    await advance_time(100)
    assert not request.done()
    if start_callback_first:
        service._start_deferred_callbacks([ToolCall(name=tool.name, result=accepted)])

    caplog.set_level("INFO", logger="aiavatar.sts.llm.tools.grokbot_tool")
    assert tool.receive_response(accepted.task_id, status="canceled") is True
    assert pending.cancelled()
    assert accepted.task_id in caplog.text
    for status in ("progress", "done", "canceled"):
        assert tool.receive_response(accepted.task_id, "too late", status=status) is False
    with pytest.raises(asyncio.CancelledError):
        await request
    assert tool.get_running_tasks(context_id="ctx") == []
    assert not tool._running_tasks

    if not start_callback_first:
        service._start_deferred_callbacks([ToolCall(name=tool.name, result=accepted)])
    callbacks = list(tool._background_tasks)
    await asyncio.gather(*callbacks, return_exceptions=True)
    assert all(callback.cancelled() for callback in callbacks)
    assert not tool._background_tasks
    assert completed == []


@pytest.mark.asyncio
async def test_canceled_callback_preserves_newer_request(make_tool):
    sent = asyncio.Queue()

    def handle(request):
        sent.put_nowait(json.loads(request.content)["request_id"])
        return httpx.Response(202)

    tool = make_tool(handle)
    first = asyncio.create_task(tool.invoke_grokbot("original", {"task_id": "original"}))
    second = asyncio.create_task(tool.invoke_grokbot("corrected", {"task_id": "corrected"}))
    assert {await asyncio.wait_for(sent.get(), 1) for _ in range(2)} == {"original", "corrected"}
    assert tool.receive_response("original", "old progress", status="progress") is True
    assert tool.receive_response("corrected", "new progress", status="progress") is True

    assert tool.receive_response("original", status="canceled") is True
    with pytest.raises(asyncio.CancelledError):
        await first
    assert set(tool._running_tasks) == {"corrected"}
    assert tool._running_tasks["corrected"]["progress"] == "new progress\n"
    assert not second.done()
    assert tool.receive_response("corrected", "corrected answer", status="done") is True
    assert (await second)["answer"] == "corrected answer"
    assert not tool._running_tasks


@pytest.mark.asyncio
async def test_canceled_callback_interrupts_pending_post(make_tool):
    sent = asyncio.Event()
    post_exited = asyncio.Event()

    async def handle(request):
        sent.set()
        try:
            await asyncio.Event().wait()
        finally:
            post_exited.set()

    tool = make_tool(handle)
    request = asyncio.create_task(tool.invoke_grokbot("question", {"task_id": "posting"}))
    await asyncio.wait_for(sent.wait(), 1)
    assert tool.receive_response("posting", status="canceled") is True
    assert tool._running_tasks["posting"]["future"].cancelled()
    for status in ("progress", "done", "canceled"):
        assert tool.receive_response("posting", "too late", status=status) is False
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, 1)
    assert post_exited.is_set()
    assert not tool._running_tasks
    # make_tool's teardown also verifies the interrupted HTTP client was closed.


@pytest.mark.asyncio
@pytest.mark.parametrize("send_failure", [False, True])
async def test_canceled_callback_before_post_returns_never_notifies(
    make_tool, tmp_path, send_failure,
):
    sent = asyncio.Event()
    completed = []
    requests = []

    def handle(request):
        requests.append(asyncio.current_task())
        request_id = json.loads(request.content)["request_id"]
        assert tool.receive_response(request_id, status="canceled") is True
        sent.set()
        return httpx.Response(503 if send_failure else 202)

    tool = make_tool(handle)

    @tool.on_completed
    async def on_completed(result, metadata):
        completed.append(result)

    service = make_service(tool, tmp_path)
    results = [result async for result in service.execute_tool(tool.name, {"query": "question"}, {})]
    await asyncio.wait_for(sent.wait(), 1)
    with pytest.raises(asyncio.CancelledError):
        await requests[0]
    service._start_deferred_callbacks([ToolCall(name=tool.name, result=results[0])])
    await asyncio.gather(*tool._background_tasks, return_exceptions=True)
    assert completed == []
    assert not tool._running_tasks
    assert not tool._background_tasks


@pytest.mark.asyncio
async def test_canceled_callback_does_not_override_done(make_tool):
    def handle(request):
        request_id = json.loads(request.content)["request_id"]
        assert tool.receive_response(request_id, "final answer", status="done") is True
        assert tool.receive_response(request_id, status="canceled") is False
        return httpx.Response(202)

    tool = make_tool(handle)
    assert (await tool.invoke_grokbot("question", {"task_id": "done"}))["answer"] == "final answer"
    assert tool.receive_response("done", status="canceled") is False
    assert tool.receive_response("unknown", status="canceled") is False
    assert not tool._running_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_close", [False, True])
async def test_close_cancels_requests_and_deferred_completion(make_tool, tmp_path, cancel_before_close):
    sent = asyncio.Event()
    completed = []

    def handle(request):
        sent.set()
        return httpx.Response(202)

    tool = make_tool(handle)

    @tool.on_completed
    async def on_completed(result, metadata):
        completed.append(result)

    service = make_service(tool, tmp_path)
    results = [result async for result in service.execute_tool(tool.name, {"query": "question"}, {})]
    if not cancel_before_close:
        service._start_deferred_callbacks([ToolCall(name=tool.name, result=results[0])])
    await asyncio.wait_for(sent.wait(), 1)
    active = {entry["task"] for entry in tool._running_tasks.values()} | tool._background_tasks
    assert active
    if cancel_before_close:
        assert tool.receive_response(results[0].task_id, status="canceled") is True
        assert tool._running_tasks[results[0].task_id]["future"].cancelled()
        # Close must still find and join the canceled request before it gets to run cleanup.
        assert not tool._background_tasks
        assert all(not task.done() for task in active)
    await tool.aclose()
    assert all(task.done() for task in active)
    assert not tool._background_tasks
    assert not tool._running_tasks
    assert completed == []
