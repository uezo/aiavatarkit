import asyncio
import pytest
from aiavatar.sts.llm.base import Tool, ToolCallResult, ToolCall, LLMServiceDummy


def make_tool(func, **kwargs):
    return Tool(
        name="test_tool",
        spec={
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "test",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        },
        func=func,
        **kwargs
    )


def make_service_with_tool(tool):
    svc = LLMServiceDummy(system_prompt="test", model="dummy", response_text="ok")
    svc.add_tool(tool, use_original=True)
    return svc


# --- ToolCallResult / ToolCall ---

def test_tool_call_result_task_id():
    tr = ToolCallResult(data={"msg": "hello"}, task_id="abc-123")
    assert tr.task_id == "abc-123"
    assert tr.data == {"msg": "hello"}


def test_tool_call_to_dict_with_task_id():
    tr = ToolCallResult(data={"msg": "hello"}, task_id="abc-123")
    tc = ToolCall(id="1", name="test", arguments='{"q":"hi"}', result=tr)
    d = tc.to_dict()
    assert d["result"]["task_id"] == "abc-123"
    assert d["result"]["data"] == {"msg": "hello"}


def test_tool_call_to_dict_without_task_id():
    tr = ToolCallResult(data={"msg": "hello"})
    tc = ToolCall(id="1", name="test", arguments='{"q":"hi"}', result=tr)
    d = tc.to_dict()
    assert "task_id" not in d["result"]


# --- Synchronous execution (no on_completed) ---

@pytest.mark.asyncio
async def test_sync_execution():
    async def my_func(query: str):
        return {"answer": query}

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hello"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert len(results) == 1
    assert results[0].data == {"answer": "hello"}
    assert results[0].is_final is True
    assert results[0].task_id is None


# --- Immediate background (on_completed, no timeout) ---

@pytest.mark.asyncio
async def test_immediate_background():
    completed = []

    async def my_func(query: str):
        await asyncio.sleep(0.05)
        return {"answer": query}

    tool = make_tool(my_func)

    @tool.on_completed
    async def handle_completed(result, metadata):
        completed.append({"result": result, "metadata": metadata})

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hello"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    # Should return immediately with immediate_message and task_id
    assert len(results) == 1
    assert results[0].data["message"] == tool.immediate_message
    assert results[0].data["task_id"] == results[0].task_id
    assert results[0].task_id is not None
    assert results[0].is_final is True
    assert results[0].deferred_callback is not None

    # on_completed not called yet (deferred)
    assert len(completed) == 0

    # Simulate caller starting deferred callbacks after response completes
    tc = ToolCall(id="1", name="test_tool", arguments='{"query": "hello"}', result=results[0])
    svc._start_deferred_callbacks([tc])

    # Wait for background task
    await asyncio.sleep(0.15)

    assert len(completed) == 1
    assert completed[0]["result"] == {"answer": "hello"}
    assert completed[0]["metadata"]["task_id"] == results[0].task_id
    assert "submitted_at" in completed[0]["metadata"]
    assert completed[0]["metadata"]["arguments"] == {"query": "hello"}


# --- on_submitted callback ---

@pytest.mark.asyncio
async def test_on_submitted_called():
    submitted = []
    completed = []

    async def my_func(query: str):
        return {"answer": query}

    tool = make_tool(my_func)

    @tool.on_submitted
    async def handle_submitted(task_id, metadata):
        submitted.append({"task_id": task_id, "metadata": metadata})

    @tool.on_completed
    async def handle_completed(result, metadata):
        completed.append(result)

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hi"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    # on_submitted should have been called before returning
    assert len(submitted) == 1
    assert submitted[0]["task_id"] == results[0].task_id
    assert submitted[0]["metadata"]["context_id"] == "c1"


# --- Background with timeout: completes in time ---

@pytest.mark.asyncio
async def test_timeout_completes_in_time():
    completed = []

    async def my_func(query: str):
        await asyncio.sleep(1)
        return {"answer": query}

    tool = make_tool(my_func, background_timeout=5.0)

    @tool.on_completed
    async def handle_completed(result, metadata):
        completed.append(result)

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "fast"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    # Should return the actual result (sync path)
    assert len(results) == 1
    assert results[0].data == {"answer": "fast"}
    assert results[0].task_id is None
    assert results[0].is_final is True

    # on_completed should NOT be called (sync return)
    await asyncio.sleep(0.1)
    assert len(completed) == 0


# --- Background with timeout: times out ---

@pytest.mark.asyncio
async def test_timeout_falls_back_to_background():
    completed = []

    async def my_func(query: str):
        await asyncio.sleep(5)
        return {"answer": query}

    tool = make_tool(my_func, background_timeout=2.0)

    @tool.on_completed
    async def handle_completed(result, metadata):
        completed.append({"result": result, "metadata": metadata})

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "slow"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    # Should return immediate_message (timed out)
    assert len(results) == 1
    assert results[0].data["message"] == tool.immediate_message
    assert results[0].data["task_id"] == results[0].task_id
    assert results[0].task_id is not None
    assert results[0].deferred_callback is not None

    # on_completed not called yet (deferred)
    assert len(completed) == 0

    # Simulate caller starting deferred callbacks after response completes
    tc = ToolCall(id="1", name="test_tool", arguments='{"query": "slow"}', result=results[0])
    svc._start_deferred_callbacks([tc])

    # Wait for background to finish
    await asyncio.sleep(5)

    assert len(completed) == 1
    assert completed[0]["result"] == {"answer": "slow"}


# --- Error handling in background ---

@pytest.mark.asyncio
async def test_background_error_calls_on_completed_with_none():
    completed = []

    async def my_func(query: str):
        raise RuntimeError("boom")

    tool = make_tool(my_func)

    @tool.on_completed
    async def handle_completed(result, metadata):
        completed.append(result)

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "fail"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert results[0].task_id is not None
    assert results[0].deferred_callback is not None

    # Simulate caller starting deferred callbacks after response completes
    tc = ToolCall(id="1", name="test_tool", arguments='{"query": "fail"}', result=results[0])
    svc._start_deferred_callbacks([tc])

    await asyncio.sleep(0.1)

    assert len(completed) == 1
    assert completed[0] is None


# --- metadata passed to func ---

@pytest.mark.asyncio
async def test_metadata_passed_to_func():
    received_metadata = []

    async def my_func(query: str, metadata: dict = None):
        received_metadata.append(metadata)
        return {"answer": query}

    tool = make_tool(my_func)

    @tool.on_completed
    async def handle_completed(result, metadata):
        pass

    svc = make_service_with_tool(tool)

    async for _ in svc.execute_tool("test_tool", {"query": "hi"}, {"context_id": "c1", "user_id": "u1"}):
        pass

    await asyncio.sleep(0.1)

    assert len(received_metadata) == 1
    assert received_metadata[0]["context_id"] == "c1"
    assert received_metadata[0]["user_id"] == "u1"
    # Enriched metadata includes task_id and submitted_at for background tools
    assert "task_id" in received_metadata[0]
    assert "submitted_at" in received_metadata[0]


# --- Background tasks are cleaned up ---

@pytest.mark.asyncio
async def test_background_tasks_cleaned_up():
    async def my_func(query: str):
        await asyncio.sleep(0.05)
        return {"answer": query}

    tool = make_tool(my_func)

    @tool.on_completed
    async def handle_completed(result, metadata):
        pass

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hi"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    # Task should NOT be started yet (deferred)
    assert len(tool._background_tasks) == 0
    assert results[0].deferred_callback is not None

    # Simulate caller starting deferred callbacks after response completes
    tc = ToolCall(id="1", name="test_tool", arguments='{"query": "hi"}', result=results[0])
    svc._start_deferred_callbacks([tc])

    # Task should be in the set while running
    assert len(tool._background_tasks) == 1

    await asyncio.sleep(0.15)

    # Task should be cleaned up after completion
    assert len(tool._background_tasks) == 0


# --- Custom immediate_message ---

@pytest.mark.asyncio
async def test_custom_immediate_message():
    async def my_func(query: str):
        return {"answer": query}

    tool = make_tool(my_func, immediate_message="Working on it...")

    @tool.on_completed
    async def handle_completed(result, metadata):
        pass

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hi"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert results[0].data["message"] == "Working on it..."


# --- Async generator (streaming, no on_completed) ---

@pytest.mark.asyncio
async def test_async_generator_execution():
    async def my_func(query: str):
        yield {"step": 1}
        yield {"step": 2}
        yield ({"step": 3, "answer": query}, True)

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hello"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    # dict yields -> is_final=False
    assert results[0].data == {"step": 1}
    assert results[0].is_final is False
    assert results[1].data == {"step": 2}
    assert results[1].is_final is False
    # tuple yield -> (data, is_final)
    assert results[2].data == {"step": 3, "answer": "hello"}
    assert results[2].is_final is True


@pytest.mark.asyncio
async def test_async_generator_text_streaming():
    async def my_func(query: str):
        yield "processing..."
        yield "done!"

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hello"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert results[0].text == "processing..."
    assert results[0].is_final is False
    assert results[1].text == "done!"
    assert results[1].is_final is False


# --- ToolCallResult returned directly from func ---

@pytest.mark.asyncio
async def test_tool_call_result_direct_return():
    async def my_func(query: str):
        return ToolCallResult(data={"answer": query}, is_final=True, task_id="custom-id")

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hello"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert len(results) == 1
    assert results[0].data == {"answer": "hello"}
    assert results[0].task_id == "custom-id"
    assert results[0].is_final is True


# --- Async generator yielding ToolCallResult directly ---

@pytest.mark.asyncio
async def test_async_generator_yields_tool_call_result():
    async def my_func(query: str):
        yield ToolCallResult(data={"progress": 50}, is_final=False)
        yield ToolCallResult(data={"answer": query}, is_final=True)

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "hello"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert results[0].data == {"progress": 50}
    assert results[0].is_final is False
    assert results[1].data == {"answer": "hello"}
    assert results[1].is_final is True


# --- structured_content ---

def test_tool_call_result_structured_content():
    tr = ToolCallResult(
        data={"raw": "data"},
        structured_content={"key": "value", "nested": {"a": 1}}
    )
    assert tr.structured_content == {"key": "value", "nested": {"a": 1}}
    assert tr.data == {"raw": "data"}


def test_tool_call_result_structured_content_default_none():
    tr = ToolCallResult(data={"msg": "hello"})
    assert tr.structured_content is None


def test_tool_call_to_dict_with_structured_content():
    tr = ToolCallResult(
        data={"msg": "hello"},
        structured_content={"ui_data": [1, 2, 3]}
    )
    tc = ToolCall(id="1", name="test", arguments='{"q":"hi"}', result=tr)
    d = tc.to_dict()
    assert d["result"]["structured_content"] == {"ui_data": [1, 2, 3]}


def test_tool_call_to_dict_without_structured_content():
    tr = ToolCallResult(data={"msg": "hello"})
    tc = ToolCall(id="1", name="test", arguments='{"q":"hi"}', result=tr)
    d = tc.to_dict()
    assert "structured_content" not in d["result"]


@pytest.mark.asyncio
async def test_structured_content_direct_return():
    async def my_func(query: str):
        return ToolCallResult(
            data={"answer": query},
            structured_content={"display": {"title": query, "type": "info"}}
        )

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "weather"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert len(results) == 1
    assert results[0].data == {"answer": "weather"}
    assert results[0].structured_content == {"display": {"title": "weather", "type": "info"}}


@pytest.mark.asyncio
async def test_structured_content_async_generator():
    async def my_func(query: str):
        yield ToolCallResult(data={"progress": 50}, is_final=False, structured_content={"status": "loading"})
        yield ToolCallResult(data={"answer": query}, is_final=True, structured_content={"status": "complete", "results": [1, 2, 3]})

    tool = make_tool(my_func)
    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "test"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert results[0].structured_content == {"status": "loading"}
    assert results[1].structured_content == {"status": "complete", "results": [1, 2, 3]}


@pytest.mark.asyncio
async def test_structured_content_background_timeout_completes():
    completed = []

    async def my_func(query: str):
        await asyncio.sleep(0.1)
        return ToolCallResult(
            data={"answer": query},
            structured_content={"card": {"title": query}}
        )

    tool = make_tool(my_func, background_timeout=5.0)

    @tool.on_completed
    async def handle_completed(result, metadata):
        completed.append(result)

    svc = make_service_with_tool(tool)

    results = []
    async for tr in svc.execute_tool("test_tool", {"query": "fast"}, {"context_id": "c1", "user_id": "u1"}):
        results.append(tr)

    assert len(results) == 1
    assert results[0].data == {"answer": "fast"}
    assert results[0].structured_content == {"card": {"title": "fast"}}
