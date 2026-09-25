import asyncio
import hashlib
import json
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from aiavatar.sts.llm import LLMServiceDummy, Tool
from aiavatar.sts.llm.tools import memory as memory_module
from aiavatar.sts.llm.tools.memory import MemoryTool


USER_ID = "user-one"
METADATA = {"user_id": USER_ID, "context_id": "context-one"}


def decision(kind, *operations, status="update", reason="Requested update"):
    return {
        "status": status,
        "kind": kind,
        "operations": list(operations),
        "reason": reason,
    }


def add(content):
    return {"action": "add", "content": content}


class FakeUpdateClient:
    """An injected client that never constructs an SDK or accesses the network."""

    def __init__(self, *responses, responder=None):
        self.responses = list(responses)
        self.responder = responder
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = await self.responder(kwargs) if self.responder else self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_tool(tmp_path, client=None, **kwargs):
    options = {
        "character_memory_dir": tmp_path / "character",
        "update_client": client or FakeUpdateClient(),
        "update_model": "test-memory-model",
    }
    options.update(kwargs)
    return MemoryTool(**options)


def contents(document):
    return [entry["content"] for entry in document["entries"]]


def json_snapshot(tmp_path):
    return {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*.json")}


def memory_debug_events(caplog):
    prefix = "MemoryTool: "
    return [
        json.loads(record.getMessage()[len(prefix):])
        for record in caplog.records
        if record.name == memory_module.__name__ and record.getMessage().startswith(prefix)
    ]


@pytest.mark.asyncio
async def test_debug_is_silent_by_default(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=memory_module.__name__)
    client = FakeUpdateClient(decision("shared", add("I prefer tea.")))
    tool = make_tool(tmp_path, client)

    result = await tool.save_memory("Remember my drink preference.", metadata=METADATA)

    assert result["status"] == "saved"
    assert tool.debug is False
    assert not [record for record in caplog.records if record.name == memory_module.__name__]


@pytest.mark.asyncio
async def test_debug_logs_correlated_input_request_raw_decision_and_result(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=memory_module.__name__)
    plan = decision("shared", add("I prefer tea."), reason="The user stated a drink preference.")
    raw_response = json.dumps(plan)

    class PrivateFakeClient(FakeUpdateClient):
        api_key = "TEST-ONLY-CLIENT-KEY"

        def __repr__(self):
            return "TEST-ONLY-CLIENT-REPR"

    client = PrivateFakeClient(raw_response)
    tool = make_tool(tmp_path, client, debug=True, reasoning_effort="high")
    identifiers = {
        **METADATA,
        "session_id": "session-one",
        "transaction_id": "transaction-one",
        "task_id": "task-one",
        "channel": "websocket",
    }
    metadata = {**identifiers, "api_key": "TEST-ONLY-METADATA-KEY", "extra": "TEST-ONLY-METADATA-EXTRA"}
    content = "Remember my drink preference."

    result = await tool.save_memory(content, metadata=metadata)

    assert result["status"] == "saved"
    events = memory_debug_events(caplog)
    by_event = {event["event"]: event for event in events}
    assert {"input", "request", "raw_response", "decision", "result"} <= by_event.keys()
    operation_ids = {event["operation_id"] for event in events}
    assert len(operation_ids) == 1 and next(iter(operation_ids))
    for event in events:
        for key, value in identifiers.items():
            assert event[key] == value
        assert "api_key" not in event
        assert "extra" not in event
    assert by_event["input"]["content"] == content
    assert by_event["input"]["forget"] is False
    assert by_event["input"]["model"] == "test-memory-model"
    assert by_event["input"]["reasoning_effort"] == "high"
    assert by_event["input"]["character_memory_dir"] == str(tmp_path / "character")
    assert by_event["input"]["shared_memory_dir"] == str(tmp_path / "shared")
    assert by_event["request"]["request"] == client.calls[0]
    assert by_event["raw_response"]["content"] == raw_response
    assert by_event["decision"]["plan"] == plan
    assert by_event["result"]["result"] == result
    for name in ("request", "raw_response", "decision", "result"):
        assert by_event[name]["attempt"] == 1
    logged = json.dumps(events)
    for private_value in (
        "TEST-ONLY-CLIENT-KEY", "TEST-ONLY-CLIENT-REPR",
        "TEST-ONLY-METADATA-KEY", "TEST-ONLY-METADATA-EXTRA",
    ):
        assert private_value not in logged


@pytest.mark.asyncio
async def test_debug_logs_invalid_raw_response_before_error_without_saving(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=memory_module.__name__)
    invalid_response = "This is not a JSON response."
    tool = make_tool(tmp_path, FakeUpdateClient(invalid_response), debug=True)

    result = await tool.save_memory("Remember this.", metadata=METADATA)

    assert result["status"] == "error"
    assert result["error"] == "invalid_update"
    events = memory_debug_events(caplog)
    raw = next(event for event in events if event["event"] == "raw_response")
    error = next(event for event in events if event["event"] == "error")
    assert raw["content"] == invalid_response
    assert raw["attempt"] == error["attempt"] == 1
    assert events.index(raw) < events.index(error)
    assert error["exception_type"] == "JSONDecodeError"
    assert error["stage"]
    assert len({event["operation_id"] for event in events}) == 1
    assert not any(event["event"] == "decision" for event in events)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_debug_error_does_not_log_updater_exception_text(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=memory_module.__name__)
    tool = make_tool(
        tmp_path, FakeUpdateClient(RuntimeError("TEST-ONLY-UPDATER-EXCEPTION-SECRET")), debug=True,
    )

    result = await tool.save_memory("Remember this.", metadata=METADATA)

    assert result["status"] == "error"
    events = memory_debug_events(caplog)
    error = next(event for event in events if event["event"] == "error")
    assert error["exception_type"] == "RuntimeError"
    assert "TEST-ONLY-UPDATER-EXCEPTION-SECRET" not in caplog.text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_debug_concurrent_requests_keep_operation_and_user_correlation(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=memory_module.__name__)
    both_started = asyncio.Event()
    arrivals = 0

    async def respond(kwargs):
        nonlocal arrivals
        request = json.loads(kwargs["messages"][-1]["content"])
        arrivals += 1
        if arrivals == 2:
            both_started.set()
        await both_started.wait()
        return decision("shared", add(request["new_information"]))

    tool = make_tool(tmp_path, FakeUpdateClient(responder=respond), debug=True)
    changes = {"user-one": "I prefer tea.", "user-two": "I prefer coffee."}

    results = await asyncio.wait_for(asyncio.gather(*[
        tool.save_memory(content, metadata={"user_id": user_id, "context_id": f"context-{user_id}"})
        for user_id, content in changes.items()
    ]), timeout=5)

    assert all(result["status"] == "saved" for result in results)
    events = memory_debug_events(caplog)
    operation_ids = {event["operation_id"] for event in events}
    assert len(operation_ids) == 2
    seen_users = set()
    for operation_id in operation_ids:
        operation_events = [event for event in events if event["operation_id"] == operation_id]
        input_event = next(event for event in operation_events if event["event"] == "input")
        user_id = input_event["user_id"]
        seen_users.add(user_id)
        assert input_event["content"] == changes[user_id]
        assert {event["user_id"] for event in operation_events} == {user_id}
        assert {event["context_id"] for event in operation_events} == {f"context-{user_id}"}
        request_event = next(event for event in operation_events if event["event"] == "request")
        request_data = json.loads(request_event["request"]["messages"][-1]["content"])
        assert request_data["new_information"] == changes[user_id]
        result_event = next(event for event in operation_events if event["event"] == "result")
        assert contents(result_event["result"]) == [changes[user_id]]
    assert seen_users == set(changes)


@pytest.mark.asyncio
async def test_initialization_and_missing_reads_do_not_create_files(tmp_path):
    client = FakeUpdateClient()
    tool = make_tool(tmp_path, client)

    assert await tool.read_memory(USER_ID, "character") == {"revision": 0, "entries": []}
    assert await tool.read_memory(USER_ID, "shared") == {"revision": 0, "entries": []}
    assert await tool.get_prompt(USER_ID) == ""
    assert list(tmp_path.iterdir()) == []
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("character_directory_exists", [False, True], ids=["missing-directory", "missing-user-file"])
async def test_shared_only_startup_loads_shared_without_creating_character_memory(
    tmp_path, character_directory_exists,
):
    character_dir = tmp_path / "new-character"
    if character_directory_exists:
        character_dir.mkdir()
    shared_path = tmp_path / "shared" / f"{USER_ID}.json"
    shared_path.parent.mkdir()
    shared_document = {
        "revision": 4,
        "entries": [{"id": "drink", "content": "I prefer tea."}],
    }
    shared_path.write_text(json.dumps(shared_document, indent=2) + "\n", encoding="utf-8")
    shared_bytes = shared_path.read_bytes()
    paths_before = set(tmp_path.rglob("*"))
    client = FakeUpdateClient()
    tool = MemoryTool(character_memory_dir=character_dir, update_client=client)

    prompt = await tool.get_prompt(USER_ID)

    assert "## Shared memories for this user" in prompt
    assert "- I prefer tea." in prompt
    assert "## Character-specific memories" not in prompt
    assert await tool.read_memory(USER_ID, "character") == {"revision": 0, "entries": []}
    assert await tool.read_memory(USER_ID, "shared") == shared_document
    assert character_dir.exists() == character_directory_exists
    assert not (character_dir / f"{USER_ID}.json").exists()
    assert set(tmp_path.rglob("*")) == paths_before
    assert shared_path.read_bytes() == shared_bytes
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("character_dir", "expected_shared_dir"), [
    ("memories/kuroha", "memories/shared"),
    ("memories/kuroha/", "memories/shared"),
    ("kuroha", "shared"),
], ids=["omitted-shared", "trailing-slash", "simple-relative"])
async def test_shared_directory_defaults_to_character_parent(
    tmp_path, monkeypatch, character_dir, expected_shared_dir,
):
    monkeypatch.chdir(tmp_path)
    client = FakeUpdateClient(decision("shared", add("I prefer tea.")))
    tool = MemoryTool(character_memory_dir=character_dir, update_client=client)

    assert tool.character_memory_dir == Path(character_dir)
    assert tool.shared_memory_dir == Path(expected_shared_dir)
    assert not tool.shared_memory_dir.is_absolute()
    assert list(tmp_path.iterdir()) == []
    assert await tool.get_prompt(USER_ID) == ""
    assert list(tmp_path.iterdir()) == []
    assert client.calls == []

    result = await tool.save_memory("Remember my drink preference.", metadata=METADATA)

    assert result["status"] == "saved"
    assert (tmp_path / expected_shared_dir / f"{USER_ID}.json").is_file()
    assert not (tmp_path / character_dir).exists()


@pytest.mark.asyncio
async def test_explicit_shared_directory_overrides_default_without_initialization_io(tmp_path):
    client = FakeUpdateClient(decision("shared", add("I prefer tea.")))
    shared_dir = tmp_path / "elsewhere" / "user-facts"
    tool = MemoryTool(
        character_memory_dir=tmp_path / "memories" / "kuroha",
        shared_memory_dir=shared_dir,
        update_client=client,
    )

    assert tool.shared_memory_dir == shared_dir
    assert list(tmp_path.iterdir()) == []

    result = await tool.save_memory("Remember my drink preference.", metadata=METADATA)

    assert result["status"] == "saved"
    assert (shared_dir / f"{USER_ID}.json").is_file()
    assert not (tmp_path / "memories").exists()


@pytest.mark.asyncio
async def test_equal_character_and_shared_directories_fail_before_updater_or_write(tmp_path):
    client = FakeUpdateClient()
    directory = tmp_path / "memory"
    tool = MemoryTool(
        character_memory_dir=directory,
        shared_memory_dir=directory,
        update_client=client,
    )

    with pytest.raises(ValueError):
        await tool.read_memory(USER_ID, "shared")
    result = await tool.save_memory("Remember my drink preference.", metadata=METADATA)

    assert result["status"] == "error"
    assert client.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["character", "shared"])
async def test_add_uses_updater_classification_and_safe_user_filename(tmp_path, kind):
    client = FakeUpdateClient(decision(kind, add("Keep this memory.")))
    tool = make_tool(tmp_path, client)
    user_id = "../ユーザー/one"

    result = await tool.save_memory("Remember this.", metadata={"user_id": user_id})

    assert result["status"] == "saved"
    assert result["kind"] == kind
    assert result["revision"] == 1
    document = await tool.read_memory(user_id, kind)
    assert result["entries"] == document["entries"]
    assert contents(document) == ["Keep this memory."]
    assert document["entries"][0]["id"]
    filename = memory_module._user_filename(user_id)
    assert (tmp_path / kind / filename).is_file()
    assert "/" not in filename and "\\" not in filename
    assert filename.startswith("%2E%2E%2F")
    assert list(tmp_path.rglob("*.json")) == [tmp_path / kind / filename]
    other_kind = "shared" if kind == "character" else "character"
    assert await tool.read_memory(user_id, other_kind) == {"revision": 0, "entries": []}
    assert client.calls[0]["model"] == "test-memory-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [
    "uezo", "user-one", "b4f601e6-a022-4ff3-8cd8-b377c2ec9b09",
])
async def test_simple_user_ids_have_readable_filenames(tmp_path, user_id):
    client = FakeUpdateClient(decision("character", add("User-specific memory.")))
    tool = make_tool(tmp_path, client)

    result = await tool.save_memory("Remember this.", metadata={"user_id": user_id})

    assert result["status"] == "saved"
    assert (tmp_path / "character" / f"{user_id}.json").is_file()
    assert contents(await tool.read_memory(user_id, "character")) == ["User-specific memory."]


def test_filename_encoding_preserves_distinct_ids_on_case_insensitive_filesystems():
    user_ids = [
        "uezo", "Uezo", "UEZO", "user-one", "user_one", "user.one",
        "user/one", "user\\one", "user%2fone", "user%2Fone", "user~one",
        "user", "user.previous", "user.lock", "user.json", "../user",
    ]
    filenames = [memory_module._user_filename(user_id) for user_id in user_ids]

    assert len({filename.casefold() for filename in filenames}) == len(user_ids)
    assert memory_module._user_filename("Uezo") == "%55ezo.json"
    assert memory_module._user_filename("user/one") == "user%2Fone.json"
    assert memory_module._user_filename("user%2fone") == "user%252fone.json"
    assert all("/" not in filename and "\\" not in filename for filename in filenames)
    assert all("." not in filename[:-5] for filename in filenames)
    assert "user.previous.json" not in filenames
    assert "user.lock.json" not in filenames


@pytest.mark.parametrize(("user_id", "filename"), [
    ("con", "%63on.json"), ("prn", "%70rn.json"), ("aux", "%61ux.json"),
    ("nul", "%6Eul.json"), ("com1", "%63om1.json"), ("lpt9", "%6Cpt9.json"),
])
def test_windows_device_names_are_escaped(user_id, filename):
    assert memory_module._user_filename(user_id) == filename


@pytest.mark.asyncio
async def test_long_user_ids_have_bounded_distinct_filenames(tmp_path):
    user_ids = ["長いユーザー名" * 40 + ending for ending in ("a", "b")]
    client = FakeUpdateClient(*[
        decision("character", add(f"Memory {index}.")) for index in range(2)
    ])
    tool = make_tool(tmp_path, client)
    filenames = []

    for index, user_id in enumerate(user_ids):
        result = await tool.save_memory("Remember this.", metadata={"user_id": user_id})
        filename = memory_module._user_filename(user_id)
        filenames.append(filename)
        assert result["status"] == "saved"
        assert len(filename.encode("utf-8")) <= 185
        assert filename.endswith("~" + hashlib.sha256(user_id.encode("utf-8")).hexdigest() + ".json")
        assert (tmp_path / "character" / filename).is_file()
        assert contents(await tool.read_memory(user_id, "character")) == [f"Memory {index}."]
    assert filenames[0].casefold() != filenames[1].casefold()


@pytest.mark.asyncio
async def test_hex_user_id_does_not_read_or_overwrite_another_users_legacy_memory(tmp_path):
    original_user = "uezo"
    hex_user_id = hashlib.sha256(original_user.encode("utf-8")).hexdigest()
    original_document = {"revision": 3, "entries": [{"id": "original", "content": "Private original memory."}]}
    legacy_path = tmp_path / "character" / f"{hex_user_id}.json"
    legacy_path.parent.mkdir()
    legacy_path.write_text(json.dumps(original_document), encoding="utf-8")
    original_bytes = legacy_path.read_bytes()
    client = FakeUpdateClient(decision("character", add("Hex user's own memory.")))
    tool = make_tool(tmp_path, client)

    assert await tool.read_memory(hex_user_id, "character") == {"revision": 0, "entries": []}
    result = await tool.save_memory("Remember my fact.", metadata={"user_id": hex_user_id})

    assert result["status"] == "saved"
    assert memory_module._user_filename(hex_user_id).startswith("%")
    assert (legacy_path.parent / memory_module._user_filename(hex_user_id)).is_file()
    assert contents(await tool.read_memory(hex_user_id, "character")) == ["Hex user's own memory."]
    assert await tool.read_memory(original_user, "character") == original_document
    assert legacy_path.read_bytes() == original_bytes
    updater_memories = json.loads(client.calls[0]["messages"][-1]["content"])["memories"]
    assert updater_memories["character"]["entries"] == []


@pytest.mark.asyncio
async def test_legacy_memory_stays_at_legacy_path_when_read_and_updated(tmp_path):
    digest = hashlib.sha256(USER_ID.encode("utf-8")).hexdigest()
    legacy_path = tmp_path / "character" / f"{digest}.json"
    legacy_path.parent.mkdir()
    before = {
        "revision": 7,
        "entries": [
            {"id": "name", "content": "Call me Alex."},
            {"id": "drink", "content": "I prefer tea."},
        ],
    }
    legacy_path.write_text(json.dumps(before), encoding="utf-8")
    client = FakeUpdateClient(decision("character", {
        "action": "replace", "id": "name", "content": "Call me Sam.",
    }))
    tool = make_tool(tmp_path, client)

    assert await tool.read_memory(USER_ID, "character") == before
    assert "Call me Alex." in await tool.get_prompt(USER_ID)
    result = await tool.save_memory("Call me Sam.", metadata=METADATA)

    assert result["status"] == "saved"
    assert result["revision"] == 8
    assert json.loads(legacy_path.read_text(encoding="utf-8")) == {
        "revision": 8,
        "entries": [{"id": "name", "content": "Call me Sam."}, before["entries"][1]],
    }
    backup = legacy_path.with_suffix(".previous.json")
    assert json.loads(backup.read_text(encoding="utf-8")) == before
    assert not (legacy_path.parent / f"{USER_ID}.json").exists()
    assert not (legacy_path.parent / f"{USER_ID}.previous.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_content", [
    json.dumps({"revision": 2, "entries": [{"id": "new", "content": "Readable memory."}]}),
    "{invalid JSON",
], ids=["valid-new", "corrupt-new"])
async def test_both_readable_and_legacy_files_fail_without_fallback_or_write(tmp_path, new_content):
    directory = tmp_path / "character"
    directory.mkdir()
    digest = hashlib.sha256(USER_ID.encode("utf-8")).hexdigest()
    (directory / f"{digest}.json").write_text(json.dumps({
        "revision": 1, "entries": [{"id": "old", "content": "Legacy memory."}],
    }), encoding="utf-8")
    (directory / f"{USER_ID}.json").write_text(new_content, encoding="utf-8")
    before = json_snapshot(tmp_path)
    client = FakeUpdateClient()
    tool = make_tool(tmp_path, client)

    with pytest.raises(ValueError):
        await tool.read_memory(USER_ID, "character")
    result = await tool.save_memory("Remember this.", metadata=METADATA)

    assert result["status"] == "error"
    assert client.calls == []
    assert json_snapshot(tmp_path) == before
    assert list(tmp_path.rglob("*.lock")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True], ids=["readable", "legacy"])
async def test_memory_file_symlinks_are_rejected_before_updater_or_write(tmp_path, legacy):
    target = tmp_path / "target.json"
    target.write_text(json.dumps({
        "revision": 1, "entries": [{"id": "target", "content": "Must remain untouched."}],
    }), encoding="utf-8")
    directory = tmp_path / "character"
    directory.mkdir()
    stem = hashlib.sha256(USER_ID.encode("utf-8")).hexdigest() if legacy else USER_ID
    link = directory / f"{stem}.json"
    link.symlink_to(target)
    original_bytes = target.read_bytes()
    client = FakeUpdateClient()
    tool = make_tool(tmp_path, client)

    with pytest.raises(ValueError):
        await tool.read_memory(USER_ID, "character")
    result = await tool.save_memory("Remember this.", metadata=METADATA)

    assert result["status"] == "error"
    assert client.calls == []
    assert target.read_bytes() == original_bytes
    assert link.is_symlink()
    assert list(tmp_path.rglob("*.lock")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {}, {"reasoning_effort": None}, {"reasoning_effort": "low"}, {"reasoning_effort": "high"},
], ids=["default", "explicit-none", "low", "high"])
async def test_default_model_and_optional_reasoning_effort(tmp_path, options):
    client = FakeUpdateClient(decision("character", add("Use polite speech.")))
    tool = MemoryTool(
        character_memory_dir=tmp_path / "character",
        shared_memory_dir=tmp_path / "shared",
        update_client=client,
        **options,
    )

    result = await tool.save_memory("Remember my speech preference.", metadata=METADATA)

    assert result["status"] == "saved"
    assert client.calls[0]["model"] == "gpt-6-sol"
    if options.get("reasoning_effort") is None:
        assert "reasoning_effort" not in client.calls[0]
    else:
        assert client.calls[0]["reasoning_effort"] == options["reasoning_effort"]


@pytest.mark.asyncio
async def test_replace_preserves_unrelated_entry_and_previous_revision(tmp_path):
    client = FakeUpdateClient(decision("character", add("Call me Alex."), add("I prefer tea.")))
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember my name and drink preference.", metadata=METADATA)
    before = await tool.read_memory(USER_ID, "character")
    name_entry, unrelated_entry = before["entries"]
    client.responses.append(decision("character", {
        "action": "replace", "id": name_entry["id"], "content": "Call me Sam.",
    }))

    result = await tool.save_memory("Call me Sam from now on.", metadata=METADATA)

    assert result["status"] == "saved"
    after = await tool.read_memory(USER_ID, "character")
    assert after["revision"] == before["revision"] + 1
    assert after["entries"] == [
        {"id": name_entry["id"], "content": "Call me Sam."}, unrelated_entry,
    ]
    backup = tmp_path / "character" / f"{USER_ID}.previous.json"
    assert json.loads(backup.read_text(encoding="utf-8")) == before
    updater_input = json.dumps(client.calls[-1]["messages"], ensure_ascii=False)
    assert "Call me Alex." in updater_input
    assert "I prefer tea." in updater_input
    assert "Call me Sam from now on." in updater_input


@pytest.mark.asyncio
async def test_condition_specific_replace_does_not_erase_other_conditions(tmp_path):
    client = FakeUpdateClient(decision("character", add("Use polite speech at work."), add("Use casual speech at home.")))
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember the speaking rules.", metadata=METADATA)
    before = await tool.read_memory(USER_ID, "character")
    client.responses.append(decision("character", {
        "action": "replace",
        "id": before["entries"][0]["id"],
        "content": "Use concise polite speech at work.",
    }))

    result = await tool.save_memory("At work, also keep speech concise.", metadata=METADATA)

    assert result["status"] == "saved"
    after = await tool.read_memory(USER_ID, "character")
    assert contents(after) == ["Use concise polite speech at work.", "Use casual speech at home."]
    assert after["entries"][1] == before["entries"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unchanged", "needs_clarification"])
async def test_nonupdate_decision_preserves_files_and_revision(tmp_path, status):
    client = FakeUpdateClient(decision("character", add("I prefer tea.")))
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember my preference.", metadata=METADATA)
    before = json_snapshot(tmp_path)
    client.responses.append(decision("character", status=status, reason="No change is needed."))

    result = await tool.save_memory("Remember my preference.", metadata=METADATA)

    assert result["status"] == status
    assert json_snapshot(tmp_path) == before
    assert (await tool.read_memory(USER_ID, "character"))["revision"] == 1


@pytest.mark.asyncio
async def test_delete_requires_explicit_forget_and_preserves_other_entries(tmp_path):
    client = FakeUpdateClient(decision("character", add("Call me Alex."), add("I prefer tea.")))
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember these details.", metadata=METADATA)
    before = await tool.read_memory(USER_ID, "character")
    delete = decision("character", {"action": "delete", "id": before["entries"][0]["id"]})
    client.responses.extend([delete, delete])
    files_before = json_snapshot(tmp_path)

    denied = await tool.save_memory("Forget my name.", metadata=METADATA)

    assert denied["status"] == "error"
    assert json_snapshot(tmp_path) == files_before
    result = await tool.save_memory("Forget my name.", metadata=METADATA, forget=True)
    assert result["status"] == "saved"
    assert await tool.read_memory(USER_ID, "character") == {
        "revision": 2, "entries": [before["entries"][1]],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    "not JSON",
    "[]",
    decision("unknown", add("Unexpected memory.")),
    decision("character", {"action": "replace", "id": "missing", "content": "Changed."}),
    decision("character", {"action": "delete", "id": "missing"}),
    decision("character", add("Do not commit this."), {"action": "overwrite", "content": "Invalid."}),
    decision("character", {"action": "add", "content": "   "}),
    decision("character", {"action": "add", "content": {"invalid": "object"}}),
], ids=["invalid-json", "invalid-root", "invalid-kind", "missing-replace-id", "missing-delete-id", "mixed-valid-invalid", "blank-content", "invalid-content-type"])
async def test_invalid_decision_fails_atomically_without_modifying_memory(tmp_path, response):
    client = FakeUpdateClient(decision("character", add("Existing memory.")), response)
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember this.", metadata=METADATA)
    before = json_snapshot(tmp_path)

    forget = isinstance(response, dict) and any(
        operation.get("action") == "delete" for operation in response["operations"]
    )
    result = await tool.save_memory("Update my memory.", metadata=METADATA, forget=forget)

    assert result["status"] == "error"
    assert json_snapshot(tmp_path) == before
    assert contents(await tool.read_memory(USER_ID, "character")) == ["Existing memory."]


@pytest.mark.asyncio
async def test_update_client_failure_preserves_existing_memory(tmp_path):
    client = FakeUpdateClient(decision("shared", add("Run the checks before publishing.")), RuntimeError("Updater unavailable"))
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember the publishing procedure.", metadata=METADATA)
    before = json_snapshot(tmp_path)

    result = await tool.save_memory("Revise the procedure.", metadata=METADATA)

    assert result["status"] == "error"
    assert json_snapshot(tmp_path) == before


@pytest.mark.asyncio
async def test_atomic_replace_failure_preserves_current_memory(tmp_path, monkeypatch):
    client = FakeUpdateClient(
        decision("character", add("Existing memory.")),
        decision("character", add("This write must fail.")),
    )
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember this.", metadata=METADATA)
    current_path = tmp_path / "character" / f"{USER_ID}.json"
    current_bytes = current_path.read_bytes()
    original_replace = memory_module.os.replace

    def fail_current_replace(source, destination):
        if destination == current_path:
            raise OSError("Simulated atomic replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(memory_module.os, "replace", fail_current_replace)

    result = await tool.save_memory("Remember another detail.", metadata=METADATA)

    assert result["status"] == "error"
    assert current_path.read_bytes() == current_bytes
    assert await tool.read_memory(USER_ID, "character") == json.loads(current_bytes)
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.asyncio
async def test_corrupt_existing_json_is_not_overwritten(tmp_path):
    client = FakeUpdateClient()
    tool = make_tool(tmp_path, client)
    digest = hashlib.sha256(USER_ID.encode("utf-8")).hexdigest()
    current_path = tmp_path / "character" / f"{digest}.json"
    current_path.parent.mkdir()
    current_path.write_text("{invalid JSON", encoding="utf-8")
    before = json_snapshot(tmp_path)

    result = await tool.save_memory("Remember a new fact.", metadata=METADATA)

    assert result["status"] == "error"
    assert json_snapshot(tmp_path) == before
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {}, {"user_id": ""}, {"user_id": "   "}])
async def test_missing_user_id_fails_before_updater_or_filesystem_write(tmp_path, metadata):
    client = FakeUpdateClient()
    tool = make_tool(tmp_path, client)

    result = await tool.save_memory("Remember this.", metadata=metadata)

    assert result["status"] == "error"
    assert client.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_default_shared_directory_shares_user_facts_and_procedures_between_characters(tmp_path):
    first_client = FakeUpdateClient(
        decision("character", add("First character calls me Captain.")),
        decision("shared", add("I prefer tea.")),
        decision("shared", add("Check the calendar before booking.")),
        decision("shared", add("Another user's private procedure.")),
    )
    second_client = FakeUpdateClient(decision("character", add("Second character calls me Partner.")))
    first = make_tool(tmp_path, first_client, character_memory_dir=tmp_path / "first-character")
    second = make_tool(tmp_path, second_client, character_memory_dir=tmp_path / "second-character")

    assert first.shared_memory_dir == second.shared_memory_dir == tmp_path / "shared"
    await first.save_memory("Remember this character's nickname for me.", metadata=METADATA)
    await first.save_memory("Remember my drink preference.", metadata=METADATA)
    await first.save_memory("Remember my booking procedure.", metadata=METADATA)
    await second.save_memory("Remember the other character's nickname for me.", metadata=METADATA)
    await first.save_memory("Remember a private procedure.", metadata={"user_id": "other-user"})

    assert contents(await first.read_memory(USER_ID, "character")) == ["First character calls me Captain."]
    assert contents(await second.read_memory(USER_ID, "character")) == ["Second character calls me Partner."]
    assert contents(await second.read_memory(USER_ID, "shared")) == [
        "I prefer tea.", "Check the calendar before booking.",
    ]
    assert await second.read_memory("other-user", "character") == {"revision": 0, "entries": []}
    assert contents(await second.read_memory("other-user", "shared")) == ["Another user's private procedure."]


@pytest.mark.asyncio
async def test_prompt_contains_only_active_plaintext_for_requested_kind(tmp_path):
    client = FakeUpdateClient(
        decision("character", add("Use polite speech.")),
        decision("shared", add("Check the calendar before booking.")),
    )
    tool = make_tool(tmp_path, client)
    await tool.save_memory("Remember the speech preference.", metadata=METADATA)
    await tool.save_memory("Remember the booking procedure.", metadata=METADATA)
    before = await tool.read_memory(USER_ID, "character")
    client.responses.append(decision("character", {
        "action": "replace", "id": before["entries"][0]["id"], "content": "Use casual speech.",
    }))
    await tool.save_memory("Use casual speech instead.", metadata=METADATA)

    combined = await tool.get_prompt(USER_ID)
    character = await tool.get_prompt(USER_ID, "character")
    shared = await tool.get_prompt(USER_ID, "shared")

    assert "Use casual speech." in combined and "Check the calendar before booking." in combined
    assert "Use casual speech." in character and "Check the calendar before booking." not in character
    assert "Check the calendar before booking." in shared and "Use casual speech." not in shared
    assert "Use polite speech." not in combined
    assert '"revision"' not in combined and '"entries"' not in combined
    for kind in ("character", "shared"):
        for entry in (await tool.read_memory(USER_ID, kind))["entries"]:
            assert entry["id"] not in combined


@pytest.mark.asyncio
async def test_concurrent_instances_retry_without_losing_updates(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=memory_module.__name__)
    both_read = asyncio.Event()
    arrivals = 0

    def responder_for(content):
        calls = 0

        async def respond(kwargs):
            nonlocal arrivals, calls
            calls += 1
            if calls == 1:
                arrivals += 1
                if arrivals == 2:
                    both_read.set()
                await both_read.wait()
            else:
                other_content = "Remember A." if content == "Remember B." else "Remember B."
                assert other_content in json.dumps(kwargs["messages"])
            return decision("character", add(content))

        return respond

    first_client = FakeUpdateClient(responder=responder_for("Remember A."))
    second_client = FakeUpdateClient(responder=responder_for("Remember B."))
    first = make_tool(tmp_path, first_client, debug=True)
    second = make_tool(tmp_path, second_client, debug=True)

    results = await asyncio.wait_for(asyncio.gather(
        first.save_memory("Remember A.", metadata=METADATA),
        second.save_memory("Remember B.", metadata=METADATA),
    ), timeout=5)

    assert [result["status"] for result in results] == ["saved", "saved"]
    document = await first.read_memory(USER_ID, "character")
    assert document["revision"] == 2
    assert set(contents(document)) == {"Remember A.", "Remember B."}
    assert len(first_client.calls) + len(second_client.calls) == 3
    events = memory_debug_events(caplog)
    retries = [event for event in events if event["event"] == "retry"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    retried_events = [
        event for event in events if event["operation_id"] == retries[0]["operation_id"]
    ]
    assert [event["attempt"] for event in retried_events if event["event"] == "request"] == [1, 2]
    result_event = next(event for event in retried_events if event["event"] == "result")
    assert result_event["attempt"] == 2
    assert result_event["result"]["status"] == "saved"


@pytest.mark.asyncio
async def test_conflict_retries_are_bounded_and_never_commit_stale_update(tmp_path):
    interfering_client = FakeUpdateClient()
    interfering = make_tool(tmp_path, interfering_client)
    count = 0

    async def always_conflict(kwargs):
        nonlocal count
        count += 1
        interfering_client.responses.append(decision("character", add(f"Concurrent update {count}.")))
        result = await interfering.save_memory(f"Remember update {count}.", metadata=METADATA)
        assert result["status"] == "saved"
        return decision("character", add("Stale update must not be committed."))

    client = FakeUpdateClient(responder=always_conflict)
    tool = make_tool(tmp_path, client, max_conflict_retries=2)

    result = await asyncio.wait_for(tool.save_memory("Save while another writer is active.", metadata=METADATA), timeout=5)

    assert result["status"] == "error"
    assert len(client.calls) == 3
    document = await tool.read_memory(USER_ID, "character")
    assert document["revision"] == 3
    assert contents(document) == [f"Concurrent update {number}." for number in range(1, 4)]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unchanged", "needs_clarification"])
async def test_nonupdate_decision_retries_after_concurrent_change(tmp_path, status):
    writer_client = FakeUpdateClient(
        decision("character", add("Existing memory.")),
        decision("character", add("Concurrently added memory.")),
    )
    writer = make_tool(tmp_path, writer_client)
    await writer.save_memory("Remember this.", metadata=METADATA)
    calls = 0

    async def respond(kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            result = await writer.save_memory("Add another memory.", metadata=METADATA)
            assert result["status"] == "saved"
        else:
            snapshot = json.loads(kwargs["messages"][-1]["content"])["memories"]["character"]
            assert snapshot["revision"] == 2
            assert "Concurrently added memory." in contents(snapshot)
        return decision("character", status=status)

    client = FakeUpdateClient(responder=respond)
    tool = make_tool(tmp_path, client)

    result = await tool.save_memory("Remember the same information.", metadata=METADATA)

    assert result["status"] == status
    assert result["revision"] == 2
    assert len(client.calls) == 2
    assert contents(result) == ["Existing memory.", "Concurrently added memory."]


@pytest.mark.asyncio
async def test_cancellation_during_updater_never_writes_memory(tmp_path):
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def respond(kwargs):
        started.set()
        await blocked.wait()
        return decision("character", add("Must not be written."))

    tool = make_tool(tmp_path, FakeUpdateClient(responder=respond))
    task = asyncio.create_task(tool.save_memory("Remember this.", metadata=METADATA))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert list(tmp_path.iterdir()) == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_commit_worker(tmp_path, monkeypatch):
    tool = make_tool(tmp_path, FakeUpdateClient(decision("character", add("Committed memory."))))
    original_commit = tool._commit
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_commit(*args):
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release the commit worker")
            return original_commit(*args)
        finally:
            finished.set()

    monkeypatch.setattr(tool, "_commit", blocked_commit)
    task = asyncio.create_task(tool.save_memory("Remember this.", metadata=METADATA))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert finished.is_set()
        assert contents(await tool.read_memory(USER_ID, "character")) == ["Committed memory."]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 5)


@pytest.mark.asyncio
async def test_builtin_tool_dispatch_injects_metadata(tmp_path):
    client = FakeUpdateClient(decision("character", add("Use polite speech.")))
    tool = make_tool(tmp_path, client)
    llm = LLMServiceDummy(context_manager=SimpleNamespace())
    assert isinstance(tool, Tool)
    assert tool.name == "save_memory"
    assert "metadata" not in tool.spec["function"]["parameters"]["properties"]
    llm.add_tool(tool, use_original=True)
    results = [result async for result in llm.execute_tool(
        name="save_memory", arguments={"content": "Remember my speech preference."}, metadata=METADATA,
    )]

    assert len(results) == 1
    assert results[0].data["status"] == "saved"
    assert contents(await tool.read_memory(USER_ID, "character")) == ["Use polite speech."]


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", ["saved", "error"])
async def test_background_tool_acknowledges_before_save_and_delivers_final_result(tmp_path, final_status):
    started = asyncio.Event()
    release = asyncio.Event()
    callbacks = []

    async def respond(kwargs):
        started.set()
        await release.wait()
        if final_status == "error":
            return RuntimeError("Updater unavailable")
        return decision("character", add("Use polite speech."))

    tool = make_tool(tmp_path, FakeUpdateClient(responder=respond))

    @tool.on_completed
    async def on_completed(result, metadata):
        callbacks.append((result, metadata))

    llm = LLMServiceDummy(context_manager=SimpleNamespace())
    llm.add_tool(tool, use_original=True)

    async def execute_tool():
        return [result async for result in llm.execute_tool(
            name="save_memory",
            arguments={"content": "Remember my speech preference."},
            metadata=METADATA,
        )]

    callback_task = None
    try:
        results = await asyncio.wait_for(execute_tool(), timeout=5)
        callback_task = asyncio.create_task(results[0].deferred_callback())
        await asyncio.wait_for(started.wait(), timeout=5)

        assert len(results) == 1
        assert results[0].data["message"] == tool.immediate_message
        assert results[0].task_id
        assert "status" not in results[0].data
        assert callbacks == []
        assert await tool.read_memory(USER_ID, "character") == {"revision": 0, "entries": []}

        release.set()
        await asyncio.wait_for(asyncio.shield(callback_task), timeout=5)

        assert len(callbacks) == 1
        result, metadata = callbacks[0]
        assert result["status"] == final_status
        assert metadata["user_id"] == USER_ID
        assert metadata["context_id"] == METADATA["context_id"]
        assert metadata["task_id"] == results[0].task_id
        assert metadata["arguments"] == {"content": "Remember my speech preference."}
        expected = ["Use polite speech."] if final_status == "saved" else []
        assert contents(await tool.read_memory(USER_ID, "character")) == expected
    finally:
        release.set()
        if callback_task is not None:
            await asyncio.wait_for(callback_task, timeout=5)
