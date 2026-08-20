import asyncio
import json
import time

import httpx


ASTERISK_URL = "http://127.0.0.1:18088/ari"
HARNESS_URL = "http://127.0.0.1:18080"
ARI_AUTH = ("aiavatar-local", "local-only-change-me")


async def wait_for(predicate, *, timeout=15.0, interval=0.1):
    deadline = time.monotonic() + timeout
    last_value = None
    while time.monotonic() < deadline:
        last_value = await predicate()
        if last_value:
            return last_value
        await asyncio.sleep(interval)
    raise TimeoutError(f"Condition was not met; last value: {last_value!r}")


async def main():
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{ASTERISK_URL}/channels",
            auth=ARI_AUTH,
            params={
                "endpoint": "Local/6000@media-source",
                "app": "aiavatar",
                "appArgs": "inbound",
                "callerId": "AIAvatar local test <6000>",
                "timeout": 30,
            },
        )
        response.raise_for_status()

        async def connected_call():
            state = (await client.get(f"{HARNESS_URL}/test/state")).json()
            for session_id, call in state["calls"].items():
                if call["media_connected"]:
                    return session_id, call

        session_id, before = await wait_for(connected_call)
        pipeline_session_id = before["pipeline_session_id"]
        before_bytes = before["bytes_received"]

        response = await client.post(f"{HARNESS_URL}/test/tone/{session_id}")
        response.raise_for_status()

        async def echoed_audio():
            state = (await client.get(f"{HARNESS_URL}/test/state")).json()
            call = state["calls"].get(session_id)
            if call and call["bytes_received"] >= before_bytes + 24_000:
                return call

        after = await wait_for(echoed_audio)
        response = await client.post(f"{HARNESS_URL}/test/hangup/{session_id}")
        response.raise_for_status()

        async def cleaned_up():
            state = (await client.get(f"{HARNESS_URL}/test/state")).json()
            if session_id not in state["calls"] and not state["adapter_sessions"]:
                return state

        final_state = await wait_for(cleaned_up)
        print(json.dumps({
            "result": "passed",
            "session_id": session_id,
            "sent_pcm_bytes": 32_000,
            "echoed_pcm_bytes": after["bytes_received"] - before_bytes,
            "echoed_frames": after["frames_received"] - before["frames_received"],
            "ari_cleanup": not final_state["calls"],
            "adapter_cleanup": not final_state["adapter_sessions"],
            "pipeline_finalized": pipeline_session_id in final_state["finalized"],
        }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
