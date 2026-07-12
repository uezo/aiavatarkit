from typing import Awaitable, Callable, Dict, List


def is_previous_response_not_found(error: dict) -> bool:
    """Return True when an API error says previous_response_id no longer exists."""
    if not isinstance(error, dict):
        return False

    error = error.get("error", error)
    if not isinstance(error, dict):
        return False

    code = str(error.get("code") or "").lower()
    param = str(error.get("param") or "").lower()
    message = str(error.get("message") or "").lower()
    is_not_found = (
        "not_found" in code
        or "not found" in message
        or "does not exist" in message
    )
    refers_to_previous_response = (
        param == "previous_response_id"
        or "previous_response_id" in code
        or "previous_response_id" in message
        or "previous response" in message
    )
    return is_not_found and refers_to_previous_response


async def build_recovery_input(
    context_id: str,
    user_id: str,
    current_input: List[Dict],
    system_prompt_params: Dict,
    get_initial_messages: Callable[..., Awaitable[List[Dict]]],
    context_manager,
) -> List[Dict]:
    """Rebuild a Responses API input from the locally persisted conversation."""
    initial_messages = await get_initial_messages(
        context_id, user_id, system_prompt_params
    )
    histories = await context_manager.get_histories(context_id)
    return list(initial_messages or []) + histories + current_input
