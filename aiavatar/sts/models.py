from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from uuid import uuid4
from .llm import ToolCall


@dataclass(frozen=True, kw_only=True)
class Interruption:
    """Interruption facts scoped to the conversation of the interrupted response."""

    interrupted_response_context_id: Optional[str]
    interrupted_response_transaction_id: str
    spoken_text: Optional[str] = None
    answered_through_transaction_id: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class RequestContext:
    interruption: Optional[Interruption] = None
    track_interruption: bool = False


@dataclass(kw_only=True)
class STSRequest:
    type: str = "start"
    session_id: str = None
    user_id: str = None
    context_id: str = None
    transaction_id: str = field(default_factory=lambda: str(uuid4()))
    text: str = None
    audio_data: bytes = None
    audio_duration: float = 0
    files: List[Dict[str, str]] = None
    system_prompt_params: Dict[str, Any] = None
    allow_merge: bool = True
    wait_in_queue: bool = False
    block_barge_in: bool = False
    channel: str = None
    quick_response_text: str = None
    quick_response_voice_text: str = None
    quick_response_audio: bytes = None
    skip_quick_response: bool = False
    request_context: Optional[RequestContext] = None
    metadata: Dict[str, Any] = None


@dataclass(kw_only=True)
class STSResponse:
    type: str
    session_id: str = None
    user_id: str = None
    context_id: str = None
    transaction_id: str = None
    text: str = None
    voice_text: str = None
    language: str = None
    audio_data: bytes = None
    tool_call: ToolCall = None
    metadata: dict = None
    structured_content: dict = None
