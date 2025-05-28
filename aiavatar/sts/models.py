from dataclasses import dataclass
from typing import List, Dict, Any
from .llm import ToolCall


@dataclass
class STSRequest:
    type: str = "start"
    session_id: str = None
    user_id: str = None
    context_id: str = None
    text: str = None
    audio_data: bytes = None
    audio_duration: float = 0
    files: List[Dict[str, str]] = None
    system_prompt_params: Dict[str, Any] = None


@dataclass
class STSResponse:
    type: str
    session_id: str = None
    user_id: str = None
    context_id: str = None
    text: str = None
    voice_text: str = None
    language: str = None
    audio_data: bytes = None
    tool_call: ToolCall = None
    metadata: dict = None
