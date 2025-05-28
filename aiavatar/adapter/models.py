from typing import List, Dict, Optional, Union
from pydantic import BaseModel


class AvatarControlRequest(BaseModel):
    animation_name: Optional[str] = None
    animation_duration: Optional[float] = None
    face_name: Optional[str] = None
    face_duration: Optional[float] = None


class AIAvatarRequest(BaseModel):
    type: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    text: Optional[str] = None
    audio_data: Optional[Union[bytes, str]] = None
    files: Optional[List[Dict[str, str]]] = None
    system_prompt_params: Optional[Dict] = None
    metadata: Optional[Dict] = None


class AIAvatarResponse(BaseModel):
    type: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    text: Optional[str] = None
    voice_text: Optional[str] = None
    language: Optional[str] = None
    avatar_control_request: Optional[AvatarControlRequest] = None
    audio_data: Optional[Union[bytes, str]] = None
    metadata: Optional[Dict] = None


class AIAvatarException(Exception):
    def __init__(self, message: str, response: AIAvatarResponse = None):
        super().__init__(message)
        self.response = response or None
