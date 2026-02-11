from typing import Optional, Dict
from pydantic import BaseModel


class STTRequest(BaseModel):
    type: str  # start, data, config, stop
    session_id: Optional[str] = None
    audio_data: Optional[str] = None  # base64 encoded
    metadata: Optional[Dict] = None


class STTResponse(BaseModel):
    type: str  # connected, partial, final, error, stop
    session_id: Optional[str] = None
    text: Optional[str] = None
    is_final: bool = False
    metadata: Optional[Dict] = None
