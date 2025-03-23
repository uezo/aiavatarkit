from pydantic import BaseModel, Field
from typing import List


class APIResponse(BaseModel):
    message: str = Field(..., example="Message from API", description="Message from API")


class ErrorResponse(BaseModel):
    error: str = Field(..., example="Error message from API", description="Error message from API")


class ListenerStatusResponse(BaseModel):
    is_listening: bool = Field(..., description="Whether the listener is listening")


class GetAvatarStatusResponse(BaseModel):
    current_face: str = Field(..., example="fun", description="Name of current face expression")
    current_animation: str = Field(..., example="wave_hands", description="Name of current animation")


class SpeechRequest(BaseModel):
    text: str = Field(..., example="[face:joy]Hi, let's talk with me!", description="Text to speech with face and animation tag")


class GetIsSpeakingResponse(BaseModel):
    is_speaking: bool = Field(..., description="Whether the avatar is speaking")


class FaceRequest(BaseModel):
    name: str = Field(..., example="fun", description="Name of face expression to set")
    duration: float = Field(4.0, example=4.0, description="Duration in seconds for how long the face expression should last")


class GetFaceResponse(BaseModel):
    current_face: str = Field(..., example="fun", description="Name of current face expression")


class AnimationRequest(BaseModel):
    name: str = Field(..., example="wave_hands", description="Name of animation to set")
    duration: float = Field(4.0, example=4.0, description="Duration in seconds for how long the animation should last")


class GetAnimationResponse(BaseModel):
    current_animation: str = Field(..., example="wave_hands", description="Name of current animation")


class ChatRequest(BaseModel):
    text: str = Field(..., example="こんにちは！", description="Text message to send to ChatProcessor")


class GetHistoriesResponse(BaseModel):
    histories: list[dict] = Field(..., example='[{"role": "user", "content": "こんにちは"}, {"role": "assistant", "content": "こんにちは！何かお手伝いすることはありますか？"}]', description="Histories cached in ChatProcessor")


class LogRequest(BaseModel):
    count: int = Field(50, example=50, description="Lines from tail to read")


class LogResponse(BaseModel):
    lines: List[str] = Field(default=[], examples=["[INFO] 2024-05-05 00:25:08,070 : AzureWakeWordListener: Hello", "[INFO] 2024-05-05 00:25:13,949 : User: Hello", "[INFO] 2024-05-05 00:25:13,949 : AI: Hello! What's up?"], description="List of lines in log file")
