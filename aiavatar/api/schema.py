from pydantic import BaseModel, Field
from typing import List


class APIResponse(BaseModel):
    message: str = Field(..., example="Message from API", description="Message from API")


class ErrorResponse(BaseModel):
    error: str = Field(..., example="Error message from API", description="Error message from API")


class WakewordStartRequest(BaseModel):
    wakewords: List[str] = Field(default=[], description="List of wakewords to start chat.")


class WakewordStatusResponse(BaseModel):
    is_listening: bool = Field(..., description="Whether the WakewordListener is listening")
    thread_name: str|None = Field(None, example="Thread-2", description="The id of the thread wakeword listener is running in")


class SpeechRequest(BaseModel):
    text: str = Field(..., example="[face:joy]Hi, let's talk with me!", description="Text to speech with face and animation tag")


class FaceRequest(BaseModel):
    name: str = Field(..., example="fun", description="Name of face expression to set")
    duration: float = Field(4.0, example=4.0, description="Duration in seconds for how long the face expression should last")


class AnimationRequest(BaseModel):
    name: str = Field(..., example="wave_hands", description="Name of animation to set")
    duration: float = Field(4.0, example=4.0, description="Duration in seconds for how long the animation should last")


class LogRequest(BaseModel):
    count: int = Field(50, example=50, description="Lines from tail to read")


class LogResponse(BaseModel):
    lines: List[str] = Field(default=[], examples=["[INFO] 2024-05-05 00:25:08,070 : AzureWakeWordListener: Hello", "[INFO] 2024-05-05 00:25:13,949 : User: Hello", "[INFO] 2024-05-05 00:25:13,949 : AI: Hello! What's up?"], description="List of lines in log file")
