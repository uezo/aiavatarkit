from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import aiavatar
import aiavatar.speech


class APIResponse(BaseModel):
    message: str = Field(..., example="Message from API", description="Message from API")


class ErrorResponse(BaseModel):
    error: str = Field(..., example="Error message from API", description="Error message from API")


class WakewordStartRequest(BaseModel):
    wakewords: List[str] = Field(default=[], description="List of wakewords to start chat.")


class WakewordStatusResponse(BaseModel):
    is_listening: bool = Field(..., description="Whether the WakewordListener is listening")
    thread_name: Optional[str] = Field(None, example="Thread-2", description="The id of the thread wakeword listener is running in")


class GetAvatarStatusResponse(BaseModel):
    is_speaking: bool = Field(..., description="Whether the avatar is speaking")
    current_face: str = Field(..., example="fun", description="Name of current face expression")
    current_animation: str = Field(..., example="wave_hands", description="Name of current animation")


class SpeechRequest(BaseModel):
    text: str = Field(..., example="[face:joy]Hi, let's talk with me!", description="Text to speech with face and animation tag")


class GetIsSpeakingResponse(BaseModel):
    is_speaking: bool = Field(..., description="Whether the avatar is speaking")


class SpeechConfigBase(BaseModel):
    # Common
    base_url: Optional[str] = Field(None, example="http://127.0.0.1:50021", description="Base url for speech service")
    rate: Optional[int] = Field(None, example="16000", description="Sample rate")
    device_index: Optional[int] = Field(None, example="1", description="Output device index")
    playback_margin: Optional[float] = Field(None, example="0.1", description="Margin in seconds after playback")
    use_subprocess: Optional[bool] = Field(None, example=True, description="Enable or disable the use of subprocess for TTS")
    subprocess_timeout: Optional[float] = Field(None, example="5.0", description="Timeout duration for the subprocess in seconds")
    # VOICEVOX
    speaker_id: Optional[int] = Field(None, example="1", description="ID of the speaker to use in VOICEVOX")
    # OpenAI/Azure
    api_key: Optional[str] = Field(None, example="sk-xxxxxxxxxxxx", description="API key for accessing the TTS service")
    # OpenAI
    voice: Optional[str] = Field(None, example="alloy", description="Voice model to use for OpenAI TTS")
    model: Optional[str] = Field(None, example="whisper-1", description="Model name for OpenAI TTS")
    speed: Optional[float] = Field(None, example="1.0", description="Speech speed multiplier")
    # Azure
    region: Optional[str] = Field(None, example="japaneast", description="Azure region for TTS service")
    speaker_name: Optional[str] = Field(None, example="en-US-JennyMultilingualNeural", description="Name of the speaker to use in Azure TTS")
    speaker_gender: Optional[str] = Field(None, example="Female", description="Gender of the speaker for Azure TTS")
    lang: Optional[str] = Field(None, example="en-US", description="Language code for TTS")

    @classmethod
    def from_speech_controller_base(cls, sc: aiavatar.speech.SpeechControllerBase):
        return cls(
            base_url=getattr(sc, "base_url", None),
            rate=getattr(sc, "rate", None),
            device_index=getattr(sc, "device_index", None),
            playback_margin=getattr(sc, "playback_margin", None),
            use_subprocess=getattr(sc, "use_subprocess", None),
            subprocess_timeout=getattr(sc, "subprocess_timeout", None),
            speaker_id=getattr(sc, "speaker_id", None),
            api_key=getattr(sc, "api_key", None),
            voice=getattr(sc, "voice", None),
            model=getattr(sc, "model", None),
            speed=getattr(sc, "speed", None),
            region=getattr(sc, "region", None),
            speaker_name=getattr(sc, "speaker_name", None),
            speaker_gender=getattr(sc, "speaker_gender", None),
            lang=getattr(sc, "lang", None),
        )


class SpeechConfigRequest(SpeechConfigBase): ...


class SpeechConfigResponse(SpeechConfigBase): ...


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


class ChatProcessorConfig(BaseModel):
    # Common fields
    api_key: Optional[str] = Field(None, example="sk-xxxxxxxxxxxx", description="API key for accessing the chat service")
    model: Optional[str] = Field(None, example="gpt-4o", description="Model name for the chat service")
    temperature: Optional[float] = Field(None, example=1.0, description="Temperature setting for the chat model")
    max_tokens: Optional[int] = Field(None, example=200, description="Maximum tokens for the response")
    functions: Optional[Dict] = Field(None, example={}, description="Functions for the chat model")
    system_message_content: Optional[str] = Field(None, example="System message content", description="Content of the system message")
    history_count: Optional[int] = Field(None, example=10, description="Number of history entries to retain")
    history_timeout: Optional[float] = Field(None, example=60.0, description="Timeout for history entries in seconds")
    # ChatGPTProcessor
    base_url: Optional[str] = Field(None, example="http://127.0.0.1:8080", description="Base URL for the chat service")
    parse_function_call_in_response: Optional[bool] = Field(None, example=True, description="Whether to parse function call in response")
    # GeminiProcessor
    system_message_content_acknowledgement_content: Optional[str] = Field(None, example="了解しました。", description="Acknowledgement content for system message")

    @classmethod
    def from_chat_processor(cls, processor) -> "ChatProcessorConfig":
        return cls(
            api_key=getattr(processor, "api_key", None),
            base_url=getattr(processor, "base_url", None),
            model=getattr(processor, "model", None),
            temperature=getattr(processor, "temperature", None),
            max_tokens=getattr(processor, "max_tokens", None),
            functions=getattr(processor, "functions", None),
            parse_function_call_in_response=getattr(processor, "parse_function_call_in_response", None),
            system_message_content=getattr(processor, "system_message_content", None),
            system_message_content_acknowledgement_content=getattr(processor, "system_message_content_acknowledgement_content", None),
            history_count=getattr(processor, "history_count", None),
            history_timeout=getattr(processor, "history_timeout", None),
        )


class ChatProcessorConfigRequest(ChatProcessorConfig): ...


class ChatProcessorConfigResponse(ChatProcessorConfig): ...


class LogRequest(BaseModel):
    count: int = Field(50, example=50, description="Lines from tail to read")


class LogResponse(BaseModel):
    lines: List[str] = Field(default=[], examples=["[INFO] 2024-05-05 00:25:08,070 : AzureWakeWordListener: Hello", "[INFO] 2024-05-05 00:25:13,949 : User: Hello", "[INFO] 2024-05-05 00:25:13,949 : AI: Hello! What's up?"], description="List of lines in log file")
