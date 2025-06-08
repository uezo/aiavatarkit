import collections
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from ..sts.vad import SpeechDetector
from ..sts.stt import SpeechRecognizer
from ..sts.llm import LLMService
from ..sts.tts import SpeechSynthesizer
from ..sts import STSPipeline

logger = logging.getLogger(__name__)


# Speech-to-Text
class STTConfig(BaseModel):
    """
    STTConfig is a data model that holds configuration settings for Speech-to-Text (STT) components.
    
    All fields are optional and default to None if not provided.
    """
    language: Optional[str] = Field(
        default=None,
        description="The primary language for speech recognition, e.g., 'en-US', 'ja-JP', etc."
    )
    alternative_languages: Optional[List[str]] = Field(
        default=None,
        description="A list of alternative languages for speech recognition."
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout for speech recognition requests in seconds."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class STTConfigResponse(BaseModel):
    type: str = Field(
        description="Type of Speech-to-Text component"
    )
    name: Optional[str] = Field(
        description="Registered name of Speech-to-Text component"
    )
    config: STTConfig = Field(
        description="Configuration of Speech-to-Text component"
    )


class UpdateSTTConfigRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="Registered name of Speech-to-Text component"
    )
    config: STTConfig = Field(
        description="Configuration of Speech-to-Text component"
    )


# LLM
class LLMConfig(BaseModel):
    """
    LLMConfig is a data model that holds configuration settings for invoking a large language model (LLM).
    
    All fields are optional and default to None if not provided.
    """
    system_prompt: Optional[str] = Field(
        default=None,
        description="The system prompt that sets initial instructions or context for the LLM."
    )
    model: Optional[str] = Field(
        default=None,
        description="The name of the model to use, e.g., 'gpt-4o-mini', 'gpt-4', etc."
    )
    temperature: Optional[float] = Field(
        default=None,
        description="The temperature for text generation. Higher values increase randomness (range 0.0 to 1.0)."
    )
    split_chars: Optional[List[str]] = Field(
        default=None,
        description="A list of delimiter characters used by the LLM to split responses."
    )
    option_split_chars: Optional[List[str]] = Field(
        default=None,
        description="A list of delimiter characters used for splitting optional elements."
    )
    option_split_threshold: Optional[float] = Field(
        default=None,
        description="A threshold value to control splitting of optional parts based on score or confidence."
    )
    voice_text_tag: Optional[str] = Field(
        default=None,
        description="A tag used for speech synthesis or recognition, e.g., 'en-US', 'ja-JP', etc."
    )
    use_dynamic_tools: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable dynamic tool invocation. If True, tools are selected dynamically."
    )
    context_manager: Optional[str] = Field(
        default=None,
        description="The name of the context manager class, stored as a string."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class LLMConfigResponse(BaseModel):
    type: str = Field(
        description="Type of LLM component"
    )
    name: str = Field(
        description="Registered name of LLM component"
    )
    config: LLMConfig = Field(
        description="Configuration of LLM component"
    )


class UpdateLLMConfigRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="Registered name of LLM component"
    )
    config: LLMConfig = Field(
        description="Configuration of LLM component"
    )


# Text-to-Speech
class TTSConfig(BaseModel):
    """
    TTSConfig is a data model that holds configuration settings for Text-to-Speech (TTS) components.
    
    All fields are optional and default to None if not provided.
    """
    style_mapper: Optional[Dict[str, str]] = Field(
        default=None,
        description="A dictionary mapping style keywords to TTS-specific style values."
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout for speech synthesis requests in seconds."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class TTSConfigResponse(BaseModel):
    type: str = Field(
        description="Type of Text-to-Speech component"
    )
    name: str = Field(
        description="Registered name of Text-to-Speech component"
    )
    config: TTSConfig = Field(
        description="Configuration of Text-to-Speech component"
    )


class UpdateTTSConfigRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="Registered name of Text-to-Speech component"
    )
    config: TTSConfig = Field(
        description="Configuration of Text-to-Speech component"
    )


# STS Pipeline
class STSComponent(BaseModel):
    type: str = Field(
        description="Type of component"
    )
    name: str = Field(
        description="Registered name of component"
    )


class STSComponentResponse(BaseModel):
    stts: List[STSComponent] = Field(
        default=None,
        description="The registered Speech-To-Text (STT) components"
    )
    llms: List[STSComponent] = Field(
        default=None,
        description="The registered LLM components"
    )
    ttss: List[STSComponent] = Field(
        default=None,
        description="The registered Text-To-Speech (TTS) components"
    )


class UpdateSTSComponentRequest(BaseModel):
    """
    STSComponentRequest is a data model for overriding which components to use
    in the speech-to-speech pipeline. Specify the registered component names (keys)
    for STT, LLM, and TTS as needed.
    """
    stt: Optional[str] = Field(
        default=None,
        description="The name of the registered Speech-To-Text (STT) component to use."
    )
    llm: Optional[str] = Field(
        default=None,
        description="The name of the registered Large Language Model (LLM) component to use."
    )
    tts: Optional[str] = Field(
        default=None,
        description="The name of the registered Text-To-Speech (TTS) component to use."
    )


# System
class LogResponse(BaseModel):
    lines: List[str] = Field(default=[], description="List of lines in log file")


# Router
class ConfigAPI:
    DEFAULT_COMPONENT_KEY = "default"

    def __init__(
        self,
        sts: STSPipeline,
        *,
        vads: Dict[str, SpeechDetector] = None,
        stts: Dict[str, SpeechRecognizer] = None,
        llms: Dict[str, LLMService] = None,
        ttss: Dict[str, SpeechSynthesizer] = None,
        logfile_path: str = None,
    ):
        self.sts = sts
        self._vads = vads or {}
        self._stts = stts or {}
        self._llms = llms or {}
        self._ttss = ttss or {}
        self.logfile_path = logfile_path

    @property
    def vads(self):
        return self._vads or {self.DEFAULT_COMPONENT_KEY: self.sts.vad}

    def add_vad(self, name: str, vad: SpeechDetector):
        self._vads[name] = vad

    @property
    def stts(self):
        return self._stts or {self.DEFAULT_COMPONENT_KEY: self.sts.stt}

    def add_stt(self, name: str, stt: SpeechRecognizer):
        self._stts[name] = stt

    @property
    def llms(self):
        return self._llms or {self.DEFAULT_COMPONENT_KEY: self.sts.llm}

    def add_llm(self, name: str, llm: LLMService):
        self._llms[name] = llm

    @property
    def ttss(self):
        return self._ttss or {self.DEFAULT_COMPONENT_KEY: self.sts.tts}

    def add_tts(self, name: str, tts: SpeechSynthesizer):
        self._ttss[name] = tts

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/stt/config",
            tags=["Speech-to-Text"],
            summary="Get Speech-to-Text configuration",
            description="Retrieve the current configuration settings for the Speech-to-Text component",
            response_description="Current STT configuration including language, timeout, and debug settings",
            responses={
                200: {"description": "Successfully retrieved STT configuration"},
                404: {"description": "STT component not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_stt_config(
            name: Optional[str] = Query(
                None, 
                description="Name of specific STT component to retrieve. If not provided, returns current active component"
            )
        ) -> STTConfigResponse:
            """
            Get detailed configuration for Speech-to-Text component.
            
            This endpoint returns comprehensive configuration information including:
            - Language settings and alternatives
            - Timeout configurations  
            - Debug mode status
            - Component-specific settings (URLs, etc.)
            """
            try:
                if name:
                    stt = self.stts.get(name)
                else:
                    # Update current component if name is not specified
                    stt = self.sts.stt

                if not stt:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"Speech-to-Text service not found: {name}"
                    )

                stt_config = STTConfig(
                    language=stt.language,
                    alternative_languages=stt.alternative_languages,
                    timeout=getattr(stt.http_client.timeout, "timeout", None) if hasattr(stt, "http_client") and stt.http_client else None,
                    debug=stt.debug
                )

                # Add any additional attributes that might exist on specific STT implementations
                if hasattr(stt, "model"):
                    stt_config.model = stt.model
                if hasattr(stt, "base_url"):
                    stt_config.base_url = stt.base_url
                if hasattr(stt, "sample_rate"):
                    stt_config.sample_rate = stt.sample_rate

                return STTConfigResponse(
                    type=stt.__class__.__name__,
                    name=next((k for k, v in self.stts.items() if v == stt), None),
                    config=stt_config
                )
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error retrieving STT configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving STT configuration"
                )

        @router.post(
            "/stt/config",
            tags=["Speech-to-Text"],
            summary="Update Speech-to-Text configuration",
            description="Update configuration settings for the Speech-to-Text component",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Configuration updated successfully"},
                404: {"description": "STT component not found"},
                422: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_stt_config(request: UpdateSTTConfigRequest) -> Dict[str, Any]:
            """
            Update Speech-to-Text component configuration.
            
            This endpoint allows updating various STT settings including:
            - Language and alternative languages
            - Timeout values
            - Debug mode
            - Component-specific parameters
            
            Only non-null values in the request will be updated.
            """
            try:
                if request.name:
                    stt = self.stts.get(request.name)
                else:
                    # Update current component if name is not specified
                    stt = self.sts.stt

                if not stt:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"Speech-to-Text service not found: {request.name}"
                    )

                updated = {}
                for k, v in request.config.model_dump().items():
                    if v is not None and hasattr(stt, k):
                        try:
                            stt.__setattr__(k, v)
                            updated[k] = v
                        except Exception as attr_ex:
                            logger.warning(f"Failed to set attribute {k} to {v}: {attr_ex}")

                return updated
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error updating STT configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating STT configuration"
                )

        @router.get(
            "/llm/config",
            tags=["LLM"],
            summary="Get LLM configuration",
            description="Retrieve the current configuration settings for the Large Language Model component",
            response_description="Current LLM configuration including model, temperature, and prompt settings",
            responses={
                200: {"description": "Successfully retrieved LLM configuration"},
                404: {"description": "LLM component not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_llm_config(
            name: Optional[str] = Query(
                None, 
                description="Name of specific LLM component to retrieve. If not provided, returns current active component"
            )
        ) -> LLMConfigResponse:
            """
            Get detailed configuration for Large Language Model component.
            
            This endpoint returns comprehensive configuration information including:
            - Model name and parameters (temperature, max_tokens)
            - System prompt and text processing settings
            - Dynamic tools configuration
            - Context manager settings
            - Debug mode status
            """
            try:
                if name:
                    llm = self.llms.get(name)
                else:
                    # Update current component if name is not specified
                    llm = self.sts.llm

                if not llm:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"LLM service not found: {name}"
                    )

                llm_config = LLMConfig(
                    system_prompt=llm.system_prompt,
                    model=llm.model,
                    temperature=llm.temperature,
                    split_chars=llm.split_chars,
                    option_split_chars=llm.option_split_chars,
                    option_split_threshold=llm.option_split_threshold,
                    voice_text_tag=llm.voice_text_tag,
                    use_dynamic_tools=llm.use_dynamic_tools,
                    context_manager=llm.context_manager.__class__.__name__,
                    debug=llm.debug
                )

                # Add any additional attributes that might exist on specific LLM implementations
                if hasattr(llm, "base_url"):
                    llm_config.base_url = llm.base_url
                if hasattr(llm, "max_tokens"):
                    llm_config.max_tokens = llm.max_tokens
                if hasattr(llm, "thinking_budget"):
                    llm_config.thinking_budget = llm.thinking_budget

                return LLMConfigResponse(
                    type=llm.__class__.__name__,
                    name=next((k for k, v in self.llms.items() if v == llm), None),
                    config=llm_config
                )
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error retrieving LLM configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving LLM configuration"
                )

        @router.post(
            "/llm/config",
            tags=["LLM"],
            summary="Update LLM configuration",
            description="Update configuration settings for the Large Language Model component",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Configuration updated successfully"},
                404: {"description": "LLM component not found"},
                422: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_llm_config(request: UpdateLLMConfigRequest) -> Dict[str, Any]:
            """
            Update Large Language Model component configuration.
            
            This endpoint allows updating various LLM settings including:
            - Model name and parameters (temperature, max_tokens)
            - System prompt and text processing settings
            - Dynamic tools configuration
            - Debug mode
            
            Note: Context manager settings cannot be updated through this endpoint.
            Only non-null values in the request will be updated.
            """
            try:
                if request.name:
                    llm = self.llms.get(request.name)
                else:
                    # Update current component if name is not specified
                    llm = self.sts.llm

                if not llm:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"LLM service not found: {request.name}"
                    )

                updated = {}
                for k, v in request.config.model_dump().items():
                    if v is not None and hasattr(llm, k) and k != "context_manager":
                        try:
                            llm.__setattr__(k, v)
                            updated[k] = v
                        except Exception as attr_ex:
                            logger.warning(f"Failed to set attribute {k} to {v}: {attr_ex}")

                return updated
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error updating LLM configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating LLM configuration"
                )

        @router.get(
            "/tts/config",
            tags=["Text-to-Speech"],
            summary="Get Text-to-Speech configuration",
            description="Retrieve the current configuration settings for the Text-to-Speech component",
            response_description="Current TTS configuration including voice, speaker, and style settings",
            responses={
                200: {"description": "Successfully retrieved TTS configuration"},
                404: {"description": "TTS component not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_tts_config(
            name: Optional[str] = Query(
                None, 
                description="Name of specific TTS component to retrieve. If not provided, returns current active component"
            )
        ) -> TTSConfigResponse:
            """
            Get detailed configuration for Text-to-Speech component.
            
            This endpoint returns comprehensive configuration information including:
            - Voice and speaker settings
            - Style mapping configurations
            - Timeout values
            - Audio format settings
            - Debug mode status
            """
            try:
                if name:
                    tts = self.ttss.get(name)
                else:
                    # Update current component if name is not specified
                    tts = self.sts.tts

                if not tts:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"Text-to-Speech service not found: {name}"
                    )

                tts_config = TTSConfig(
                    style_mapper=tts.style_mapper,
                    timeout=getattr(tts.http_client.timeout, "timeout", None) if hasattr(tts, "http_client") and tts.http_client else None,
                    debug=tts.debug
                )

                # Add any additional attributes that might exist on specific TTS implementations
                if hasattr(tts, "model"):
                    tts_config.model = tts.model
                if hasattr(tts, "voice"):
                    tts_config.voice = tts.voice
                if hasattr(tts, "speaker"):
                    tts_config.speaker = tts.speaker
                if hasattr(tts, "base_url"):
                    tts_config.base_url = tts.base_url
                if hasattr(tts, "url"):
                    tts_config.url = tts.url
                if hasattr(tts, "speed"):
                    tts_config.speed = tts.speed
                if hasattr(tts, "pitch"):
                    tts_config.pitch = tts.pitch
                if hasattr(tts, "volume"):
                    tts_config.volume = tts.volume

                return TTSConfigResponse(
                    type=tts.__class__.__name__,
                    name=next((k for k, v in self.ttss.items() if v == tts), None),
                    config=tts_config
                )
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error retrieving TTS configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving TTS configuration"
                )

        @router.post(
            "/tts/config",
            tags=["Text-to-Speech"],
            summary="Update Text-to-Speech configuration",
            description="Update configuration settings for the Text-to-Speech component",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Configuration updated successfully"},
                404: {"description": "TTS component not found"},
                422: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_tts_config(request: UpdateTTSConfigRequest) -> Dict[str, Any]:
            """
            Update Text-to-Speech component configuration.
            
            This endpoint allows updating various TTS settings including:
            - Voice and speaker settings
            - Style mapping configurations
            - Timeout values
            - Audio format parameters
            - Debug mode
            
            Only non-null values in the request will be updated.
            """
            try:
                if request.name:
                    tts = self.ttss.get(request.name)
                else:
                    # Update current component if name is not specified
                    tts = self.sts.tts

                if not tts:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"Text-to-Speech service not found: {request.name}"
                    )

                updated = {}
                for k, v in request.config.model_dump().items():
                    if v is not None and hasattr(tts, k):
                        try:
                            tts.__setattr__(k, v)
                            updated[k] = v
                        except Exception as attr_ex:
                            logger.warning(f"Failed to set attribute {k} to {v}: {attr_ex}")

                return updated
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error updating TTS configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating TTS configuration"
                )

        @router.get(
            "/sts/component",
            tags=["Pipeline"],
            summary="Get pipeline components",
            description="Retrieve all registered Speech-to-Speech pipeline components",
            response_description="List of available STT, LLM, and TTS components",
            responses={
                200: {"description": "Successfully retrieved pipeline components"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_sts_component() -> STSComponentResponse:
            """
            Get all registered Speech-to-Speech pipeline components.
            
            This endpoint returns information about all available components:
            - Speech-to-Text (STT) components
            - Large Language Model (LLM) components
            - Text-to-Speech (TTS) components
            
            Each component includes its type (class name) and registered name.
            """
            try:
                return STSComponentResponse(
                    stts=[STSComponent(type=v.__class__.__name__, name=k) for k, v in self.stts.items()],
                    llms=[STSComponent(type=v.__class__.__name__, name=k) for k, v in self.llms.items()],
                    ttss=[STSComponent(type=v.__class__.__name__, name=k) for k, v in self.ttss.items()]
                )
            except Exception as ex:
                logger.error(f"Error retrieving STS components: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving STS components"
                )

        @router.post(
            "/sts/component",
            tags=["Pipeline"],
            summary="Update pipeline components",
            description="Switch the active components used in the Speech-to-Speech pipeline",
            response_description="Dictionary of successfully switched components",
            responses={
                200: {"description": "Components switched successfully"},
                400: {"description": "Invalid component name"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_sts_component(request: UpdateSTSComponentRequest) -> Dict[str, Any]:
            """
            Switch the active components used in the Speech-to-Speech pipeline.
            
            This endpoint allows switching between different registered components:
            - STT: Switch the active Speech-to-Text component
            - LLM: Switch the active Large Language Model component
            - TTS: Switch the active Text-to-Speech component
            
            Only specify the components you want to switch. Unspecified components remain unchanged.
            """
            try:
                switched = {}

                if request.stt and request.stt in self.stts:
                    self.sts.stt = self.stts[request.stt]
                    switched["stt"] = request.stt
                elif request.stt and request.stt not in self.stts:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"STT component not found: {request.stt}"
                    )

                if request.llm and request.llm in self.llms:
                    self.sts.llm = self.llms[request.llm]
                    switched["llm"] = request.llm
                elif request.llm and request.llm not in self.llms:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"LLM component not found: {request.llm}"
                    )

                if request.tts and request.tts in self.ttss:
                    self.sts.tts = self.ttss[request.tts]
                    switched["tts"] = request.tts
                elif request.tts and request.tts not in self.ttss:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"TTS component not found: {request.tts}"
                    )

                return switched
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error switching STS components: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while switching STS components"
                )

        @router.get(
            "/system/log",
            tags=["System"],
            summary="Get recent system logs",
            description="Retrieve the most recent system log entries",
            response_description="List of recent log lines",
            responses={
                200: {"description": "Successfully retrieved log entries"},
                404: {"description": "Log file not configured or not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_system_log(
            count: int = Query(
                100, 
                description="Number of recent log lines to retrieve", 
                ge=1, 
                le=1000
            )
        ) -> LogResponse:
            """
            Get recent system log entries.
            
            This endpoint retrieves the most recent log entries from the configured log file.
            Useful for monitoring system status and troubleshooting issues.
            """
            try:
                if not self.logfile_path:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Log file not configured"
                    )

                with open(self.logfile_path, "r", encoding="utf-8") as f:
                    deque_lines = collections.deque(f, maxlen=count)
                    return LogResponse(lines=list(deque_lines))
            
            except HTTPException:
                raise
            except FileNotFoundError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Log file not found: {self.logfile_path}"
                )
            except Exception as ex:
                logger.error(f"Error reading log file: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while reading log file"
                )

        return router
