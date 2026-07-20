import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from ...sts.tts import SpeechSynthesizer

logger = logging.getLogger(__name__)


class TtsConfig(BaseModel):
    """
    TtsConfig is a data model that holds configuration settings for Text-to-Speech (TTS) components.

    All fields are optional and default to None if not provided.
    """
    speaker: Optional[str] = Field(
        default=None,
        description="The voice speaker name or ID for speech synthesis."
    )
    style_mapper: Optional[Dict[str, str]] = Field(
        default=None,
        description="A mapping of style keywords to speaker style values."
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


class TtsConfigResponse(BaseModel):
    type: str = Field(
        description="Type of TTS component"
    )
    config: TtsConfig = Field(
        description="Configuration of TTS component"
    )


class UpdateTtsConfigRequest(BaseModel):
    config: TtsConfig = Field(
        description="Configuration of TTS component"
    )


class TtsConfigAPI:
    def __init__(
        self,
        tts: SpeechSynthesizer
    ):
        self.tts = tts

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/config/tts",
            tags=["Config"],
            summary="Get TTS configuration",
            description="Retrieve the current configuration settings for the TTS component",
            response_description="Current TTS configuration including speaker, style, and timeout settings",
            responses={
                200: {"description": "Successfully retrieved TTS configuration"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_tts() -> TtsConfigResponse:
            try:
                tts = self.tts

                return TtsConfigResponse(
                    type=tts.__class__.__name__,
                    config=TtsConfig(**tts.get_config()),
                )

            except Exception as ex:
                logger.error(f"Error retrieving TTS configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving TTS configuration"
                )

        @router.post(
            "/config/tts",
            tags=["Config"],
            summary="Update TTS configuration",
            description="Update configuration settings for the TTS component. Only non-null values will be updated.",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Successfully updated TTS configuration"},
                400: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_config_tts(request: UpdateTtsConfigRequest) -> Dict[str, Any]:
            try:
                tts = self.tts
                return tts.set_config(request.config.model_dump())

            except Exception as ex:
                logger.error(f"Error updating TTS configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating TTS configuration"
                )

        return router
