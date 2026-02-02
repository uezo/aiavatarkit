import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from ...sts.stt import SpeechRecognizer

logger = logging.getLogger(__name__)


class SttConfig(BaseModel):
    """
    SttConfig is a data model that holds configuration settings for Speech-to-Text (STT) components.

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
    max_retries: Optional[int] = Field(
        default=None,
        description="Maximum number of retry attempts for failed requests."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class SttConfigResponse(BaseModel):
    type: str = Field(
        description="Type of STT component"
    )
    config: SttConfig = Field(
        description="Configuration of STT component"
    )


class UpdateSttConfigRequest(BaseModel):
    config: SttConfig = Field(
        description="Configuration of STT component"
    )


class SttConfigAPI:
    def __init__(
        self,
        stt: SpeechRecognizer
    ):
        self.stt = stt

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/config/stt",
            tags=["Config"],
            summary="Get STT configuration",
            description="Retrieve the current configuration settings for the Speech-to-Text component",
            response_description="Current STT configuration including language, timeout, and debug settings",
            responses={
                200: {"description": "Successfully retrieved STT configuration"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_stt() -> SttConfigResponse:
            try:
                stt = self.stt

                return SttConfigResponse(
                    type=stt.__class__.__name__,
                    config=SttConfig(**stt.get_config()),
                )

            except Exception as ex:
                logger.error(f"Error retrieving STT configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving STT configuration"
                )

        @router.post(
            "/config/stt",
            tags=["Config"],
            summary="Update STT configuration",
            description="Update configuration settings for the STT component. Only non-null values will be updated.",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Successfully updated STT configuration"},
                400: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_config_stt(request: UpdateSttConfigRequest) -> Dict[str, Any]:
            try:
                stt = self.stt
                return stt.set_config(request.config.model_dump())

            except Exception as ex:
                logger.error(f"Error updating STT configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating STT configuration"
                )

        return router
