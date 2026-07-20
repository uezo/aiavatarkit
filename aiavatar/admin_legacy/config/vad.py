import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from ...sts.vad import SpeechDetector

logger = logging.getLogger(__name__)


class VadConfig(BaseModel):
    """
    VadConfig is a data model that holds configuration settings for Voice Activity Detection (VAD) components.

    All fields are optional and default to None if not provided.
    """
    volume_db_threshold: Optional[float] = Field(
        default=None,
        description="Volume threshold in dB for speech detection, e.g., -50.0. Set to null to disable volume-based filtering."
    )
    silence_duration_threshold: Optional[float] = Field(
        default=None,
        description="Duration of silence in seconds required to end a recording, e.g., 0.5."
    )
    max_duration: Optional[float] = Field(
        default=None,
        description="Maximum recording duration in seconds, e.g., 10.0."
    )
    min_duration: Optional[float] = Field(
        default=None,
        description="Minimum recording duration in seconds. Recordings shorter than this are discarded, e.g., 0.2."
    )
    sample_rate: Optional[int] = Field(
        default=None,
        description="Audio sample rate in Hz, e.g., 16000."
    )
    channels: Optional[int] = Field(
        default=None,
        description="Number of audio channels, e.g., 1 for mono."
    )
    preroll_buffer_count: Optional[int] = Field(
        default=None,
        description="Number of audio frames to buffer before speech is detected, e.g., 5."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class VadConfigResponse(BaseModel):
    type: str = Field(
        description="Type of VAD component"
    )
    config: VadConfig = Field(
        description="Configuration of VAD component"
    )


class UpdateVadConfigRequest(BaseModel):
    config: VadConfig = Field(
        description="Configuration of VAD component"
    )


class VadConfigAPI:
    def __init__(
        self,
        vad: SpeechDetector
    ):
        self.vad = vad

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/config/vad",
            tags=["Config"],
            summary="Get VAD configuration",
            description="Retrieve the current configuration settings for the VAD component",
            response_description="Current VAD configuration including volume threshold, silence/duration settings, and debug mode",
            responses={
                200: {"description": "Successfully retrieved VAD configuration"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_vad() -> VadConfigResponse:
            try:
                vad = self.vad

                return VadConfigResponse(
                    type=vad.__class__.__name__,
                    config=VadConfig(**vad.get_config()),
                )

            except Exception as ex:
                logger.error(f"Error retrieving VAD configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving VAD configuration"
                )

        @router.post(
            "/config/vad",
            tags=["Config"],
            summary="Update VAD configuration",
            description="Update configuration settings for the VAD component. Only non-null values will be updated.",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Successfully updated VAD configuration"},
                400: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_config_vad(request: UpdateVadConfigRequest) -> Dict[str, Any]:
            try:
                vad = self.vad
                return vad.set_config(request.config.model_dump())

            except Exception as ex:
                logger.error(f"Error updating VAD configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating VAD configuration"
                )

        return router
