import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from ...sts.pipeline import STSPipeline

logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """
    PipelineConfig is a data model that holds configuration settings for the STS pipeline.

    All fields are optional and default to None if not provided.
    """
    wakewords: Optional[List[str]] = Field(
        default=None,
        description="A list of wakewords to activate the pipeline."
    )
    wakeword_timeout: Optional[float] = Field(
        default=None,
        description="Timeout in seconds after which the pipeline requires a wakeword again."
    )
    merge_request_threshold: Optional[float] = Field(
        default=None,
        description="Time threshold in seconds for merging consecutive requests."
    )
    merge_request_prefix: Optional[str] = Field(
        default=None,
        description="Prefix text added when merging consecutive requests."
    )
    timestamp_interval_seconds: Optional[float] = Field(
        default=None,
        description="Interval in seconds for inserting timestamps into requests. 0 to disable."
    )
    timestamp_prefix: Optional[str] = Field(
        default=None,
        description="Prefix text for inserted timestamps."
    )
    timestamp_timezone: Optional[str] = Field(
        default=None,
        description="Timezone for inserted timestamps, e.g., 'UTC', 'Asia/Tokyo'."
    )
    voice_recorder_enabled: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable voice recording."
    )
    invoke_queue_idle_timeout: Optional[float] = Field(
        default=None,
        description="Idle timeout in seconds for the invoke queue worker."
    )
    invoke_timeout: Optional[float] = Field(
        default=None,
        description="Timeout in seconds for a single invoke operation."
    )
    use_invoke_queue: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to use the invoke queue for sequential processing."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class PipelineConfigResponse(BaseModel):
    config: PipelineConfig = Field(
        description="Configuration of STS pipeline"
    )


class UpdatePipelineConfigRequest(BaseModel):
    config: PipelineConfig = Field(
        description="Configuration of STS pipeline"
    )


class PipelineConfigAPI:
    def __init__(
        self,
        pipeline: STSPipeline
    ):
        self.pipeline = pipeline

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/config/pipeline",
            tags=["Config"],
            summary="Get pipeline configuration",
            description="Retrieve the current configuration settings for the STS pipeline",
            response_description="Current pipeline configuration including wakewords, timestamps, and queue settings",
            responses={
                200: {"description": "Successfully retrieved pipeline configuration"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_pipeline() -> PipelineConfigResponse:
            try:
                pipeline = self.pipeline

                return PipelineConfigResponse(
                    config=PipelineConfig(**pipeline.get_config()),
                )

            except Exception as ex:
                logger.error(f"Error retrieving pipeline configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving pipeline configuration"
                )

        @router.post(
            "/config/pipeline",
            tags=["Config"],
            summary="Update pipeline configuration",
            description="Update configuration settings for the STS pipeline. Only non-null values will be updated.",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Successfully updated pipeline configuration"},
                400: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_config_pipeline(request: UpdatePipelineConfigRequest) -> Dict[str, Any]:
            try:
                pipeline = self.pipeline
                return pipeline.set_config(request.config.model_dump())

            except Exception as ex:
                logger.error(f"Error updating pipeline configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating pipeline configuration"
                )

        return router
