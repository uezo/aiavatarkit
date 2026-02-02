import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from ...sts.llm import LLMService

logger = logging.getLogger(__name__)


class LlmConfig(BaseModel):
    """
    LlmConfig is a data model that holds configuration settings for Large Language Model (LLM) components.

    All fields are optional and default to None if not provided.
    """
    system_prompt: Optional[str] = Field(
        default=None,
        description="The system prompt that sets initial instructions or context for the LLM."
    )
    model: Optional[str] = Field(
        default=None,
        description="The name of the model to use, e.g., 'gpt-4o-mini', 'claude-haiku-4-5', etc."
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
    option_split_threshold: Optional[int] = Field(
        default=None,
        description="A threshold value to control splitting of optional parts based on character count."
    )
    split_on_control_tags: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to split text on control tags like [face:xxx] or [animation:xxx]."
    )
    voice_text_tag: Optional[str] = Field(
        default=None,
        description="A tag used for voice text extraction, e.g., 'voice'."
    )
    initial_messages: Optional[List[dict]] = Field(
        default=None,
        description="A list of initial messages to prepend to the conversation context."
    )
    use_dynamic_tools: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable dynamic tool invocation."
    )
    debug: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether to enable debug mode. If True, detailed logs are output."
    )

    class Config:
        extra = "allow"


class LlmConfigResponse(BaseModel):
    type: str = Field(
        description="Type of LLM component"
    )
    config: LlmConfig = Field(
        description="Configuration of LLM component"
    )


class UpdateLlmConfigRequest(BaseModel):
    config: LlmConfig = Field(
        description="Configuration of LLM component"
    )


class LlmConfigAPI:
    def __init__(
        self,
        llm: LLMService
    ):
        self.llm = llm

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/config/llm",
            tags=["Config"],
            summary="Get LLM configuration",
            description="Retrieve the current configuration settings for the LLM component",
            response_description="Current LLM configuration including model, temperature, and prompt settings",
            responses={
                200: {"description": "Successfully retrieved LLM configuration"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_llm() -> LlmConfigResponse:
            try:
                llm = self.llm

                return LlmConfigResponse(
                    type=llm.__class__.__name__,
                    config=LlmConfig(**llm.get_config()),
                )

            except Exception as ex:
                logger.error(f"Error retrieving LLM configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving LLM configuration"
                )

        @router.post(
            "/config/llm",
            tags=["Config"],
            summary="Update LLM configuration",
            description="Update configuration settings for the LLM component. Only non-null values will be updated.",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Successfully updated LLM configuration"},
                400: {"description": "Invalid configuration parameters"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_config_llm(request: UpdateLlmConfigRequest) -> Dict[str, Any]:
            try:
                llm = self.llm
                return llm.set_config(request.config.model_dump())

            except Exception as ex:
                logger.error(f"Error updating LLM configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating LLM configuration"
                )

        return router
