import logging
import re
from typing import Optional
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from ..sts.models import STSRequest, STSResponse
from ..adapter.base import Adapter
from .auth import create_api_key_dependency

logger = logging.getLogger(__name__)


class APIResponse(BaseModel):
    """
    APIResponse is a standard response model for successful API operations.
    
    Used to return a simple success message or status information.
    """
    message: str = Field(
        ..., 
        example="Message from API", 
        description="Success message from API operation"
    )


class SpeechRequest(BaseModel):
    """
    SpeechRequest contains text for avatar speech synthesis and performance.

    Supports control tags for face expressions and animations embedded in text.
    """
    text: str = Field(
        ...,
        example="[face:joy]Hi, let's talk with me!",
        description="Text to synthesize with optional face and animation control tags"
    )
    session_id: Optional[str] = Field(
        default=None,
        example="local_session",
        description="Session Id to route the response to a specific client (e.g., WebSocket connection)"
    )


class ChatRequest(BaseModel):
    """
    ChatRequest contains a text message for conversation processing.

    Used to send messages to the AIAvatar's conversation processing pipeline.
    """
    text: str = Field(
        ...,
        example="こんにちは！",
        description="Text message to send to the conversation processor"
    )
    session_id: Optional[str] = Field(
        default=None,
        example="local_session",
        description="Session Id to route the response to a specific client (e.g., WebSocket connection)"
    )
    user_id: Optional[str] = Field(
        default=None,
        example="user_001",
        description="User Id for conversation context management (e.g., per-user memory)"
    )


class ControlAPI:
    def __init__(self, adapter: Adapter, default_session_id: str = None):
        self.adapter = adapter
        self.default_session_id = default_session_id

    def remove_control_tags(self, text: str) -> str:
        clean_text = text
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(
            "/avatar/perform", 
            tags=["Avatar Control"], 
            summary="Perform avatar speech with controls",
            description="Synthesize speech and perform avatar controls (face/animation) based on embedded tags",
            response_description="Performance completed successfully",
            responses={
                200: {"description": "Avatar performance completed successfully"},
                422: {"description": "Invalid text or control tags"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_avatar_perform(request: SpeechRequest) -> APIResponse:
            """
            Perform comprehensive avatar actions including speech, face, and animation.
            
            This endpoint processes text with embedded control tags:
            - Synthesizes speech from the provided text (excluding control tags)
            - Parses control tags for face expressions and animations
            - Executes synchronized avatar performance
            
            Control tag format:
            - Face: [face:expression_name] (e.g., [face:joy])
            - Animation: [animation:animation_name] (e.g., [animation:wave_hands])
            
            Example: "[face:joy]Hello there! [animation:wave_hands]Nice to meet you!"
            """
            try:
                session_id = request.session_id or self.default_session_id
                voice = await self.adapter.sts.tts.synthesize(text=self.remove_control_tags(request.text))

                await self.adapter.stop_response(session_id, "_")
                await self.adapter.handle_response(STSResponse(
                    type="chunk",
                    session_id=session_id,
                    text=request.text,
                    voice_text=self.remove_control_tags(request.text),
                    audio_data=voice,
                    metadata={}
                ))

                return APIResponse(message="Avatar performance completed successfully")
            
            except Exception as ex:
                logger.exception(f"Error performing avatar actions")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while performing avatar actions"
                )

        @router.post(
            "/conversation", 
            tags=["Avatar Control"], 
            summary="Send message to conversation processor",
            description="Send a text message to the AIAvatar conversation processing pipeline",
            response_description="Success message indicating conversation processing completion",
            responses={
                200: {"description": "Message processed successfully"},
                400: {"description": "AIAvatar is not listening"},
                422: {"description": "Invalid message format"},
                500: {"description": "Internal server error"}
            }
        )
        async def processor_chat(request: ChatRequest) -> APIResponse:
            """
            Send a text message to the conversation processing pipeline.
            
            This endpoint processes a text message through the complete STS pipeline:
            - Validates that the avatar is currently listening
            - Invokes the Speech-to-Speech pipeline with the text input
            - Processes LLM response and generates appropriate avatar actions
            - Handles TTS synthesis and avatar control synchronization
            
            The message will be processed as if it were spoken input, triggering
            the full conversation flow including context management and response generation.
            """
            try:
                session_id = request.session_id or self.default_session_id
                context_id = self.adapter.sts.vad.get_session_data(session_id, "context_id") if session_id else None
                async for resp in self.adapter.sts.invoke(STSRequest(session_id=session_id, user_id=request.user_id, context_id=context_id, text=request.text)):
                    if resp.type == "start":
                        if session_id:
                            self.adapter.sts.vad.set_session_data(session_id, "context_id", resp.context_id)
                    await self.adapter.handle_response(resp)

                return APIResponse(message="Message processed successfully")
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error processing conversation message: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while processing conversation message"
                )

        return router


def setup_control_api(
    app: FastAPI,
    *,
    adapter: Adapter,
    default_session_id: str = None,
    api_key: str = None,
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []
    app.include_router(
        ControlAPI(adapter=adapter, default_session_id=default_session_id).get_router(),
        dependencies=deps,
    )
