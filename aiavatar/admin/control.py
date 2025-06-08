import asyncio
import logging
import re
from typing import Union
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from aiavatar.sts.models import STSRequest
from aiavatar import AIAvatar, AIAvatarResponse, AIAvatarClientBase

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


class ListenerRequest(BaseModel):
    session_id: str = Field(
        ..., 
        example="local_session", 
        description="Session Id for the conversation"
    )


class ListenerStatusResponse(BaseModel):
    """
    ListenerStatusResponse returns the current listening status of the AIAvatar.
    
    Indicates whether the avatar is actively listening for voice input.
    """
    is_listening: bool = Field(
        ..., 
        description="Whether the avatar listener is currently active and listening for voice input"
    )
    session_id: str = Field(
        ..., 
        example="local_session", 
        description="Current Session Id for the conversation"
    )


class GetAvatarStatusResponse(BaseModel):
    """
    GetAvatarStatusResponse returns the current avatar state information.
    
    Provides the current face expression and animation status.
    """
    current_face: str = Field(
        ..., 
        example="fun", 
        description="Name of the currently active face expression"
    )
    current_animation: str = Field(
        ..., 
        example="wave_hands", 
        description="Name of the currently active animation"
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


class GetIsSpeakingResponse(BaseModel):
    """
    GetIsSpeakingResponse returns the current speech status of the avatar.
    
    Indicates whether the avatar is currently performing speech synthesis.
    """
    is_speaking: bool = Field(
        ..., 
        description="Whether the avatar is currently speaking or performing speech synthesis"
    )


class FaceRequest(BaseModel):
    """
    FaceRequest specifies face expression settings for the avatar.
    
    Allows setting a specific face expression with duration control.
    """
    name: str = Field(
        ..., 
        example="fun", 
        description="Name of the face expression to set (e.g., 'joy', 'angry', 'neutral')"
    )
    duration: float = Field(
        4.0, 
        example=4.0, 
        description="Duration in seconds for how long the face expression should be active"
    )


class GetFaceResponse(BaseModel):
    """
    GetFaceResponse returns the current face expression status.
    
    Provides information about the currently active face expression.
    """
    current_face: str = Field(
        ..., 
        example="fun", 
        description="Name of the currently active face expression"
    )


class AnimationRequest(BaseModel):
    """
    AnimationRequest specifies animation settings for the avatar.
    
    Allows triggering specific animations with duration control.
    """
    name: str = Field(
        ..., 
        example="wave_hands", 
        description="Name of the animation to perform (e.g., 'wave_hands', 'nod', 'bow')"
    )
    duration: float = Field(
        4.0, 
        example=4.0, 
        description="Duration in seconds for how long the animation should be active"
    )


class GetAnimationResponse(BaseModel):
    """
    GetAnimationResponse returns the current animation status.
    
    Provides information about the currently active animation.
    """
    current_animation: str = Field(
        ..., 
        example="wave_hands", 
        description="Name of the currently active animation"
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


class GetHistoriesResponse(BaseModel):
    """
    GetHistoriesResponse returns conversation history data.
    
    Contains the cached conversation history from the context manager.
    """
    histories: list[dict] = Field(
        ..., 
        example='[{"role": "user", "content": "こんにちは"}, {"role": "assistant", "content": "こんにちは！何かお手伝いすることはありますか？"}]', 
        description="List of conversation history entries with role and content information"
    )


class ControlAPI:
    def __init__(self, aiavatr_app: Union[AIAvatar, AIAvatarClientBase]):
        self.aiavatr_app = aiavatr_app
        self.current_session_id = None

    def is_listening(self) -> bool:
        return not (self.aiavatr_app.send_microphone_task is None or self.aiavatr_app.send_microphone_task.cancelled()) \
                    and not (self.aiavatr_app.receive_response_task is None or self.aiavatr_app.receive_response_task.cancelled())

    def remove_control_tags(self, text: str) -> str:
        clean_text = text
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(
            "/listener/start", 
            tags=["Listener"], 
            summary="Start voice listener",
            description="Start the AIAvatar voice listener to begin processing audio input",
            response_description="Status message indicating whether listener was started or already running",
            responses={
                200: {"description": "Listener started successfully or already running"},
                500: {"description": "Internal server error"}
            }
        )
        async def listener_start(request: ListenerRequest) -> APIResponse:
            """
            Start the AIAvatar voice listener.
            
            This endpoint initiates the voice listening process, enabling the avatar to:
            - Capture audio input from the microphone
            - Process voice activity detection
            - Trigger speech recognition and conversation processing
            
            If the listener is already active, returns a status message without changes.
            """
            try:
                if not self.is_listening():
                    asyncio.create_task(self.aiavatr_app.start_listening(session_id=request.session_id))
                    self.current_session_id = request.session_id
                    return APIResponse(message="Listener start requested")
                else:
                    return APIResponse(message="Listener already running")
            
            except Exception as ex:
                logger.error(f"Error starting listener: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while starting listener"
                )

        @router.post(
            "/listener/stop", 
            tags=["Listener"], 
            summary="Stop voice listener",
            description="Stop the AIAvatar voice listener and cease audio input processing",
            response_description="Status message confirming listener stop request",
            responses={
                200: {"description": "Listener stopped successfully"},
                500: {"description": "Internal server error"}
            }
        )
        async def listener_stop() -> APIResponse:
            """
            Stop the AIAvatar voice listener.
            
            This endpoint terminates the voice listening process:
            - Stops microphone audio capture
            - Cancels active voice processing tasks
            - Ends the current listening session
            
            Safe to call even if listener is not currently active.
            """
            try:
                await self.aiavatr_app.stop_listening(session_id=self.current_session_id)
                return APIResponse(message="Listener stop requested")
            
            except Exception as ex:
                logger.error(f"Error stopping listener: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while stopping listener"
                )

        @router.get(
            "/listener/status", 
            tags=["Listener"], 
            summary="Get voice listener status",
            description="Check whether the AIAvatar voice listener is currently active",
            response_description="Current listening status",
            responses={
                200: {"description": "Successfully retrieved listener status"},
                500: {"description": "Internal server error"}
            }
        )
        async def listener_status() -> ListenerStatusResponse:
            """
            Get the current status of the AIAvatar voice listener.
            
            This endpoint returns whether the avatar is actively listening for voice input.
            The listener is considered active when both microphone capture and response
            processing tasks are running.
            """
            try:
                return ListenerStatusResponse(
                    is_listening=self.is_listening(),
                    session_id=self.current_session_id
                )
            
            except Exception as ex:
                logger.error(f"Error retrieving listener status: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving listener status"
                )

        @router.get(
            "/avatar/status", 
            tags=["Avatar"], 
            summary="Get avatar status",
            description="Retrieve the current status of avatar face and animation",
            response_description="Current avatar state including face expression and animation",
            responses={
                200: {"description": "Successfully retrieved avatar status"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_avatar_status() -> GetAvatarStatusResponse:
            """
            Get the current status of the avatar.
            
            This endpoint returns comprehensive avatar state information:
            - Current active face expression
            - Current active animation
            
            Useful for monitoring avatar state and synchronizing external systems.
            """
            try:
                return GetAvatarStatusResponse(
                    current_face=self.aiavatr_app.face_controller.current_face,
                    current_animation=self.aiavatr_app.animation_controller.current_animation
                )
            
            except Exception as ex:
                logger.error(f"Error retrieving avatar status: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving avatar status"
                )

        @router.post(
            "/avatar/face", 
            tags=["Avatar"], 
            summary="Set avatar face expression",
            description="Set a specific face expression for the avatar with duration control",
            response_description="Success message confirming face expression was set",
            responses={
                200: {"description": "Face expression set successfully"},
                422: {"description": "Invalid face expression name or duration"},
                500: {"description": "Internal server error"}
            }
        )
        async def avatar_face(request: FaceRequest) -> APIResponse:
            """
            Set the avatar's face expression.
            
            This endpoint updates the avatar's facial expression:
            - Sets the specified face expression
            - Applies the expression for the given duration
            - Automatically reverts to neutral after duration expires
            
            Common face expressions include: 'joy', 'angry', 'sorrow', 'surprised', 
            'fun', 'neutral', 'think'.
            """
            try:
                await self.aiavatr_app.face_controller.set_face(request.name, request.duration)
                return APIResponse(message="Face expression set successfully")
            
            except Exception as ex:
                logger.error(f"Error setting face expression: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while setting face expression"
                )

        @router.get(
            "/avatar/face", 
            tags=["Avatar"], 
            summary="Get current face expression",
            description="Retrieve the currently active face expression of the avatar",
            response_description="Current face expression name",
            responses={
                200: {"description": "Successfully retrieved current face expression"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_avatar_face() -> GetFaceResponse:
            """
            Get the avatar's current face expression.
            
            This endpoint returns the name of the currently active face expression.
            Useful for monitoring avatar state and ensuring synchronization with
            external systems or user interfaces.
            """
            try:
                current_face = self.aiavatr_app.face_controller.current_face
                return GetFaceResponse(current_face=current_face)
            
            except Exception as ex:
                logger.error(f"Error retrieving current face expression: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving face expression"
                )

        @router.post(
            "/avatar/animation", 
            tags=["Avatar"], 
            summary="Set avatar animation",
            description="Trigger a specific animation for the avatar with duration control",
            response_description="Success message confirming animation was triggered",
            responses={
                200: {"description": "Animation triggered successfully"},
                422: {"description": "Invalid animation name or duration"},
                500: {"description": "Internal server error"}
            }
        )
        async def avatar_animation(request: AnimationRequest) -> APIResponse:
            """
            Trigger an avatar animation.
            
            This endpoint starts a specific animation sequence:
            - Triggers the specified animation
            - Runs the animation for the given duration
            - Automatically returns to idle state after completion
            
            Common animations include: 'wave_hands', 'nod', 'bow', 'clap', 
            'point', 'thumbs_up'.
            """
            try:
                await self.aiavatr_app.animation_controller.animate(request.name, request.duration)
                return APIResponse(message="Animation triggered successfully")
            
            except Exception as ex:
                logger.error(f"Error triggering animation: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while triggering animation"
                )

        @router.get(
            "/avatar/animation", 
            tags=["Avatar"], 
            summary="Get current animation",
            description="Retrieve the currently active animation of the avatar",
            response_description="Current animation name",
            responses={
                200: {"description": "Successfully retrieved current animation"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_avatar_animation() -> GetAnimationResponse:
            """
            Get the avatar's current animation.
            
            This endpoint returns the name of the currently active animation.
            Useful for monitoring avatar state and ensuring synchronization with
            external systems or user interfaces.
            """
            try:
                current_animation = self.aiavatr_app.animation_controller.current_animation
                return GetAnimationResponse(current_animation=current_animation)
            
            except Exception as ex:
                logger.error(f"Error retrieving current animation: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving animation"
                )

        @router.post(
            "/avatar/perform", 
            tags=["Avatar"], 
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
                voice = await self.aiavatr_app.sts.tts.synthesize(text=self.remove_control_tags(request.text))
                avatar_control_request = self.aiavatr_app.parse_avatar_control_request(request.text)

                await self.aiavatr_app.stop_response("_", "_")
                await self.aiavatr_app.perform_response(AIAvatarResponse(
                    type="control_api",
                    audio_data=voice,
                    avatar_control_request=avatar_control_request
                ))
                
                return APIResponse(message="Avatar performance completed successfully")
            
            except Exception as ex:
                logger.error(f"Error performing avatar actions: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while performing avatar actions"
                )

        @router.post(
            "/conversation", 
            tags=["Conversation"], 
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
                if not self.is_listening():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="AIAvatar is not listening. Please start the listener first."
                    )

                context_id = self.aiavatr_app.sts.vad.get_session_data(self.current_session_id, "context_id")
                async for resp in self.aiavatr_app.sts.invoke(STSRequest(session_id=self.current_session_id, context_id=context_id, text=request.text)):
                    if resp.type == "start":
                        self.aiavatr_app.sts.vad.set_session_data(self.current_session_id, "context_id", resp.context_id)
                    await self.aiavatr_app.handle_response(resp)

                return APIResponse(message="Message processed successfully")
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error processing conversation message: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while processing conversation message"
                )

        @router.get(
            "/conversation/histories", 
            tags=["Conversation"], 
            summary="Get conversation history",
            description="Retrieve the current conversation history from the context manager",
            response_description="List of conversation history entries",
            responses={
                200: {"description": "Successfully retrieved conversation history"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_processor_histories() -> GetHistoriesResponse:
            """
            Get the current conversation history.
            
            This endpoint retrieves conversation history from the context manager:
            - Returns all messages in the current conversation context
            - Includes both user inputs and assistant responses
            - Maintains chronological order of the conversation
            
            Useful for displaying conversation history in user interfaces or
            for debugging conversation flow.
            """
            try:
                context_id = self.aiavatr_app.sts.vad.get_session_data(self.current_session_id, "context_id")
                histories = await self.aiavatr_app.sts.llm.context_manager.get_histories(context_id)
                return GetHistoriesResponse(histories=histories)
            
            except Exception as ex:
                logger.error(f"Error retrieving conversation histories: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving conversation histories"
                )

        return router
