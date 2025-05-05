import asyncio
import collections
import logging
import re
import traceback
from fastapi import APIRouter
from fastapi.responses import Response
from aiavatar.sts.models import STSRequest
from aiavatar import AIAvatar, AIAvatarResponse
from .schema import *


logger = logging.getLogger(__name__)


def get_router(aiavatr_app: AIAvatar, logfile_path: str) -> APIRouter:
    api_router = APIRouter()

    def is_listening() -> bool:
       return not (aiavatr_app.send_microphone_task is None or aiavatr_app.send_microphone_task.cancelled()) \
                and not (aiavatr_app.receive_response_task is None or aiavatr_app.receive_response_task.cancelled())

    def remove_control_tags(text: str) -> str:
        clean_text = text
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    @api_router.post("/listener/start", tags=["Listener"], name="Start listener")
    async def listener_start(response: Response) -> APIResponse:
        try:
            if not is_listening():
                asyncio.create_task(aiavatr_app.start_listening())

                return APIResponse(message="start requested")
            else:
                return APIResponse(message="already running")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/listener/stop", tags=["Listener"], name="Stop listener")
    async def listener_stop() -> APIResponse:
        await aiavatr_app.stop_listening("local_session")
        return APIResponse(message="stop requested")


    @api_router.get("/listener/status", tags=["Listener"], name="Check the status of listener")
    async def listener_status() -> ListenerStatusResponse:
        return ListenerStatusResponse(
            is_listening=is_listening()
        )


    @api_router.get("/avatar/status", tags=["Avatar"], name="Get avatar status")
    async def get_avatar_status(response: Response) -> GetAvatarStatusResponse:
        try:
            return GetAvatarStatusResponse(
                current_face=aiavatr_app.face_controller.current_face,
                current_animation=aiavatr_app.animation_controller.current_animation
            )

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/face", tags=["Avatar"], name="Set face expression")
    async def avatar_face(request: FaceRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.face_controller.set_face(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/avatar/face", tags=["Avatar"], name="Get current face expression")
    async def get_avatar_face(response: Response) -> GetFaceResponse:
        try:
            current_face = aiavatr_app.face_controller.current_face
            return GetFaceResponse(current_face=current_face)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/animation", tags=["Avatar"], name="Set animation")
    async def avatar_animation(request: AnimationRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.animation_controller.animate(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/avatar/animation", tags=["Avatar"], name="Get current animation")
    async def get_avatar_animation(response: Response) -> GetAnimationResponse:
        try:
            current_animation = aiavatr_app.animation_controller.current_animation
            return GetAnimationResponse(current_animation=current_animation)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")

    @api_router.post("/avatar/perform", tags=["Avatar"], name="Perform voice, face and animation")
    async def post_avatar_perform(request: SpeechRequest):
        voice = await aiavatr_app.sts.tts.synthesize(text=remove_control_tags(request.text))
        avatar_control_request = aiavatr_app.parse_avatar_control_request(request.text)

        await aiavatr_app.stop_response("_", "_")
        await aiavatr_app.perform_response(AIAvatarResponse(
            type="control_api",
            audio_data=voice,
            avatar_control_request=avatar_control_request
        ))


    @api_router.post("/conversation", tags=["Conversation"], name="Send message to AIAvatar")
    async def processor_chat(request: ChatRequest, response: Response) -> APIResponse:
        try:
            if not is_listening():
                response.status_code = 400
                return APIResponse(message="AIAvatar is not listening.")

            context_id = aiavatr_app.sts.vad.get_session_data("local_session", "context_id")
            async for resp in aiavatr_app.sts.invoke(STSRequest(session_id="local_session", context_id=context_id, text=request.text)):
                if resp.type == "start":
                    aiavatr_app.sts.vad.set_session_data("local_session", "context_id", resp.context_id)
                await aiavatr_app.handle_response(resp)

            return APIResponse(message="Done.")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/conversation/histories", tags=["Conversation"], name="Get current histories")
    async def get_processor_histories(response: Response) -> GetHistoriesResponse:
        context_id = aiavatr_app.sts.vad.get_session_data("local_session", "context_id")
        return GetHistoriesResponse(
            histories=await aiavatr_app.sts.llm.context_manager.get_histories(context_id)
        )


    @api_router.delete("/conversation/histories", tags=["Conversation"], name="Delete histories")
    async def delete_processor_histories(response: Response) -> APIResponse:
        aiavatr_app.last_request_at = 0     # Make current context timeout
        return APIResponse(message="deleted")


    @api_router.post("/system/log", tags=["System"], name="See the recent log")
    async def system_log(request: LogRequest, response: Response) -> LogResponse:
        try:
            with open(logfile_path, "r", encoding="utf-8") as f:
                deque_lines = collections.deque(f, maxlen=request.count)
                return LogResponse(lines=deque_lines)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")

    return api_router
