import collections
import logging
import traceback
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import Response
from aiavatar import AIAvatar
from .schema import *


logger = logging.getLogger(__name__)


def get_router(aiavatr_app: AIAvatar, logfile_path: str) -> APIRouter:
    api_router = APIRouter()

    @api_router.post("/listener/start", tags=["Listener"], name="Start listener")
    async def listener_start(request: ListenerStartRequest, response: Response, background_tasks: BackgroundTasks) -> APIResponse:
        try:
            if not aiavatr_app.adapter.is_listening:
                if request.wakewords:
                    aiavatr_app.wakewords = request.wakewords

                background_tasks.add_task(aiavatr_app.start_listening)

                return APIResponse(message="start requested")
            else:
                return APIResponse(message="already running")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/listener/stop", tags=["Listener"], name="Stop listener")
    async def listener_stop() -> APIResponse:
        aiavatr_app.adapter.stop_listening()
        return APIResponse(message="stop requested")


    @api_router.get("/listener/status", tags=["Listener"], name="Check the status of listener")
    async def listener_status() -> ListenerStatusResponse:
        return ListenerStatusResponse(
            is_listening=aiavatr_app.adapter.is_listening
        )


    @api_router.get("/avatar/status", tags=["Avatar"], name="Get avatar status")
    async def get_avatar_status(response: Response) -> GetAvatarStatusResponse:
        try:
            return GetAvatarStatusResponse(
                current_face=aiavatr_app.adapter.face_controller.current_face,
                current_animation=aiavatr_app.adapter.animation_controller.current_animation
            )

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/face", tags=["Avatar"], name="Set face expression")
    async def avatar_face(request: FaceRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.adapter.face_controller.set_face(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/avatar/face", tags=["Avatar"], name="Get current face expression")
    async def get_avatar_face(response: Response) -> GetFaceResponse:
        try:
            current_face = aiavatr_app.adapter.face_controller.current_face
            return GetFaceResponse(current_face=current_face)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/animation", tags=["Avatar"], name="Set animation")
    async def avatar_animation(request: AnimationRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.adapter.animation_controller.animate(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/avatar/animation", tags=["Avatar"], name="Get current animation")
    async def get_avatar_animation(response: Response) -> GetAnimationResponse:
        try:
            current_animation = aiavatr_app.adapter.animation_controller.current_animation
            return GetAnimationResponse(current_animation=current_animation)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/chat", tags=["Chat"], name="Send message to AIAvatar")
    async def processor_chat(request: ChatRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.chat(request.text)
            return APIResponse(message="accepted")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/chat/histories", tags=["Chat"], name="Get current histories")
    async def get_processor_histories(response: Response) -> GetHistoriesResponse:
        return GetHistoriesResponse(
            histories=await aiavatr_app.sts.llm.context_manager.get_histories(aiavatr_app.current_context_id)
        )


    @api_router.delete("/chat/histories", tags=["Chat"], name="Delete histories")
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
