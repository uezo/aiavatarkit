import collections
import logging
import traceback
from fastapi import APIRouter
from fastapi.responses import Response
from aiavatar import AIAvatar
from .schema import *


logger = logging.getLogger(__name__)


def get_router(aiavatr_app: AIAvatar, logfile_path: str) -> APIRouter:
    api_router = APIRouter()

    @api_router.post("/wakeword/start", tags=["Wakeword Listener"], name="Start wakeword listener")
    async def wakeword_start(request: WakewordStartRequest, response: Response) -> APIResponse:
        try:
            if not aiavatr_app.is_wakeword_listener_listening():
                aiavatr_app.start_listening_wakeword(False)

                if not aiavatr_app.is_wakeword_listener_listening():
                    response.status_code = 500
                    return ErrorResponse(error="failed")

                if request.wakewords:
                    aiavatr_app.wakeword_listener.wakewords = request.wakewords

                return APIResponse(message="started")
            else:
                return APIResponse(message="already running")
        
        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/wakeword/stop", tags=["Wakeword Listener"], name="Stop wakeword listener")
    async def wakeword_stop() -> APIResponse:
        aiavatr_app.stop_listening_wakeword()
        return APIResponse(message="stop requested")


    @api_router.get("/wakeword/status", tags=["Wakeword Listener"], name="See the status of wakeword listener")
    async def wakeword_status() -> WakewordStatusResponse:
        return WakewordStatusResponse(
            is_listening=aiavatr_app.is_wakeword_listener_listening(),
            thread_name=aiavatr_app.wakeword_listener_thread.name if aiavatr_app.wakeword_listener_thread else None
        )


    @api_router.post("/avatar/speech", tags=["Avatar"], name="Speak text with face expression and animation")
    async def avatar_speech(request: SpeechRequest, response: Response) -> APIResponse:
        try:
            avreq = aiavatr_app.avatar_controller.parse(request.text)
            await aiavatr_app.avatar_controller.perform(avreq)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/face", tags=["Avatar"], name="Set face expression")
    async def avatar_face(request: FaceRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.avatar_controller.face_controller.set_face(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/animation", tags=["Avatar"], name="Set animation")
    async def avatar_animation(request: AnimationRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.avatar_controller.animation_controller.animate(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


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
