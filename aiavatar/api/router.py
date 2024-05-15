import asyncio
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


    @api_router.get("/avatar/status", tags=["Avatar"], name="Get avatar status")
    async def get_avatar_status(response: Response) -> GetAvatarStatusResponse:
        try:
            return GetAvatarStatusResponse(
                is_speaking=aiavatr_app.avatar_controller.speech_controller.is_speaking(),
                current_face=aiavatr_app.avatar_controller.face_controller.current_face,
                current_animation=aiavatr_app.avatar_controller.animation_controller.current_animation
            )

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/speech", tags=["Avatar"], name="Speak text with face expression and animation")
    async def avatar_speech(request: SpeechRequest, response: Response) -> APIResponse:
        try:
            avreq = aiavatr_app.avatar_controller.parse(request.text)
            await aiavatr_app.avatar_controller.perform(avreq)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/avatar/speech/is_speaking", tags=["Avatar"], name="Check whether the avatar is speaking")
    async def get_avatar_is_speaking(response: Response) -> GetIsSpeakingResponse:
        try:
            return GetIsSpeakingResponse(is_speaking=aiavatr_app.avatar_controller.speech_controller.is_speaking())

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/avatar/speech/configuration", tags=["Avatar"], name="Update SpeechController configurations")
    async def avatar_speech_config(request: SpeechConfigRequest, response: Response) -> SpeechConfigResponse:
        sc = aiavatr_app.avatar_controller.speech_controller

        # Update configuration if specified
        for field, value in request.model_dump(exclude_unset=True).items():
            if value is not None and hasattr(sc, field):
                setattr(sc, field, value)

        # Clear cache after updating configuration
        sc.clear_cache()

        # Return updated configuration
        resp = SpeechConfigResponse.from_speech_controller_base(sc)
        resp.api_key = "************"
        return resp


    @api_router.get("/avatar/speech/configuration", tags=["Avatar"], name="Get current SpeechController configurations")
    async def get_avatar_speech_config(response: Response) -> SpeechConfigResponse:
        resp = SpeechConfigResponse.from_speech_controller_base(
            aiavatr_app.avatar_controller.speech_controller
        )
        resp.api_key = "************"
        return resp


    @api_router.post("/avatar/face", tags=["Avatar"], name="Set face expression")
    async def avatar_face(request: FaceRequest, response: Response) -> APIResponse:
        try:
            await aiavatr_app.avatar_controller.face_controller.set_face(request.name, request.duration)
            return APIResponse(message="success")

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/avatar/face", tags=["Avatar"], name="Get current face expression")
    async def get_avatar_face(response: Response) -> GetFaceResponse:
        try:
            current_face = aiavatr_app.avatar_controller.face_controller.current_face
            return GetFaceResponse(current_face=current_face)

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


    @api_router.get("/avatar/animation", tags=["Avatar"], name="Get current animation")
    async def get_avatar_animation(response: Response) -> GetAnimationResponse:
        try:
            current_animation = aiavatr_app.avatar_controller.animation_controller.current_animation
            return GetAnimationResponse(current_animation=current_animation)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.post("/processor/chat", tags=["Chat Processor"], name="Send message to ChatProcessor")
    async def processor_chat(request: ChatRequest, response: Response) -> APIResponse:
        try:
            response_text = ""
            avatar_task = asyncio.create_task(aiavatr_app.avatar_controller.start())
            stream_buffer = ""
            async for t in aiavatr_app.chat_processor.chat(request.text):
                stream_buffer += t
                for spc in aiavatr_app.split_chars:
                    stream_buffer = stream_buffer.replace(spc, spc + "|")
                sp = stream_buffer.split("|")
                if len(sp) > 1: # >1 means `|` is found (splited at the end of sentence)
                    sentence = sp.pop(0)
                    stream_buffer = "".join(sp)
                    aiavatr_app.avatar_controller.set_text(sentence)
                    response_text += sentence
                await asyncio.sleep(0.01)   # wait slightly in every loop not to use up CPU

            if stream_buffer:
                aiavatr_app.avatar_controller.set_text(stream_buffer)
                response_text += stream_buffer

            aiavatr_app.avatar_controller.set_stop()
            await avatar_task

            return APIResponse(message=response_text)

        except Exception as ex:
            response.status_code = 500
            return ErrorResponse(error=f"error: {ex}\n{traceback.format_exc()}")


    @api_router.get("/processor/histories", tags=["Chat Processor"], name="Get current histories")
    async def get_processor_histories(response: Response) -> GetHistoriesResponse:
        return GetHistoriesResponse(
            histories=aiavatr_app.chat_processor.histories
        )


    @api_router.delete("/processor/histories", tags=["Chat Processor"], name="Delete histories")
    async def delete_processor_histories(response: Response) -> APIResponse:
        aiavatr_app.chat_processor.reset_histories()
        return APIResponse(message="deleted")


    @api_router.post("/processor/configuration", tags=["Chat Processor"], name="Update ChatProcessor configurations")
    async def processor_config(request: ChatProcessorConfigRequest, response: Response) -> ChatProcessorConfigResponse:
        proc = aiavatr_app.chat_processor

        # Update configuration if specified
        for field, value in request.model_dump(exclude_unset=True).items():
            if value is not None and hasattr(proc, field):
                setattr(proc, field, value)

        # Clear histories after updating configuration
        proc.histories.clear()

        # Return updated configuration
        resp = ChatProcessorConfigResponse.from_chat_processor(proc)
        resp.api_key = "************"
        return resp


    @api_router.get("/processor/configuration", tags=["Chat Processor"], name="Get current ChatProcessor configurations")
    async def get_processor_config(response: Response) -> ChatProcessorConfigResponse:
        resp = ChatProcessorConfigResponse.from_chat_processor(aiavatr_app.chat_processor)
        resp.api_key = "************"
        return resp


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
