import base64
from logging import getLogger, NullHandler
import numpy
import time
import traceback
from typing import Callable
import aiohttp
import sounddevice

class SpeechListenerBase:
    def __init__(self, api_key: str, on_speech_recognized: Callable, volume_threshold: int=3000, timeout: float=1.0, detection_timeout: float=0.0, min_duration: float=0.3, max_duration: float=20.0, lang: str="ja-JP", rate: int=44100, channels: int=1, device_index: int=-1):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.on_speech_recognized = on_speech_recognized
        self.volume_threshold = volume_threshold
        self.timeout = timeout
        self.detection_timeout = detection_timeout
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.lang = lang
        self.channels = channels
        self.rate = rate
        self.device_index = device_index
        self.is_listening = False

    def record_audio(self, device_index) -> bytes:
        audio_data = []

        def callback(in_data, frame_count, time_info, status):
            audio_data.append(in_data.copy())

        try:
            stream = sounddevice.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.rate,
                dtype=numpy.int16,
                callback=callback
            )

            start_time = time.time()
            is_recording = False
            silence_start_time = time.time()
            is_silent = False
            last_detected_time = time.time()
            stream.start()

            while stream.active:
                current_time = time.time()
                volume = numpy.linalg.norm(audio_data[-10:]) / 50 if audio_data else 0

                if not is_recording:
                    if volume > self.volume_threshold:
                        audio_data = audio_data[-100:]  # Use 100ms data before start recording
                        is_recording = True
                        start_time = current_time

                else:
                    if volume <= self.volume_threshold:
                        if is_silent:
                            if current_time - silence_start_time > self.timeout:
                                # Timeouot
                                recorded_length = current_time - start_time - self.timeout

                                if recorded_length < self.min_duration:
                                    self.logger.info(f"Too short: {recorded_length}")
                                    is_recording = False
                                    audio_data.clear()

                                else:
                                    return b"".join(audio_data)
                            else:
                                # Continue silent
                                pass

                        else:
                            # Start silent
                            silence_start_time = current_time
                            is_silent = True

                    else:
                        # Detecting voice
                        is_silent = False
                        last_detected_time = current_time

                    if current_time - start_time > self.max_duration:
                        self.logger.info(f"Max recording duration reached: {current_time - start_time}")
                        is_recording = False
                        audio_data.clear()

                if self.detection_timeout > 0 and time.time() - last_detected_time > self.detection_timeout:
                    self.logger.info(f"Voice detection timeout: {self.detection_timeout}")
                    break

        except Exception as ex:
            self.logger.error(f"Error at record_audio: {str(ex)}\n{traceback.format_exc()}")

        finally:
            stream.stop()
            stream.close()

        # Return empty bytes
        return b"".join([])

    async def transcribe(self, audio_data: list) -> str:
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        request_body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.rate,
                "languageCode": self.lang,
            },
            "audio": {
                "content": audio_b64
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}",
                json=request_body
            ) as resp:
                j = await resp.json()

                if resp.status != 200:
                    self.logger.error(f"Failed in recognition: {resp.status}\n{j}")
                    return None

                if j.get("results"):
                    if j["results"][0]["alternatives"][0].get("transcript"):
                        return j["results"][0]["alternatives"][0]["transcript"]

        return None

    async def start_listening(self):
        self.is_listening = True

        try:
            self.logger.info(f"Listening... ({self.__class__.__name__})")
            while self.is_listening:
                audio_data = self.record_audio(self.device_index)

                if audio_data:
                    recognized_text = await self.transcribe(audio_data)

                    if recognized_text:
                        await self.on_speech_recognized(recognized_text)
                    else:
                        self.logger.info("No speech recognized")
                
                else:
                    # Stop listening when no recorded data
                    break

            self.logger.info(f"Stopped listening ({self.__class__.__name__})")

        except Exception as ex:
            self.logger.error(f"Error at start_listening: {str(ex)}\n{traceback.format_exc()}")

        finally:
            self.is_listening = False

    def stop_listening(self):
        self.is_listening = False
