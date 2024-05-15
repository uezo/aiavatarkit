from abc import ABC, abstractmethod
import base64
import collections
from logging import getLogger, NullHandler
import numpy
import time
import traceback
from typing import Callable
import aiohttp
import sounddevice


class RequestListenerBase(ABC):
    @abstractmethod
    async def get_request(self):
        ...


class WakewordListenerBase(ABC):
    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...


class NoiseLevelDetector:
    def __init__(self, rate: int=16000, channels: int=1, device_index: int=-1):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.channels = channels
        self.rate = rate
        self.device_index = device_index

    def get_volume_db(self, data: numpy.ndarray[numpy.int16], ref: int=32768) -> float:
        amplitude = numpy.max(numpy.abs(data))
        if amplitude == 0:
            amplitude = 1   # Return 1 to calculate dB
        return float(20 * numpy.log10(amplitude / ref))

    def get_noise_level(self) -> float:
        with sounddevice.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.rate,
                dtype=numpy.int16
            ) as stream:

            audio_data = collections.deque(maxlen=60)

            self.logger.info("Measuring noise levels...")

            while stream.active:
                data, overflowed = stream.read(int(self.rate * 0.05))
                if overflowed:
                    self.logger.warning("Audio buffer has overflowed")

                volume_db = self.get_volume_db(data)
                print(f"Current: {volume_db:.2f}dB", end="\r")
                audio_data.append(volume_db)
            
                if len(audio_data) == audio_data.maxlen:
                    median_db = numpy.median(audio_data)
                    print(f"Noise level: {median_db:.2f}dB")
                    return median_db


class SpeechListenerBase:
    def __init__(self, api_key: str, on_speech_recognized: Callable, volume_threshold: int=-50, timeout: float=1.0, detection_timeout: float=0.0, min_duration: float=0.3, max_duration: float=20.0, lang: str="ja-JP", rate: int=16000, channels: int=1, device_index: int=-1):
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

    def get_volume_db(self, data: numpy.ndarray[numpy.int16], ref: int=32768) -> float:
        amplitude = numpy.max(numpy.abs(data))
        if amplitude == 0:
            amplitude = 1   # Return 1 to calculate dB
        return float(20 * numpy.log10(amplitude / ref))

    def record_audio(self, device_index) -> bytes:
        audio_data = []

        try:
            stream = sounddevice.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.rate,
                dtype=numpy.int16
            )

            start_time = time.time()
            is_recording = False
            silence_start_time = time.time()
            is_silent = False
            last_detected_time = time.time()
            stream.start()

            while stream.active:
                data, overflowed = stream.read(int(self.rate * 0.05))   # Process frames in 0.05 sec
                if overflowed:
                    self.logger.warning("Audio buffer has overflowed")

                audio_data.append(data)

                current_time = time.time()
                volume = self.get_volume_db(data)

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
        # ToDo: make stream inactive in `record_audio()` to stop listening immediately
        self.is_listening = False
