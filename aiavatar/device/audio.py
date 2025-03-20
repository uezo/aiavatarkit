import asyncio
import collections
import io
import logging
import numpy as np
import queue
import threading
from typing import AsyncGenerator
import wave
import pyaudio

logger = logging.getLogger(__name__)


class AudioDevice:
    def __init__(self, input_device: int = -1, output_device: int = -1):
        self._p = pyaudio.PyAudio()

        if isinstance(input_device, int):
            if input_device < 0:
                input_device_info = self.get_default_input_device_info()
                input_device = input_device_info["index"]
            else:
                input_device_info = self.get_device_info(input_device)
        elif isinstance(input_device, str):
            input_device_info = self.get_input_device_by_name(input_device)
            if input_device_info is None:
                input_device_info = self.get_default_input_device_info()
            input_device = input_device_info["index"]

        self.input_device = input_device
        self.input_device_info = input_device_info

        if isinstance(output_device, int):
            if output_device < 0:
                output_device_info = self.get_default_output_device_info()
                output_device = output_device_info["index"]
            else:
                output_device_info = self.get_device_info(output_device)
        elif isinstance(output_device, str):
            output_device_info = self.get_output_device_by_name(output_device)
            if output_device_info is None:
                output_device_info = self.get_default_output_device_info()
            output_device = output_device_info["index"]

        self.output_device = output_device
        self.output_device_info = output_device_info

    def normalize_device_info(self, info: dict) -> dict:
        normalized = {
            "index": info.get("index"),
            "name": info.get("name"),
            "max_input_channels": info.get("maxInputChannels"),
            "max_output_channels": info.get("maxOutputChannels"),
            "default_sample_rate": info.get("defaultSampleRate")
        }
        return normalized

    def get_default_input_device_info(self) -> dict:
        try:
            info = self._p.get_default_input_device_info()
            info["index"] = info.get("index", 0)
            return self.normalize_device_info(info)
        except Exception as ex:
            devices = self.get_audio_devices()
            for d in devices:
                if d["max_input_channels"] > 0:
                    return d
            raise Exception("Input devices not found")

    def get_default_output_device_info(self) -> dict:
        try:
            info = self._p.get_default_output_device_info()
            info["index"] = info.get("index", 0)
            return self.normalize_device_info(info)
        except Exception as ex:
            devices = self.get_audio_devices()
            for d in devices:
                if d["max_output_channels"] > 0:
                    return d
            raise Exception("Output devices not found")

    def get_device_info(self, index: int) -> dict:
        info = self._p.get_device_info_by_index(index)
        info["index"] = index
        return self.normalize_device_info(info)

    def get_input_device_by_name(self, name: str) -> dict:
        for d in self.get_audio_devices():
            if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
                return d
        return None

    def get_output_device_by_name(self, name: str) -> dict:
        for d in self.get_audio_devices():
            if d["max_output_channels"] > 0 and name.lower() in d["name"].lower():
                return d
        return None

    def get_input_device_with_prompt(self, prompt: str = None) -> dict:
        print("==== Input devices ====")
        for d in self.get_audio_devices():
            if d["max_input_channels"] > 0:
                print(f'{d["index"]}: {d["name"]}')
        idx = input(prompt or "Input device index (Skip to use default): ")
        if idx.strip() == "":
            return self.get_default_input_device_info()
        else:
            return self.get_device_info(int(idx.strip()))

    def get_output_device_with_prompt(self, prompt: str = None) -> dict:
        print("==== Output devices ====")
        for d in self.get_audio_devices():
            if d["max_output_channels"] > 0:
                print(f'{d["index"]}: {d["name"]}')
        idx = input(prompt or "Output device index (Skip to use default): ")
        if idx.strip() == "":
            return self.get_default_output_device_info()
        else:
            return self.get_device_info(int(idx.strip()))

    def get_audio_devices(self) -> list:
        devices = []
        count = self._p.get_device_count()
        for i in range(count):
            info = self._p.get_device_info_by_index(i)
            info["index"] = i
            devices.append(self.normalize_device_info(info))
        return devices

    def list_audio_devices(self):
        for d in self.get_audio_devices():
            print(d)

    def terminate(self):
        self._p.terminate()


class AudioPlayer:
    def __init__(self, device_index: int, chunk_size: int = 1024):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

        self.to_wave = None
        self.p = pyaudio.PyAudio()
        self.play_stream = None
        self.device_index = device_index
        self.chunk_size = chunk_size

        self.wave_params = None
        self.is_playing = False
        self.stop_event = threading.Event()

    def is_wave_params_changed(self, current_params: wave._wave_params):
        return self.wave_params is None or current_params is None \
            or self.wave_params.nchannels != current_params.nchannels \
            or self.wave_params.sampwidth != current_params.sampwidth \
            or self.wave_params.framerate != current_params.framerate

    def play(self, content: bytes):
        try:
            self.stop_event.clear()
            self.is_playing = True

            if self.to_wave:
                wave_content = self.to_wave(content)
            else:
                wave_content = content

            if wave_content:
                with wave.open(io.BytesIO(wave_content), "rb") as wf:
                    current_params = wf.getparams()
                    if not self.play_stream or self.is_wave_params_changed(current_params):
                        if self.play_stream:
                            self.play_stream.stop_stream()
                            self.play_stream.close()
                            self.play_stream = None

                        self.wave_params = current_params
                        self.play_stream = self.p.open(
                            format=self.p.get_format_from_width(self.wave_params.sampwidth),
                            channels=self.wave_params.nchannels,
                            rate=self.wave_params.framerate,
                            output=True,
                            output_device_index=self.device_index,
                            frames_per_buffer=self.chunk_size
                        )

                    data = wf.readframes(self.chunk_size)
                    while data:
                        if self.stop_event.is_set():
                            break
                        self.play_stream.write(data)
                        data = wf.readframes(self.chunk_size)

        except Exception as ex:
            logger.error(f"Error at play: {ex}", exc_info=True)

        finally:
            self.is_playing = False

    def process_queue(self):
        while True:
            data = self.queue.get()
            if data is None:
                break

            self.play(data)

    def add(self, audio_bytes: bytes):
        self.queue.put(audio_bytes)

    def cancel(self):
        while not self.queue.empty():
            self.queue.get()

    def stop(self):
        self.queue.put(None)
        self.thread.join()
        self.stop_event.set()


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, device_index: int = -1, channels: int = 1, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.is_listening = False

    async def start_stream(self) -> AsyncGenerator[bytes, None]:
        p = pyaudio.PyAudio()
        pyaudio_stream = p.open(
            rate=self.sample_rate,
            channels=self.channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index
        )
        self.is_listening = True

        try:
            while self.is_listening:
                yield pyaudio_stream.read(self.chunk_size, exception_on_overflow=False)
                await asyncio.sleep(0.0001)
        finally:
            pyaudio_stream.stop_stream()
            pyaudio_stream.close()
            logger.info("PyAudio stream closed.")

    def stop_stream(self):
        self.is_listening = False


class NoiseLevelDetector:
    def __init__(self, rate: int = 16000, channels: int = 1, device_index: int = -1):
        self.channels = channels
        self.rate = rate
        self.device_index = device_index
        self.chunk = int(self.rate * 0.05)

    def get_volume_db(self, data: np.ndarray, ref: int = 32768) -> float:
        amplitude = np.max(np.abs(data))
        if amplitude == 0:
            amplitude = 1  # Return 1 to avoid 0 div
        return float(20 * np.log10(amplitude / ref))

    def get_noise_level(self) -> float:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index if self.device_index >= 0 else None,
                frames_per_buffer=self.chunk
            )
            audio_data = collections.deque(maxlen=60)
            logger.info("Measuring noise levels...")

            while True:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                except Exception as e:
                    logger.warning("Audio buffer has overflowed")
                    continue

                audio_array = np.frombuffer(data, dtype=np.int16)
                volume_db = self.get_volume_db(audio_array)
                print(f"Current: {volume_db:.2f}dB", end="\r")
                audio_data.append(volume_db)

                if len(audio_data) == audio_data.maxlen:
                    median_db = np.median(audio_data)
                    print(f"Noise level: {median_db:.2f}dB")
                    return median_db

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
