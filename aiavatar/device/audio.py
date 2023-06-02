from pyaudio import PyAudio

class AudioDevice:
    @classmethod
    def get_default_input_device_info(cls):
        return PyAudio().get_default_input_device_info()

    @classmethod
    def get_default_output_device_info(cls):
        return PyAudio().get_default_output_device_info()

    @classmethod
    def get_device_info(cls, index: int):
        return PyAudio().get_device_info_by_index(index)

    @classmethod
    def get_input_device_by_name(cls, name: str):
        for d in AudioDevice.get_audio_devices():
            if d["maxInputChannels"] > 0:
                if name.lower() in d["name"].lower():
                    return d
        return None

    @classmethod
    def get_output_device_by_name(cls, name: str):
        for d in AudioDevice.get_audio_devices():
            if d["maxOutputChannels"] > 0:
                if name.lower() in d["name"].lower():
                    return d
        return None

    @classmethod
    def get_audio_devices(cls):
        devices = []
        pa = PyAudio()
        for i in range(pa.get_device_count()):
            devices.append(pa.get_device_info_by_index(i))
        return devices

    @classmethod
    def list_audio_devices(cls):
        devices = cls.get_audio_devices()
        print("Available audio devices:")
        for d in devices:
            print(f"{d['index']}: {d['name']}")
