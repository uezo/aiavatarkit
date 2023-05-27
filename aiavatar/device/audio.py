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
