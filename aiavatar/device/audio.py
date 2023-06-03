from pyaudio import PyAudio

class AudioDevice:
    @classmethod
    def get_default_input_device_info(cls):
        pa = PyAudio()
        ret = pa.get_default_input_device_info()
        pa.terminate()
        return ret

    @classmethod
    def get_default_output_device_info(cls):
        pa = PyAudio()
        ret = pa.get_default_output_device_info()
        pa.terminate()
        return ret

    @classmethod
    def get_device_info(cls, index: int):
        pa = PyAudio()
        ret = pa.get_device_info_by_index(index)
        pa.terminate()
        return ret

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
    def get_input_device_with_prompt(cls, prompt: str=None):
        print("==== Input devices ====")
        for d in AudioDevice.get_audio_devices():
            if d["maxInputChannels"] > 0:
                print(f'{d["index"]}: {d["name"]}')
        idx = input(prompt or "Index of microphone device (Skip to use default): ")
        if idx == "":
            return cls.get_default_input_device_info()
        else:
            return cls.get_device_info(int(idx))

    @classmethod
    def get_output_device_with_prompt(cls, prompt: str=None):
        print("==== Output devices ====")
        for d in AudioDevice.get_audio_devices():
            if d["maxOutputChannels"] > 0:
                print(f'{d["index"]}: {d["name"]}')
        idx = input(prompt or "Index of speaker device (Skip to use default): ")
        if idx == "":
            return cls.get_default_output_device_info()
        else:
            return cls.get_device_info(int(idx))

    @classmethod
    def get_audio_devices(cls):
        devices = []
        pa = PyAudio()
        for i in range(pa.get_device_count()):
            devices.append(pa.get_device_info_by_index(i))
        pa.terminate()
        return devices

    @classmethod
    def list_audio_devices(cls):
        devices = cls.get_audio_devices()
        print("Available audio devices:")
        for d in devices:
            print(f"{d['index']}: {d['name']}")
