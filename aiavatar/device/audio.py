import sounddevice

class AudioDevice:
    def __init__(self, input_device: int=-1, output_device: int=-1):
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

    @classmethod
    def get_default_input_device_info(cls):
        device_list = sounddevice.query_devices()
        for idx in sounddevice.default.device:
            if device_list[idx]["max_input_channels"] > 0:
                return device_list[idx]

    @classmethod
    def get_default_output_device_info(cls):
        device_list = sounddevice.query_devices()
        for idx in sounddevice.default.device:
            if device_list[idx]["max_output_channels"] > 0:
                return device_list[idx]

    @classmethod
    def get_device_info(cls, index: int):
        return sounddevice.query_devices(index)

    @classmethod
    def get_input_device_by_name(cls, name: str):
        for d in sounddevice.query_devices():
            if d["max_input_channels"] > 0:
                if name.lower() in d["name"].lower():
                    return d
        return None

    @classmethod
    def get_output_device_by_name(cls, name: str):
        for d in sounddevice.query_devices():
            if d["max_output_channels"] > 0:
                if name.lower() in d["name"].lower():
                    return d
        return None

    @classmethod
    def get_input_device_with_prompt(cls, prompt: str=None):
        print("==== Input devices ====")
        for d in sounddevice.query_devices():
            if d["max_input_channels"] > 0:
                print(f'{d["index"]}: {d["name"]}')
        idx = input(prompt or "Index of microphone device (Skip to use default): ")
        if idx == "":
            return cls.get_default_input_device_info()
        else:
            return cls.get_device_info(int(idx))

    @classmethod
    def get_output_device_with_prompt(cls, prompt: str=None):
        print("==== Output devices ====")
        for d in sounddevice.query_devices():
            if d["max_output_channels"] > 0:
                print(f'{d["index"]}: {d["name"]}')
        idx = input(prompt or "Index of speaker device (Skip to use default): ")
        if idx == "":
            return cls.get_default_output_device_info()
        else:
            return cls.get_device_info(int(idx))

    @classmethod
    def get_audio_devices(cls):
        return sounddevice.query_devices()

    @classmethod
    def list_audio_devices(cls):
        print(cls.get_audio_devices())
