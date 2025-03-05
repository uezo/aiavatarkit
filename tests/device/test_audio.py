from aiavatar.device import AudioDevice

def test_init():
    audio_device = AudioDevice()

    assert audio_device.input_device >= 0
    assert isinstance(audio_device.input_device_info["name"], str)
    assert audio_device.input_device_info["max_input_channels"] > 0
    assert audio_device.output_device >= 0
    assert isinstance(audio_device.output_device_info["name"], str)
    assert audio_device.output_device_info["max_output_channels"] > 0

def test_init_index():
    audio_device = AudioDevice(input_device=1, output_device=2)

    assert audio_device.input_device == 1
    assert isinstance(audio_device.input_device_info["name"], str)
    assert audio_device.output_device == 2
    assert isinstance(audio_device.output_device_info["name"], str)

def test_init_name():
    audio_device = AudioDevice(input_device="マイク", output_device="スピーカー")

    assert "マイク" in audio_device.input_device_info["name"]
    assert audio_device.input_device_info["max_input_channels"] > 0
    assert "スピーカー" in audio_device.output_device_info["name"]
    assert audio_device.output_device_info["max_output_channels"] > 0

def test_get_default_input_device_info():
    audio_device = AudioDevice()

    d = audio_device.get_default_input_device_info()
    assert d["index"] >= 0
    assert d["index"] < 1000
    assert d["name"] is not None

def test_get_default_output_device_info():
    audio_device = AudioDevice()

    d = audio_device.get_default_output_device_info()
    assert d["index"] >= 0
    assert d["index"] < 1000
    assert d["name"] is not None

def test_get_input_device_by_name():
    audio_device = AudioDevice()

    d = audio_device.get_input_device_by_name("マイク")
    assert d is not None
    assert d["index"] >= 0
    assert d["max_input_channels"] > 0

    d = audio_device.get_input_device_by_name("_aiavater_dummy_")
    assert d is None

def test_get_output_device_by_name():
    audio_device = AudioDevice()

    d = audio_device.get_output_device_by_name("スピーカー")
    assert d is not None
    assert d["index"] >= 0
    assert d["max_output_channels"] > 0

    d = audio_device.get_output_device_by_name("_aiavater_dummy_")
    assert d is None

def test_get_audio_devices():
    audio_device = AudioDevice()

    devices = audio_device.get_audio_devices()
    assert len(devices) >= 2

    for d in devices:
        assert d["index"] >= 0
        assert d["index"] < 1000
        assert d["name"] is not None
