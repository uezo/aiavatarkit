from aiavatar.device import AudioDevice

def test_get_default_input_device_info():
    d = AudioDevice.get_default_input_device_info()
    assert d["index"] >= 0
    assert d["index"] < 1000
    assert d["name"] is not None

def test_get_default_output_device_info():
    d = AudioDevice.get_default_output_device_info()
    assert d["index"] >= 0
    assert d["index"] < 1000
    assert d["name"] is not None

def test_get_audio_devices():
    devices = AudioDevice.get_audio_devices()
    assert len(devices) >= 2

    for d in devices:
        assert d["index"] >= 0
        assert d["index"] < 1000
        assert d["name"] is not None
