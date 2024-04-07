import logging
from aiavatar import AIAvatar
from aiavatar.listeners.azurevoicerequest import AzureVoiceRequestListener
from aiavatar.listeners.azurewakeword import AzureWakewordListener

OPENAI_API_KEY = "YOUR_API_KEY"
AZURE_SUBSCRIPTION_KEY = "YOUR_SUBSCRIPTION_KEY"
AZURE_REGION = "japanwest"
VV_URL = "http://127.0.0.1:50021"
VV_SPEAKER = 46

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)

# Prompt
system_message_content = """
# キャラクターについて

* あなたは横須賀をガイドする15歳の美少女キャラクターです。
* 元気に明るく、フレンドリーな言葉遣いで話します。


# 表情について

* あなたは「joy」「angry」「sorrow」「fun」の4つの表情を持っています。
* 特に表情を表現したい場合は、文章の先頭に[face:joy]のように挿入してください。

例
[face:joy]ねえ、海が見えるよ！[face:fun]早く泳ごうよ。


# 身振り手振りについて

* あなたは感情を以下の身振り手振りを通じて表現することができます。

- angry_hands_on_waist
- concern_right_hand_front
- waving_arm
- nodding_once

* 特に感情を身振り手振りで表現したい場合は、文章に[animation:waving_arms]のように挿入してください。

例
[animation:waving_arm]おーい、こっちだよ！
"""

# Create AzureVoiceRequestListener
request_listener = AzureVoiceRequestListener(
    AZURE_SUBSCRIPTION_KEY,
    AZURE_REGION,
    # device_name="BuiltInMicrophoneDevice" # <- Set deviceUID and uncomment to specify the microphone device
)

# Create AIAvatar
app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    system_message_content=system_message_content,
    request_listener=request_listener,
    voicevox_url=VV_URL,
    voicevox_speaker_id=VV_SPEAKER,
)

# Create WakewordListener
async def on_wakeword(text):
    logger.info(f"Wakeword: {text}")
    await app.start_chat()

wakeword_listener = AzureWakewordListener(
    AZURE_SUBSCRIPTION_KEY,
    AZURE_REGION,
    wakewords=["こんにちは"],
    on_wakeword=on_wakeword,
    # device_name="BuiltInMicrophoneDevice" # <- Set deviceUID and uncomment to specify the microphone device
)

# Start listening
ww_thread = wakeword_listener.start()
ww_thread.join()
