import asyncio
import logging
from aiavatar import AIAvatar

GOOGLE_API_KEY = "YOUR API KEY"
OPENAI_API_KEY = "YOUR API KEY"
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
system_message_content = """あなたは「joy」「angry」「sorrow」「fun」の4つの表情を持っています。
特に表情を表現したい場合は、文章の先頭に[face:joy]のように挿入してください。

例
[face:joy]ねえ、海が見えるよ！[face:fun]早く泳ごうよ。
"""

# Create AIAvatar
app = AIAvatar(
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    VV_URL,
    VV_SPEAKER,
    system_message_content=system_message_content,
)

# Start AIAvatar
asyncio.run(app.start())
