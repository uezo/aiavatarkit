"""Response text preparation and speech synthesis for one Realtime turn."""

from dataclasses import dataclass, replace
import re
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from aiavatar.sts.tts import SpeechSynthesizer


# LLMService's streaming parser also owns conversation state. These rules only
# transform response text and do not require constructing a separate LLM.
_CONTROL_TAG = re.compile(r'''\[\w+:[^\]]*\]|</?[A-Za-z_]\w*(?:"[^"]*"|'[^']*'|[^"'<>])*>''')
_INCOMPLETE_TAG = re.compile(r"</?(?:[A-Za-z_][^<>]*)?$|\[[A-Za-z_][^\]]*$")


def voice_text(text: str) -> str:
    if incomplete := _INCOMPLETE_TAG.search(text):
        text = text[:incomplete.start()]
    return _CONTROL_TAG.sub("", text).strip()


class VoiceTextFilter:
    """Select spoken regions while retaining tag state across sentences."""

    def __init__(self, tags: list[str]):
        self.pattern = re.compile(r"<(/?)(" + "|".join(re.escape(tag) for tag in tags) + r")>") if tags else None
        self.active_tags: list[str] = []
        self.seen_tag = False

    def feed(self, text: str) -> str:
        if self.pattern is None:
            return voice_text(text)
        parts = []
        position = 0
        for token in _CONTROL_TAG.finditer(text):
            match = self.pattern.fullmatch(token.group())
            if match is None:
                continue
            if self.active_tags:
                parts.append(text[position:token.start()])
            closing, tag = match.groups()
            if not closing:
                self.active_tags.append(tag)
                self.seen_tag = True
            elif self.active_tags and self.active_tags[-1] == tag:
                self.active_tags.pop()
            position = token.end()
        if self.active_tags:
            parts.append(text[position:])
        return voice_text("".join(parts))


@dataclass
class SpeechChunk:
    text: str
    voice_text: str
    audio_data: bytes | None = None
    styled_text: str | None = None


class ResponseOutput:
    """Prepare one response without owning its queue, worker, or delivery."""

    def __init__(self):
        self.text = ""
        self.buffer = ""

    def append(self, delta: str, sentence_end: str) -> Iterator[str]:
        self.text += delta
        self.buffer += delta
        if len(self.buffer) > 8000 or len(self.text) > 100000:
            raise RuntimeError("Realtime response text exceeds the buffering limit")
        position = 0
        segment_start = 0
        while position < len(self.buffer):
            if self.buffer[position] in "<[":
                if tag := _CONTROL_TAG.match(self.buffer, position):
                    position = tag.end()
                    continue
                suffix = self.buffer[position:]
                # Wait for tags split across text deltas, including punctuation
                # inside a quoted attribute. Ordinary comparisons remain text.
                if re.match(r"</?[A-Za-z_]|\[[A-Za-z_]", suffix):
                    break
            if self.buffer[position] in sentence_end:
                yield self.buffer[segment_start:position + 1]
                segment_start = position + 1
            position += 1
        self.buffer = self.buffer[segment_start:]

    def flush(self) -> Iterator[str]:
        yield self.buffer
        self.buffer = ""

    def prepare(self, text: str, voice_filter: VoiceTextFilter) -> SpeechChunk:
        return SpeechChunk(text=text, voice_text=voice_filter.feed(text))

    def finish(self, voice_text_tags: list[str], voice_filter: VoiceTextFilter) -> tuple[str, SpeechChunk | None]:
        final_voice_text = VoiceTextFilter(voice_text_tags).feed(self.text)
        # Wait until completion before falling back to untagged text, so text
        # outside selected regions cannot leak into speech while streaming.
        if voice_text_tags and not voice_filter.seen_tag:
            final_voice_text = voice_text(self.text)
            if final_voice_text:
                return final_voice_text, SpeechChunk(
                    text="", voice_text=final_voice_text, styled_text=self.text,
                )
        return final_voice_text, None

    async def synthesize(self, chunk: SpeechChunk, tts: "SpeechSynthesizer", language: str | None) -> SpeechChunk:
        audio = await tts.synthesize(
            chunk.voice_text,
            style_info={"styled_text": chunk.text if chunk.styled_text is None else chunk.styled_text, "info": {}},
            language=language,
        ) if chunk.voice_text else None
        return replace(chunk, audio_data=audio)
