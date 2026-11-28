"""Tell the LLM what the user heard when they spoke over the assistant."""

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
import logging
from typing import Optional

from .models import STSRequest, STSResponse
from .pipeline import STSPipeline
from .playback import Interruption, PlaybackLedger

logger = logging.getLogger(__name__)

DEFAULT_SPOKEN_TEMPLATE = (
    "$When the user started speaking, assistant audio had played up to the point "
    "shown below. Use this to interpret what the user is referring to or correcting, "
    "and to judge whether they are adding to their own earlier message. Do not assume "
    "the user had already heard the rest of that response or any later reply in the "
    "conversation history.\n\n"
    "Assistant audio played so far (estimated):\n{spoken_text}\n\n"
    "Current user message:\n{request_text}"
)
DEFAULT_BEFORE_AUDIO_TEMPLATE = (
    "$The user spoke again before audio for the previous request began. "
    "Interpret the current message as a possible addition or correction "
    "to that request, whose answer had not played when they started speaking.\n\n"
    "Previous user request:\n{previous_request}\n\n"
    "Current user message:\n{request_text}"
)
DEFAULT_DISCARDED_TEMPLATE = (
    "$The following assistant speech was canceled before playback started.\n"
    "{discarded_text}\n\n{request_text}"
)
_CAPTURE_KEY = "interruption_context"


@dataclass(frozen=True)
class _Capture:
    """Marks a user turn, with what it interrupted when the user started speaking."""

    interruption: Optional[Interruption] = None


@dataclass
class _Session:
    ledger: PlaybackLedger
    # Transaction IDs and raw text of the latest unanswered user turns in one conversation.
    chain_context_id: Optional[str] = None
    chain: list[tuple[str, str]] = field(default_factory=list)


class InterruptionContext:
    """Add notes about interrupted assistant speech to the next user message.

    The adapter reports what it delivered and what the client played. When the user
    starts speaking, this plugin captures what that speech interrupts. Once the
    pipeline resolves the conversation, an ``on_before_start`` hook records the user's
    turn and adds the interruption note, so ``start`` and every ``on_before_llm`` hook
    see it. Discard reports arrive after the pipeline sends ``stop``, so the note about
    discarded speech is added later, in an ``on_before_llm`` hook. ``bind`` registers
    these hooks on a pipeline.
    """

    def __init__(
        self,
        *,
        get_active_transaction_id: Callable[[str], Optional[str]],
        spoken_template: str = DEFAULT_SPOKEN_TEMPLATE,
        before_audio_template: str = DEFAULT_BEFORE_AUDIO_TEMPLATE,
        discarded_template: str = DEFAULT_DISCARDED_TEMPLATE,
        debug: bool = False,
    ):
        self.spoken_template = spoken_template
        self.before_audio_template = before_audio_template
        self.discarded_template = discarded_template
        self.debug = debug
        self._get_active_transaction_id = get_active_transaction_id
        self._sessions: dict[str, _Session] = {}
        self._sts: Optional[STSPipeline] = None

    def bind(self, sts: STSPipeline, *, channels: str | Collection[str] | None = None) -> None:
        """Register the plugin's hooks on the pipeline and its VAD.

        Notes go only to requests from ``channels``, or from every channel when it is None.
        A plugin binds to one pipeline, once.
        """
        if self._sts is not None:
            raise RuntimeError("InterruptionContext is already bound to a pipeline")
        if not callable(getattr(sts.vad, "on_speech_started", None)):
            raise ValueError("InterruptionContext requires a VAD with on_speech_started")
        self._sts = sts
        sts.vad.on_speech_started(self._capture)
        sts.on_before_start(self._add_interruption_note, channels=channels)
        sts.on_before_llm(self._add_discarded_note, channels=channels)
        sts.on_finalize(self.close_session)
        sts.on_shutdown(self.close)

    @staticmethod
    def get_interruption(request: STSRequest) -> Optional[Interruption]:
        """Return what the request interrupted when the user started speaking, if anything."""
        capture = (request.metadata or {}).get(_CAPTURE_KEY)
        return capture.interruption if isinstance(capture, _Capture) else None

    # Facts reported by the adapter

    def open_session(self, session_id: str) -> None:
        """Start fresh tracking when the adapter opens a session."""
        self._sessions[session_id] = _Session(PlaybackLedger(debug=self.debug))

    def get_ledger(self, session_id: str) -> Optional[PlaybackLedger]:
        session = self._sessions.get(session_id)
        return session.ledger if session is not None else None

    def observe(self, response: STSResponse) -> None:
        """Record a response that the adapter is about to deliver."""
        if ledger := self.get_ledger(response.session_id):
            ledger.observe(response)

    def mark_answered(self, session_id: str, transaction_id: Optional[str]) -> None:
        """Record that the client received answer audio or an ordinary final."""
        if ledger := self.get_ledger(session_id):
            ledger.mark_answered(transaction_id)

    def handle_playback_event(self, session_id: str, /, **metadata) -> None:
        """Record a playback report from the client."""
        if ledger := self.get_ledger(session_id):
            ledger.handle_playback_event(**metadata)

    def track(self, request: STSRequest) -> None:
        """Mark a user request that does not come from the VAD, such as typed text."""
        request.metadata = {**(request.metadata or {}), _CAPTURE_KEY: _Capture()}

    def snapshot(self, session_id: str) -> Optional[Interruption]:
        """Return what an utterance starting now would interrupt."""
        ledger = self.get_ledger(session_id)
        if ledger is None:
            return None
        return ledger.snapshot(self._get_active_transaction_id(session_id))

    async def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def close(self) -> None:
        self._sessions.clear()

    # Pipeline hooks

    def _capture(self, session_id: str, metadata: dict) -> None:
        if session_id in self._sessions:
            metadata[_CAPTURE_KEY] = _Capture(self.snapshot(session_id))

    async def _add_interruption_note(self, request: STSRequest) -> None:
        session = self._sessions.get(request.session_id)
        capture = (request.metadata or {}).get(_CAPTURE_KEY)
        if session is None or not isinstance(capture, _Capture):
            return

        # Record this turn before start, so the next utterance can refer to it.
        interruption = capture.interruption
        chain = session.chain if session.chain_context_id == request.context_id else []
        session.chain_context_id = request.context_id
        session.chain = []
        if not request.recognized_text:
            return

        preceding = []
        if interruption is not None and interruption.interrupted_response_context_id == request.context_id:
            if interruption.spoken_text:
                request.text = self.spoken_template.format(
                    spoken_text=interruption.spoken_text, request_text=request.text,
                )
            else:
                preceding = _unanswered_requests(chain, interruption)
                if preceding:
                    request.text = self.before_audio_template.format(
                        previous_request="\n".join(text for _, text in preceding),
                        request_text=request.text,
                    )
            if self.debug and (interruption.spoken_text or preceding):
                # Name the quoted response so clips with the same text can be told apart in the playback log.
                logger.info(
                    "Interruption note: transaction=%s, interrupted_transaction=%s",
                    request.transaction_id, interruption.interrupted_response_transaction_id,
                )
        session.chain = [*preceding, (request.transaction_id, request.recognized_text)]

    async def _add_discarded_note(self, request: STSRequest) -> None:
        session = self._sessions.get(request.session_id)
        if session is None or not request.text:
            return
        if discarded := session.ledger.take_discarded_audio(request.context_id):
            request.text = self.discarded_template.format(
                discarded_text="\n".join(discarded), request_text=request.text,
            )


def _unanswered_requests(chain: list[tuple[str, str]], interruption: Interruption) -> list[tuple[str, str]]:
    """Return the turns after the last answered one, through the interrupted one."""
    start = 0
    for index, (transaction_id, _) in enumerate(chain):
        if transaction_id == interruption.answered_through_transaction_id:
            start = index + 1
        if transaction_id == interruption.interrupted_response_transaction_id:
            return chain[start:index + 1]
    return []
