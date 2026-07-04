import logging
import time
from typing import Any, Iterable, Optional, Union

import openai

from .base import TurnEndDecision, TurnEndGate, TurnEndGateContext

logger = logging.getLogger(__name__)


class LLMTurnEndGate(TurnEndGate):
    """Text-based turn-end gate backed by an OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        *,
        openai_client: openai.AsyncOpenAI,
        model: str = "gpt-4.1-mini",
        name: str = "llm",
        depends_on: Optional[Union[str, Iterable[str]]] = None,
        timeout: Optional[float] = 10.0,
        request_timeout: Optional[float] = 2.0,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        system_prompt: Optional[str] = None,
        run_in_background: bool = True,
        debug: bool = False,
    ):
        self.openai_client = openai_client
        self.model = model
        self.name = name
        self.run_in_background = run_in_background
        if depends_on is None:
            self.depends_on = []
        elif isinstance(depends_on, str):
            self.depends_on = [depends_on]
        else:
            self.depends_on = list(depends_on)
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt or (
            "You are a strict turn-end detector for spoken conversation. "
            "Given the current user transcript, answer with exactly one token: "
            "WAIT if the user is likely still thinking, hesitating, or about to continue; "
            "END if the utterance is ready to send to the assistant. "
            "Do not explain."
        )
        self.debug = debug

    def _should_skip(self, context: Optional[TurnEndGateContext]) -> bool:
        if not self.depends_on:
            return False
        if context is None:
            return True
        return not any(context.is_waiting(gate_name) for gate_name in self.depends_on)

    def should_run_in_background(self, context: Optional[TurnEndGateContext]) -> bool:
        return not self._should_skip(context)

    def _build_messages(
        self,
        *,
        text: str,
        recorded_duration: float,
        silence_duration: float,
        context: Optional[TurnEndGateContext],
    ):
        previous = []
        if context is not None:
            for gate_name, decision in context.decisions.items():
                previous.append(
                    f"{gate_name}: {'WAIT' if not decision.should_end else 'PASS'}"
                    f" reason={decision.reason}"
                )

        user_content = (
            f"Transcript:\n{text}\n\n"
            f"Recorded speech duration: {recorded_duration:.3f}s\n"
            f"Current silence duration: {silence_duration:.3f}s\n"
        )
        if previous:
            user_content += "Previous gate decisions:\n" + "\n".join(previous) + "\n"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, content: str) -> bool:
        normalized = (content or "").strip().lower()
        if normalized.startswith("wait"):
            return False
        if normalized.startswith("end") or normalized.startswith("complete") or normalized.startswith("pass"):
            return True
        logger.warning("LLM turn-end gate returned unexpected content: %r", content)
        return True

    async def should_end_turn(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        channels: int,
        recorded_duration: float,
        silence_duration: float,
        session_id: str,
        text: Optional[str] = None,
        session: Any = None,
        context: Optional[TurnEndGateContext] = None,
    ) -> TurnEndDecision:
        if self._should_skip(context):
            if self.debug:
                logger.info(
                    "LLM Turn: PASS skipped session=%s, depends_on=%s",
                    session_id,
                    self.depends_on,
                )
            return TurnEndDecision(should_end=True, reason="llm_skipped")

        normalized_text = (text or "").strip()
        if not normalized_text:
            return TurnEndDecision(should_end=True, reason="llm_no_text")

        params = {
            "model": self.model,
            "messages": self._build_messages(
                text=normalized_text,
                recorded_duration=recorded_duration,
                silence_duration=silence_duration,
                context=context,
            ),
            "max_tokens": 1,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.request_timeout is not None:
            params["timeout"] = self.request_timeout

        started_at = time.perf_counter()
        try:
            response = await self.openai_client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
        except Exception:
            elapsed = time.perf_counter() - started_at
            logger.error(
                "Error in LLM turn-end gate after %.3f sec; passing turn end",
                elapsed,
                exc_info=True,
            )
            return TurnEndDecision(should_end=True, reason="llm_error")
        elapsed = time.perf_counter() - started_at

        should_end = self._parse_response(content)
        if self.debug:
            logger.info(
                "LLM Turn: %s session=%s, elapsed=%.3f, content=%r, timeout=%s",
                "PASS complete" if should_end else "WAIT incomplete",
                session_id,
                elapsed,
                content,
                self.timeout,
            )

        return TurnEndDecision(
            should_end=should_end,
            confidence=None,
            reason="llm_complete" if should_end else "llm_incomplete",
            timeout=None if should_end else self.timeout,
        )
