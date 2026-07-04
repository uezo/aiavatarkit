import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import TurnEndDecision, TurnEndGate, TurnEndGateContext

logger = logging.getLogger(__name__)


@dataclass
class _TurnEndGateHoldState:
    active: bool = False
    timeout: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    wait_decisions: Dict[str, TurnEndDecision] = field(default_factory=dict)
    pending_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)


class TurnEndGateManager:
    """Coordinates turn-end gates and owns gate wait/pending state."""

    def __init__(
        self,
        gates: Optional[List[TurnEndGate]] = None,
        *,
        debug: bool = False,
        log: Optional[logging.Logger] = None,
    ):
        self.gates = gates or []
        self.debug = debug
        self.logger = log or logger
        self._states: Dict[str, _TurnEndGateHoldState] = {}

    def has_gates(self) -> bool:
        return bool(self.gates)

    def reset_session(self, session_id: str, session: Any = None):
        state = self._states.pop(session_id, None)
        if state is not None:
            for task in state.pending_tasks.values():
                if task.done():
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        self.logger.error(
                            "Error in pending turn-end gate task during reset: session=%s",
                            session_id,
                            exc_info=True,
                        )
                else:
                    task.cancel()
        if session is not None:
            self._sync_session_debug_state(session, _TurnEndGateHoldState())

    async def should_end_turn(
        self,
        *,
        session: Any,
        audio: Optional[bytes] = None,
        audio_factory: Optional[Callable[[], bytes]] = None,
        sample_rate: int,
        channels: int,
        recorded_duration: float,
        silence_duration: float,
        silence_duration_threshold: float,
        text: Optional[str] = None,
    ) -> bool:
        if not self.gates:
            return True

        session_id = session.session_id
        hold_duration = max(0.0, silence_duration - silence_duration_threshold)
        state = self._states.setdefault(session_id, _TurnEndGateHoldState())

        if state.active:
            self._harvest_pending_tasks(session_id, state, hold_duration)
            if not state.wait_decisions and not state.pending_tasks:
                self.reset_session(session_id, session=session)
                return True
            self._sync_session_debug_state(session, state)
            if state.timeout is not None and hold_duration >= state.timeout:
                if self.debug:
                    self.logger.info(
                        "Turn-end gates: FORCE timeout session=%s, hold_duration=%.3f, timeout=%.3f, reasons=%s",
                        session_id,
                        hold_duration,
                        state.timeout,
                        state.reasons,
                )
                self.reset_session(session_id, session=session)
                return True
            return False

        if audio is None:
            if audio_factory is None:
                raise ValueError("audio or audio_factory must be provided")
            audio = audio_factory()

        decisions = []
        context = TurnEndGateContext()
        try:
            for gate in self.gates:
                gate_name = self._gate_name(gate)
                if getattr(gate, "run_in_background", False):
                    should_run_in_background = getattr(gate, "should_run_in_background", None)
                    if callable(should_run_in_background) and not should_run_in_background(context):
                        decision = await self._run_gate(
                            gate,
                            audio=audio,
                            sample_rate=sample_rate,
                            channels=channels,
                            recorded_duration=recorded_duration,
                            silence_duration=silence_duration,
                            session_id=session_id,
                            text=text,
                            session=session,
                            context=context,
                        )
                        context.add_decision(gate_name, decision)
                        decisions.append((gate_name, gate, decision))
                        continue

                    context_snapshot = TurnEndGateContext(decisions=dict(context.decisions))
                    task = asyncio.create_task(
                        self._run_gate(
                            gate,
                            audio=audio,
                            sample_rate=sample_rate,
                            channels=channels,
                            recorded_duration=recorded_duration,
                            silence_duration=silence_duration,
                            session_id=session_id,
                            text=text,
                            session=session,
                            context=context_snapshot,
                        )
                    )
                    state.pending_tasks[gate_name] = task
                    pending_decision = TurnEndDecision(
                        should_end=False,
                        reason=f"{gate_name}_pending",
                        timeout=getattr(gate, "timeout", None),
                        pending=True,
                    )
                    context.add_decision(gate_name, pending_decision)
                    decisions.append((gate_name, gate, pending_decision))
                    continue

                decision = await self._run_gate(
                    gate,
                    audio=audio,
                    sample_rate=sample_rate,
                    channels=channels,
                    recorded_duration=recorded_duration,
                    silence_duration=silence_duration,
                    session_id=session_id,
                    text=text,
                    session=session,
                    context=context,
                )
                context.add_decision(gate_name, decision)
                decisions.append((gate_name, gate, decision))
        except Exception:
            self.logger.error("Error in turn-end gate; ending turn with default behavior", exc_info=True)
            self.reset_session(session_id, session=session)
            return True

        wait_decisions = {}
        for gate_name, gate, decision in decisions:
            if self.debug:
                self.logger.info(
                    "Turn-end gate[%s]: %s session=%s, confidence=%s, reason=%s, timeout=%s, hold_duration=%.3f",
                    gate_name,
                    "PENDING" if decision.pending else ("PASS" if decision.should_end else "WAIT"),
                    session_id,
                    decision.confidence,
                    decision.reason,
                    decision.timeout,
                    hold_duration,
                )

            if not decision.should_end:
                wait_decisions[gate_name] = decision

        if not wait_decisions:
            self.reset_session(session_id, session=session)
            return True

        state.active = True
        state.wait_decisions.update(wait_decisions)
        self._recalculate_wait_state(state)
        self._sync_session_debug_state(session, state)
        if self.debug:
            if state.timeout is None:
                self.logger.info(
                    "Turn-end gates: WAIT latched session=%s, no force timeout, reasons=%s",
                    session_id,
                    state.reasons,
                )
            else:
                self.logger.info(
                    "Turn-end gates: WAIT latched session=%s, will force after %.3f sec of additional silence, reasons=%s",
                    session_id,
                    state.timeout,
                    state.reasons,
                )
        return False

    async def _run_gate(self, gate: TurnEndGate, **kwargs) -> TurnEndDecision:
        decision = await gate.should_end_turn(**kwargs)
        if isinstance(decision, bool):
            return TurnEndDecision(should_end=decision)
        return decision

    def _harvest_pending_tasks(self, session_id: str, state: _TurnEndGateHoldState, hold_duration: float):
        done_gate_names = [
            gate_name
            for gate_name, task in state.pending_tasks.items()
            if task.done()
        ]
        for gate_name in done_gate_names:
            task = state.pending_tasks.pop(gate_name)
            try:
                decision = task.result()
            except asyncio.CancelledError:
                state.wait_decisions.pop(gate_name, None)
                continue
            except Exception:
                state.wait_decisions.pop(gate_name, None)
                self.logger.error(
                    "Error in pending turn-end gate task; ignoring result: session=%s, gate=%s",
                    session_id,
                    gate_name,
                    exc_info=True,
                )
                continue

            if self.debug:
                self.logger.info(
                    "Turn-end gate[%s]: %s completed session=%s, confidence=%s, reason=%s, timeout=%s, hold_duration=%.3f",
                    gate_name,
                    "PASS" if decision.should_end else "WAIT",
                    session_id,
                    decision.confidence,
                    decision.reason,
                    decision.timeout,
                    hold_duration,
                )

            if not decision.should_end:
                state.wait_decisions[gate_name] = decision
            else:
                state.wait_decisions.pop(gate_name, None)

        if done_gate_names:
            self._recalculate_wait_state(state)

    def _recalculate_wait_state(self, state: _TurnEndGateHoldState):
        state.reasons = [
            decision.reason or "wait"
            for decision in state.wait_decisions.values()
        ]
        timeout_values = [
            decision.timeout
            for decision in state.wait_decisions.values()
            if decision.timeout is not None
        ]
        if len(timeout_values) != len(state.wait_decisions):
            state.timeout = None
        else:
            state.timeout = max(timeout_values) if timeout_values else None

    def _sync_session_debug_state(self, session: Any, state: _TurnEndGateHoldState):
        if not hasattr(session, "turn_end_gate_hold_active"):
            return
        session.turn_end_gate_hold_active = state.active
        session.turn_end_gate_hold_timeout = state.timeout
        session.turn_end_gate_hold_reasons = list(state.reasons)

    def _gate_name(self, gate: TurnEndGate) -> str:
        return getattr(gate, "name", gate.__class__.__name__)
