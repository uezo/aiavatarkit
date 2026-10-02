# Semantic turn taking

Short acknowledgments such as "uh-huh" and "yeah" are a natural part of
conversation. `TurnTakingGate` helps prevent them from unintentionally
interrupting the assistant, while still allowing users to ask a question,
give an answer, or otherwise take a turn.

- [Quick start with Jev](#quick-start-with-jev)
- [Built-in gates](#built-in-gates)
- [Combining multiple gates](#combining-multiple-gates)
- [Playback and barge-in policies](#playback-and-barge-in-policies)
- [Deep dive](#deep-dive)
- [Diagnostics and limitations](#diagnostics-and-limitations)

## Quick start with Jev

Use `SileroStreamSpeechDetector` so recognized user text is available before the
turn-taking decision. In an existing [WebSocket app](adapters-websocket.md), reuse
your configured `speech_recognizer`, `llm`, and `tts`, and set `TYPESAFE_API_KEY`
in the environment. Add the gate and forward the browser's playback events:

```python
import os
import httpx
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_taking_gates.jev import JevTurnTakingGate

jev_http_client = httpx.AsyncClient()
jev_gate = JevTurnTakingGate(
    http_client=jev_http_client,
    api_key=os.environ["TYPESAFE_API_KEY"],
    discard_threshold=0.5,
    response_end_grace_seconds=0.3,
    debug=True,
)
vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    on_recording_started_min_duration=1.5,
    turn_taking_gate=jev_gate,
)
aiavatar_app = AIAvatarWebSocketServer(
    vad=vad,
    stt=speech_recognizer,
    llm=llm,
    tts=tts,
    mute_on_barge_in=True,
)

@aiavatar_app.on_request
async def receive_playback(request):
    if request.type != "playback" or not aiavatar_app.can_handle(request.session_id):
        return
    if vad.turn_taking_gate is not None:
        vad.turn_taking_gate.handle_playback_event(
            request.session_id, **(request.metadata or {}),
        )
```

Keep the app's existing routes and startup/shutdown handling. The HTTP client is
application-owned: call `await jev_http_client.aclose()` during shutdown, after
active sessions have finished. It can also be shared with `JevTurnEndGate`.

Use the maintained [2D or 3D Web client](../examples/websocket/README.md) with
**Barge-in enabled** so microphone input continues during assistant playback.
These pages already send the playback notifications required by the gate.
Custom clients must send the same
[playback events](../examples/websocket/README.md#playback-context-for-turn-taking).
The standard WebSocket adapter initializes and closes gate sessions through the
VAD; no additional session-start or disconnect hooks are needed for Jev.

With these example settings:

- Short utterances during playback go to Jev. A take-turn probability at or
  below `0.5` is discarded; a higher probability starts normal request handling.
- Utterances reaching `1.5` seconds stop playback through the existing barge-in
  callback, then bypass turn-taking classification when finalized.
- Replies in the last `0.3` seconds of the confirmed final audio chunk bypass
  classification so the user can answer promptly.

`0.5` and `0.3` are example choices, not library defaults. Enable INFO logging
for `aiavatar.sts.vad.turn_taking_gates` to see the context and decisions produced
by `debug=True`. Try both a listening acknowledgment and an interrupting question
while the assistant speaks; see [diagnostics](#diagnostics-and-limitations) to
distinguish a Jev decision from a bypass.

## Built-in gates

### JevTurnTakingGate

Jev classifies the conversational role of the user's recognized text using the
assistant's estimated spoken prefix and the known full text of started audio
chunks. It can distinguish an acknowledgment during an explanation from a short
answer to a question currently being spoken. The full text may include unread
words; it is context, not evidence that the user has heard them.

Import `JevTurnTakingGate` from `aiavatar.sts.vad.turn_taking_gates.jev`.
Its required constructor arguments are `http_client` and `api_key`.

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | `"jev-latest"` | Jev model to use. |
| `discard_threshold` | `0.2` | Discard when the take-turn probability is at or below this value. Lower values allow more inputs. |
| `request_timeout` | `1.0` | Total API request deadline in seconds; also applied to HTTPX I/O timeouts. |
| `response_end_grace_seconds` | `0.0` | Allow replies near the confirmed end of the response without classification; zero disables this policy. |
| `skip_condition` | `None` | Use the VAD-supplied duration policy unless an explicit predicate is provided. |
| `instructions` | Built-in instructions | Optional replacement instructions for Jev. |
| `debug` | `False` | Log playback context and classification decisions. |

`probability` means the probability that the user should take the turn, not
confidence in whichever boolean label was returned. For example, with a
threshold of `0.5`, a probability of `0.1` rejects the input and `0.8` allows it.

Timeouts allow the current input with reason `jev_timeout`. Other API errors or
invalid responses allow it with `jev_error`. Cancellation propagates instead of
being converted into an accepted turn. When used together with a turn-end gate,
Jev's turn-taking request runs after the turn-end gate has released the recording.

### SessionAllowTurnTakingGate

Use this gate when the application knows that a particular session should be
allowed to answer, such as while asking a confirmation question. It grants one
evaluation an explicit allowance, without calling a model:

```python
from aiavatar.sts.vad.turn_taking_gates.session_allow import SessionAllowTurnTakingGate

session_allow = SessionAllowTurnTakingGate(default_expires_in=300.0)

def begin_confirmation(session_id):
    # Call while asking the question so an overlapping answer can be accepted.
    session_allow.allow(session_id, expires_in=30.0, reason="confirmation_answer")

def process_answer(session_id, text):
    session_allow.release(session_id)
    # Continue the application's answer handling here.

def cleanup_connection(session_id):
    session_allow.release(session_id)
```

Place it before Jev in a Manager, as shown below. An active allowance returns
`True` regardless of the input text and is consumed once. Without an allowance,
or after expiry, it returns `False` so the Manager can ask the next gate. It does
not provide a force-reject mode. Used alone, it rejects classified inputs that
have no active allowance.

`allow(session_id, expires_in=None, reason="session_allow")` uses
`default_expires_in` when no expiry is supplied. Repeated calls replace the
pending allowance rather than queueing allowances. `get_allowance(session_id)`
inspects it without consuming it; `release()` and `reset_session()` clear it.
Expiry uses monotonic time and is checked lazily, without a background timer.

The allowance applies to the **next evaluation of this gate**, which may not be
the next utterance. A shared bypass or an earlier gate's acceptance can skip its
evaluation. Release any unused allowance when handling the answer and on
disconnect; Manager session cleanup does not clear child allowances. Use a
[custom gate](#custom-gates) when acceptance must also depend on the answer text.

## Combining multiple gates

`TurnTakingGateManager` implements the same interface as a single gate. Pass it
to the VAD's existing `turn_taking_gate` argument. Using the gates above, configure
the Manager before constructing the VAD and WebSocket server:

```python
from aiavatar.sts.vad.turn_taking_gates import TurnTakingGateManager

manager = TurnTakingGateManager(
    gates=[session_allow, jev_gate],
    response_end_grace_seconds=0.3,
    debug=True,
)
vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    on_recording_started_min_duration=1.5,
    turn_taking_gate=manager,
)
# Construct the WebSocket server and forward playback events as in the quick start.
```

Children run in list order. The first `should_take_turn=True` accepts the input
and stops evaluation. `False` continues to the next gate; all `False` results
discard the input. An empty Manager allows input. This is an OR policy, unlike
the all-must-pass policy used by turn-end gates.

Set **`skip_condition` and `response_end_grace_seconds` on the Manager**. These
policies run once before any child classifier. Child values for these options
are ignored, even if the Manager uses its defaults. With
`manager.skip_condition=None`, the VAD supplies its duration-based default; the
Manager does not inherit a child's condition. Jev's model, instructions,
threshold, request timeout, and debug settings still apply.

The Manager owns one playback history and one active decision per session. All
children receive the same frozen spoken-prefix and full-text snapshot through
their `should_take_turn()` methods. Playback events go to the Manager, and no
child playback sessions need to be created.

A provider failure allows the input through normal VAD evaluation; it is not
treated as `False` to continue the chain. Jev timeout/error decisions also allow
it immediately. Cancellation propagates. The Manager adds no overall timeout,
so sequential classifiers can add their individual waiting times.

## Playback and barge-in policies

### Interrupting longer utterances

The quick start combines semantic classification of short utterances with
duration-based early interruption. With `mute_on_barge_in=True`, the WebSocket
server stops playback when `on_recording_started` fires. Adjust its threshold on
the VAD, for example:

```python
vad.on_recording_started_min_duration = 3.0
```

When a turn-taking gate is configured, the default recording-start condition
uses duration only; `on_recording_started_min_text_length` does not trigger it.
Without a gate, the default still allows either duration or text length.
An explicit `should_trigger_recording_started` callback overrides these defaults.

With `skip_condition=None`, the VAD supplies the matching classification bypass:
`recorded_duration >= vad.on_recording_started_min_duration`. A matching finalized
input passes with reason `turn_take_skipped`. Each input uses its own duration;
no skip flag carries over to the next recording.

Stopping playback does not finalize the recording or bypass a turn-end gate.
The duration includes internal pauses and excludes current trailing silence;
it is not a sum of voiced samples. A `max_duration` flush reports total recording
duration, including trailing silence, and the skip condition uses that value.
The existing stop operation does not cancel generation or add suppression of
later audio chunks from the response.

For interruption only after a turn-taking decision, set the server's
`mute_on_barge_in=False`. This disables its recording-start stop callback, while
the browser's Barge-in option must remain enabled to capture the user. The
VAD-supplied duration bypass still applies unless explicitly overridden.

### Custom skip conditions

`skip_condition(text, recorded_duration)` is a synchronous predicate. An explicit
condition on the gate overrides the VAD default. For example, to classify long
utterances as well:

```python
jev_gate = JevTurnTakingGate(
    http_client=jev_http_client,
    api_key=os.environ["TYPESAFE_API_KEY"],
    skip_condition=lambda text, recorded_duration: False,
)
```

For multiple gates, set this condition on the Manager instead. Changing the skip
condition does not change when `on_recording_started` fires. Keep these policies
aligned if that callback stops playback: an input already stopped by the
callback can otherwise be rejected by classification.

### Replies near the end of a response

`response_end_grace_seconds=0.3` allows inputs in the final 300 ms of the
**confirmed last audio chunk** without a classifier call, with reason
`playback_response_end_grace`. The default is `0.0` (disabled).

This uses the estimated playback position at the user's speech end, including
when the audio has naturally finished by decision time. Earlier chunks, gaps,
unconfirmed final chunks, and interrupted audio do not qualify. A new chunk
invalidates the previous final designation. The policy intentionally accepts
acknowledgments in that final window too; it does not change Jev's threshold.

## Deep dive

### From recognized speech to a request

When configured, [turn-end gates](vad-turn-end.md) first decide whether the user
has finished speaking. A turn-taking gate then decides whether that completed
utterance should start a new request.

The VAD evaluates the gate before firing `on_speech_detected`. Only accepted
utterances reach those callbacks, including the pipeline callback that creates
and invokes an `STSRequest`. The VAD does not create the request itself. A
rejected utterance never reaches the pipeline's `accepted` notification, request
queue, or response-stop operation.

No application speech-filter callback or special registration order is needed.
Callbacks run in registration order and their return values do not stop later
callbacks. Ordinary callback exceptions are logged and the next handler runs;
cancellation propagates. Closing a session during one accepted-input callback
does not suppress later callbacks.

The gate does not suppress partial-recognition or recording-start events. A
detector supplying no recognized text passes through without classification;
recognition is not moved from the pipeline into the VAD. Text requests sent
directly with `invoke` also do not pass through the VAD gate.

### Playback events and progress estimation

The maintained Web client sends `type="playback"` control messages separately
from microphone `data` messages. Each decoded audio chunk gets a `playback_id`.
Its start event carries the full chunk text, duration in seconds, and the
response's top-level `transaction_id`. The WebSocket adapter propagates that
transaction ID from the pipeline without an application response hook.

`gate.handle_playback_event(session_id, **metadata)` routes events as follows:

| `event` | Session method | Purpose |
| --- | --- | --- |
| `start` | `start_playback()` | Store the chunk and its server-side monotonic start time. |
| `end` | `end_playback()` | Record natural completion (`completed=True`) or interruption. |
| `final` | `mark_response_final()` | Identify the response's last audio chunk without changing its timing. |

The method accepts `session_id` positionally or by keyword and ignores unrelated
metadata fields. It updates existing, open sessions only. Missing/closed sessions
and unknown events return `False`; valid events return the delegated method's
result. Only the literal boolean `True` marks an end as completed. An end event
for an older playback ID does not stop a newer start. Duration expiry also ends
estimated playback if an end event is lost.

A normal response `final` lets the browser identify the last audio chunk, even
if it has already started or naturally ended. Interrupted/error responses do
not establish a final chunk, and an empty playback queue is not sufficient.
See the [client protocol](../examples/websocket/README.md#playback-context-for-turn-taking)
for message shapes and custom-page integration.

The gate retains started chunks for the current transaction. A different
transaction replaces that history; previous responses are not retained. Missing,
empty, or invalid transaction IDs use independent single-chunk tracking.

The VAD supplies `vad_performance.speech_end_at` to estimate what was spoken at
the **end of the user's speech**, rather than when classification begins. The
session subtracts the delay from current UTC time to that timezone-aware
timestamp from its monotonic clock. Future timestamps contribute zero delay;
missing, invalid, or timezone-naive timestamps use decision time.

For each chunk started by that target time, the estimate includes its spoken
text. A partial chunk contributes
`text[:floor(len(text) * elapsed / duration_seconds)]`. Prefixes accumulate within
the transaction, so a speech end in an earlier chunk or a known gap can still be
evaluated after the next chunk starts. Ended or expired playback without a
transaction ID keeps the single-chunk bypass behavior.

At a known boundary after a completed chunk, if the next chunk has zero
estimated characters, the classifier receives a continuation hint: the next
chunk's first two characters (or all of its text if shorter), followed by `…`.
This also applies to a known gap. It does not apply inside the preceding chunk,
after an interrupted partial chunk, on the first chunk, or after the next chunk
has at least one estimated character. `PlaybackEstimate.assistant_spoken_text`
and timing stay unchanged; `evaluation_text` adds the `continuation_hint`.

The same snapshot contains `assistant_full_text`: all text from the transaction's
chunks that have started by decision time, including unread portions. Without
a transaction ID, this is only the current chunk's full text. Queued chunks that
have not started and text not yet generated are unknown to the gate. Both text
fields are frozen before awaiting classification.

Jev uses this full text to interpret answers to a question currently being read.
A question occurring later in the full text alone does not make every listening
acknowledgment an answer. Supplying full text improves the available context but
does not guarantee acceptance of every short answer.

### Session ownership and cancellation

The gate keeps one `TurnTakingSession` per connection's unique session ID. Gates
can be shared across connections and VAD instances with different session IDs.
Playback state, the VAD-supplied default skip condition, and pending decisions
belong to that helper; the gate does not depend on VAD internals.

Silero and SileroStream initialize the helper when creating their VAD session.
Normal recording resets preserve it. Session deletion closes it and cancels
pending decisions; `await vad.finalize_session(session_id)` also waits for cleanup.
The standard WebSocket adapter performs this finalization on disconnect.

A newly evaluated finalized input cancels the previous pending decision without
waiting for it to finish, even if the new input bypasses classification. Only
the latest decision remains eligible. Superseded results or exceptions propagate
as `asyncio.CancelledError`, including when a classifier suppresses its initial
cancellation. Decisions already returned to the pipeline are unaffected.

For use outside a VAD, register state with
`gate.get_session(session_id, create=True, default_skip_condition=...)`. The
default `create=False` only looks up an existing helper. Call
`await gate.evaluate(session_id, text, recorded_duration=..., recording_id=...,
speech_end_at=...)` for the managed decision flow; missing or closed sessions
reject input rather than being recreated.

`gate.close_session(session_id)` removes and synchronously closes the helper.
Await its `aclose()` if a helper was returned to finish pending cleanup.
`gate.create_session(...)` creates an unregistered helper for direct use; callers
own its `should_take_turn()` calls and cleanup. Inspect registered playback with
`gate.get_session(session_id)` and, when present,
`session.estimate_playback(speech_end_at=...)`.

Use a new session ID for each connection. Playback bridges must reject controls
from disconnected connections before updating state. Evaluation obtains the
helper when it starts and checks ownership again after classification. Neither
evaluation nor playback controls recreate missing state. Provider resources,
including Jev's HTTP client, still require separate application shutdown cleanup.

### Custom gates

Subclass `TurnTakingGate` and implement `should_take_turn()`. The supplied spoken
prefix includes any continuation hint; accept `assistant_full_text` even when
your classifier does not use it. For example, with an application-provided async
`classify_turn(user_text, assistant_spoken_text, assistant_full_text)` function:

```python
from typing import Optional
from aiavatar.sts.vad.turn_taking_gates import TurnTakingDecision, TurnTakingGate

class CustomTurnTakingGate(TurnTakingGate):
    async def should_take_turn(
        self,
        user_text: Optional[str],
        assistant_spoken_text: Optional[str],
        *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
    ) -> TurnTakingDecision:
        take_turn = await classify_turn(user_text, assistant_spoken_text, assistant_full_text)
        return TurnTakingDecision(take_turn, None, "custom_classifier")

custom_gate = CustomTurnTakingGate(response_end_grace_seconds=0.3, debug=True)
# Pass custom_gate as turn_taking_gate, or include it in a Manager.
```

Use `probability=None` if the implementation has no probability score. Handle
expected provider failures by allowing the turn with a diagnostic reason, and
preserve `asyncio.CancelledError`. If defining a constructor, call
`super().__init__()` to initialize the shared policies and registry.

`base.py` owns the interface, immutable `TurnTakingDecision`, session registry,
and managed `evaluate()` flow. `session.py` owns playback estimation, bypasses,
and cancellation; `manager.py` composes classifiers. Provider implementations
only need to classify the supplied context. Generic package exports do not
import the Jev implementation.

## Diagnostics and limitations

With `debug=True` and INFO logging enabled, loggers under
`aiavatar.sts.vad.turn_taking_gates` report session creation/closure, playback
events, finalized input, bypasses, and decisions. Session logs include
`transaction_id`, `assistant_spoken_text`, `assistant_full_text`,
`evaluation_text`, `continuation_hint`, `is_final_chunk`, and `remaining_seconds`.
The Jev API log labels the supplemented evaluation text `assistant_spoken_text`
and includes `user_text`, `should_take_turn`, `probability`, `reason`, and elapsed
time. Manager debug logs identify each evaluated child's order, class, and result;
each child's own debug setting controls its classifier diagnostics.

`event=superseded` means newer finalized input replaced a pending decision.
Repeated cleanup does not duplicate `event=session_closed`. These lifecycle
events describe the helper, not network connection establishment. Ignored
playback controls are logged by the gate unless rejected earlier by the transport.
With debug disabled, these diagnostic logs are silent; failure warnings remain.

Keep the following boundaries in mind:

- Classification is skipped when there is no recognized text, no playback at
  the estimated time, or no estimated spoken prefix. A configured skip condition
  or final-response grace can also accept an input without consulting any gate.
- Progress is an approximation at the user's speech end, not speech onset.
  Network delay and uneven text-to-audio alignment affect it. The continuation
  hint previews unread text; it does not claim the user heard it.
- The gate cannot predict chunks it has not received. It retains only the
  current response transaction. A response-ID mismatch with ongoing generation
  is not a reason to discard user speech.
- `recording_id` is optional diagnostic metadata. Missing or unknown recording
  IDs do not reject input.
- A partial transcript can be displayed before a turn-taking decision. The Web
  client's optional [separate transcript display](../examples/websocket/README.md#recognition-preview)
  keeps those partial results from replacing the assistant's message.
- VAD minimum-duration filtering and recognized-text validation still apply
  before turn-taking evaluation. Displaying a partial transcript does not
  guarantee that an utterance reaches this gate or becomes a request.

---

[← Speech detector (VAD)](vad.md) · [Documentation index](../README.md#-documentation)
