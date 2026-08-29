# Semantic turn end

Acoustic VAD ends a turn when the user has been silent long enough. That is wrong whenever
someone pauses mid-thought — after a conjunction, during a filler, or while recalling a
name. Turn-end gates run *after* the silence threshold is reached and decide whether the
utterance is really finished.

AIAvatarKit supports semantic VAD by combining acoustic VAD with optional turn-end gates. The acoustic VAD first detects a turn-end candidate from silence, then turn-end gates inspect audio, recognized text, or model-specific signals to decide whether the user's utterance is semantically complete.

`SileroSpeechDetector` and `SileroStreamSpeechDetector` can use built-in gates such as Smart Turn, Filler-only, Namo Turn, and LLM-based gates. You can also implement your own `TurnEndGate` when you need domain-specific turn-end logic. Gates are called only after `silence_duration_threshold` has already been reached. All gates must pass to end the turn. If any gate returns "wait", the detector keeps the current recording open until the user resumes speaking or the waiting gate's timeout forces the turn to end.

This is useful for utterances that contain a short pause but are likely to continue, such as trailing conjunctions, filler phrases, or incomplete requests.

```python
vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    turn_end_gates=[my_gate],
)
```

Gate timeouts are measured after `silence_duration_threshold` has been reached. For example, with `silence_duration_threshold=0.5` and a gate timeout of `2.0`, the longest silence wait is approximately 2.5 seconds.

Gate coordination is handled by `TurnEndGateManager`. VAD detectors only ask the manager whether the current turn-end candidate should end or keep recording. The manager keeps per-session wait state, passes previous gate decisions through `TurnEndGateContext`, and uses the longest timeout among gates that returned wait. If the detector reaches `max_duration`, it still ends the turn even when a gate is holding it, so gate waits cannot keep the recording buffer open forever.

Gates can opt into background execution by setting `run_in_background=True`. Background gates do not block audio processing while they are pending. While pending, their `timeout` is used as a provisional wait timeout, so a detector can be configured with only background gates. When a background gate finishes, its result replaces the provisional pending decision. If the result is still pending when the timeout expires, it is ignored and the turn ends.

`SmartTurnEndGate` and `NamoTurnEndGate` use one ONNX Runtime session per gate instance and serialize inference with an internal lock. This is fine for typical usage because gates run only after a VAD turn-end candidate, not for every audio chunk. For very high concurrency, create separate gate instances per detector or worker process, or add a small gate/session pool if turn-end gate latency becomes visible.

## Smart Turn Gate

`SmartTurnEndGate` uses [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) to classify the current recorded audio as complete or incomplete.

```sh
pip install "aiavatar[smart-turn]"
```

```python
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.turn_end_gates.smart_turn import SmartTurnEndGate

turn_end_gate = SmartTurnEndGate(
    threshold=0.5,
    timeout=1.5,
    debug=True,
)

vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    turn_end_gates=[turn_end_gate],
    debug=True,
)
```

To use a local Smart Turn ONNX model instead of downloading from Hugging Face, set `model_path`:

```python
turn_end_gate = SmartTurnEndGate(
    model_path="/models/smart-turn-v3.2-cpu.onnx",
)
```

## Filler-Only Gate

`FillerOnlyTurnEndGate` waits longer when the recognized text is only a filler phrase, or ends with a trailing filler phrase, such as "えっと", "あの", "um", or "uh". It normalizes text before matching, so spaces, punctuation, and symbols are ignored; for example, "えっと。" matches "えっと". One-character fillers such as "あ" are not used for trailing-filler matching, and short replies that can be meaningful answers, such as "うん", are not included in the default filler list.

This gate is most useful with `SileroStreamSpeechDetector`, because it needs recognized text.

```python
from aiavatar.sts.vad.turn_end_gates import FillerOnlyTurnEndGate, FillerPhrase

filler_gate = FillerOnlyTurnEndGate(
    name="filler",
    fillers=[
        FillerPhrase("あの", match="suffix", timeout=6.0),
        FillerPhrase("えっと", match="suffix"),
        "um",  # str means exact match
    ],
    timeout=5.0,
    debug=True,
)
```

## Namo Turn Gate

`NamoTurnEndGate` uses [videosdk-live/NAMO-Turn-Detector-v1](https://github.com/videosdk-live/NAMO-Turn-Detector-v1) to classify recognized text as end-of-turn or not-end-of-turn. It is most useful with `SileroStreamSpeechDetector`, because the stream detector can pass accumulated partial recognition text to the gate.

```sh
pip install "aiavatar[namo-turn]"
```

This gate is part of the built-in application's default stack, so the `aiavatar` command
offers to install the extra for you on first run — see
[Getting started](getting-started.md#semantic-vad-dependencies). In your own scripts, install
it yourself.

```python
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates.filler import FillerOnlyTurnEndGate
from aiavatar.sts.vad.turn_end_gates.namo_turn import NamoTurnEndGate

filler_gate = FillerOnlyTurnEndGate(
    name="filler",
    timeout=5.0,
)

turn_end_gate = NamoTurnEndGate(
    name="namo",
    language="ja",   # Japanese model. Use language=None for the multilingual model.
    threshold=0.5,
    force_end_phrases=["こんにちは"],
    timeout=1.5,
    debug=True,
)

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    segment_silence_threshold=0.05,
    silence_duration_threshold=0.5,
    turn_end_gates=[
        filler_gate,
        turn_end_gate,
    ],
    debug=True,
)
```

`threshold` is the minimum predicted probability of class 1 ("End of Turn"). Higher values require stronger evidence before ending the turn, so they hold the turn more often.

`force_end_phrases` is an optional list of exact utterances that should end the turn without running the model. Matching ignores case, whitespace, full-width variants, punctuation, and symbols, so `"こんにちは。"` matches `"こんにちは"`. Longer utterances such as `"こんにちは、今日は相談があります"` do not match.

For long recordings, Namo keeps the end of the recognized text when tokenized text exceeds the model limit, because turn-end detection depends most on the final words. If no text is available, `NamoTurnEndGate` defaults to ending the turn. You can change this with `no_text_should_end=False`.

To run Namo from local files without downloading from Hugging Face, set both `model_path` and `tokenizer_path`:

```python
turn_end_gate = NamoTurnEndGate(
    language="ja",
    model_path="/models/namo/model_quant.onnx",
    tokenizer_path="/models/namo/tokenizer",
)
```

## LLM Turn Gate

`LLMTurnEndGate` uses an OpenAI-compatible Chat Completions client to make a slower but more flexible text-based decision. It is useful as a second-stage gate after a cheaper gate has already decided to wait. It runs in the background by default, so the current audio receive loop is not blocked while waiting for the LLM response.

Pass a long-lived client instance to the constructor. The gate reuses that client instead of creating one per decision, so the underlying HTTP connection pool can be reused.

```python
from openai import AsyncOpenAI

from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates import FillerOnlyTurnEndGate
from aiavatar.sts.vad.turn_end_gates.llm import LLMTurnEndGate
from aiavatar.sts.vad.turn_end_gates.namo_turn import NamoTurnEndGate

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

filler_gate = FillerOnlyTurnEndGate(
    name="filler",
    timeout=5.0,
)

namo_gate = NamoTurnEndGate(
    name="namo",
    language="ja",
    threshold=0.5,
    timeout=1.5,
)

llm_gate = LLMTurnEndGate(
    openai_client=openai_client,
    model="gpt-4.1-mini",
    depends_on=["filler", "namo"],
    timeout=10.0,
    request_timeout=2.0,
    debug=True,
)

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    silence_duration_threshold=0.5,
    turn_end_gates=[
        filler_gate,
        namo_gate,
        llm_gate,
    ],
)
```

When `depends_on` is set, the LLM gate runs only if one of the named previous gates returned
wait. The value can be a string or a list of gate names. In the example above, normal
utterances do not call the LLM; the LLM is only called when the filler or Namo gate waits.

A background LLM gate contributes its timeout as soon as it starts, even while its result is
still pending. The manager uses the longest timeout among all waiting and pending gates; it
does not wait for one gate's timeout and then add another.

For example, suppose Namo returns wait with `timeout=3.0` and the LLM gate has `timeout=5.0`.
The times below start after `silence_duration_threshold` has already been reached:

```text
0 seconds  Namo returns wait; the background LLM gate starts.
3 seconds  The LLM may still be pending. The turn stays open because its 5-second timeout applies.
5 seconds  If the LLM is still pending, the turn ends.
```

If the LLM returns an end decision after 2 seconds, the Namo timeout remains and the turn can
end after 3 seconds. If it returns an end decision after 4 seconds, Namo's timeout has already
passed, so the turn ends then. If the LLM returns wait, the 5-second timeout remains.

`timeout` controls how long the gate may keep the turn open. `request_timeout` separately
limits the LLM API request itself.

`LLMTurnEndGate` accepts `temperature` and `reasoning_effort`, but only passes them to the API when they are explicitly set. Use the option supported by the model you choose.

## Session Hold Gate

`SessionHoldTurnEndGate` lets application logic hold the next turn-end candidate for a specific session. Unlike gates that inspect the current audio or recognized text, this gate is armed in advance when the application knows that the next answer may need more thinking time.

For example, a restaurant-search assistant may ask the user to choose a cuisine, describe their preferences, or recall an area or budget. These answers often begin with hesitation and contain pauses, such as "Well... maybe Italian." The assistant can prefix questions like these with a control tag such as `<require_restaurant_preferences />`. When the tag is detected, the session hold gate allows a longer pause for the next user answer instead of ending the turn at the normal silence threshold.

```python
from aiavatar.sts.llm import LLMResponse
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.turn_end_gates.session_hold import SessionHoldTurnEndGate

REQUIRE_PREFERENCES_TAG = "<require_restaurant_preferences />"

session_hold_gate = SessionHoldTurnEndGate(debug=True)

vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    turn_end_gates=[session_hold_gate],
    debug=True,
)

# Pass vad to AIAvatar when creating aiavatar_app.

# Prompt convention for the LLM:
# Prefix a question with <require_restaurant_preferences /> when its answer may
# require the user to recall details, compare options, or think aloud.
# Example:
# <require_restaurant_preferences />What kind of food are you in the mood for?

@aiavatar_app.sts.process_llm_chunk
async def hold_restaurant_preference_answer(
    llm_chunk: LLMResponse,
    session_id: str,
    user_id: str,
) -> dict:
    if REQUIRE_PREFERENCES_TAG in (llm_chunk.text or ""):
        session_hold_gate.hold(
            session_id,
            timeout=3.0,
            reason="restaurant_preferences",
        )
    return {}
```

The tag is detected while the LLM response is streaming, before the next user turn, and AIAvatarKit removes control tags such as this one from the synthesized voice text. With the settings above, the normal turn-end candidate is detected after 0.5 seconds of silence, and the armed gate can keep the recording open for up to 3.0 additional seconds. The hold is consumed by that candidate; subsequent turns use the normal silence threshold unless another tagged response arms the gate again.

## Custom Gate

Implement `TurnEndGate` to plug in your own decision logic. Gates receive the current recorded audio, timing information, the session id, recognized text when available, and a context containing previous gate decisions.

```python
from aiavatar.sts.vad.turn_end_gates import TurnEndDecision, TurnEndGate, TurnEndGateContext

class MyTurnEndGate(TurnEndGate):
    async def should_end_turn(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        channels: int,
        recorded_duration: float,
        silence_duration: float,
        session_id: str,
        text: str | None = None,
        session=None,
        context: TurnEndGateContext | None = None,
    ) -> TurnEndDecision:
        if text and text.endswith("and"):
            return TurnEndDecision(should_end=False, confidence=0.9, reason="continues", timeout=3.0)
        return TurnEndDecision(should_end=True, confidence=0.9, reason="complete")
```

## See also

- [Speech detector](vad.md) — the detectors that gates plug into
- [Audio filters](vad-filters.md) — processing audio before detection

---

[← Documentation index](../README.md#-documentation)
