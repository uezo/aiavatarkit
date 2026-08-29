# Guardrail

Guardrails inspect a request before it reaches the model, or a response after the model has
produced it. Because responses stream, a response guardrail cannot block in time — so
instead of blocking, it interrupts: the avatar stops mid-sentence and speaks a correction.
In a conversation that is usually the better behaviour anyway, and it is what a person
would do.

Which of the two you get is decided by `applies_to` — see
[Blocking or correcting](#blocking-or-correcting).

A guardrail is your own class. Implement `apply()`, return a `GuardrailRespose`, and append
it to `llm.guardrails`.

```python
from aiavatar.sts.llm import Guardrail, GuardrailRespose

# Define guardrails
class RequestGuardrail(Guardrail):
    async def apply(self, context_id, user_id, text, files = None, system_prompt_params = None):
        if text.lower() == "problematic input":
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="block",
                text="The problematic input has been blocked."  # Immediately returns this message to the user
            )
        elif text.lower() == "hello":
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="replace",
                text="こんにちは"   # Replaces the original request text with this value
            )
        else:
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=False
            )

class ResponseGuardrail(Guardrail):
    async def apply(self, context_id, user_id, text, files = None, system_prompt_params = None):
        if "ramen" in text.lower():
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="replace",
                text="The problematic output has been blocked." # Emits an additional replacement chunk for the response
            )
        else:
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=False
            )

# Apply guardrails
service.guardrails.append(RequestGuardrail(applies_to="request"))
service.guardrails.append(ResponseGuardrail(applies_to="response"))
```

**NOTE:** When multiple guardrails are defined, they run in parallel.
Processing stops when all guardrails have finished evaluating or when the first guardrail returns a response with `is_triggered=True`.

**NOTE:** Response guardrails are evaluated only after the LLM response stream finishes, so
the problematic output does reach the user before the guardrail acts on it. That is a
deliberate trade: the user is never left waiting on a check that may not fire. For voice the
correction interrupts playback, and for text the client replaces what it displayed — see
[Interrupting speech mid-answer](#interrupting-speech-mid-answer).

## Blocking or correcting

There is one switch, and it is `applies_to`. Where you attach a guardrail decides whether it
can stop an answer or only correct one after the fact.

| `applies_to` | Runs | What waits for it | Can it prevent the output? |
| --- | --- | --- | --- |
| `"request"` | Before the LLM is called | The whole turn — nothing is generated until it returns | Yes: `"block"` stops the turn outright |
| `"response"` | After generation, before the turn closes | The end of the turn; the answer has already been sent | No: it can only interrupt and correct |
| `"both"` | Both of the above | Both | On the request side only |

```python
llm.guardrails.append(MyGuardrail(applies_to="request"))    # Blocking
llm.guardrails.append(MyGuardrail(applies_to="response"))   # Correcting
```

**Neither one is fire-and-forget.** Both are awaited; nothing is dispatched to a background
task and abandoned. What differs is what is already in flight when the wait happens.

On the request side nothing has been produced yet, so the wait is dead air: the user has
finished speaking and the avatar has not started. Every millisecond a request guardrail
takes is added to the gap before the first word, on every single turn — including the vast
majority where the guardrail finds nothing.

On the response side every audio chunk is already with the client and playing. The wait
overlaps with speech instead of with silence, which is why a response guardrail feels
asynchronous even though the pipeline is awaiting it. What it actually delays is the *end*
of the turn — the context update and the point at which the avatar is ready for the next
input — not the answer the user is currently hearing.

### Why a response guardrail cannot block

This is not a gap in the API — it falls out of streaming. The pipeline synthesizes each
speakable unit and ships it the moment it exists; that overlap is where the sub-second
response time comes from. To vet an answer *before* the user hears any of it, you would have
to hold every chunk back until generation finished and the guardrail returned. First audio
would then arrive after the whole answer had been generated **and** checked, instead of after
the first clause.

So the two are mutually exclusive: you can stream, or you can pre-screen the complete
answer, not both. AIAvatarKit streams, and the response guardrail corrects afterwards. If
your domain genuinely cannot tolerate the original reaching the user, put the check on the
request side where blocking works, constrain the model with the system prompt and
[tools that answer from a template](tools.md), or buffer the response in your own
application layer and accept the latency.

Guardrails on the same side do run concurrently with each other. `apply_guardrails` starts
them all as tasks and takes the first trigger, so adding a second guardrail to a side costs
no extra latency — see [First trigger wins](#first-trigger-wins).

**Which to choose.** Anything that must never be said belongs on the request side, where
blocking actually works — and you pay for it in silence before every answer. Anything you
can afford to correct after the fact belongs on the response side, which costs the user
nothing until it fires.

A useful split is a cheap, fast check on the request (a keyword list, a classifier you host)
and an expensive one on the response (an LLM-based judge). The fast check keeps the worst
input out; the slow one catches what got through, without ever making a well-behaved turn
wait for it.

`"both"` runs the same guardrail on both sides. That is occasionally what you want — one
policy applied to what the user said and to what the avatar answered — but it doubles the
calls, and the request-side half is on the critical path.

## Actions

`GuardrailRespose` carries an `action`, and the two sides of the pipeline treat it
differently. Knowing which is which saves a confusing afternoon.

| `action` | Request guardrail | Response guardrail |
| --- | --- | --- |
| `"replace"` | Request text is swapped for `text` before the LLM sees it | Playback is interrupted and `text` is spoken instead |
| `"block"` | The turn short-circuits: `text` is spoken and the LLM is never called | Not inspected — same as `"replace"` |
| anything else, e.g. `"warn"` | Nothing happens; the request proceeds unchanged | Not inspected — same as `"replace"` |

The request side inspects the action and falls through when it recognises neither
`"replace"` nor `"block"`. The response side does not inspect it at all: any triggered
response guardrail produces a correction.

## Interrupting speech mid-answer

A response guardrail runs after the LLM stream finishes — but the avatar is almost certainly
still talking at that point. Generation completes in a second or two; speaking several
sentences takes considerably longer. The correction therefore lands **while the earlier audio
is still playing**, and the pipeline uses that to cut it off.

The mechanism is worth knowing, because it explains what your client has to do:

1. The guardrail's corrective `LLMResponse` carries `guradrail_name`.
2. The pipeline synthesizes it like any other chunk and sets
   `metadata["is_guardrail_triggered"] = True` on it.
3. The adapter sees that flag, calls `stop_response()` to halt whatever is currently
   playing, and then delivers the corrective audio.

So the user hears the avatar stop mid-sentence and say the correction — the spoken
equivalent of "sorry, scratch that". Nothing in your guardrail has to arrange this; returning
a triggered response with `text` is enough.

What the flag does depends on the channel:

| Adapter | On `is_guardrail_triggered` |
| --- | --- |
| WebSocket | Sends a `stop` message to the client, then the corrective audio |
| Twilio Voice | Stops playback, then sends the corrective audio |
| Asterisk | Stops playback; the corrective chunk's audio is not sent |
| HTTP (SSE) | Emits the chunk as `message_replace` instead of `message`, so a text client replaces what it already displayed |

Text clients need to handle the replacement themselves. Voice clients get the interruption
for free, provided they honour the `stop` message — the bundled browser client does.

Because the correction arrives late by design, keep it short and self-contained. It is spoken
into a conversation that has already gone somewhere else.

## Warn-only guardrails

Every triggered guardrail is logged at warning level before anything else happens, whatever
its action:

```
Guardrail for request 'MyGuardrail' triggered: action=warn, text=None
```

So on the **request** side you get a log-only guardrail by returning a triggered response
with an action the pipeline does not act on:

```python
class WatchOnlyGuardrail(Guardrail):
    async def apply(self, context_id, user_id, text, files=None, system_prompt_params=None):
        if "cancel my subscription" in text.lower():
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="warn",      # Logged; the request still goes through unchanged
            )
        return GuardrailRespose(guardrail_name=self.name, is_triggered=False)
```

On the **response** side this does not work — the action is ignored and `text` is spoken,
which for a warn-only guardrail would be `None`. Log inside `apply()` and report the
guardrail as not triggered instead:

```python
class ResponseWatchGuardrail(Guardrail):
    async def apply(self, context_id, user_id, text, files=None, system_prompt_params=None):
        if "ramen" in text.lower():
            logger.warning("Response mentioned ramen: context_id=%s", context_id)
        return GuardrailRespose(guardrail_name=self.name, is_triggered=False)
```

That keeps the observation and leaves the conversation alone. Route it wherever your other
warnings go — see [Administration](admin.md).

## First trigger wins

Guardrails run in parallel and the pipeline takes the **first** one to report
`is_triggered=True`, then cancels the rest. With several guardrails in flight, which one
that is depends on completion order, not on the order you registered them. Do not rely on
one guardrail seeing what another decided; put ordered logic inside a single guardrail.

A guardrail that only ever returns `is_triggered=False`, as in the warn-only pattern above,
never wins the race and so never suppresses another guardrail's action.

## Reading the result downstream

When a guardrail produces a response, the emitted `LLMResponse` carries the guardrail's
name. Note the field is spelled `guradrail_name` — a typo that is part of the public
surface, so match it exactly:

```python
if llm_response.guradrail_name:
    ...
```

## See also

- [LLM](llm.md) — the service guardrails attach to
- [Tools](tools.md) — returning exact values without the model rephrasing them
- [Administration](admin.md) — reviewing what was said

---

[← Documentation index](../README.md#-documentation)
