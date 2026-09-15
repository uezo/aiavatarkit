"""A small, stateless classifier for short listener acknowledgments."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
import tomllib
from urllib.parse import urlsplit
from xml.etree import ElementTree


@dataclass(frozen=True)
class NodDecision:
    id: str
    phrase: str | None = None
    finish_reason: str = "unknown"
    elapsed_ms: float = 0.0
    utterance_id: str | None = None
    acknowledged_text: str = ""
    text_length: int = 0
    outcome: str = "candidate"


def _validate_candidates(candidates):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Nod requires a nonempty candidates list")
    ids, phrases = set(), set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidates must be objects")
        identifier = candidate.get("id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier)
                or identifier in ids or identifier in ("none", "invalid", "empty")):
            raise ValueError("Candidate IDs must be unique lowercase identifiers, excluding reserved values")
        ids.add(identifier)
        phrase = candidate.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip() or phrase in phrases:
            raise ValueError("Candidate phrases must be unique nonempty strings")
        phrases.add(phrase)
        if not isinstance(candidate.get("description"), str) or not candidate["description"].strip():
            raise ValueError("Candidates need nonempty descriptions")


def load_profile(path):
    """Load prompt and candidates from one TOML file before the async loop."""
    profile = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _validate_candidates(profile.get("candidates"))
    prompt = profile.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Nod prompt must not be empty")
    return profile


class NodEngine:
    """Share this classifier across sessions; conversation state belongs to NodSession.

    An injected HTTP client is caller-owned and may carry its own authentication.
    Otherwise httpx is imported and a client is created on the first decision.
    Close sessions before closing their shared engine.
    """

    def __init__(self, candidates, prompt, *, client=None, api_key=None,
                 model="gpt-5.6-luna", base_url="https://api.openai.com/v1",
                 request_options=None):
        _validate_candidates(candidates)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Nod prompt must not be empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Nod model must not be empty")
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query, or fragment")
        options = deepcopy({} if request_options is None else request_options)
        if not isinstance(options, dict):
            raise ValueError("request_options must be an object")
        if {"model", "messages", "stream"}.intersection(options):
            raise ValueError("request_options cannot override model, messages, or stream")
        if "max_tokens" in options and "max_completion_tokens" in options:
            raise ValueError("Set only one completion-token limit")
        self._candidates = deepcopy(candidates)
        self._choices = {item["id"]: item["phrase"] for item in candidates}
        self._phrase_ids = {item["phrase"]: item["id"] for item in candidates}
        self.prompt = prompt.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._request_options = options
        self._client = client
        self._owns_client = client is None
        self._api_key = api_key
        self._closed = False

    @property
    def candidates(self):
        return deepcopy(self._candidates)

    @classmethod
    def from_profile(cls, path, **kwargs):
        """Construct from a TOML profile containing prompt and candidates."""
        profile = load_profile(path)
        return cls(profile["candidates"], profile["prompt"], **kwargs)

    def build_payload(self, user_input):
        if not isinstance(user_input, str):
            raise TypeError("user_input must be text")
        options = "\n相槌の候補文言：\n" + "\n".join(
            f'「{item["phrase"]}」: {item["description"]}' for item in self._candidates
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.prompt + options},
                         {"role": "user", "content": user_input}],
            "temperature": 0, "max_tokens": 32, "stream": False,
        }
        host = urlsplit(self.base_url).hostname
        if host == "api.openai.com" and self.model == "gpt-5.6-luna":
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload["reasoning_effort"] = "none"
        elif host == "openrouter.ai":
            payload.update(provider={"sort": "latency", "allow_fallbacks": False},
                           reasoning={"enabled": False})
        # An explicit token-limit style replaces the default instead of sending both.
        if "max_tokens" in self._request_options:
            payload.pop("max_completion_tokens", None)
        if "max_completion_tokens" in self._request_options:
            payload.pop("max_tokens", None)
        payload.update(deepcopy(self._request_options))
        return payload

    def _parse_output(self, value):
        if not value:
            return "empty"
        # Accept only one text-only element, without attributes or XML declarations.
        if not re.fullmatch(r"<nod_assistant\s*/>|<nod_assistant>[^<]*</nod_assistant>", value):
            return "invalid"
        try:
            phrase = ElementTree.fromstring(value).text or ""
        except ElementTree.ParseError:
            return "invalid"
        return self._phrase_ids.get(phrase, "invalid") if phrase else "none"

    async def decide(self, user_input):
        """Return only a known candidate or a silent outcome; provider errors propagate."""
        if self._closed:
            raise RuntimeError("NodEngine is closed")
        payload = self.build_payload(user_input)
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=5.0)
        started = perf_counter()
        kwargs = {"json": payload}
        if self._api_key:
            kwargs["headers"] = {"Authorization": "Bearer " + self._api_key}
        response = await self._client.post(self.base_url + "/chat/completions", **kwargs)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Malformed nod response")
        result = choices[0]
        message = result.get("message")
        if not isinstance(message, dict):
            raise ValueError("Malformed nod response")
        content = message.get("content")
        value = content.strip() if isinstance(content, str) else ""
        identifier = self._parse_output(value)
        finish = result.get("finish_reason")
        if finish not in ("stop", "length", "content_filter", "tool_calls", "error"):
            finish = "unknown"
        return NodDecision(id=identifier, phrase=self._choices.get(identifier), finish_reason=finish,
                           elapsed_ms=(perf_counter() - started) * 1000,
                           outcome="candidate" if identifier in self._choices else identifier)

    async def close(self):
        if not self._closed:
            self._closed = True
            if self._owns_client and self._client is not None:
                await self._client.aclose()

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("NodEngine is closed")
        return self

    async def __aexit__(self, *exc):
        await self.close()
