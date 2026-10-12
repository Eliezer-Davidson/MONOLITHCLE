# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Pluggable VLM boundary. Phase 1 default: gemini-3.5-flash (GA —
the -preview tier 503s for minutes under load, measured 2026-07-21).
Transient errors back off exponentially; 429s wait longer."""
from __future__ import annotations
import io, os, time
from typing import Protocol
import numpy as np
from PIL import Image

class VLM(Protocol):
    def ask(self, text: str, images: list[np.ndarray]) -> str: ...

class FakeVLM:
    def __init__(self, replies: list[str]):
        self._replies = list(replies); self.calls = []
    def ask(self, text: str, images: list[np.ndarray]) -> str:
        self.calls.append((text, len(images)))
        return self._replies.pop(0)

_TRANSIENT = ("503", "UNAVAILABLE", "500", "INTERNAL", "504", "DEADLINE",
              "Connection", "unreachable", "timed out")
# Conditions that can never resolve themselves on retry: a missing key
# (raised by _call_once as a plain RuntimeError) or an auth rejection from
# the API. Retrying these just burns the full backoff schedule before the
# real error surfaces.
_FATAL = ("GEMINI_API_KEY unset", "401", "403", "PERMISSION_DENIED", "UNAUTHENTICATED")

class GeminiVLM:
    def __init__(self, model: str = "gemini-3.5-flash", max_retries: int = 8):
        self.model, self.max_retries = model, max_retries

    def _call_once(self, text: str, images: list[np.ndarray]) -> str:
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY unset (see .env.example)")
        parts = [types.Part.from_text(text=text)]
        for a in images:
            buf = io.BytesIO(); Image.fromarray(a).save(buf, "PNG")
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
        # Bind the client to a name and keep it alive for the whole call.
        # `genai.Client(...).models.generate_content(...)` as a bare chained
        # expression is a real footgun: google.genai.Client defines __del__
        # (calls close(), which tears down the shared httpx transport), and
        # `.models` does not hold a back-reference to the Client wrapper --
        # only to its lower-level `_api_client`. CPython's refcounting GC
        # frees the temporary Client (firing __del__ -> close()) right after
        # `.models` is fetched, i.e. BEFORE generate_content()'s network call
        # runs, which then fails deterministically with "Cannot send a
        # request, as the client has been closed." every single time --
        # observed in practice: all 8 outer retries failed identically since
        # this isn't network flakiness, it's guaranteed-every-call GC timing.
        client = genai.Client(api_key=key)
        r = client.models.generate_content(
            model=self.model, contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=4096))
        return r.text or ""

    def ask(self, text: str, images: list[np.ndarray]) -> str:
        for attempt in range(self.max_retries):
            try:
                return self._call_once(text, images)
            except Exception as e:
                s = str(e)
                # Permanently fatal: can't possibly resolve by retrying, so
                # re-raise immediately without sleeping or counting attempts.
                if any(t in s for t in _FATAL): raise
                if attempt == self.max_retries - 1: raise
                # An empty/whitespace-only message is far more likely a raw
                # socket-level failure than a permanent error — treat it as
                # transient rather than falling to the flat "other" wait.
                if not s.strip():                            wait = min(120, 10 * 2 ** attempt)
                elif "429" in s or "RESOURCE_EXHAUSTED" in s: wait = 60 * 2 ** attempt
                elif any(t in s for t in _TRANSIENT):         wait = min(120, 10 * 2 ** attempt)
                else:                                          wait = 10
                time.sleep(wait)
        raise RuntimeError("unreachable")
