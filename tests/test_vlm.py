import numpy as np, pytest
from lens.vlm import FakeVLM, GeminiVLM

def test_fake_vlm_sequences_replies():
    v = FakeVLM(["a", "b"])
    img = np.zeros((4, 4, 3), np.uint8)
    assert v.ask("q1", [img]) == "a" and v.ask("q2", [img]) == "b"
    assert v.calls[0][0] == "q1"

def test_gemini_retries_transient(monkeypatch):
    v = GeminiVLM(max_retries=3)
    tries = []
    def flaky(text, images):
        tries.append(1)
        if len(tries) < 3: raise RuntimeError("503 UNAVAILABLE")
        return "ok"
    monkeypatch.setattr(v, "_call_once", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert v.ask("q", []) == "ok" and len(tries) == 3

def _no_sleep_allowed(monkeypatch):
    """Any call to time.sleep during a permanently-fatal error is a bug:
    it means the code retried something that can never succeed."""
    def boom(seconds):
        raise AssertionError(f"time.sleep({seconds}) called — should not retry a fatal error")
    monkeypatch.setattr("time.sleep", boom)

def test_missing_api_key_raises_immediately_no_retry(monkeypatch):
    v = GeminiVLM(max_retries=8)
    calls = []
    def missing_key(text, images):
        calls.append(1)
        raise RuntimeError("GEMINI_API_KEY unset (see .env.example)")
    monkeypatch.setattr(v, "_call_once", missing_key)
    _no_sleep_allowed(monkeypatch)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        v.ask("q", [])
    assert len(calls) == 1

def test_auth_rejection_raises_immediately_no_retry(monkeypatch):
    v = GeminiVLM(max_retries=8)
    calls = []
    def unauthorized(text, images):
        calls.append(1)
        raise RuntimeError("403 PERMISSION_DENIED: API key invalid")
    monkeypatch.setattr(v, "_call_once", unauthorized)
    _no_sleep_allowed(monkeypatch)
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        v.ask("q", [])
    assert len(calls) == 1

def test_empty_message_exception_classified_transient(monkeypatch):
    v = GeminiVLM(max_retries=3)
    tries = []
    def flaky(text, images):
        tries.append(1)
        if len(tries) < 3: raise RuntimeError("")  # e.g. raw socket failure
        return "ok"
    monkeypatch.setattr(v, "_call_once", flaky)
    waits = []
    monkeypatch.setattr("time.sleep", lambda s: waits.append(s))
    assert v.ask("q", []) == "ok" and len(tries) == 3
    # Transient schedule is min(120, 10 * 2**attempt): 10, then 20.
    # The flat "other" wait would produce [10, 10] instead — this
    # assertion fails under that misclassification.
    assert waits == [10, 20]
