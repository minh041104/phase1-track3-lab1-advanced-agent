import json
import urllib.error
import urllib.request

from src.reflexion_lab.runtime import OpenAICompatibleRuntime


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_openai_runtime_retries_after_urlerror():
    runtime = OpenAICompatibleRuntime(
        model="test-model",
        api_key="test-key",
        timeout_s=1,
        max_retries=2,
        retry_backoff_s=0,
    )
    original_urlopen = urllib.request.urlopen
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError(TimeoutError("timed out"))
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"total_tokens": 7},
            }
        )

    urllib.request.urlopen = fake_urlopen
    try:
        text, metrics = runtime._chat_text(
            [{"role": "user", "content": "hello"}]
        )
    finally:
        urllib.request.urlopen = original_urlopen

    assert text == "ok"
    assert metrics.token_count == 7
    assert calls["count"] == 2
