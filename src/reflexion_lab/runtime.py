from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

from .mock_runtime import FIRST_ATTEMPT_WRONG
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


@dataclass
class TextResponse:
    text: str
    token_count: int
    latency_ms: int


@dataclass
class Metrics:
    token_count: int
    latency_ms: int


class LabRuntime(Protocol):
    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> TextResponse:
        ...

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, Metrics]:
        ...

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, Metrics]:
        ...


def _context_text(example: QAExample) -> str:
    return "\n\n".join(
        f"Title: {chunk.title}\nText: {chunk.text}" for chunk in example.context
    )


def _rough_token_count(*parts: str) -> int:
    text = " ".join(part for part in parts if part)
    return max(1, len(text) // 4)


class MockRuntime:
    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> TextResponse:
        if example.qid not in FIRST_ATTEMPT_WRONG:
            answer = example.gold_answer
        elif agent_type == "react":
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        elif attempt_id == 1 and not reflection_memory:
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        else:
            answer = example.gold_answer

        token_count = _rough_token_count(
            example.question,
            _context_text(example),
            "\n".join(reflection_memory),
            answer,
        )
        latency_ms = 120 + (attempt_id * 35) + (40 if agent_type == "reflexion" else 0)
        return TextResponse(text=answer, token_count=token_count, latency_ms=latency_ms)

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, Metrics]:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            judge = JudgeResult(
                score=1,
                reason="Final answer matches the gold answer after normalization.",
            )
        elif normalize_answer(answer) == "london":
            judge = JudgeResult(
                score=0,
                reason=(
                    "The answer stopped at the birthplace city and never completed "
                    "the second hop to the river."
                ),
                missing_evidence=[
                    "Need to identify the river that flows through London.",
                ],
            )
        else:
            judge = JudgeResult(
                score=0,
                reason="The final answer selected the wrong second-hop entity.",
                missing_evidence=["Need to ground the answer in the second paragraph."],
                spurious_claims=[answer],
            )

        metrics = Metrics(
            token_count=_rough_token_count(example.gold_answer, answer, judge.reason),
            latency_ms=45,
        )
        return judge, metrics

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, Metrics]:
        strategy = (
            "Do the second hop explicitly: birthplace city -> river through that city."
            if example.qid == "hp2"
            else "Verify the final entity against the second paragraph before answering."
        )
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=(
                "A partial first-hop answer is not enough; the final answer must "
                "complete all hops."
            ),
            next_strategy=strategy,
        )
        metrics = Metrics(
            token_count=_rough_token_count(reflection.lesson, reflection.next_strategy),
            latency_ms=55,
        )
        return reflection, metrics


class OpenAICompatibleRuntime:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: int = 60,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> TextResponse:
        reflection_block = (
            "\n".join(f"- {entry}" for entry in reflection_memory)
            if reflection_memory
            else "- None"
        )
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Attempt: {attempt_id}\n"
            f"Agent type: {agent_type}\n\n"
            f"Context:\n{_context_text(example)}\n\n"
            f"Reflection memory:\n{reflection_block}\n\n"
            "Return only the final short answer."
        )
        text, metrics = self._chat_text(
            [
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        return TextResponse(text=text.strip(), **metrics.__dict__)

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, Metrics]:
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Gold answer:\n{example.gold_answer}\n\n"
            f"Predicted answer:\n{answer}\n\n"
            f"Context:\n{_context_text(example)}"
        )
        text, metrics = self._chat_text(
            [
                {"role": "system", "content": EVALUATOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        payload = self._parse_json(text)
        judge = JudgeResult.model_validate(payload)
        return judge, metrics

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, Metrics]:
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Attempt:\n{attempt_id}\n\n"
            f"Context:\n{_context_text(example)}\n\n"
            f"Judge result:\n{judge.model_dump_json(indent=2)}"
        )
        text, metrics = self._chat_text(
            [
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )
        payload = self._parse_json(text)
        payload.setdefault("attempt_id", attempt_id)
        payload.setdefault("failure_reason", judge.reason)
        reflection = ReflectionEntry.model_validate(payload)
        return reflection, metrics

    def _chat_text(self, messages: list[dict[str, str]]) -> tuple[str, Metrics]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    response_body = response.read().decode("utf-8")
                latency_ms = int((time.perf_counter() - start) * 1000)
                payload = json.loads(response_body)
                message = payload["choices"][0]["message"]["content"]
                usage = payload.get("usage", {})
                token_count = usage.get("total_tokens") or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )
                if not token_count:
                    token_count = _rough_token_count(
                        *(item["content"] for item in messages),
                        message,
                    )
                return message, Metrics(token_count=token_count, latency_ms=latency_ms)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"LLM request failed with HTTP {exc.code}: {detail}"
                )
                if not self._should_retry_http(exc.code) or attempt > self.max_retries:
                    raise last_error from exc
            except (TimeoutError, socket.timeout) as exc:
                last_error = RuntimeError(
                    f"LLM request timed out after {self.timeout_s}s"
                )
                if attempt > self.max_retries:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"LLM request failed: {exc.reason}")
                if not self._should_retry_url_error(exc) or attempt > self.max_retries:
                    raise last_error from exc

            time.sleep(self._retry_delay(attempt))

        raise RuntimeError("LLM request failed after retries.") from last_error

    def _retry_delay(self, attempt: int) -> float:
        return self.retry_backoff_s * (2 ** (attempt - 1))

    @staticmethod
    def _should_retry_http(status_code: int) -> bool:
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    @staticmethod
    def _should_retry_url_error(exc: urllib.error.URLError) -> bool:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        return isinstance(reason, OSError)

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Expected JSON object, got: {text}") from None
            return json.loads(cleaned[start : end + 1])


def create_runtime(
    mode: str,
    timeout_s: int = 60,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> LabRuntime:
    load_dotenv()
    normalized_mode = mode.strip().lower()
    if normalized_mode == "mock":
        return MockRuntime()
    if normalized_mode in {"openai", "openai-compatible"}:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for openai mode.")
        if not model:
            raise RuntimeError("Missing OPENAI_MODEL for openai mode.")
        return OpenAICompatibleRuntime(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
    raise ValueError(f"Unsupported mode: {mode}")
