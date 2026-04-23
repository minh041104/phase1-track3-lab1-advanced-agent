from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .mock_runtime import FAILURE_MODE_BY_QID
from .runtime import LabRuntime, MockRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: LabRuntime = field(default_factory=MockRuntime)
    adaptive_max_attempts: bool = False
    memory_limit: int = 3

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            actor_result = self.runtime.actor_answer(
                example,
                attempt_id,
                self.agent_type,
                reflection_memory,
            )
            judge, judge_metrics = self.runtime.evaluator(example, actor_result.text)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=actor_result.text,
                score=judge.score,
                reason=judge.reason,
                token_estimate=actor_result.token_count + judge_metrics.token_count,
                latency_ms=actor_result.latency_ms + judge_metrics.latency_ms,
            )

            final_answer = actor_result.text
            final_score = judge.score

            if judge.score != 1 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, reflection_metrics = self.runtime.reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(
                    f"Lesson: {reflection.lesson} Strategy: {reflection.next_strategy}"
                )
                self._compress_reflection_memory(reflection_memory)
                trace.reflection = reflection
                trace.token_estimate += reflection_metrics.token_count
                trace.latency_ms += reflection_metrics.latency_ms

            traces.append(trace)
            if judge.score == 1:
                break
            if self.adaptive_max_attempts and self._should_stop_early(traces):
                break

        total_tokens = sum(trace.token_estimate for trace in traces)
        total_latency = sum(trace.latency_ms for trace in traces)
        failure_mode = self._infer_failure_mode(example, traces, final_score)

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )

    def _infer_failure_mode(
        self,
        example: QAExample,
        traces: list[AttemptTrace],
        final_score: int,
    ) -> str:
        if final_score == 1:
            return "none"

        normalized_answers = [normalize_answer(trace.answer) for trace in traces]
        if len(normalized_answers) > 1 and len(set(normalized_answers)) == 1:
            return "looping"
        if (
            self.agent_type == "reflexion"
            and len(normalized_answers) > 1
            and len(set(normalized_answers)) > 1
        ):
            return "reflection_overfit"
        return FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

    def _should_stop_early(self, traces: list[AttemptTrace]) -> bool:
        if len(traces) < 2:
            return False

        last_answer = normalize_answer(traces[-1].answer)
        prev_answer = normalize_answer(traces[-2].answer)
        return last_answer == prev_answer and traces[-1].score == 0 and traces[-2].score == 0

    def _compress_reflection_memory(self, reflection_memory: list[str]) -> None:
        if self.memory_limit <= 0 or len(reflection_memory) <= self.memory_limit:
            return

        deduped: list[str] = []
        seen: set[str] = set()
        for item in reversed(reflection_memory):
            key = item.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            compact = item if len(item) <= 240 else f"{item[:237]}..."
            deduped.append(compact)
            if len(deduped) == self.memory_limit:
                break

        reflection_memory[:] = list(reversed(deduped))


class ReActAgent(BaseAgent):
    def __init__(self, runtime: LabRuntime | None = None) -> None:
        super().__init__(
            agent_type="react",
            max_attempts=1,
            runtime=runtime or MockRuntime(),
            adaptive_max_attempts=False,
            memory_limit=0,
        )


class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        max_attempts: int = 3,
        runtime: LabRuntime | None = None,
        adaptive_max_attempts: bool = True,
        memory_limit: int = 3,
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            runtime=runtime or MockRuntime(),
            adaptive_max_attempts=adaptive_max_attempts,
            memory_limit=memory_limit,
        )
