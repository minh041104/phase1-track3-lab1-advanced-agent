from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.runtime import Metrics, MockRuntime, TextResponse
from src.reflexion_lab.schemas import JudgeResult, ReflectionEntry
from src.reflexion_lab.utils import load_dataset


class LoopingRuntime:
    def actor_answer(self, example, attempt_id, agent_type, reflection_memory):
        return TextResponse(text="London", token_count=10, latency_ms=10)

    def evaluator(self, example, answer):
        return (
            JudgeResult(
                score=0,
                reason="Repeated the same wrong first-hop answer.",
                missing_evidence=["Need the second hop."],
            ),
            Metrics(token_count=5, latency_ms=5),
        )

    def reflector(self, example, attempt_id, judge):
        return (
            ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Do not stop at the intermediate entity.",
                next_strategy="Resolve the second hop before answering.",
            ),
            Metrics(token_count=5, latency_ms=5),
        )


def test_react_agent_stays_on_first_wrong_answer():
    examples = {example.qid: example for example in load_dataset("data/hotpot_mini.json")}

    record = ReActAgent(runtime=MockRuntime()).run(examples["hp2"])

    assert not record.is_correct
    assert record.predicted_answer == "London"
    assert record.attempts == 1
    assert record.failure_mode == "incomplete_multi_hop"


def test_reflexion_agent_recovers_after_reflection():
    examples = {example.qid: example for example in load_dataset("data/hotpot_mini.json")}

    record = ReflexionAgent(max_attempts=3, runtime=MockRuntime()).run(examples["hp2"])

    assert record.is_correct
    assert record.predicted_answer == "River Thames"
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[0].reflection is not None


def test_adaptive_max_attempts_stops_looping_runtime():
    examples = {example.qid: example for example in load_dataset("data/hotpot_mini.json")}

    record = ReflexionAgent(
        max_attempts=5,
        runtime=LoopingRuntime(),
        adaptive_max_attempts=True,
    ).run(examples["hp2"])

    assert not record.is_correct
    assert record.attempts == 2
    assert record.failure_mode == "looping"


def test_memory_compression_keeps_latest_unique_items():
    agent = ReflexionAgent(memory_limit=2)
    reflection_memory = [
        "Lesson: one Strategy: first",
        "Lesson: two Strategy: second",
        "Lesson: one Strategy: first",
        "Lesson: three Strategy: third",
    ]

    agent._compress_reflection_memory(reflection_memory)

    assert reflection_memory == [
        "Lesson: one Strategy: first",
        "Lesson: three Strategy: third",
    ]
