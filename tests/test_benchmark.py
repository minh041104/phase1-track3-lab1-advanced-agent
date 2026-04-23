import time

from run_benchmark import _run_agent_batch
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.utils import load_dataset


class DelayedAgent:
    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    def run(self, example):
        time.sleep(self.delays.get(example.qid, 0.0))
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type="react",
            predicted_answer=example.gold_answer,
            is_correct=True,
            attempts=1,
            token_estimate=1,
            latency_ms=1,
            failure_mode="none",
            reflections=[],
            traces=[],
        )


def test_parallel_batch_preserves_dataset_order():
    examples = load_dataset("data/hotpot_mini.json")[:4]
    delays = {
        "hp1": 0.08,
        "hp2": 0.01,
        "hp3": 0.05,
        "hp4": 0.02,
    }

    records = _run_agent_batch(
        agent_name="react",
        agent=DelayedAgent(delays),
        examples=examples,
        progress_every=10,
        workers=3,
    )

    assert [record.qid for record in records] == [example.qid for example in examples]
