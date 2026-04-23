# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.5 | 1.0 | 0.5 |
| Avg attempts | 1 | 1.5 | 0.5 |
| Avg token estimate | 78 | 153.62 | 75.62 |
| Avg latency (ms) | 200 | 405 | 205 |

## Failure modes
```json
{
  "overall": {
    "none": 12,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  },
  "react": {
    "none": 4,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  },
  "reflexion": {
    "none": 8
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- adaptive_max_attempts
- memory_compression

## Discussion
Reflexion improved exact match whenever the first answer stopped after an intermediate hop or jumped to an unsupported entity. The main tradeoff was higher attempt count, higher token usage, and higher latency because the agent spent extra budget on evaluator and reflector calls. The remaining failure modes should be inspected by comparing the first wrong answer, the judge reason, and the next strategy recorded in the trace. In a real 100+ example HotpotQA run, this report should be expanded with concrete examples of when reflection memory helped, when the evaluator was too weak, and when extra attempts only increased cost without improving the final answer.
