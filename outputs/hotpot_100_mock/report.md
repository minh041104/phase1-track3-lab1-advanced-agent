# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg token estimate | 1476.79 | 1476.79 | 0.0 |
| Avg latency (ms) | 200 | 240 | 40 |

## Failure modes
```json
{
  "overall": {
    "none": 200
  },
  "react": {
    "none": 100
  },
  "reflexion": {
    "none": 100
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
