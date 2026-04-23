# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: openai
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.77 | 0.89 | 0.12 |
| Avg attempts | 1 | 1.25 | 0.25 |
| Avg token estimate | 2944.09 | 4249.03 | 1304.94 |
| Avg latency (ms) | 4561.15 | 8987.17 | 4426.02 |

## Failure modes
```json
{
  "overall": {
    "none": 166,
    "wrong_final_answer": 23,
    "looping": 9,
    "reflection_overfit": 2
  },
  "react": {
    "none": 77,
    "wrong_final_answer": 23
  },
  "reflexion": {
    "none": 89,
    "looping": 9,
    "reflection_overfit": 2
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts
- memory_compression

## Discussion
Reflexion improved exact match whenever the first answer stopped after an intermediate hop or jumped to an unsupported entity. The main tradeoff was higher attempt count, higher token usage, and higher latency because the agent spent extra budget on evaluator and reflector calls. The remaining failure modes should be inspected by comparing the first wrong answer, the judge reason, and the next strategy recorded in the trace. In a real 100+ example HotpotQA run, this report should be expanded with concrete examples of when reflection memory helped, when the evaluator was too weak, and when extra attempts only increased cost without improving the final answer.
