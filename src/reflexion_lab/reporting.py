from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .schemas import ReportPayload, RunRecord


def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)

    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {
            "count": len(rows),
            "em": round(mean(1.0 if row.is_correct else 0.0 for row in rows), 4),
            "avg_attempts": round(mean(row.attempts for row in rows), 4),
            "avg_token_estimate": round(mean(row.token_estimate for row in rows), 2),
            "avg_latency_ms": round(mean(row.latency_ms for row in rows), 2),
        }

    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {
            "em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4),
            "attempts_abs": round(
                summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"],
                4,
            ),
            "tokens_abs": round(
                summary["reflexion"]["avg_token_estimate"]
                - summary["react"]["avg_token_estimate"],
                2,
            ),
            "latency_abs": round(
                summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"],
                2,
            ),
        }
    return summary


def failure_breakdown(records: list[RunRecord]) -> dict:
    overall = Counter(record.failure_mode for record in records)
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1

    breakdown: dict[str, dict[str, int]] = {"overall": dict(overall)}
    for agent_type in sorted(grouped):
        breakdown[agent_type] = dict(grouped[agent_type])
    return breakdown


def _default_extensions(mode: str) -> list[str]:
    extensions = [
        "structured_evaluator",
        "reflection_memory",
        "benchmark_report_json",
    ]
    if mode == "mock":
        extensions.append("mock_mode_for_autograding")
    return extensions


def build_report(
    records: list[RunRecord],
    dataset_name: str,
    mode: str = "mock",
    extensions: list[str] | None = None,
) -> ReportPayload:
    report_examples = [
        {
            "qid": record.qid,
            "agent_type": record.agent_type,
            "gold_answer": record.gold_answer,
            "predicted_answer": record.predicted_answer,
            "is_correct": record.is_correct,
            "attempts": record.attempts,
            "failure_mode": record.failure_mode,
            "reflection_count": len(record.reflections),
        }
        for record in records
    ]

    discussion = (
        "Reflexion improved exact match whenever the first answer stopped after an "
        "intermediate hop or jumped to an unsupported entity. The main tradeoff was "
        "higher attempt count, higher token usage, and higher latency because the "
        "agent spent extra budget on evaluator and reflector calls. The remaining "
        "failure modes should be inspected by comparing the first wrong answer, the "
        "judge reason, and the next strategy recorded in the trace. In a real 100+ "
        "example HotpotQA run, this report should be expanded with concrete examples "
        "of when reflection memory helped, when the evaluator was too weak, and when "
        "extra attempts only increased cost without improving the final answer."
    )

    return ReportPayload(
        meta={
            "dataset": dataset_name,
            "mode": mode,
            "num_records": len(records),
            "agents": sorted({record.agent_type for record in records}),
        },
        summary=summarize(records),
        failure_modes=failure_breakdown(records),
        examples=report_examples,
        extensions=extensions or _default_extensions(mode),
        discussion=discussion,
    )


def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    summary = report.summary
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
