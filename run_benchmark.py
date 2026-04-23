from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import time
from pathlib import Path

import typer
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.runtime import create_runtime
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


def _select_examples(
    dataset_path: str,
    limit: int | None,
    offset: int,
    shuffle: bool,
    seed: int,
):
    examples = load_dataset(dataset_path)
    selected = list(examples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if offset:
        selected = selected[offset:]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    if not selected:
        raise typer.BadParameter("No examples selected after applying offset/limit.")
    return selected


def _resolve_extensions(mode: str, adaptive_attempts: bool, memory_limit: int) -> list[str]:
    extensions = [
        "structured_evaluator",
        "reflection_memory",
        "benchmark_report_json",
    ]
    if mode == "mock":
        extensions.append("mock_mode_for_autograding")
    if adaptive_attempts:
        extensions.append("adaptive_max_attempts")
    if memory_limit > 0:
        extensions.append("memory_compression")
    return extensions


def _run_agent_batch(
    agent_name: str,
    agent,
    examples,
    progress_every: int,
    workers: int,
):
    total = len(examples)
    records = [None] * total
    started_at = time.perf_counter()
    print(
        f"[cyan]Running[/cyan] {agent_name} on {total} examples "
        f"(workers={workers})..."
    )

    if workers <= 1:
        for index, example in enumerate(examples, start=1):
            record = agent.run(example)
            records[index - 1] = record

            should_log = (
                progress_every > 0
                and (index == 1 or index % progress_every == 0 or index == total)
            )
            if should_log:
                elapsed_s = time.perf_counter() - started_at
                print(
                    f"[cyan]{agent_name}[/cyan] {index}/{total} "
                    f"(qid={example.qid}, correct={int(record.is_correct)}, "
                    f"attempts={record.attempts}, elapsed={elapsed_s:.1f}s)"
                )
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=agent_name) as pool:
            future_to_item = {
                pool.submit(agent.run, example): (index, example.qid)
                for index, example in enumerate(examples)
            }
            completed = 0
            for future in as_completed(future_to_item):
                index, qid = future_to_item[future]
                try:
                    record = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"{agent_name} failed on qid={qid}: {exc}"
                    ) from exc

                records[index] = record
                completed += 1

                should_log = (
                    progress_every > 0
                    and (
                        completed == 1
                        or completed % progress_every == 0
                        or completed == total
                    )
                )
                if should_log:
                    elapsed_s = time.perf_counter() - started_at
                    print(
                        f"[cyan]{agent_name}[/cyan] {completed}/{total} "
                        f"(qid={qid}, correct={int(record.is_correct)}, "
                        f"attempts={record.attempts}, elapsed={elapsed_s:.1f}s)"
                    )

    total_elapsed_s = time.perf_counter() - started_at
    print(f"[green]Completed[/green] {agent_name} in {total_elapsed_s:.1f}s")
    return [record for record in records if record is not None]


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
    limit: int | None = None,
    offset: int = 0,
    shuffle: bool = False,
    seed: int = 13,
    adaptive_attempts: bool = True,
    memory_limit: int = 3,
    progress_every: int = 5,
    workers: int = 1,
    timeout_s: int = 60,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> None:
    if workers < 1:
        raise typer.BadParameter("--workers must be at least 1.")
    if timeout_s < 1:
        raise typer.BadParameter("--timeout-s must be at least 1.")
    if max_retries < 0:
        raise typer.BadParameter("--max-retries must be at least 0.")
    if retry_backoff_s < 0:
        raise typer.BadParameter("--retry-backoff-s must be at least 0.")

    examples = _select_examples(
        dataset_path=dataset,
        limit=limit,
        offset=offset,
        shuffle=shuffle,
        seed=seed,
    )
    runtime = create_runtime(
        mode,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(
        max_attempts=reflexion_attempts,
        runtime=runtime,
        adaptive_max_attempts=adaptive_attempts,
        memory_limit=memory_limit,
    )

    print(
        f"Benchmark mode={mode}, dataset={Path(dataset).name}, "
        f"selected_examples={len(examples)}, workers={workers}, "
        f"timeout_s={timeout_s}, max_retries={max_retries}"
    )

    react_records = _run_agent_batch(
        "react",
        react,
        examples,
        progress_every,
        workers,
    )
    reflexion_records = _run_agent_batch(
        "reflexion",
        reflexion,
        examples,
        progress_every,
        workers,
    )
    all_records = react_records + reflexion_records

    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(
        all_records,
        dataset_name=Path(dataset).name,
        mode=mode,
        extensions=_resolve_extensions(mode, adaptive_attempts, memory_limit),
    )
    json_path, md_path = save_report(report, out_path)

    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(f"Selected examples: {len(examples)}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
