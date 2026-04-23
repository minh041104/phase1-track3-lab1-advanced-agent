from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import typer
from rich import print

from src.reflexion_lab.schemas import QAExample

app = typer.Typer(add_completion=False)


def _load_hotpotqa(path: str | Path) -> list[dict[str, Any]]:
    raw_text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".jsonl"):
        return [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    payload = json.loads(raw_text)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported HotpotQA file format.")


def _build_context_chunks(
    record: dict[str, Any],
    supporting_only: bool,
    max_contexts: int,
) -> list[dict[str, str]]:
    supporting_titles = {title for title, _sent_id in record.get("supporting_facts", [])}
    chunks: list[dict[str, str]] = []

    for item in record.get("context", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if supporting_only and title not in supporting_titles:
            continue
        text = " ".join(sentences).strip()
        if not text:
            continue
        chunks.append({"title": title, "text": text})
        if max_contexts > 0 and len(chunks) >= max_contexts:
            break

    return chunks


def convert_hotpotqa_record(
    record: dict[str, Any],
    index: int,
    supporting_only: bool = False,
    max_contexts: int = 0,
) -> dict[str, Any] | None:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    if not question or not answer:
        return None

    chunks = _build_context_chunks(
        record=record,
        supporting_only=supporting_only,
        max_contexts=max_contexts,
    )
    if not chunks:
        return None

    level = str(record.get("level", "medium")).lower()
    difficulty = level if level in {"easy", "medium", "hard"} else "medium"

    payload = {
        "qid": str(record.get("_id") or record.get("id") or f"hotpot_{index}"),
        "difficulty": difficulty,
        "question": question,
        "gold_answer": answer,
        "context": chunks,
    }
    return QAExample.model_validate(payload).model_dump()


def convert_hotpotqa_records(
    records: list[dict[str, Any]],
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 13,
    supporting_only: bool = False,
    max_contexts: int = 0,
) -> list[dict[str, Any]]:
    selected = list(records)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit is not None and limit > 0:
        selected = selected[:limit]

    converted: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        item = convert_hotpotqa_record(
            record,
            index=index,
            supporting_only=supporting_only,
            max_contexts=max_contexts,
        )
        if item is not None:
            converted.append(item)
    return converted


@app.command()
def main(
    source_path: str,
    output_path: str,
    limit: int | None = 100,
    shuffle: bool = True,
    seed: int = 13,
    supporting_only: bool = False,
    max_contexts: int = 0,
) -> None:
    records = _load_hotpotqa(source_path)
    converted = convert_hotpotqa_records(
        records=records,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
        supporting_only=supporting_only,
        max_contexts=max_contexts,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, indent=2), encoding="utf-8")

    print(f"[green]Saved[/green] {output}")
    print(f"Converted examples: {len(converted)}")


if __name__ == "__main__":
    app()
